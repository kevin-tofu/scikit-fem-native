from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy.linalg import qr
from scipy.sparse import bmat, csr_matrix, hstack, isspmatrix_csr

from .basis import DiscreteField
from ._skfn import (
    CrossBilinearAssembler,
    LinearFormAssembler,
    build_triangle_supermesh,
)


def _native_num_threads(value):
    if value is None:
        return 0
    if isinstance(value,bool) or not isinstance(value,int) or value<1:
        raise ValueError("num_threads must be a positive integer")
    from .runtime import available_num_threads
    return min(value,available_num_threads())


class BoundarySurface:
    """Triangulated search geometry backed by parent high-order facet shapes."""

    def __init__(
        self,facet_basis,*,geometry_tolerance=None,max_subdivision_level=5
    ):
        self.basis=facet_basis
        scale=np.linalg.norm(
            facet_basis.mesh.p.max(axis=1)-facet_basis.mesh.p.min(axis=1)
        )
        tolerance=(
            1e-6*max(scale,1.)
            if geometry_tolerance is None else float(geometry_tolerance)
        )
        points=[];triangles=[];parents=[];search_faces=[];point_lookup={}
        reference_tangents=[];self.maximum_subdivision_level=0
        for face_index,(local,element) in enumerate(zip(
            facet_basis.local_faces,facet_basis.parent_elements
        )):
            local=tuple(local);element=int(element)
            is_triangle=len(local) in (3,6)
            corner_count=3 if is_triangle else 4
            reference_corners=facet_basis.elem.elem.doflocs[
                np.asarray(local[:corner_count] if len(local)!=9 else (local[0],local[2],local[8],local[6]))
            ]
            reference_tangents.append(np.stack((
                reference_corners[1]-reference_corners[0],
                reference_corners[-1]-reference_corners[0],
            ),axis=1))
            element_coordinates=facet_basis.mesh.p[:,facet_basis.mesh.t[:,element]]

            def reference(uv):
                r,s=uv
                if is_triangle:
                    return (1-r-s)*reference_corners[0]+r*reference_corners[1]+s*reference_corners[2]
                weights=np.array([(1-r)*(1-s),r*(1-s),r*s,(1-r)*s])
                return weights@reference_corners

            def physical(uv):
                shape,_=facet_basis.volume_basis._evaluate_reference(
                    reference(uv)[:,None]
                )
                return element_coordinates@shape[0]

            def point_index(uv):
                key=(face_index,)+tuple(np.round(uv,14))
                if key not in point_lookup:
                    point_lookup[key]=len(points);points.append(physical(uv))
                return point_lookup[key]

            def subdivide(vertices,level):
                xyz=np.array([physical(vertex) for vertex in vertices])
                mids=np.array([
                    .5*(vertices[0]+vertices[1]),
                    .5*(vertices[1]+vertices[2]),
                    .5*(vertices[2]+vertices[0]),
                ])
                exact=np.array([physical(mid) for mid in mids])
                chord=np.array([
                    .5*(xyz[0]+xyz[1]),.5*(xyz[1]+xyz[2]),.5*(xyz[2]+xyz[0])
                ])
                error=float(np.max(np.linalg.norm(exact-chord,axis=1)))
                if error>tolerance and level<max_subdivision_level:
                    subdivide(np.array([vertices[0],mids[0],mids[2]]),level+1)
                    subdivide(np.array([mids[0],vertices[1],mids[1]]),level+1)
                    subdivide(np.array([mids[2],mids[1],vertices[2]]),level+1)
                    subdivide(np.array([mids[0],mids[1],mids[2]]),level+1)
                    return
                self.maximum_subdivision_level=max(
                    self.maximum_subdivision_level,level
                )
                triangles.append(tuple(point_index(vertex) for vertex in vertices))
                parents.append(element)
                search_faces.append(face_index)

            if is_triangle:
                subdivide(np.array([[0.,0.],[1.,0.],[0.,1.]]),0)
            else:
                subdivide(np.array([[0.,0.],[1.,0.],[1.,1.]]),0)
                subdivide(np.array([[0.,0.],[1.,1.],[0.,1.]]),0)
        self.points=np.asarray(points,dtype=float).T
        self.triangles=np.asarray(triangles,dtype=np.int64).T
        self.parents=np.asarray(parents,dtype=np.int64)
        self.search_faces=np.asarray(search_faces,dtype=np.int64)
        self.reference_tangents=tuple(reference_tangents)
        self.geometry_tolerance=tolerance

    @property
    def components(self):
        return self.basis.elem._dim

    @property
    def local_nodes(self):
        return self.basis.mesh.t.shape[0]

    def element_nodes(self, search_triangle):
        return self.basis.mesh.t[:,self.parents[search_triangle]]

    def evaluate(self, search_triangle, physical_points):
        element=self.parents[search_triangle]
        nodes=self.basis.mesh.t[:,element]
        coordinates=self.basis.mesh.p[:,nodes]
        if coordinates.shape[1] in (4,10):
            guess=np.full((len(physical_points),3),.25)
        else:
            guess=np.full((len(physical_points),3),.5)
        volume=self.basis.volume_basis
        for point_index,point in enumerate(physical_points):
            xi=guess[point_index]
            for _ in range(15):
                shape,reference_gradient=volume._evaluate_reference(xi[:,None])
                residual=coordinates@shape[0]-point
                jacobian=coordinates@reference_gradient[0]
                increment=np.linalg.solve(jacobian,residual)
                xi-=increment
                if np.linalg.norm(increment)<1e-12:
                    break
            else:
                raise ValueError("isoparametric inverse mapping did not converge")
            guess[point_index]=xi
        values=np.empty((len(physical_points),coordinates.shape[1]))
        gradients=np.empty((len(physical_points),coordinates.shape[1],3))
        normals=np.empty((len(physical_points),3))
        reference_tangents=self.reference_tangents[
            self.search_faces[search_triangle]
        ]
        centroid=coordinates.mean(axis=1)
        for q,xi in enumerate(guess):
            shape,reference_gradient=volume._evaluate_reference(xi[:,None])
            jacobian=coordinates@reference_gradient[0]
            values[q]=shape[0]
            gradients[q]=reference_gradient[0]@np.linalg.inv(jacobian)
            tangents=jacobian@reference_tangents
            normal=np.cross(tangents[:,0],tangents[:,1])
            normal/=np.linalg.norm(normal)
            point=coordinates@shape[0]
            if np.dot(normal,point-centroid)<0.:
                normal=-normal
            normals[q]=normal
        return values,gradients,normals


def _cross2(a, b):
    return a[0]*b[1]-a[1]*b[0]


def _clip(subject, clip_triangle, tolerance):
    polygon=[np.asarray(point) for point in subject]
    orientation=np.sign(sum(
        _cross2(clip_triangle[(i+1)%3]-clip_triangle[i],
                clip_triangle[(i+2)%3]-clip_triangle[(i+1)%3])
        for i in range(3)
    )) or 1.
    for i in range(3):
        a,b=clip_triangle[i],clip_triangle[(i+1)%3];edge=b-a
        output=[];previous=polygon[-1] if polygon else None
        for current in polygon:
            current_inside=orientation*_cross2(edge,current-a)>=-tolerance
            previous_inside=orientation*_cross2(edge,previous-a)>=-tolerance
            if current_inside != previous_inside:
                segment=current-previous
                denominator=_cross2(segment,edge)
                if abs(denominator)>tolerance:
                    parameter=_cross2(a-previous,edge)/denominator
                    output.append(previous+parameter*segment)
            if current_inside:
                output.append(current)
            previous=current
        polygon=output
        if not polygon:
            break
    return polygon


def _barycentric(point, triangle):
    matrix=np.column_stack((triangle[1]-triangle[0],triangle[2]-triangle[0]))
    coordinates=np.linalg.solve(matrix,point-triangle[0])
    return np.array([1.-coordinates.sum(),coordinates[0],coordinates[1]])


@dataclass(frozen=True)
class SupermeshDiagnostics:
    total_pair_count: int
    candidate_pair_count: int
    overlap_pair_count: int
    integration_triangle_count: int
    overlap_area: float
    noncoplanar_rejection_count: int
    maximum_plane_gap: float
    master_search_triangle_count: int = 0
    slave_search_triangle_count: int = 0
    maximum_subdivision_level: int = 0
    created_overlap_pair_count: int = 0
    disappeared_overlap_pair_count: int = 0
    pattern_reused: bool = False
    update_count: int = 0
    orientation_mismatch_count: int = 0
    maximum_normal_opposition_error: float = 0.0


@dataclass(frozen=True)
class ContactFacetSearchResult:
    """Parent facets participating in piecewise-3D projected overlap."""

    master_parent_facets: np.ndarray
    slave_parent_facets: np.ndarray
    projection_tolerance: float
    candidate_pair_count: int
    overlap_pair_count: int
    maximum_plane_gap: float


def contact_projection_tolerance(
    master_points, master_triangles, slave_points, slave_triangles,
    *, tolerance=1e-10, relative_tolerance=0.05,
):
    """Choose a gap tolerance from the median contact-facet edge length."""

    lengths=[]
    for points,triangles in (
        (master_points,master_triangles),(slave_points,slave_triangles)
    ):
        points=np.asarray(points,dtype=float)
        triangles=np.asarray(triangles,dtype=np.int64)
        if points.shape[0]!=3: points=points.T
        if triangles.shape[0]!=3: triangles=triangles.T
        xyz=np.transpose(points[:,triangles],(2,1,0))
        for first,second in ((0,1),(1,2),(2,0)):
            lengths.extend(np.linalg.norm(
                xyz[:,second]-xyz[:,first],axis=1
            ).tolist())
    positive=np.asarray(lengths,dtype=float)
    positive=positive[positive>float(tolerance)]
    characteristic=(float(np.median(positive)) if positive.size else 0.)
    return max(float(tolerance),float(relative_tolerance)*characteristic)


def find_contact_facets(
    master_points, master_triangles, slave_points, slave_triangles,
    *, tolerance=1e-10, projection_tolerance=None,
    relative_projection_tolerance=0.05, num_threads=None,
):
    """Find overlapping parent facets using the native piecewise-3D search."""

    projection=(
        contact_projection_tolerance(
            master_points,master_triangles,slave_points,slave_triangles,
            tolerance=tolerance,
            relative_tolerance=relative_projection_tolerance,
        )
        if projection_tolerance is None else float(projection_tolerance)
    )
    built=build_triangle_supermesh(
        np.ascontiguousarray(master_points,dtype=np.float64),
        np.ascontiguousarray(master_triangles,dtype=np.int64),
        np.ascontiguousarray(slave_points,dtype=np.float64),
        np.ascontiguousarray(slave_triangles,dtype=np.int64),
        float(tolerance),projection,_native_num_threads(num_threads),
    )
    return ContactFacetSearchResult(
        np.unique(np.asarray(built["master_indices"],dtype=np.int64)),
        np.unique(np.asarray(built["slave_indices"],dtype=np.int64)),
        projection,int(built["candidate_count"]),
        int(built["overlap_count"]),float(built["maximum_plane_gap"]),
    )


def _aabb_candidates(master_xyz,slave_xyz,tolerance):
    """Yield overlapping triangle AABB pairs without formulation knowledge."""
    master_xyz=np.asarray(master_xyz,dtype=np.float64)
    slave_xyz=np.asarray(slave_xyz,dtype=np.float64)
    master_min=master_xyz.min(axis=1)-float(tolerance)
    master_max=master_xyz.max(axis=1)+float(tolerance)
    slave_min=slave_xyz.min(axis=1)-float(tolerance)
    slave_max=slave_xyz.max(axis=1)+float(tolerance)
    for master_index,(lower,upper) in enumerate(zip(master_min,master_max)):
        selected=np.flatnonzero(np.all(
            (slave_max>=lower)&(slave_min<=upper),axis=1
        ))
        for slave_index in selected:
            yield master_index,int(slave_index)


@dataclass(frozen=True)
class MortarTraceData:
    """One side of an interface evaluated at shared physical quadrature points."""

    shape_values: np.ndarray
    physical_gradients: np.ndarray | None
    outward_normals: np.ndarray
    quadrature_weights: np.ndarray
    parent_facets: np.ndarray
    parent_elements: np.ndarray
    dofs: np.ndarray
    coordinates: np.ndarray


@dataclass(frozen=True)
class CrossTabulation:
    """Formulation-neutral test-space tabulation on shared quadrature points."""

    dofs: np.ndarray
    shape_values: np.ndarray
    size: int

    def __post_init__(self):
        dofs=np.asarray(self.dofs,dtype=np.int64)
        shape=np.asarray(self.shape_values,dtype=np.float64)
        if dofs.ndim!=3 or shape.ndim!=3:
            raise ValueError("cross tabulation arrays must have three axes")
        if dofs.shape[0]!=shape.shape[0]:
            raise ValueError("cross tabulation entity counts differ")
        if dofs.shape[1]!=shape.shape[2]:
            raise ValueError("cross tabulation local basis sizes differ")
        if int(self.size)<0:
            raise ValueError("cross tabulation size must be non-negative")
        object.__setattr__(self,"dofs",dofs)
        object.__setattr__(self,"shape_values",shape)


@dataclass(frozen=True)
class SupermeshSearch:
    """Reusable planar triangle topology and overlap integration state."""

    def __init__(
        self,master_triangles,slave_triangles,*,components=1,
        tolerance=1e-10,projection_tolerance=None,num_threads=None,
    ):
        master_triangles=np.asarray(master_triangles,dtype=np.int64)
        slave_triangles=np.asarray(slave_triangles,dtype=np.int64)
        if master_triangles.shape[0]!=3:
            master_triangles=master_triangles.T
        if slave_triangles.shape[0]!=3:
            slave_triangles=slave_triangles.T
        self.master_triangles=np.ascontiguousarray(master_triangles)
        self.slave_triangles=np.ascontiguousarray(slave_triangles)
        self.components=components
        self.tolerance=float(tolerance)
        self.projection_tolerance=projection_tolerance
        self.num_threads=num_threads
        self.integration=None

    def build(self,master_points,slave_points):
        self.integration=TriangleSupermesh(
            master_points,self.master_triangles,
            slave_points,self.slave_triangles,
            components=self.components,tolerance=self.tolerance,
            projection_tolerance=self.projection_tolerance,
            num_threads=self.num_threads,
        )
        return self.integration

    def update(self,master_points,slave_points,*,num_threads=None):
        if self.integration is None:
            return self.build(master_points,slave_points)
        return self.integration.update(
            master_points,slave_points,
            num_threads=self.num_threads if num_threads is None else num_threads,
        )


class TriangleSupermesh:
    """Reusable P1 coupling quadrature for coplanar nonmatching triangles."""

    def __init__(
        self, master_points, master_triangles, slave_points, slave_triangles,
        *, components=1, tolerance=1e-10, projection_tolerance=None,
        num_threads=None,
    ):
        self._initialize(
            master_points,master_triangles,slave_points,slave_triangles,
            components=components,tolerance=tolerance,
            projection_tolerance=projection_tolerance,
            num_threads=num_threads,
        )

    def _initialize(
        self,master_points,master_triangles,slave_points,slave_triangles,
        *,components,tolerance,projection_tolerance,
        master_surface=None,slave_surface=None,num_threads=None,
    ):
        if np.isscalar(components):
            row_components=column_components=int(components)
        else:
            row_components,column_components=map(int,components)
        master_points=np.asarray(master_points,dtype=float)
        slave_points=np.asarray(slave_points,dtype=float)
        master_triangles=np.asarray(master_triangles,dtype=np.int64)
        slave_triangles=np.asarray(slave_triangles,dtype=np.int64)
        if master_points.shape[0]!=3: master_points=master_points.T
        if slave_points.shape[0]!=3: slave_points=slave_points.T
        if master_triangles.shape[0]!=3: master_triangles=master_triangles.T
        if slave_triangles.shape[0]!=3: slave_triangles=slave_triangles.T
        row_dofs=[];column_dofs=[];row_shape=[];column_shape=[];weights=[]
        row_gradients=[];column_gradients=[]
        row_normal_gradient=[];column_normal_gradient=[]
        quadrature_coordinates=[];master_normals=[];slave_normals=[];gaps=[]
        master_parent_facets=[];slave_parent_facets=[]
        master_parent_elements=[];slave_parent_elements=[]
        orientation_errors=[]
        candidates=0;overlaps=0;area_total=0.;noncoplanar=0;maximum_gap=0.
        projection_tolerance=(
            tolerance if projection_tolerance is None
            else float(projection_tolerance)
        )
        if master_surface is None and slave_surface is None:
            return self._initialize_planar_native(
                master_points,master_triangles,
                slave_points,slave_triangles,
                row_components=row_components,
                column_components=column_components,
                tolerance=tolerance,
                projection_tolerance=projection_tolerance,
                num_threads=num_threads,
            )
        a=.445948490915965;b=.091576213509771
        quadrature_bary=np.array([
            [a,a,1-2*a],[a,1-2*a,a],[1-2*a,a,a],
            [b,b,1-2*b],[b,1-2*b,b],[1-2*b,b,b],
        ])
        quadrature_weights=np.array(
            [.223381589678011]*3+[.109951743655322]*3
        )
        master_xyz_all=master_points[:,master_triangles].transpose(2,1,0)
        slave_xyz_all=slave_points[:,slave_triangles].transpose(2,1,0)
        for master_index,slave_index in _aabb_candidates(
            master_xyz_all,slave_xyz_all,max(tolerance,projection_tolerance)
        ):
            master_nodes=master_triangles[:,master_index]
            slave_nodes=slave_triangles[:,slave_index]
            master_xyz=master_points[:,master_nodes].T
            tangent0=master_xyz[1]-master_xyz[0]
            normal=np.cross(tangent0,master_xyz[2]-master_xyz[0])
            norm=np.linalg.norm(normal)
            if norm<=tolerance: continue
            normal/=norm;tangent0/=np.linalg.norm(tangent0)
            tangent1=np.cross(normal,tangent0)
            project=lambda xyz:np.column_stack(((xyz-master_xyz[0])@tangent0,
                                                (xyz-master_xyz[0])@tangent1))
            master_2d=project(master_xyz)
            slave_xyz=slave_points[:,slave_nodes].T;candidates+=1
            distances=(slave_xyz-master_xyz[0])@normal
            gap=float(np.max(np.abs(distances)));maximum_gap=max(maximum_gap,gap)
            if gap>projection_tolerance:
                noncoplanar+=1;continue
            slave_2d=project(slave_xyz)
            polygon=_clip(master_2d,slave_2d,tolerance)
            if len(polygon)<3: continue
            pair_has_area=False
            for i in range(1,len(polygon)-1):
                triangle=np.array([polygon[0],polygon[i],polygon[i+1]])
                area=abs(_cross2(triangle[1]-triangle[0],triangle[2]-triangle[0]))/2.
                if area<=tolerance: continue
                pair_has_area=True;area_total+=area
                master_values=[];slave_values=[];physical_points=[]
                for bary in quadrature_bary:
                    point=bary@triangle
                    master_values.append(_barycentric(point,master_2d))
                    slave_values.append(_barycentric(point,slave_2d))
                    physical_points.append(
                        master_xyz[0]+point[0]*tangent0+point[1]*tangent1
                    )
                physical_points=np.asarray(physical_points)
                slave_search_values=np.asarray(slave_values)
                slave_search_normal=np.cross(
                    slave_xyz[1]-slave_xyz[0],slave_xyz[2]-slave_xyz[0]
                )
                slave_search_normal/=np.linalg.norm(slave_search_normal)
                if np.dot(slave_search_normal,normal)>0:
                    slave_search_normal=-slave_search_normal
                if master_surface is not None:
                    master_values,master_gradients,master_q_normals=master_surface.evaluate(
                        master_index,np.asarray(physical_points)
                    )
                    master_element_nodes=master_surface.element_nodes(master_index)
                else:
                    master_element_nodes=master_nodes
                if slave_surface is not None:
                    slave_values,slave_gradients,slave_q_normals=slave_surface.evaluate(
                        slave_index,np.asarray(physical_points)
                    )
                    slave_element_nodes=slave_surface.element_nodes(slave_index)
                else:
                    slave_element_nodes=slave_nodes
                row_shape.append(master_values);column_shape.append(slave_values)
                quadrature_coordinates.append(physical_points)
                if master_surface is None:
                    master_q_normals=np.broadcast_to(
                        normal,physical_points.shape
                    )
                    slave_q_normals=np.broadcast_to(
                        slave_search_normal,physical_points.shape
                    )
                master_normals.append(master_q_normals)
                opposition=np.linalg.norm(master_q_normals+slave_q_normals,axis=1)
                orientation_errors.extend(opposition)
                # A mortar interface has one geometric normal convention.  Keep
                # the independently evaluated outward normal in the diagnostic,
                # then expose an exactly opposing pair to flux kernels.
                slave_q_normals=-master_q_normals
                slave_normals.append(slave_q_normals)
                master_parent_facets.append(
                    master_surface.search_faces[master_index]
                    if master_surface is not None else master_index
                )
                slave_parent_facets.append(
                    slave_surface.search_faces[slave_index]
                    if slave_surface is not None else slave_index
                )
                master_parent_elements.append(
                    master_surface.parents[master_index]
                    if master_surface is not None else master_index
                )
                slave_parent_elements.append(
                    slave_surface.parents[slave_index]
                    if slave_surface is not None else slave_index
                )
                slave_physical=slave_search_values@slave_xyz
                gaps.append(np.einsum(
                    "qi,qi->q",slave_physical-physical_points,
                    master_q_normals,
                ))
                if master_surface is not None:
                    row_gradients.append(master_gradients)
                    column_gradients.append(slave_gradients)
                    row_normal_gradient.append(
                        np.einsum(
                            "qni,qi->qn",master_gradients,master_q_normals
                        )
                    )
                    column_normal_gradient.append(
                        np.einsum(
                            "qni,qi->qn",slave_gradients,slave_q_normals
                        )
                    )
                weights.append(area*quadrature_weights)
                row_dofs.append([[row_components*node+c for c in range(row_components)] for node in master_element_nodes])
                column_dofs.append([[column_components*node+c for c in range(column_components)] for node in slave_element_nodes])
            overlaps+=int(pair_has_area)
        if not row_dofs:
            raise ValueError("triangle surfaces have no positive-area overlap")
        self._row_gradients=(
            np.asarray(row_gradients,dtype=np.float64)
            if row_gradients else None
        )
        self._column_gradients=(
            np.asarray(column_gradients,dtype=np.float64)
            if column_gradients else None
        )
        self._row_physical_gradients=self._row_gradients
        self._column_physical_gradients=self._column_gradients
        self._native=CrossBilinearAssembler(
            np.asarray(row_dofs,dtype=np.int64),
            np.asarray(column_dofs,dtype=np.int64),
            np.asarray(row_shape,dtype=np.float64),
            np.asarray(column_shape,dtype=np.float64),
            np.asarray(weights,dtype=np.float64),
            self._row_gradients,
            self._column_gradients,
        )
        self._matrix=csr_matrix(
            (self._native.values,self._native.indices,self._native.indptr),
            shape=(self._native.rows,self._native.columns),copy=False,
        )
        self._coefficient_shape=np.asarray(weights).shape
        self._row_dofs=np.asarray(row_dofs,dtype=np.int64)
        self._column_dofs=np.asarray(column_dofs,dtype=np.int64)
        self._row_shape=np.asarray(row_shape,dtype=np.float64)
        self._column_shape=np.asarray(column_shape,dtype=np.float64)
        self._weights=np.asarray(weights,dtype=np.float64)
        self.global_coordinates=np.asarray(
            quadrature_coordinates,dtype=np.float64
        )
        self.master_normals=np.asarray(master_normals,dtype=np.float64)
        self.slave_normals=np.asarray(slave_normals,dtype=np.float64)
        self.gap=np.asarray(gaps,dtype=np.float64)
        self._master_parent_facets=np.asarray(master_parent_facets,dtype=np.int64)
        self._slave_parent_facets=np.asarray(slave_parent_facets,dtype=np.int64)
        self._master_parent_elements=np.asarray(master_parent_elements,dtype=np.int64)
        self._slave_parent_elements=np.asarray(slave_parent_elements,dtype=np.int64)
        self._row_normal_gradient=(
            np.asarray(row_normal_gradient,dtype=np.float64)
            if row_normal_gradient else None
        )
        self._column_normal_gradient=(
            np.asarray(column_normal_gradient,dtype=np.float64)
            if column_normal_gradient else None
        )
        self.master_size=self._native.rows
        self.slave_size=self._native.columns
        orientation_error=float(np.max(orientation_errors,initial=0.))
        self.diagnostics=SupermeshDiagnostics(
            master_triangles.shape[1]*slave_triangles.shape[1],
            candidates,overlaps,len(weights),area_total,
            noncoplanar,maximum_gap,
            orientation_mismatch_count=int(np.count_nonzero(
                np.asarray(orientation_errors)>1e-8
            )),
            maximum_normal_opposition_error=orientation_error,
        )

    def _initialize_planar_native(
        self,master_points,master_triangles,
        slave_points,slave_triangles,*,row_components,
        column_components,tolerance,projection_tolerance,num_threads=None,
    ):
        old_native=getattr(self,"_native",None)
        old_matrix=getattr(self,"_matrix",None)
        old_row_dofs=getattr(self,"_row_dofs",None)
        old_column_dofs=getattr(self,"_column_dofs",None)
        old_pairs=getattr(self,"_pair_set",set())
        update_count=getattr(self,"_update_count",0)
        built=build_triangle_supermesh(
            np.ascontiguousarray(master_points,dtype=np.float64),
            np.ascontiguousarray(master_triangles,dtype=np.int64),
            np.ascontiguousarray(slave_points,dtype=np.float64),
            np.ascontiguousarray(slave_triangles,dtype=np.int64),
            tolerance,projection_tolerance,
            _native_num_threads(num_threads),
        )
        master_indices=np.asarray(built["master_indices"])
        slave_indices=np.asarray(built["slave_indices"])
        master_nodes=master_triangles[:,master_indices].T
        slave_nodes=slave_triangles[:,slave_indices].T
        row_dofs=np.ascontiguousarray(
            row_components*master_nodes[...,None]
            +np.arange(row_components)
        )
        column_dofs=np.ascontiguousarray(
            column_components*slave_nodes[...,None]
            +np.arange(column_components)
        )
        row_shape=np.asarray(
            built["row_shape"],dtype=np.float64
        )
        column_shape=np.asarray(
            built["column_shape"],dtype=np.float64
        )
        weights=np.asarray(built["weights"],dtype=np.float64)
        pattern_reused=(
            old_native is not None
            and old_row_dofs.shape==row_dofs.shape
            and old_column_dofs.shape==column_dofs.shape
            and np.array_equal(old_row_dofs,row_dofs)
            and np.array_equal(old_column_dofs,column_dofs)
        )
        if pattern_reused:
            old_native.update_tabulation(
                row_shape,column_shape,weights
            )
            native=old_native
        else:
            native=CrossBilinearAssembler(
                row_dofs,column_dofs,row_shape,column_shape,weights,
            )
        pair_set=set(zip(
            map(int,master_indices),map(int,slave_indices)
        ))
        created=len(pair_set-old_pairs) if update_count else 0
        disappeared=len(old_pairs-pair_set) if update_count else 0
        self._row_dofs=row_dofs
        self._column_dofs=column_dofs
        self._row_shape=row_shape
        self._column_shape=column_shape
        self._weights=weights
        def triangle_gradients(points,triangles,indices):
            xyz=points[:,triangles[:,indices]].transpose(2,1,0)
            edges=np.stack((
                xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0]
            ),axis=2)
            metric=np.einsum("edi,edj->eij",edges,edges)
            reference_gradients=np.einsum(
                "edi,eij->edj",edges,np.linalg.inv(metric)
            )
            gradients=np.stack((
                -reference_gradients.sum(axis=2),
                reference_gradients[:,:,0],reference_gradients[:,:,1],
            ),axis=1)
            return np.broadcast_to(
                gradients[:,None,:,:],
                (len(indices),weights.shape[1],3,3),
            )
        self._row_physical_gradients=triangle_gradients(
            master_points,master_triangles,master_indices
        )
        self._column_physical_gradients=triangle_gradients(
            slave_points,slave_triangles,slave_indices
        )
        self._row_gradients=None
        self._column_gradients=None
        self._row_normal_gradient=None
        self._column_normal_gradient=None
        self.global_coordinates=np.asarray(
            built["coordinates"],dtype=np.float64
        )
        self.master_normals=np.asarray(
            built["master_normals"],dtype=np.float64
        )
        raw_slave_normals=np.asarray(
            built["slave_normals"],dtype=np.float64
        )
        orientation_errors=np.linalg.norm(
            self.master_normals+raw_slave_normals,axis=2
        )
        self.slave_normals=-self.master_normals
        self.gap=np.asarray(built["gaps"],dtype=np.float64)
        self._master_parent_facets=np.asarray(master_indices,dtype=np.int64)
        self._slave_parent_facets=np.asarray(slave_indices,dtype=np.int64)
        self._master_parent_elements=np.asarray(master_indices,dtype=np.int64)
        self._slave_parent_elements=np.asarray(slave_indices,dtype=np.int64)
        self._native=native
        self._matrix=(
            old_matrix if pattern_reused else
            csr_matrix(
                (
                    self._native.values,
                    self._native.indices,
                    self._native.indptr,
                ),
                shape=(self._native.rows,self._native.columns),
                copy=False,
            )
        )
        self._coefficient_shape=self._weights.shape
        self.master_size=row_components*master_points.shape[1]
        self.slave_size=column_components*slave_points.shape[1]
        self._matrix.resize((self.master_size,self.slave_size))
        self._master_points=np.array(master_points,copy=True)
        self._slave_points=np.array(slave_points,copy=True)
        self._master_triangles=np.array(master_triangles,copy=True)
        self._slave_triangles=np.array(slave_triangles,copy=True)
        self._components=(row_components,column_components)
        self._tolerance=float(tolerance)
        self._projection_tolerance=float(projection_tolerance)
        self._build_num_threads=num_threads
        self._pair_set=pair_set
        self._update_count=update_count
        self.diagnostics=SupermeshDiagnostics(
            master_triangles.shape[1]*slave_triangles.shape[1],
            int(built["candidate_count"]),
            int(built["overlap_count"]),
            int(built["integration_triangle_count"]),
            float(built["overlap_area"]),
            int(built["noncoplanar_rejection_count"]),
            float(built["maximum_plane_gap"]),
            created_overlap_pair_count=created,
            disappeared_overlap_pair_count=disappeared,
            pattern_reused=pattern_reused,
            update_count=update_count,
            orientation_mismatch_count=int(np.count_nonzero(
                orientation_errors>1e-8
            )),
            maximum_normal_opposition_error=float(
                np.max(orientation_errors,initial=0.)
            ),
        )

    def update(self,master_points,slave_points,*,num_threads=None):
        """Update planar coordinates and reuse the CSR pattern when possible."""
        if not hasattr(self,"_master_triangles"):
            raise NotImplementedError(
                "update() currently supports direct planar triangle "
                "supermeshes"
            )
        master_points=np.asarray(master_points,dtype=np.float64)
        slave_points=np.asarray(slave_points,dtype=np.float64)
        if master_points.shape[0]!=3:
            master_points=master_points.T
        if slave_points.shape[0]!=3:
            slave_points=slave_points.T
        if master_points.shape!=self._master_points.shape:
            raise ValueError("master point topology cannot change")
        if slave_points.shape!=self._slave_points.shape:
            raise ValueError("slave point topology cannot change")
        self._update_count+=1
        try:
            self._initialize_planar_native(
                master_points,self._master_triangles,
                slave_points,self._slave_triangles,
                row_components=self._components[0],
                column_components=self._components[1],
                tolerance=self._tolerance,
                projection_tolerance=self._projection_tolerance,
                num_threads=(
                    self._build_num_threads
                    if num_threads is None else num_threads
                ),
            )
        except Exception:
            self._update_count-=1
            raise
        return self

    @classmethod
    def from_facets(
        cls, master_basis, slave_basis, *, tolerance=1e-10,
        projection_tolerance=None,geometry_tolerance=None,
        max_subdivision_level=5,
    ):
        master=BoundarySurface(
            master_basis,geometry_tolerance=geometry_tolerance,
            max_subdivision_level=max_subdivision_level,
        )
        slave=BoundarySurface(
            slave_basis,geometry_tolerance=geometry_tolerance,
            max_subdivision_level=max_subdivision_level,
        )
        return cls._from_surfaces(
            master,slave,tolerance=tolerance,
            projection_tolerance=projection_tolerance,
        )

    @classmethod
    def _from_surfaces(
        cls,master,slave,*,tolerance,projection_tolerance,
    ):
        instance=cls.__new__(cls)
        instance._initialize(
            master.points,master.triangles,slave.points,slave.triangles,
            components=(master.components,slave.components),tolerance=tolerance,
            projection_tolerance=projection_tolerance,
            master_surface=master,slave_surface=slave,
        )
        instance.master_size=master.basis.N
        instance.slave_size=slave.basis.N
        old=instance.diagnostics
        instance.diagnostics=SupermeshDiagnostics(
            old.total_pair_count,old.candidate_pair_count,
            old.overlap_pair_count,old.integration_triangle_count,
            old.overlap_area,old.noncoplanar_rejection_count,
            old.maximum_plane_gap,
            master.triangles.shape[1],slave.triangles.shape[1],
            max(master.maximum_subdivision_level,slave.maximum_subdivision_level),
            orientation_mismatch_count=old.orientation_mismatch_count,
            maximum_normal_opposition_error=old.maximum_normal_opposition_error,
        )
        return instance

    def assemble(self, coefficient=1., *, num_threads=None):
        coefficient=np.ascontiguousarray(np.broadcast_to(
            np.asarray(coefficient,dtype=np.float64),self._coefficient_shape
        ))
        self._native.assemble(
            coefficient,"value","value",_native_num_threads(num_threads)
        )
        return self._matrix

    def trace_data(self,side):
        """Return immutable-view data for a side at the shared mortar points."""
        if side not in {"master","slave"}:
            raise ValueError("side must be master or slave")
        master=side=="master"
        return MortarTraceData(
            self._row_shape if master else self._column_shape,
            self._row_physical_gradients if master
            else self._column_physical_gradients,
            self.master_normals if master else self.slave_normals,
            self._weights,
            self._master_parent_facets if master else self._slave_parent_facets,
            self._master_parent_elements if master else self._slave_parent_elements,
            self._row_dofs if master else self._column_dofs,
            self.global_coordinates,
        )

    @property
    def master_trace(self):
        return self.trace_data("master")

    @property
    def slave_trace(self):
        return self.trace_data("slave")

    def assemble_cross_tabulation(
        self,test: CrossTabulation,trial: CrossTabulation,*,num_threads=None,
    ):
        """Assemble one rectangular value-value block from supplied tables.

        This operation has no knowledge of Mortar spaces or contact laws.  The
        caller owns the test-space construction and supplies its evaluated
        shape functions and global row numbering.
        """
        if not isinstance(test,CrossTabulation):
            raise TypeError("test must be CrossTabulation")
        if not isinstance(trial,CrossTabulation):
            raise TypeError("trial must be CrossTabulation")
        if test.dofs.shape[0]!=trial.dofs.shape[0]:
            raise ValueError("test and trial entity counts differ")
        if test.shape_values.shape[:2]!=trial.shape_values.shape[:2]:
            raise ValueError("test and trial quadrature shapes differ")
        native=CrossBilinearAssembler(
            np.ascontiguousarray(test.dofs,dtype=np.int64),
            np.ascontiguousarray(trial.dofs,dtype=np.int64),
            np.ascontiguousarray(test.shape_values,dtype=np.float64),
            np.ascontiguousarray(trial.shape_values,dtype=np.float64),
            np.ascontiguousarray(self._weights,dtype=np.float64),
        )
        native.assemble(
            np.ones(self._coefficient_shape,dtype=np.float64),
            "value","value",_native_num_threads(num_threads),
        )
        matrix=csr_matrix(
            (native.values,native.indices,native.indptr),
            shape=(native.rows,native.columns),copy=True,
        )
        matrix.resize((int(test.size),int(trial.size)))
        return matrix

    def interpolate(self,master_coefficients,slave_coefficients):
        """Interpolate both interface fields at overlap quadrature points."""
        return (
            self._interpolate_side("master",master_coefficients),
            self._interpolate_side("slave",slave_coefficients),
        )

    def _interpolate_side(self,side,coefficients):
        dofs=self._row_dofs if side=="master" else self._column_dofs
        shape=self._row_shape if side=="master" else self._column_shape
        gradients=(
            self._row_gradients
            if side=="master" else self._column_gradients
        )
        expected=self.master_size if side=="master" else self.slave_size
        coefficients=np.asarray(coefficients,dtype=np.float64)
        if coefficients.shape!=(expected,):
            raise ValueError(
                f"{side} coefficients must have shape ({expected},)"
            )
        local=coefficients[dofs]
        value=np.einsum("eqn,enc->ceq",shape,local)
        gradient=(
            None if gradients is None else
            np.einsum("eqnd,enc->cdeq",gradients,local)
        )
        if dofs.shape[2]==1:
            value=value[0]
            if gradient is not None:
                gradient=gradient[0]
        return DiscreteField(value,gradient)

    def assemble_tensor(self, coefficient, *, num_threads=None):
        """Assemble master-by-slave coupling with a component tensor."""
        coefficient=np.asarray(coefficient,dtype=np.float64)
        target=self._coefficient_shape+(
            self._row_dofs.shape[2],self._column_dofs.shape[2]
        )
        coefficient=np.ascontiguousarray(np.broadcast_to(coefficient,target))
        self._native.assemble(
            coefficient,"value","value",_native_num_threads(num_threads)
        )
        self._matrix.resize((self.master_size,self.slave_size))
        return self._matrix

    def assemble_cross(
        self, coefficient=1., *, row_kind="value", column_kind="value",
        num_threads=None,
    ):
        """Assemble a value/gradient master-by-slave contraction.

        Tensor axes after the reusable entity/quadrature axes are
        ``(row component, [row direction], column component,
        [column direction])``.
        """
        valid={"value","gradient"}
        if row_kind not in valid or column_kind not in valid:
            raise ValueError("cross basis kind must be value or gradient")
        row_gradient=row_kind=="gradient"
        column_gradient=column_kind=="gradient"
        if row_gradient and self._row_gradients is None:
            raise ValueError(
                "gradient coupling requires a supermesh created from FacetBasis"
            )
        if column_gradient and self._column_gradients is None:
            raise ValueError(
                "gradient coupling requires a supermesh created from FacetBasis"
            )
        coefficient=np.asarray(coefficient,dtype=np.float64)
        tensor_axes=[
            self._row_dofs.shape[2],
            *((self._row_gradients.shape[3],) if row_gradient else ()),
            self._column_dofs.shape[2],
            *((self._column_gradients.shape[3],) if column_gradient else ()),
        ]
        scalar=(
            row_gradient==column_gradient
            and self._row_dofs.shape[2]==self._column_dofs.shape[2]
            and coefficient.ndim<=2
        )
        if scalar:
            target=self._coefficient_shape
        else:
            target=self._coefficient_shape+tuple(tensor_axes)
        coefficient=np.ascontiguousarray(np.broadcast_to(coefficient,target))
        self._native.assemble(
            coefficient,row_kind,column_kind,_native_num_threads(num_threads)
        )
        self._matrix.resize((self.master_size,self.slave_size))
        return self._matrix

    def assemble_traces(
        self, row_weights, column_weights, *,
        row_kind="value", column_kind="value", coefficient=1.,
        num_threads=None,
    ):
        """Assemble a 2-by-2 interface block from arbitrary trace weights."""
        valid={"value","gradient","normal_gradient"}
        if row_kind not in valid or column_kind not in valid:
            raise ValueError(
                "trace kind must be value, gradient, or normal_gradient"
            )
        if "gradient" in (row_kind,column_kind) and (
            self._row_gradients is None or self._column_gradients is None
        ):
            raise ValueError(
                "full gradients require a supermesh created from FacetBasis"
            )
        blocks=[
            [self._cross("mm",row_kind,column_kind,coefficient,self.master_size,self.master_size,num_threads),
             self._cross("ms",row_kind,column_kind,coefficient,self.master_size,self.slave_size,num_threads)],
            [self._cross("sm",row_kind,column_kind,coefficient,self.slave_size,self.master_size,num_threads),
             self._cross("ss",row_kind,column_kind,coefficient,self.slave_size,self.slave_size,num_threads)],
        ]
        rw=np.asarray(row_weights,dtype=float);cw=np.asarray(column_weights,dtype=float)
        return bmat([
            [rw[0]*cw[0]*blocks[0][0],rw[0]*cw[1]*blocks[0][1]],
            [rw[1]*cw[0]*blocks[1][0],rw[1]*cw[1]*blocks[1][1]],
        ],format="csr")

    def assemble_linear_trace(
        self, trace_weights, *, trace_kind="value", coefficient=1.,
        num_threads=None,
    ):
        """Assemble master/slave interface test traces into one vector."""
        if trace_kind not in {"value","gradient","normal_gradient"}:
            raise ValueError(
                "trace kind must be value, gradient, or normal_gradient"
            )
        weights=np.asarray(trace_weights,dtype=float)
        if weights.shape!=(2,):
            raise ValueError("interface trace weights must contain two values")
        master=self._linear_trace(
            "master",trace_kind,coefficient,num_threads
        )
        slave=self._linear_trace(
            "slave",trace_kind,coefficient,num_threads
        )
        result=np.zeros(self.master_size+self.slave_size,dtype=np.float64)
        result[:len(master)]=weights[0]*master
        result[self.master_size:self.master_size+len(slave)]=weights[1]*slave
        return result

    def _linear_trace(self,side,kind,coefficient,num_threads=None):
        cache=getattr(self,"_linear_trace_assemblers",None)
        if cache is None:
            cache={};self._linear_trace_assemblers=cache
        native=cache.get((side,kind))
        dofs=self._row_dofs if side=="master" else self._column_dofs
        shape=self._trace(side,"value")
        gradients=(
            self._row_gradients if side=="master"
            else self._column_gradients
        )
        native_kind=kind
        if kind=="normal_gradient":
            shape=self._trace(side,kind)
            native_kind="value"
        if gradients is None:
            gradients=np.zeros(shape.shape+(3,),dtype=np.float64)
        if native is None:
            native=LinearFormAssembler(
                dofs,np.ascontiguousarray(shape),
                np.ascontiguousarray(gradients),self._weights,
            )
            cache[(side,kind)]=native
        components=dofs.shape[2]
        coefficient=np.asarray(coefficient,dtype=np.float64)
        if native_kind=="value":
            if coefficient.ndim and coefficient.shape[0]==components:
                coefficient=np.moveaxis(coefficient,0,-1)
            target=self._coefficient_shape+(components,)
            value=np.ascontiguousarray(np.broadcast_to(coefficient,target))
            result,_=native.assemble(
                value,None,_native_num_threads(num_threads)
            )
        else:
            dimension=gradients.shape[3]
            if (
                coefficient.ndim>=2
                and coefficient.shape[:2]==(components,dimension)
            ):
                coefficient=np.moveaxis(coefficient,(0,1),(-2,-1))
            target=self._coefficient_shape+(components,dimension)
            gradient=np.ascontiguousarray(
                np.broadcast_to(coefficient,target)
            )
            result,_=native.assemble(
                None,gradient,_native_num_threads(num_threads)
            )
        return np.asarray(result)

    def _trace(self,side,kind):
        if kind=="value":
            return self._row_shape if side=="master" else self._column_shape
        if kind=="gradient":
            return (
                self._row_gradients
                if side=="master" else self._column_gradients
            )
        trace=(
            self._row_normal_gradient
            if side=="master" else self._column_normal_gradient
        )
        if trace is None:
            raise ValueError(
                "normal gradients require a supermesh created from FacetBasis"
            )
        return trace

    def _cross(
        self,key,row_kind,column_kind,coefficient,rows,columns,num_threads=None,
    ):
        cache=getattr(self,"_trace_assemblers",None)
        if cache is None:
            cache={};self._trace_assemblers=cache
        row_side="master" if key[0]=="m" else "slave"
        column_side="master" if key[1]=="m" else "slave"
        row_dofs=(
            self._row_dofs if row_side=="master" else self._column_dofs
        )
        column_dofs=(
            self._row_dofs if column_side=="master" else self._column_dofs
        )
        row_shape=self._trace(row_side,row_kind)
        column_shape=self._trace(column_side,column_kind)
        native_row_kind=(
            "value" if row_kind=="normal_gradient" else row_kind
        )
        native_column_kind=(
            "value" if column_kind=="normal_gradient" else column_kind
        )
        row_base_shape=self._trace(row_side,"value")
        column_base_shape=self._trace(column_side,"value")
        row_gradients=(
            self._trace(row_side,"gradient")
            if native_row_kind=="gradient" else None
        )
        column_gradients=(
            self._trace(column_side,"gradient")
            if native_column_kind=="gradient" else None
        )
        if row_kind=="normal_gradient":
            row_base_shape=row_shape
        if column_kind=="normal_gradient":
            column_base_shape=column_shape
        signature=(key,row_kind,column_kind)
        native=cache.get(signature)
        if native is None:
            native=CrossBilinearAssembler(
                row_dofs,column_dofs,row_base_shape,column_base_shape,
                self._weights,row_gradients,column_gradients,
            )
            cache[signature]=native
        coefficient=np.asarray(coefficient,dtype=np.float64)
        scalar=(
            native_row_kind==native_column_kind
            and row_dofs.shape[2]==column_dofs.shape[2]
            and coefficient.ndim<=2
        )
        if scalar:
            target=self._coefficient_shape
        else:
            axes=[row_dofs.shape[2]]
            if native_row_kind=="gradient":
                axes.append(row_gradients.shape[3])
            axes.append(column_dofs.shape[2])
            if native_column_kind=="gradient":
                axes.append(column_gradients.shape[3])
            target=self._coefficient_shape+tuple(axes)
        coefficient=np.ascontiguousarray(np.broadcast_to(coefficient,target))
        native.assemble(
            coefficient,native_row_kind,native_column_kind,
            _native_num_threads(num_threads),
        )
        matrix=csr_matrix(
            (native.values,native.indices,native.indptr),
            shape=(native.rows,native.columns),copy=False,
        )
        matrix.resize((rows,columns))
        return matrix


class InterfaceSupermesh(TriangleSupermesh):
    """Facet-interface supermesh supporting triangular and quadrilateral faces.

    Quadrilateral and high-order faces are tessellated into search triangles
    while their parent ``FacetBasis`` shapes and gradients remain the fields
    used for integration.  ``TriangleSupermesh`` remains as a compatibility
    name for direct triangle-soup construction.
    """
