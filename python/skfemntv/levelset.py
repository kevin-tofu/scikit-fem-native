from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from scipy.spatial import ConvexHull

from .regions import CellRegion,FacetRegion


class CellClassification(IntEnum):
    """Sign classification of a cell sampled at all of its mesh nodes."""

    OUTSIDE = 0
    INSIDE = 1
    CUT = 2
    TOUCHING = 3


@dataclass(frozen=True)
class LevelSetDiagnostics:
    cell_count: int
    node_count: int
    inside_count: int
    outside_count: int
    cut_count: int
    touching_count: int
    tolerance: float
    minimum_value: float
    maximum_value: float


@dataclass(frozen=True)
class CutQuadratureDiagnostics:
    cell_count: int
    cut_cell_count: int
    nonempty_cell_count: int
    quadrature_point_count: int
    total_measure: float
    minimum_weight: float
    maximum_weight: float
    integration_order: int


@dataclass(frozen=True)
class CutCellQuadrature:
    """Allocation-linear quadrature for one side of a level-set interface."""

    cell_offsets: np.ndarray
    points: np.ndarray
    reference_points: np.ndarray
    weights: np.ndarray
    cells: np.ndarray
    normals: np.ndarray
    side: str
    diagnostics: CutQuadratureDiagnostics

    def cell_slice(self,cell: int) -> slice:
        if cell<0 or cell+1>=len(self.cell_offsets):
            raise IndexError("cut-quadrature cell index is out of bounds")
        return slice(int(self.cell_offsets[cell]),int(self.cell_offsets[cell+1]))


@dataclass(frozen=True)
class CellClassificationResult:
    """Global cell labels and first-class regions derived from a level set."""

    labels: np.ndarray
    inside: CellRegion
    outside: CellRegion
    cut: CellRegion
    touching: CellRegion
    diagnostics: LevelSetDiagnostics

    def region(self,classification: CellClassification | str) -> CellRegion:
        if isinstance(classification,str):
            try:
                classification=CellClassification[classification.upper()]
            except KeyError as error:
                raise ValueError(
                    f"unknown cell classification {classification!r}"
                ) from error
        try:
            value=CellClassification(classification)
        except (TypeError,ValueError) as error:
            raise ValueError(
                f"unknown cell classification {classification!r}"
            ) from error
        return {
            CellClassification.INSIDE:self.inside,
            CellClassification.OUTSIDE:self.outside,
            CellClassification.CUT:self.cut,
            CellClassification.TOUCHING:self.touching,
        }[value]

    @property
    def active(self) -> CellRegion:
        """Cells intersecting or lying on the non-positive side."""
        return self.inside|self.cut|self.touching

    def _cell_mask(self,mesh,cells: CellRegion) -> np.ndarray:
        if int(mesh.nelements)!=len(self.labels):
            raise ValueError(
                "classification and mesh have different cell counts"
            )
        mask=np.zeros(mesh.nelements,dtype=bool)
        mask[np.asarray(cells,dtype=np.int64)]=True
        return mask

    def _facet_region(self,mesh,selection,sides=None) -> FacetRegion:
        ids=np.flatnonzero(selection)
        return FacetRegion(
            ids,mesh.facets.shape[1],
            sides=(None if sides is None else np.asarray(sides)[ids]),
        )

    def active_facets(self,mesh) -> FacetRegion:
        """All background facets incident to at least one active cell."""
        active=self._cell_mask(mesh,self.active)
        parents=mesh.f2t
        selected=active[parents[0]]
        second=parents[1]>=0
        selected[second]|=active[parents[1,second]]
        return self._facet_region(mesh,selected)

    def active_boundary_facets(self,mesh) -> FacetRegion:
        """Facets separating the active mesh from its inactive exterior.

        The facet side identifies the active parent, including side one for an
        interior background facet whose second parent is active.
        """
        active=self._cell_mask(mesh,self.active)
        parents=mesh.f2t
        first=active[parents[0]]
        second=np.zeros(parents.shape[1],dtype=bool)
        has_second=parents[1]>=0
        second[has_second]=active[parents[1,has_second]]
        selected=first^second
        sides=np.where(first,0,1).astype(np.int8)
        return self._facet_region(mesh,selected,sides)

    def active_interior_facets(self,mesh) -> FacetRegion:
        """Background facets with an active cell on both sides."""
        active=self._cell_mask(mesh,self.active)
        parents=mesh.f2t
        has_second=parents[1]>=0
        selected=np.zeros(parents.shape[1],dtype=bool)
        selected[has_second]=(
            active[parents[0,has_second]]&active[parents[1,has_second]]
        )
        return self._facet_region(mesh,selected)

    def ghost_facets(self,mesh) -> FacetRegion:
        """Active interior facets incident to at least one cut cell.

        This is a formulation-neutral candidate set for ghost penalties; the
        package does not choose the penalty form or stabilization layers.
        """
        active=self._cell_mask(mesh,self.active)
        cut=self._cell_mask(mesh,self.cut)
        parents=mesh.f2t
        has_second=parents[1]>=0
        selected=np.zeros(parents.shape[1],dtype=bool)
        first=parents[0,has_second]
        second=parents[1,has_second]
        selected[has_second]=(
            active[first]&active[second]&(cut[first]|cut[second])
        )
        return self._facet_region(mesh,selected)

    def active_dofs(self,basis,*,components=None,fields=None):
        """Global DOFs supported by active cells in ``basis``."""
        if basis.mesh is not None:
            self._cell_mask(basis.mesh,self.active)
        return basis.get_dofs(
            elements=self.active,components=components,fields=fields
        )


class LevelSet:
    """A scalar level set supplied as a callable or global nodal values.

    Negative values define the inside.  Classification samples every node in
    each mesh connectivity column, including high-order nodes.
    """

    def __init__(
        self,field,*,tolerance: float | None=None,
        relative_tolerance: float=64.*np.finfo(np.float64).eps,
    ):
        if not callable(field):
            values=np.asarray(field,dtype=np.float64)
            if values.ndim!=1:
                raise ValueError("level-set nodal values must be one-dimensional")
            field=np.array(values,dtype=np.float64,copy=True)
            field.flags.writeable=False
        if tolerance is not None and (
            not np.isfinite(tolerance) or tolerance<0.
        ):
            raise ValueError("level-set tolerance must be finite and nonnegative")
        if not np.isfinite(relative_tolerance) or relative_tolerance<0.:
            raise ValueError(
                "level-set relative_tolerance must be finite and nonnegative"
            )
        self._field=field
        self.tolerance=None if tolerance is None else float(tolerance)
        self.relative_tolerance=float(relative_tolerance)

    @property
    def is_nodal(self) -> bool:
        return not callable(self._field)

    def values(self,mesh) -> np.ndarray:
        if callable(self._field):
            values=np.asarray(self._field(mesh.p),dtype=np.float64)
            if values.shape==(1,mesh.p.shape[1]):
                values=values[0]
        else:
            values=np.asarray(self._field)
        if values.shape!=(mesh.p.shape[1],):
            raise ValueError(
                "level-set field must produce one scalar per mesh node; "
                f"expected {(mesh.p.shape[1],)}, got {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            bad=int(np.flatnonzero(~np.isfinite(values))[0])
            raise ValueError(f"level-set value at node {bad} is not finite")
        result=np.array(values,dtype=np.float64,copy=True)
        result.flags.writeable=False
        return result

    def classify(self,mesh) -> CellClassificationResult:
        values=self.values(mesh)
        value_scale=max(1.,float(np.max(np.abs(values),initial=0.)))
        tolerance=(
            self.tolerance if self.tolerance is not None
            else self.relative_tolerance*value_scale
        )
        cell_values=values[np.asarray(mesh.t,dtype=np.int64)]
        negative=cell_values < -tolerance
        positive=cell_values > tolerance
        has_negative=np.any(negative,axis=0)
        has_positive=np.any(positive,axis=0)
        has_zero=np.any(~(negative|positive),axis=0)

        labels=np.full(
            mesh.nelements,CellClassification.TOUCHING,dtype=np.int8
        )
        labels[np.all(negative,axis=0)]=CellClassification.INSIDE
        labels[np.all(positive,axis=0)]=CellClassification.OUTSIDE
        labels[has_negative&has_positive]=CellClassification.CUT
        # ``has_zero`` is intentionally explicit: negative/zero and
        # positive/zero cells touch the interface without a sampled crossing.
        touching=has_zero&~(has_negative&has_positive)
        labels[touching]=CellClassification.TOUCHING
        labels.flags.writeable=False

        count=int(mesh.nelements)
        make_region=lambda kind:CellRegion(
            np.flatnonzero(labels==kind),count
        )
        inside=make_region(CellClassification.INSIDE)
        outside=make_region(CellClassification.OUTSIDE)
        cut=make_region(CellClassification.CUT)
        touching_region=make_region(CellClassification.TOUCHING)
        diagnostics=LevelSetDiagnostics(
            cell_count=count,node_count=int(mesh.p.shape[1]),
            inside_count=len(inside),outside_count=len(outside),
            cut_count=len(cut),touching_count=len(touching_region),
            tolerance=float(tolerance),minimum_value=float(np.min(values)),
            maximum_value=float(np.max(values)),
        )
        return CellClassificationResult(
            labels,inside,outside,cut,touching_region,diagnostics
        )

    def cut_quadrature(
        self,mesh,*,side: str="inside",
        classification: CellClassificationResult | None=None,
        intorder: int=1,
    ) -> CutCellQuadrature:
        """Integrate a linear level-set side on affine Tri3 or Tet4 cells.

        One centroid rule is used per clipped simplex and therefore integrates
        physical constant and linear fields exactly.  Higher-order geometry,
        nonlinear interface reconstruction, and interface quadrature are
        deliberately separate future stages.
        """
        if side not in ("inside","outside"):
            raise ValueError("cut-quadrature side must be 'inside' or 'outside'")
        if isinstance(intorder,bool) or not isinstance(intorder,(int,np.integer)):
            raise TypeError("cut-quadrature intorder must be an integer")
        intorder=int(intorder)
        if intorder<1:
            raise ValueError("cut-quadrature intorder must be positive")
        dimension=int(mesh.dim())
        expected=dimension+1
        if dimension not in (2,3) or mesh.t.shape[0]!=expected:
            raise NotImplementedError(
                "cut quadrature currently supports affine Tri3 and Tet4 meshes"
            )
        values=self.values(mesh)
        if classification is None:
            classification=self.classify(mesh)
        elif len(classification.labels)!=mesh.nelements:
            raise ValueError(
                "classification and mesh have different cell counts"
            )
        tolerance=float(classification.diagnostics.tolerance)
        reference=np.vstack((
            np.zeros((1,dimension)),np.eye(dimension)
        ))
        point_blocks=[];reference_blocks=[];weight_blocks=[]
        cell_blocks=[];normal_blocks=[]
        offsets=[0]
        sign=1. if side=="inside" else -1.
        for cell,nodes in enumerate(mesh.t.T):
            node_values=sign*values[nodes]
            vertices=_clip_simplex(reference,node_values,tolerance)
            physical_nodes=mesh.p[:,nodes]
            gradient=_linear_simplex_gradient(physical_nodes,values[nodes])
            normal=sign*gradient
            length=float(np.linalg.norm(normal))
            if length>0.: normal=normal/length
            else: normal=np.zeros(dimension,dtype=np.float64)
            local_reference,local_weights=_polytope_quadrature(
                vertices,physical_nodes,intorder
            )
            if len(local_weights):
                local_points=_map_reference(physical_nodes,local_reference)
                point_blocks.append(local_points)
                reference_blocks.append(local_reference)
                weight_blocks.append(local_weights)
                cell_blocks.append(np.full(len(local_weights),cell,dtype=np.int64))
                normal_blocks.append(np.broadcast_to(
                    normal,(len(local_weights),dimension)
                ).copy())
            offsets.append(offsets[-1]+len(local_weights))
        points=_stack(point_blocks,(0,dimension),np.float64)
        reference_points=_stack(reference_blocks,(0,dimension),np.float64)
        weights=_stack(weight_blocks,(0,),np.float64)
        cells=_stack(cell_blocks,(0,),np.int64)
        normals=_stack(normal_blocks,(0,dimension),np.float64)
        cell_offsets=np.asarray(offsets,dtype=np.int64)
        for array in (
            cell_offsets,points,reference_points,weights,cells,normals
        ):
            array.flags.writeable=False
        diagnostics=CutQuadratureDiagnostics(
            cell_count=int(mesh.nelements),
            cut_cell_count=len(classification.cut),
            nonempty_cell_count=int(np.count_nonzero(np.diff(cell_offsets))),
            quadrature_point_count=len(weights),
            total_measure=float(np.sum(weights)),
            minimum_weight=(float(np.min(weights)) if len(weights) else 0.),
            maximum_weight=float(np.max(weights,initial=0.)),
            integration_order=intorder,
        )
        return CutCellQuadrature(
            cell_offsets,points,reference_points,weights,cells,normals,
            side,diagnostics,
        )


def _stack(blocks,empty_shape,dtype):
    return (
        np.concatenate(blocks,axis=0)
        if blocks else np.empty(empty_shape,dtype=dtype)
    )


def _clip_simplex(vertices,values,tolerance):
    kept=[vertices[index] for index,value in enumerate(values) if value<=tolerance]
    for first,second in (
        (i,j) for i in range(len(vertices)) for j in range(i+1,len(vertices))
    ):
        a=float(values[first]);b=float(values[second])
        if (a < -tolerance and b > tolerance) or (
            b < -tolerance and a > tolerance
        ):
            fraction=a/(a-b)
            kept.append(vertices[first]+fraction*(vertices[second]-vertices[first]))
    if not kept:
        return np.empty((0,vertices.shape[1]),dtype=np.float64)
    unique=[]
    for point in kept:
        if not any(np.linalg.norm(point-other)<=1.e-13 for other in unique):
            unique.append(np.asarray(point,dtype=np.float64))
    return np.asarray(unique,dtype=np.float64)


def _map_reference(physical_nodes,reference_points):
    return (
        physical_nodes[:,0]
        +reference_points@(physical_nodes[:,1:]-physical_nodes[:,[0]]).T
    )


def _linear_simplex_gradient(physical_nodes,values):
    jacobian=physical_nodes[:,1:]-physical_nodes[:,[0]]
    return np.linalg.solve(jacobian.T,values[1:]-values[0])


def _polytope_quadrature(vertices,physical_nodes,intorder):
    dimension=physical_nodes.shape[0]
    if len(vertices)<dimension+1:
        return (
            np.empty((0,dimension),dtype=np.float64),
            np.empty(0,dtype=np.float64),
        )
    physical=_map_reference(physical_nodes,vertices)
    if dimension==2:
        center=physical.mean(axis=0)
        angles=np.arctan2(physical[:,1]-center[1],physical[:,0]-center[0])
        order=np.argsort(angles)
        vertices=vertices[order];physical=physical[order]
        reference_points=[];weights=[]
        for index in range(1,len(vertices)-1):
            points,local_weights=_simplex_quadrature_rule(
                vertices[[0,index,index+1]],
                physical[[0,index,index+1]],intorder,
            )
            reference_points.extend(points)
            weights.extend(local_weights)
        return np.asarray(reference_points),np.asarray(weights)
    center_reference=vertices.mean(axis=0)
    center_physical=physical.mean(axis=0)
    hull=ConvexHull(vertices)
    reference_points=[];weights=[]
    for face in hull.simplices:
        points,local_weights=_simplex_quadrature_rule(
            np.vstack((center_reference,vertices[face])),
            np.vstack((center_physical,physical[face])),intorder,
        )
        reference_points.extend(points)
        weights.extend(local_weights)
    return np.asarray(reference_points),np.asarray(weights)


def _simplex_quadrature_rule(reference,physical,intorder):
    dimension=physical.shape[1]
    barycentric,canonical_weights=_canonical_simplex_rule(
        dimension,intorder
    )
    jacobian=np.column_stack(tuple(
        physical[index]-physical[0] for index in range(1,dimension+1)
    ))
    scale=abs(float(np.linalg.det(jacobian)))
    if scale==0.:
        return [],[]
    return barycentric@reference,canonical_weights*scale


def _canonical_simplex_rule(dimension,intorder):
    if intorder==1:
        return (
            np.full((1,dimension+1),1./(dimension+1)),
            np.array([1./(2 if dimension==2 else 6)]),
        )
    if dimension==2 and intorder==2:
        barycentric=np.array([
            [2./3.,1./6.,1./6.],
            [1./6.,2./3.,1./6.],
            [1./6.,1./6.,2./3.],
        ])
        return barycentric,np.full(3,1./6.)
    if dimension==3 and intorder==2:
        a=.5854101966249685;b=.1381966011250105
        barycentric=np.full((4,4),b)
        np.fill_diagonal(barycentric,a)
        return barycentric,np.full(4,1./24.)
    count=max(2,int(np.ceil((intorder+dimension)/2.)))
    points,weights=np.polynomial.legendre.leggauss(count)
    points=(points+1.)/2.;weights=weights/2.
    barycentric=[];quadrature_weights=[]
    if dimension==2:
        for u,wu in zip(points,weights):
            for v,wv in zip(points,weights):
                x=u;y=(1.-u)*v
                barycentric.append((1.-x-y,x,y))
                quadrature_weights.append(wu*wv*(1.-u))
    else:
        for u,wu in zip(points,weights):
            for v,wv in zip(points,weights):
                for w,ww in zip(points,weights):
                    x=u;y=(1.-u)*v;z=(1.-u)*(1.-v)*w
                    barycentric.append((1.-x-y-z,x,y,z))
                    quadrature_weights.append(
                        wu*wv*ww*(1.-u)**2*(1.-v)
                    )
    return np.asarray(barycentric),np.asarray(quadrature_weights)
