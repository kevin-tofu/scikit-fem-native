from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from ._skfn import tabulate_basis_geometry
from .regions import CellRegion,FacetRegion,NodeRegion


@dataclass(frozen=True)
class GeometryDiagnostics:
    element_count: int
    quadrature_points_per_element: int
    minimum_determinant: float
    maximum_determinant: float
    minimum_scaled_determinant: float
    worst_element: int
    worst_quadrature_point: int
    determinant_tolerance: float
    negative_orientation_elements: int
    maximum_condition_number: float
    worst_condition_element: int
    worst_condition_quadrature_point: int


class _TopologyMesh:
    """Cached codimension-one topology shared by independent meshes."""

    def _local_facets(self,full=False):
        rows=self.t.shape[0]
        if self.dim()==2:
            if rows in (3,6):
                return (
                    ((0,1,3),(1,2,4),(0,2,5)) if full and rows==6
                    else ((0,1),(1,2),(0,2))
                )
            return (
                ((0,1,4),(1,2,5),(2,3,6),(0,3,7))
                if full and rows==9 else
                ((0,1),(1,2),(2,3),(0,3))
            )
        if rows in (4,10):
            faces=((0,1,2),(0,1,3),(0,2,3),(1,2,3))
            if not full or rows==4:
                return faces
            edge={(0,1):4,(1,2):5,(0,2):6,
                  (0,3):7,(1,3):8,(2,3):9}
            return tuple(
                face+tuple(
                    edge[tuple(sorted(pair))]
                    for pair in ((face[0],face[1]),
                                 (face[1],face[2]),
                                 (face[0],face[2]))
                )
                for face in faces
            )
        if rows==6:
            return ((0,2,1),(3,4,5),(0,1,4,3),
                    (1,2,5,4),(2,0,3,5))
        if rows==5:
            return ((0,3,2,1),(0,1,4),(1,2,4),(2,3,4),(3,0,4))
        if rows==8:
            return ((0,1,4,2),(0,2,6,3),(0,3,5,1),
                    (2,4,7,6),(1,5,7,4),(3,6,7,5))
        index=lambda i,j,k:i+3*j+9*k
        full_faces=(
            tuple(index(i,j,0) for j in range(3) for i in range(3)),
            tuple(index(0,j,k) for k in range(3) for j in range(3)),
            tuple(index(i,0,k) for k in range(3) for i in range(3)),
            tuple(index(i,2,k) for k in range(3) for i in range(3)),
            tuple(index(2,j,k) for k in range(3) for j in range(3)),
            tuple(index(i,j,2) for j in range(3) for i in range(3)),
        )
        return full_faces if full else tuple(
            (face[0],face[2],face[8],face[6]) for face in full_faces
        )

    def _build_topology(self):
        if hasattr(self,"_facets"):
            return
        local=self._local_facets()
        if self.dim()==3 and self.t.shape[0] in (5,6):
            found={}
            for element,nodes in enumerate(self.t.T):
                for local_index,face in enumerate(local):
                    oriented=tuple(int(nodes[i]) for i in face)
                    key=tuple(sorted(oriented))
                    found.setdefault(key,[]).append(
                        (element,local_index,oriented)
                    )
            entries=list(found.values())
            facets=np.full((4,len(entries)),-1,dtype=np.int64)
            sizes=np.empty(len(entries),dtype=np.int64)
            t2f=np.empty((len(local),self.nelements),dtype=np.int64)
            f2t=np.full((2,len(entries)),-1,dtype=np.int64)
            for facet,adjacent in enumerate(entries):
                oriented=adjacent[0][2]
                facets[:len(oriented),facet]=oriented
                facets[len(oriented):,facet]=oriented[-1]
                sizes[facet]=len(oriented)
                for side,(element,local_index,_) in enumerate(adjacent):
                    if side>1:
                        raise ValueError("non-manifold facet has more than two cells")
                    t2f[local_index,element]=facet
                    f2t[side,facet]=element
            self._facets=np.ascontiguousarray(facets)
            self._facet_sizes=sizes
            self._t2f=np.ascontiguousarray(t2f)
            self._f2t=f2t
            return
        indexing=np.hstack(tuple(self.t[np.asarray(face)] for face in local))
        sorted_indexing=np.sort(indexing,axis=0)
        canonical,first,inverse=np.unique(
            sorted_indexing,axis=1,return_index=True,return_inverse=True
        )
        facets=(
            indexing[:,first]
            if self.dim()==3 and self.t.shape[0] in (8,27)
            else canonical
        )
        t2f=inverse.reshape(len(local),self.nelements)
        flat=t2f.ravel(order="C")
        elements=np.tile(np.arange(self.nelements),len(local))
        f2t=np.full((2,facets.shape[1]),-1,dtype=np.int64)
        for facet,element in zip(flat,elements):
            row=0 if f2t[0,facet]==-1 else 1
            f2t[row,facet]=element
        self._facets=np.ascontiguousarray(facets,dtype=np.int64)
        self._t2f=np.ascontiguousarray(t2f,dtype=np.int64)
        self._f2t=f2t

    @property
    def facets(self):
        self._build_topology();return self._facets

    @property
    def t2f(self):
        self._build_topology();return self._t2f

    @property
    def f2t(self):
        self._build_topology();return self._f2t

    def boundary_facets(self):
        return np.flatnonzero(self.f2t[1]==-1)

    def interior_facets(self):
        return np.flatnonzero(self.f2t[1]!=-1)

    def elements_satisfying(self,test):
        centers=self.p[:,self.t[:_corner_count(self)]].mean(axis=1)
        mask=np.asarray(test(centers),dtype=bool)
        if mask.shape!=(self.nelements,):
            raise ValueError("element predicate must return one boolean per cell")
        return CellRegion(np.flatnonzero(mask),self.nelements)

    def facets_satisfying(self,test,boundaries_only=False,normal=None):
        candidates=(
            self.boundary_facets() if boundaries_only else
            np.arange(self.facets.shape[1],dtype=np.int64)
        )
        if hasattr(self,"_facet_sizes"):
            centers=np.column_stack([
                self.p[:,self.facets[:self._facet_sizes[facet],facet]].mean(axis=1)
                for facet in candidates
            ])
        else:
            centers=self.p[:,self.facets[:,candidates]].mean(axis=1)
        mask=np.asarray(test(centers),dtype=bool)
        if mask.shape!=(len(candidates),):
            raise ValueError("facet predicate must return one boolean per facet")
        selected=candidates[mask]
        if normal is None:
            return FacetRegion(selected,self.facets.shape[1])
        requested=np.asarray(normal,dtype=np.float64)
        if requested.shape!=(self.dim(),) or not np.all(np.isfinite(requested)):
            raise ValueError(
                f"normal must be a finite vector with shape ({self.dim()},)"
            )
        length=np.linalg.norm(requested)
        if length==0.:
            raise ValueError("normal must be nonzero")
        requested=requested/length
        sides=np.zeros(len(selected),dtype=np.int8)
        signs=np.ones(len(selected),dtype=np.int8)
        corners=_corner_count(self)
        for local,facet in enumerate(selected):
            element=int(self.f2t[0,facet])
            size=(
                int(self._facet_sizes[facet])
                if hasattr(self,"_facet_sizes") else self.facets.shape[0]
            )
            facet_nodes=self.facets[:size,facet]
            points=self.p[:,facet_nodes]
            center=points.mean(axis=1)
            element_center=self.p[:,self.t[:corners,element]].mean(axis=1)
            if self.dim()==2:
                tangent=points[:,1]-points[:,0]
                outward=np.array([tangent[1],-tangent[0]])
            else:
                outward=np.cross(
                    points[:,1]-points[:,0],points[:,2]-points[:,0]
                )
            outward_length=np.linalg.norm(outward)
            if not outward_length>0.:
                raise ValueError(f"facet {facet} has singular geometry")
            outward=outward/outward_length
            if np.dot(outward,center-element_center)<0.:
                outward=-outward
            if np.dot(requested,outward)<0.:
                if self.f2t[1,facet]>=0:
                    sides[local]=1
                else:
                    signs[local]=-1
        return FacetRegion(
            selected,self.facets.shape[1],
            sides=sides,normal_signs=signs,
        )

    @property
    def subdomains(self):
        return dict(getattr(self,"_subdomains",{}))

    def with_subdomains(self,subdomains):
        return _with_subdomains(self,subdomains)

    def _facet_connectivity(self,facets,full=True):
        ids=np.asarray(facets,dtype=np.int64).reshape(-1)
        if self.dim()==3 and self.t.shape[0] in (5,6):
            self._build_topology()
            sizes=self._facet_sizes[ids]
            if len(set(map(int,sizes)))>1:
                raise ValueError(
                    "mixed triangle/quadrilateral connectivity is ragged; "
                    "pass facet IDs directly to FacetBasis"
                )
            width=int(sizes[0]) if len(sizes) else 0
            return self.facets[:width,ids]
        local=self._local_facets(full=full)
        result=[]
        for facet in ids:
            element=int(self.f2t[0,facet])
            local_index=int(np.flatnonzero(self.t2f[:,element]==facet)[0])
            result.append(self.t[np.asarray(local[local_index]),element])
        width=len(local[0])
        return (
            np.asarray(result,dtype=np.int64).T if result else
            np.empty((width,0),dtype=np.int64)
        )


def _with_boundaries(mesh,boundaries):
    result=copy(mesh)
    facets=mesh.boundary_facets()
    if hasattr(mesh,"_facet_sizes"):
        centers=np.column_stack([
            mesh.p[:,mesh.facets[:mesh._facet_sizes[facet],facet]].mean(axis=1)
            for facet in facets
        ])
    else:
        centers=mesh.p[:,mesh.facets[:,facets]].mean(axis=1)
    result._boundaries={}
    for name,selector in boundaries.items():
        if callable(selector):
            mask=np.asarray(selector(centers),dtype=bool)
            if mask.shape!=(len(facets),):
                raise ValueError(
                    f"boundary selector {name!r} must return one "
                    "boolean per boundary facet"
                )
            result._boundaries[name]=FacetRegion(
                facets[mask],mesh.facets.shape[1]
            )
        else:
            selected=np.asarray(selector)
            if selected.dtype==bool:
                if selected.shape!=(len(facets),):
                    raise ValueError(
                        f"boundary selector {name!r} mask must contain one "
                        "value per boundary facet"
                    )
                selected=facets[selected]
            selected=np.asarray(selected,dtype=np.int64).reshape(-1)
            result._boundaries[name]=FacetRegion(
                selected,mesh.facets.shape[1]
            )
    return result


def _with_subdomains(mesh,subdomains):
    result=copy(mesh)
    centers=mesh.p[:,mesh.t[:_corner_count(mesh)]].mean(axis=1)
    result._subdomains={}
    for name,selector in subdomains.items():
        if callable(selector):
            mask=np.asarray(selector(centers),dtype=bool)
            if mask.shape!=(mesh.nelements,):
                raise ValueError(
                    f"subdomain selector {name!r} must return one "
                    "boolean per cell"
                )
            selected=np.flatnonzero(mask)
        else:
            selected=np.asarray(selector)
            if selected.dtype==bool:
                if selected.shape!=(mesh.nelements,):
                    raise ValueError(
                        f"subdomain selector {name!r} mask must contain one "
                        "value per cell"
                    )
                selected=np.flatnonzero(selected)
            selected=np.asarray(selected,dtype=np.int64).reshape(-1)
        result._subdomains[name]=CellRegion(selected,mesh.nelements)
    return result


class MeshTri(_TopologyMesh):
    """Two-dimensional triangular mesh."""

    def __init__(self,p=None,t=None):
        self.p=np.asarray(
            p if p is not None else [[0.,1.,0.],[0.,0.,1.]],
            dtype=np.float64,
        )
        self.t=np.asarray(
            t if t is not None else [[0],[1],[2]],dtype=np.int64
        )
        if self.p.ndim!=2 or self.p.shape[0]!=2:
            raise ValueError("p must have shape (2, nodes)")
        if self.t.ndim!=2 or self.t.shape[0]!=3:
            raise ValueError("t must have shape (3, elements)")
        self.t=np.sort(self.t,axis=0)
        self._boundaries={}

    @classmethod
    def init_tensor(cls,x,y):
        x,y=map(np.asarray,(x,y))
        points=np.array([[a,b] for b in y for a in x],dtype=float).T
        nx=len(x);node=lambda i,j:i+nx*j
        cells=[]
        for j in range(len(y)-1):
            for i in range(len(x)-1):
                a=node(i,j);b=node(i+1,j)
                c=node(i,j+1);d=node(i+1,j+1)
                cells.extend(((a,b,d),(a,d,c)))
        return cls(points,np.asarray(cells,dtype=np.int64).T)

    @property
    def nelements(self):
        return self.t.shape[1]

    def dim(self):
        return 2

    def _legacy_boundary_facets(self):
        found={}
        for nodes in self.t.T:
            for edge in ((1,2),(2,0),(0,1)):
                value=tuple(int(nodes[i]) for i in edge)
                key=tuple(sorted(value))
                found[key]=None if key in found else value
        return np.asarray(
            [value for value in found.values() if value is not None],
            dtype=np.int64,
        ).T

    @property
    def boundaries(self):
        return dict(self._boundaries)

    def with_boundaries(self,boundaries):
        return _with_boundaries(self,boundaries)


class MeshTri2(MeshTri):
    @classmethod
    def from_mesh(cls,mesh):
        points=[mesh.p[:,i].copy() for i in range(mesh.p.shape[1])]
        edge_nodes={};edge_rows=[[],[],[]]
        for nodes in mesh.t.T:
            for row,(a,b) in zip(edge_rows,((0,1),(1,2),(0,2))):
                edge=tuple(sorted((int(nodes[a]),int(nodes[b]))))
                if edge not in edge_nodes:
                    edge_nodes[edge]=len(points)
                    points.append(.5*(mesh.p[:,edge[0]]+mesh.p[:,edge[1]]))
                row.append(edge_nodes[edge])
        return cls(
            np.asarray(points).T,
            np.asarray([*mesh.t,*edge_rows],dtype=np.int64),
        )

    def __init__(self,p=None,t=None):
        if p is None or t is None:
            generated=type(self).from_mesh(MeshTri())
            self.p,self.t=generated.p,generated.t
            self._boundaries={}
            return
        self.p=np.asarray(p,dtype=np.float64)
        self.t=np.asarray(t,dtype=np.int64)
        if self.p.ndim!=2 or self.p.shape[0]!=2 or self.t.shape[0]!=6:
            raise ValueError("quadratic triangle mesh requires p (2,n), t (6,e)")
        self._boundaries={}

    def _legacy_boundary_facets(self):
        found={}
        edge_node={(0,1):3,(1,2):4,(0,2):5}
        for nodes in self.t.T:
            for a,b in ((1,2),(2,0),(0,1)):
                key=tuple(sorted((int(nodes[a]),int(nodes[b]))))
                midpoint=edge_node[tuple(sorted((a,b)))]
                value=(int(nodes[a]),int(nodes[b]),int(nodes[midpoint]))
                found[key]=None if key in found else value
        return np.asarray(
            [value for value in found.values() if value is not None],
            dtype=np.int64,
        ).T


class MeshQuad(_TopologyMesh):
    """Two-dimensional quadrilateral mesh."""

    def __init__(self,p=None,t=None):
        self.p=np.asarray(
            p if p is not None else
            [[0.,1.,1.,0.],[0.,0.,1.,1.]],dtype=np.float64
        )
        self.t=np.asarray(
            t if t is not None else [[0],[1],[2],[3]],dtype=np.int64
        )
        if self.p.ndim!=2 or self.p.shape[0]!=2:
            raise ValueError("p must have shape (2, nodes)")
        if self.t.ndim!=2 or self.t.shape[0]!=4:
            raise ValueError("t must have shape (4, elements)")
        self._boundaries={}

    @classmethod
    def init_tensor(cls,x,y):
        x,y=map(np.asarray,(x,y))
        points=np.array([[a,b] for b in y for a in x],dtype=float).T
        nx=len(x);node=lambda i,j:i+nx*j
        cells=[]
        for j in range(len(y)-1):
            for i in range(len(x)-1):
                cells.append((
                    node(i,j),node(i+1,j),
                    node(i+1,j+1),node(i,j+1),
                ))
        return cls(points,np.asarray(cells,dtype=np.int64).T)

    @property
    def nelements(self):
        return self.t.shape[1]

    def dim(self):
        return 2

    def _legacy_boundary_facets(self):
        found={}
        for nodes in self.t.T:
            for a,b in ((0,1),(1,2),(2,3),(3,0)):
                value=(int(nodes[a]),int(nodes[b]))
                key=tuple(sorted(value))
                found[key]=None if key in found else value
        return np.asarray(
            [value for value in found.values() if value is not None],
            dtype=np.int64,
        ).T

    @property
    def boundaries(self):
        return dict(self._boundaries)

    def with_boundaries(self,boundaries):
        return _with_boundaries(self,boundaries)


class MeshQuad2(MeshQuad):
    """Nine-node isoparametric quadrilateral mesh."""

    @classmethod
    def from_mesh(cls,mesh):
        points=[mesh.p[:,i].copy() for i in range(mesh.p.shape[1])]
        edge_nodes={};cells=[]
        for vertices in mesh.t.T:
            cell=list(map(int,vertices))
            for a,b in ((0,1),(1,2),(2,3),(3,0)):
                edge=tuple(sorted((int(vertices[a]),int(vertices[b]))))
                if edge not in edge_nodes:
                    edge_nodes[edge]=len(points)
                    points.append(.5*(mesh.p[:,edge[0]]+mesh.p[:,edge[1]]))
                cell.append(edge_nodes[edge])
            cell.append(len(points))
            points.append(mesh.p[:,vertices].mean(axis=1))
            cells.append(cell)
        return cls(np.asarray(points).T,np.asarray(cells,dtype=np.int64).T)

    def __init__(self,p=None,t=None):
        if p is None or t is None:
            generated=type(self).from_mesh(MeshQuad())
            self.p,self.t=generated.p,generated.t
            self._boundaries={}
            return
        self.p=np.asarray(p,dtype=np.float64)
        self.t=np.asarray(t,dtype=np.int64)
        if self.p.ndim!=2 or self.p.shape[0]!=2 or self.t.shape[0]!=9:
            raise ValueError(
                "quadratic quadrilateral mesh requires p (2,n), t (9,e)"
            )
        self._boundaries={}

    def _legacy_boundary_facets(self):
        found={}
        for nodes in self.t.T:
            for (a,b),midpoint in zip(
                ((0,1),(1,2),(2,3),(3,0)),(4,5,6,7)
            ):
                key=tuple(sorted((int(nodes[a]),int(nodes[b]))))
                value=(int(nodes[a]),int(nodes[b]),int(nodes[midpoint]))
                found[key]=None if key in found else value
        return np.asarray(
            [value for value in found.values() if value is not None],
            dtype=np.int64,
        ).T


class MeshTet(_TopologyMesh):
    """Minimal tetrahedral mesh container compatible with skfemntv.Basis."""

    def __init__(self, p: np.ndarray | None = None, t: np.ndarray | None = None):
        self.p = np.asarray(
            p if p is not None else
            [[0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]],
            dtype=np.float64,
        )
        self.t = np.asarray(
            t if t is not None else [[0], [1], [2], [3]], dtype=np.int64
        )
        if self.p.ndim != 2 or self.p.shape[0] != 3:
            raise ValueError("p must have shape (3, nodes)")
        if self.t.ndim != 2 or self.t.shape[0] != 4:
            raise ValueError("t must have shape (4, elements)")
        self._boundaries={}

    @classmethod
    def init_tensor(cls, x, y, z):
        x, y, z = map(np.asarray, (x, y, z))
        points = np.array(
            [[a, b, c] for c in z for b in y for a in x], dtype=float
        ).T
        nx, ny = len(x), len(y)
        node = lambda i,j,k: i + nx*(j + ny*k)
        pattern=((0,1,3,7),(0,3,2,7),(0,2,6,7),
                 (0,6,4,7),(0,4,5,7),(0,5,1,7))
        cells=[]
        for k in range(len(z)-1):
            for j in range(len(y)-1):
                for i in range(len(x)-1):
                    cube=(node(i,j,k),node(i+1,j,k),node(i,j+1,k),node(i+1,j+1,k),
                          node(i,j,k+1),node(i+1,j,k+1),node(i,j+1,k+1),node(i+1,j+1,k+1))
                    cells.extend(tuple(cube[q] for q in tet) for tet in pattern)
        return cls(points,np.asarray(cells,dtype=np.int64).T)

    @property
    def nelements(self):
        return self.t.shape[1]

    def dim(self):
        return 3

    def _legacy_boundary_facets(self):
        faces={}
        for element in range(self.nelements):
            for local in ((1,2,3),(0,3,2),(0,1,3),(0,2,1)):
                face=tuple(int(self.t[i,element]) for i in local);key=tuple(sorted(face))
                faces[key]=None if key in faces else (element,local,face)
        return np.array([v[2] for v in faces.values() if v is not None],dtype=np.int64).T

    @property
    def boundaries(self):
        return dict(getattr(self,"_boundaries",{}))

    def with_boundaries(self,boundaries):
        return _with_boundaries(self,boundaries)


class MeshTet2(MeshTet):
    """Quadratic tetrahedral mesh with six shared edge nodes per element."""

    @classmethod
    def from_mesh(cls, mesh: MeshTet):
        points = [mesh.p[:, i].copy() for i in range(mesh.p.shape[1])]
        edge_nodes = {}
        rows = [list(map(int, mesh.t[i])) for i in range(4)]
        edge_order = ((0,1),(1,2),(2,0),(0,3),(1,3),(2,3))
        edge_rows = [[] for _ in edge_order]
        for element, vertices in enumerate(mesh.t.T):
            for row, (a,b) in zip(edge_rows, edge_order):
                edge=tuple(sorted((int(vertices[a]),int(vertices[b]))))
                if edge not in edge_nodes:
                    edge_nodes[edge]=len(points)
                    points.append(0.5*(mesh.p[:,edge[0]]+mesh.p[:,edge[1]]))
                row.append(edge_nodes[edge])
        return cls(np.asarray(points).T,np.asarray(rows+edge_rows,dtype=np.int64))

    def __init__(self,p=None,t=None):
        if p is None or t is None:
            generated=type(self).from_mesh(MeshTet())
            self.p,self.t=generated.p,generated.t
            self._boundaries={}
            return
        self.p=np.asarray(p,dtype=np.float64);self.t=np.asarray(t,dtype=np.int64)
        if self.p.ndim!=2 or self.p.shape[0]!=3 or self.t.ndim!=2 or self.t.shape[0]!=10:
            raise ValueError("quadratic tetra mesh requires p (3,n) and t (10,e)")
        self._boundaries={}

    def _legacy_boundary_facets(self):
        edge={(0,1):4,(1,2):5,(0,2):6,(0,3):7,(1,3):8,(2,3):9}
        faces={};vertices=((1,2,3),(0,3,2),(0,1,3),(0,2,1))
        for e,nodes in enumerate(self.t.T):
            for face in vertices:
                full=tuple(face)+tuple(
                    edge[tuple(sorted((face[i],face[(i+1)%3])))] for i in range(3)
                )
                key=tuple(sorted(int(nodes[i]) for i in face))
                faces[key]=None if key in faces else tuple(int(nodes[i]) for i in full)
        return np.array([v for v in faces.values() if v is not None],dtype=np.int64).T


class MeshWedge1(_TopologyMesh):
    """Linear triangular-prism mesh compatible with scikit-fem's Wedge1."""

    def __init__(self,p=None,t=None):
        self.p=np.asarray(
            p if p is not None else
            [[0.,1.,0.,0.,1.,0.],
             [0.,0.,1.,0.,0.,1.],
             [0.,0.,0.,1.,1.,1.]],
            dtype=np.float64,
        )
        self.t=np.asarray(
            t if t is not None else np.arange(6)[:,None],dtype=np.int64
        )
        if self.p.ndim!=2 or self.p.shape[0]!=3:
            raise ValueError("p must have shape (3, nodes)")
        if self.t.ndim!=2 or self.t.shape[0]!=6:
            raise ValueError("t must have shape (6, elements)")
        self._boundaries={}

    @classmethod
    def init_tensor(cls,x,y,z):
        x,y,z=map(np.asarray,(x,y,z))
        base=MeshTri.init_tensor(x,y)
        count=base.p.shape[1]
        points=np.vstack((
            np.tile(base.p,(1,len(z))),
            np.repeat(z,count)[None,:],
        ))
        cells=[]
        for layer in range(len(z)-1):
            lower=layer*count
            upper=(layer+1)*count
            for triangle in base.t.T:
                a,b,c=map(int,triangle)
                cells.append((lower+a,lower+b,lower+c,
                              upper+a,upper+b,upper+c))
        return cls(points,np.asarray(cells,dtype=np.int64).T)

    @property
    def nelements(self):
        return self.t.shape[1]

    def dim(self):
        return 3

    @property
    def boundaries(self):
        return dict(self._boundaries)

    def with_boundaries(self,boundaries):
        return _with_boundaries(self,boundaries)


class MeshPyramid1(_TopologyMesh):
    """Five-node pyramid mesh; an skfemntv extension not present in scikit-fem."""

    def __init__(self,p=None,t=None):
        self.p=np.asarray(
            p if p is not None else
            [[0.,1.,1.,0.,.5],
             [0.,0.,1.,1.,.5],
             [0.,0.,0.,0.,1.]],
            dtype=np.float64,
        )
        self.t=np.asarray(
            t if t is not None else np.arange(5)[:,None],dtype=np.int64
        )
        if self.p.ndim!=2 or self.p.shape[0]!=3:
            raise ValueError("p must have shape (3, nodes)")
        if self.t.ndim!=2 or self.t.shape[0]!=5:
            raise ValueError("t must have shape (5, elements)")
        self._boundaries={}

    @property
    def nelements(self):
        return self.t.shape[1]

    def dim(self):
        return 3

    @property
    def boundaries(self):
        return dict(self._boundaries)

    def with_boundaries(self,boundaries):
        return _with_boundaries(self,boundaries)


class MeshHex(_TopologyMesh):
    def __init__(self, p: np.ndarray | None = None, t: np.ndarray | None = None):
        self.p = np.asarray(
            p if p is not None else
            [[0.,1.,0.,0.,1.,1.,0.,1.],
             [0.,0.,1.,0.,1.,0.,1.,1.],
             [0.,0.,0.,1.,0.,1.,1.,1.]], dtype=np.float64
        )
        self.t = np.asarray(
            t if t is not None else np.arange(8)[:, None], dtype=np.int64
        )
        if self.p.ndim != 2 or self.p.shape[0] != 3:
            raise ValueError("p must have shape (3, nodes)")
        if self.t.ndim != 2 or self.t.shape[0] != 8:
            raise ValueError("t must have shape (8, elements)")
        self._boundaries={}

    @classmethod
    def init_tensor(cls, x, y, z):
        x, y, z = map(np.asarray, (x, y, z))
        points = np.array(
            [[a,b,c] for c in z for b in y for a in x], dtype=float
        ).T
        nx, ny = len(x), len(y)
        node = lambda i,j,k: i + nx*(j + ny*k)
        cells = []
        for k in range(len(z)-1):
            for j in range(len(y)-1):
                for i in range(len(x)-1):
                    cells.append((
                        node(i,j,k), node(i+1,j,k), node(i,j+1,k), node(i,j,k+1),
                        node(i+1,j+1,k), node(i+1,j,k+1),
                        node(i,j+1,k+1), node(i+1,j+1,k+1),
                    ))
        return cls(points, np.asarray(cells, dtype=np.int64).T)

    @property
    def nelements(self):
        return self.t.shape[1]

    def dim(self):
        return 3

    def _legacy_boundary_facets(self):
        if self.t.shape[0] == 8:
            local_faces=((0,1,4,2),(3,6,7,5),(0,3,5,1),
                         (2,4,7,6),(0,2,6,3),(1,5,7,4))
            corners=local_faces
        else:
            index=lambda i,j,k:i+3*j+9*k
            local_faces=(
                tuple(index(i,j,0) for j in range(3) for i in range(3)),
                tuple(index(i,j,2) for j in range(3) for i in range(3)),
                tuple(index(i,0,k) for k in range(3) for i in range(3)),
                tuple(index(i,2,k) for k in range(3) for i in range(3)),
                tuple(index(0,j,k) for k in range(3) for j in range(3)),
                tuple(index(2,j,k) for k in range(3) for j in range(3)),
            )
            corners=tuple((face[0],face[2],face[8],face[6]) for face in local_faces)
        found={}
        for e,nodes in enumerate(self.t.T):
            for face,corner in zip(local_faces,corners):
                key=tuple(sorted(int(nodes[i]) for i in corner))
                found[key]=None if key in found else tuple(int(nodes[i]) for i in face)
        return np.array([v for v in found.values() if v is not None],dtype=np.int64).T

    @property
    def boundaries(self):
        return dict(getattr(self,"_boundaries",{}))

    def with_boundaries(self,boundaries):
        return _with_boundaries(self,boundaries)


class MeshHex2(MeshHex):
    @classmethod
    def from_mesh(cls,mesh:MeshHex):
        points=[];lookup={};cells=[]
        bits=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],
                       [1,1,0],[1,0,1],[0,1,1],[1,1,1]])
        grid=(0.,.5,1.)
        for vertices in mesh.t.T:
            xyz=mesh.p[:,vertices];cell=[]
            for zeta in grid:
                for eta in grid:
                    for xi in grid:
                        point=np.zeros(3)
                        for n,bit in enumerate(bits):
                            factors=np.where(bit,(xi,eta,zeta),1.-np.array((xi,eta,zeta)))
                            point+=np.prod(factors)*xyz[:,n]
                        key=tuple(np.round(point,14))
                        if key not in lookup:
                            lookup[key]=len(points);points.append(point)
                        cell.append(lookup[key])
            cells.append(cell)
        return cls(np.asarray(points).T,np.asarray(cells,dtype=np.int64).T)

    def __init__(self,p=None,t=None):
        if p is None or t is None:
            generated=type(self).from_mesh(MeshHex())
            self.p,self.t=generated.p,generated.t
            self._boundaries={}
            return
        self.p=np.asarray(p,dtype=np.float64);self.t=np.asarray(t,dtype=np.int64)
        if self.p.ndim!=2 or self.p.shape[0]!=3 or self.t.ndim!=2 or self.t.shape[0]!=27:
            raise ValueError("quadratic hex mesh requires p (3,n) and t (27,e)")
        self._boundaries={}


class _ComposableElement:
    def __mul__(self,other):
        if isinstance(other,ElementComposite):
            return ElementComposite(self,*other.elems)
        return ElementComposite(self,other)


class ElementTetP1(_ComposableElement):
    def __init__(self):
        self.doflocs = np.array(
            [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
        )


class ElementTriP1(_ComposableElement):
    def __init__(self):
        self.doflocs=np.array([[0.,0.],[1.,0.],[0.,1.]])


class ElementTriP2(_ComposableElement):
    def __init__(self):
        self.doflocs=np.array([
            [0.,0.],[1.,0.],[0.,1.],
            [.5,0.],[.5,.5],[0.,.5],
        ])


class ElementQuad1(_ComposableElement):
    def __init__(self):
        self.doflocs=np.array([
            [0.,0.],[1.,0.],[1.,1.],[0.,1.],
        ])


class ElementQuad2(_ComposableElement):
    def __init__(self):
        self.doflocs=np.array([
            [0.,0.],[1.,0.],[1.,1.],[0.,1.],
            [.5,0.],[1.,.5],[.5,1.],[0.,.5],[.5,.5],
        ])


class ElementTriP0(_ComposableElement):
    def __init__(self):
        self.doflocs=np.array([[1./3.,1./3.]])


class ElementQuad0(_ComposableElement):
    def __init__(self):
        self.doflocs=np.array([[.5,.5]])


class ElementTetP0(_ComposableElement):
    def __init__(self):
        self.doflocs=np.array([[.25,.25,.25]])


class ElementHex0(_ComposableElement):
    def __init__(self):
        self.doflocs=np.array([[.5,.5,.5]])


class ElementDG(_ComposableElement):
    """Make the wrapped nodal element discontinuous between cells."""

    def __init__(self,element):
        if isinstance(element,(ElementComposite,ElementVector,ElementDG)):
            raise TypeError("ElementDG expects a scalar element")
        self.elem=element
        self.doflocs=element.doflocs.copy()


class ElementTetP2(_ComposableElement):
    def __init__(self):
        self.doflocs=np.array(
            [[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],
             [.5,0.,0.],[.5,.5,0.],[0.,.5,0.],[0.,0.,.5],
             [.5,0.,.5],[0.,.5,.5]]
        )


class ElementWedge1(_ComposableElement):
    def __init__(self):
        self.doflocs=np.array(
            [[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],
             [0.,0.,1.],[1.,0.,1.],[0.,1.,1.]]
        )


class ElementPyramid1(_ComposableElement):
    """Rational nodal Pyramid5 element on a collapsed unit cube."""

    def __init__(self):
        self.doflocs=np.array(
            [[0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,0.],[0.,0.,1.]]
        )


class ElementHex1(_ComposableElement):
    def __init__(self):
        self.doflocs = np.array(
            [[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],
             [1.,1.,0.],[1.,0.,1.],[0.,1.,1.],[1.,1.,1.]]
        )


class ElementHex2(_ComposableElement):
    def __init__(self):
        self.doflocs=np.array(
            [[x,y,z] for z in (0.,.5,1.) for y in (0.,.5,1.)
             for x in (0.,.5,1.)]
        )


class ElementVector(_ComposableElement):
    def __init__(self, element, dim: int | None = None):
        self.elem = element
        self._dim = element.doflocs.shape[1] if dim is None else dim


class ElementComposite(_ComposableElement):
    """Ordered collection of nodal H1 subfields."""

    def __init__(self,*elements):
        if len(elements)<2:
            raise ValueError("ElementComposite requires at least two fields")
        flattened=[]
        for element in elements:
            if isinstance(element,ElementComposite):
                flattened.extend(element.elems)
            else:
                flattened.append(element)
        self.elems=tuple(flattened)


class _Field:
    def __init__(self, value, grad):
        self.value = value
        self.grad = grad

    def __array__(self, dtype=None):
        return np.asarray(self.value, dtype=dtype)


class _LazyVectorFields:
    """scikit-fem-compatible field sequence without eager global copies."""

    def __init__(self,basis):
        self._basis=basis

    def __len__(self):
        return self._basis.tabulated_shape.shape[2]*self._basis.elem._dim

    def __getitem__(self,index):
        size=len(self)
        if index<0:
            index+=size
        if index<0 or index>=size:
            raise IndexError("basis field index is out of range")
        basis=self._basis
        components=basis.elem._dim
        node,component=divmod(index,components)
        entities,quadrature=basis.dx.shape
        value=np.zeros((components,entities,quadrature))
        gradient=np.zeros((
            components,basis.mesh.dim(),entities,quadrature
        ))
        value[component]=basis.tabulated_shape[:,:,node]
        gradient[component]=basis.tabulated_gradients[
            :,:,node
        ].transpose(2,0,1)
        return (_Field(value,gradient),)

    def __iter__(self):
        return (self[index] for index in range(len(self)))


class DiscreteField:
    """Values and physical gradients evaluated at basis quadrature points."""

    def __init__(self,value,grad):
        self.value=np.asarray(value)
        self.grad=None if grad is None else np.asarray(grad)

    def __array__(self,dtype=None):
        return np.asarray(self.value,dtype=dtype)

    @property
    def shape(self):
        return self.value.shape

    @property
    def div(self):
        if (
            self.grad is None
            or self.value.ndim!=3
            or self.grad.shape[0]!=self.grad.shape[1]
        ):
            return None
        return np.einsum("iieq->eq",self.grad)

    def __getitem__(self,key):
        return self.value[key]

    @staticmethod
    def _value(other):
        return other.value if isinstance(other,DiscreteField) else other

    def __mul__(self,other):
        if not isinstance(other,(DiscreteField,np.ndarray)) and not np.isscalar(other):
            return NotImplemented
        return self.value*self._value(other)

    __rmul__=__mul__

    def __add__(self,other):
        return self.value+self._value(other)

    __radd__=__add__

    def __sub__(self,other):
        return self.value-self._value(other)

    def __rsub__(self,other):
        return self._value(other)-self.value

    def __truediv__(self,other):
        return self.value/self._value(other)

    def __rtruediv__(self,other):
        return self._value(other)/self.value

    def __pow__(self,other):
        return self.value**other

    def __neg__(self):
        return -self.value


class DofsView:
    """Selected DOF indices with the scikit-fem-style ``all()`` accessor."""

    def __init__(self,dofs,doflocs,groups=None):
        self._dofs=np.unique(np.asarray(dofs,dtype=np.int64))
        self.doflocs=doflocs
        self.groups={
            str(name):np.unique(np.asarray(values,dtype=np.int64))
            for name,values in (groups or {}).items()
        }

    @staticmethod
    def _names(names):
        return [names] if isinstance(names,str) else list(names)

    def all(self,names=None):
        if names is None:
            return self._dofs.copy()
        return self.keep(names).flatten()

    def flatten(self):
        return self._dofs.copy()

    def keep(self,names):
        selected=self._names(names)
        unknown=[name for name in selected if name not in self.groups]
        if unknown:
            raise KeyError(f"unknown DOF group {unknown[0]!r}")
        groups={name:self.groups[name] for name in selected}
        dofs=(
            np.unique(np.concatenate(tuple(groups.values())))
            if groups else np.empty(0,dtype=np.int64)
        )
        return DofsView(dofs,self.doflocs,groups)

    def drop(self,names):
        removed=set(self._names(names))
        unknown=removed-self.groups.keys()
        if unknown:
            raise KeyError(f"unknown DOF group {sorted(unknown)[0]!r}")
        return self.keep([
            name for name in self.groups if name not in removed
        ])

    def __array__(self,dtype=None):
        return np.asarray(self._dofs,dtype=dtype)

    def __len__(self):
        return len(self._dofs)


def _quad_shapes(points,quadratic):
    points=np.asarray(points)
    nq=points.shape[1]
    if quadratic:
        shape=np.empty((nq,9))
        grad=np.empty((nq,9,2))
        order=((0,0),(2,0),(2,2),(0,2),
               (1,0),(2,1),(1,2),(0,1),(1,1))
        for q,(x,y) in enumerate(points.T):
            vx=np.array([2.*(x-.5)*(x-1.),4.*x*(1.-x),
                         2.*x*(x-.5)])
            vy=np.array([2.*(y-.5)*(y-1.),4.*y*(1.-y),
                         2.*y*(y-.5)])
            dx=np.array([4.*x-3.,4.-8.*x,4.*x-1.])
            dy=np.array([4.*y-3.,4.-8.*y,4.*y-1.])
            for node,(i,j) in enumerate(order):
                shape[q,node]=vx[i]*vy[j]
                grad[q,node]=(dx[i]*vy[j],vx[i]*dy[j])
        return shape,grad
    bits=((0,0),(1,0),(1,1),(0,1))
    shape=np.empty((nq,4))
    grad=np.empty((nq,4,2))
    for q,(x,y) in enumerate(points.T):
        for node,(i,j) in enumerate(bits):
            shape[q,node]=(x if i else 1.-x)*(y if j else 1.-y)
            grad[q,node]=(
                (1. if i else -1.)*(y if j else 1.-y),
                (x if i else 1.-x)*(1. if j else -1.),
            )
    return shape,grad


def _interior_facets_2d(mesh):
    is_quad=mesh.t.shape[0] in (4,9)
    if is_quad:
        edges=((0,1),(1,2),(2,3),(3,0))
        midpoints=(4,5,6,7) if mesh.t.shape[0]==9 else None
    else:
        edges=((1,2),(2,0),(0,1))
        midpoints=(4,5,3) if mesh.t.shape[0]==6 else None
    found={}
    for nodes in mesh.t.T:
        for index,(a,b) in enumerate(edges):
            key=tuple(sorted((int(nodes[a]),int(nodes[b]))))
            value=key
            if midpoints is not None:
                value=value+(int(nodes[midpoints[index]]),)
            found.setdefault(key,[]).append(value)
    values=[
        adjacent[0] for adjacent in found.values()
        if len(adjacent)==2
    ]
    width=3 if midpoints is not None else 2
    if not values:
        return np.empty((width,0),dtype=np.int64)
    return np.asarray(values,dtype=np.int64).T


def _interior_facets_3d(mesh):
    if mesh.t.shape[0] in (5,6):
        raise NotImplementedError(
            "mixed-face interior facets are not implemented yet"
        )
    is_tet=mesh.t.shape[0] in (4,10)
    if is_tet:
        corner_faces=((0,1,2),(0,1,3),(0,2,3),(1,2,3))
        edge_map={(0,1):4,(1,2):5,(0,2):6,
                  (0,3):7,(1,3):8,(2,3):9}
        full_faces=tuple(
            tuple(face)+tuple(
                edge_map[tuple(sorted((face[i],face[(i+1)%3])))]
                for i in range(3)
            )
            for face in corner_faces
        ) if mesh.t.shape[0]==10 else corner_faces
    else:
        if mesh.t.shape[0]==8:
            corner_faces=((0,1,4,2),(0,2,6,3),(0,3,5,1),
                          (2,4,7,6),(1,5,7,4),(3,6,7,5))
            full_faces=corner_faces
        else:
            index=lambda i,j,k:i+3*j+9*k
            full_faces=(
                tuple(index(i,j,0) for j in range(3) for i in range(3)),
                tuple(index(0,j,k) for k in range(3) for j in range(3)),
                tuple(index(i,0,k) for k in range(3) for i in range(3)),
                tuple(index(i,2,k) for k in range(3) for i in range(3)),
                tuple(index(2,j,k) for k in range(3) for j in range(3)),
                tuple(index(i,j,2) for j in range(3) for i in range(3)),
            )
            corner_faces=tuple(
                (face[0],face[2],face[8],face[6])
                for face in full_faces
            )
    found={}
    for nodes in mesh.t.T:
        for corner,full in zip(corner_faces,full_faces):
            key=tuple(sorted(int(nodes[i]) for i in corner))
            value=tuple(int(nodes[i]) for i in full)
            found.setdefault(key,[]).append(value)
    values=[
        adjacent[0] for adjacent in found.values()
        if len(adjacent)==2
    ]
    width=len(full_faces[0])
    if not values:
        return np.empty((width,0),dtype=np.int64)
    return np.asarray(values,dtype=np.int64).T


def _corner_count(mesh):
    if mesh.dim()==2:
        return 3 if mesh.t.shape[0] in (3,6) else 4
    if mesh.t.shape[0] in (4,10):
        return 4
    if mesh.t.shape[0] in (5,6):
        return mesh.t.shape[0]
    return 8


def _simplex_shapes(points,dimension,quadratic):
    points=np.asarray(points)
    bary=np.vstack((1.-points.sum(axis=0),points)).T
    dl=np.vstack((-np.ones(dimension),np.eye(dimension)))
    if not quadratic:
        return bary,np.broadcast_to(
            dl,(points.shape[1],dimension+1,dimension)
        ).copy()
    pairs=(
        ((0,1),(1,2),(0,2)) if dimension==2 else
        ((0,1),(1,2),(2,0),(0,3),(1,3),(2,3))
    )
    shape=np.empty((len(bary),len(bary[0])+len(pairs)))
    grad=np.empty((len(bary),shape.shape[1],dimension))
    for q,L in enumerate(bary):
        for i in range(dimension+1):
            shape[q,i]=L[i]*(2.*L[i]-1.)
            grad[q,i]=(4.*L[i]-1.)*dl[i]
        for k,(i,j) in enumerate(pairs,start=dimension+1):
            shape[q,k]=4.*L[i]*L[j]
            grad[q,k]=4.*(L[j]*dl[i]+L[i]*dl[j])
    return shape,grad


def _hex_shapes(points,quadratic):
    points=np.asarray(points)
    dimension=points.shape[0]
    if quadratic:
        count=3**dimension
        shape=np.empty((points.shape[1],count))
        grad=np.empty((points.shape[1],count,dimension))
        for q,coordinate in enumerate(points.T):
            values=[
                np.array([2*(x-.5)*(x-1),4*x*(1-x),2*x*(x-.5)])
                for x in coordinate
            ]
            derivatives=[
                np.array([4*x-3,4-8*x,4*x-1])
                for x in coordinate
            ]
            for node,index in enumerate(np.ndindex(*(3,)*dimension)):
                index=index[::-1]
                shape[q,node]=np.prod([
                    values[d][index[d]] for d in range(dimension)
                ])
                for d in range(dimension):
                    grad[q,node,d]=derivatives[d][index[d]]*np.prod([
                        values[k][index[k]]
                        for k in range(dimension) if k!=d
                    ])
        return shape,grad
    bits=(
        ((0,0),(1,0),(1,1),(0,1)) if dimension==2 else
        ((0,0,0),(1,0,0),(0,1,0),(0,0,1),
         (1,1,0),(1,0,1),(0,1,1),(1,1,1))
    )
    shape=np.empty((points.shape[1],len(bits)))
    grad=np.empty((points.shape[1],len(bits),dimension))
    for q,coordinate in enumerate(points.T):
        for node,bit in enumerate(bits):
            factors=np.where(bit,coordinate,1.-coordinate)
            shape[q,node]=np.prod(factors)
            for d in range(dimension):
                grad[q,node,d]=(1. if bit[d] else -1.)*np.prod(
                    np.delete(factors,d)
                )
    return shape,grad


def _wedge_shapes(points):
    points=np.asarray(points)
    r,s,z=points
    triangle=np.vstack((1.-r-s,r,s)).T
    triangle_grad=np.array([[-1.,-1.],[1.,0.],[0.,1.]])
    shape=np.empty((points.shape[1],6))
    grad=np.empty((points.shape[1],6,3))
    for q in range(points.shape[1]):
        for node in range(3):
            value=triangle[q,node]
            for layer,factor,dz in ((0,1.-z[q],-1.),(1,z[q],1.)):
                index=node+3*layer
                shape[q,index]=value*factor
                grad[q,index,:2]=triangle_grad[node]*factor
                grad[q,index,2]=value*dz
    return shape,grad


def _pyramid_shapes(points):
    points=np.asarray(points)
    x,y,z=points
    scale=1.-z
    if np.any(scale<=np.finfo(float).eps):
        raise ValueError("pyramid reference gradients are singular at the apex")
    a=scale-x;b=scale-y
    shape=np.column_stack((
        a*b/scale,x*b/scale,x*y/scale,a*y/scale,z
    ))
    grad=np.empty((points.shape[1],5,3))
    grad[:,0,0]=-b/scale
    grad[:,0,1]=-a/scale
    grad[:,0,2]=-1.+x*y/scale**2
    grad[:,1,0]=b/scale
    grad[:,1,1]=-x/scale
    grad[:,1,2]=-x*y/scale**2
    grad[:,2,0]=y/scale
    grad[:,2,1]=x/scale
    grad[:,2,2]=x*y/scale**2
    grad[:,3,0]=-y/scale
    grad[:,3,1]=a/scale
    grad[:,3,2]=-x*y/scale**2
    grad[:,4]=(0.,0.,1.)
    return shape,grad


def _mesh_geometry_shapes(mesh,points):
    nodes=mesh.t.shape[0]
    if mesh.dim()==2 and nodes in (3,6):
        return _simplex_shapes(points,2,nodes==6)
    if mesh.dim()==2:
        return _quad_shapes(points,nodes==9)
    if nodes in (4,10):
        return _simplex_shapes(points,3,nodes==10)
    if nodes==6:
        return _wedge_shapes(points)
    if nodes==5:
        return _pyramid_shapes(points)
    return _hex_shapes(points,nodes==27)


def _validate_quadrature(quadrature,dimension):
    points,weights=quadrature
    points=np.asarray(points,dtype=np.float64)
    weights=np.asarray(weights,dtype=np.float64)
    if points.ndim==1 and dimension==1:
        points=points[None,:]
    if points.ndim!=2 or points.shape[0]!=dimension:
        raise ValueError(
            f"quadrature points must have shape ({dimension}, nq)"
        )
    if weights.ndim!=1 or weights.shape[0]!=points.shape[1]:
        raise ValueError("quadrature weights must have shape (nq,)")
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(weights)):
        raise ValueError("quadrature points and weights must be finite")
    return np.ascontiguousarray(points),np.ascontiguousarray(weights)


def _tensor_quadrature(dimension,intorder):
    count=max(1,(int(intorder)+2)//2)
    points,weights=np.polynomial.legendre.leggauss(count)
    points=(points+1.)/2.;weights=weights/2.
    entries=[]
    for index in np.ndindex(*(count,)*dimension):
        entries.append((
            tuple(points[i] for i in index[::-1]),
            np.prod([weights[i] for i in index]),
        ))
    return (
        np.asarray([entry[0] for entry in entries]).T,
        np.asarray([entry[1] for entry in entries]),
    )


def _simplex_quadrature(dimension,intorder):
    count=max(1,(int(intorder)+dimension+1)//2)
    points,weights=np.polynomial.legendre.leggauss(count)
    points=(points+1.)/2.;weights=weights/2.
    transformed=[];transformed_weights=[]
    for index in np.ndindex(*(count,)*dimension):
        parameters=np.array([points[i] for i in index])
        coordinate=np.empty(dimension);remaining=1.;jacobian=1.
        for axis,value in enumerate(parameters):
            coordinate[axis]=remaining*value
            if axis<dimension-1:
                power=dimension-axis-1
                jacobian*=((1.-value)**power)
                remaining*=1.-value
        transformed.append(coordinate)
        transformed_weights.append(
            jacobian*np.prod([weights[i] for i in index])
        )
    return np.asarray(transformed).T,np.asarray(transformed_weights)


def _wedge_quadrature(intorder):
    triangle,triangle_weights=_simplex_quadrature(2,intorder)
    line,line_weights=_tensor_quadrature(1,intorder)
    points=[];weights=[]
    for k,z in enumerate(line[0]):
        for q in range(triangle.shape[1]):
            points.append((triangle[0,q],triangle[1,q],z))
            weights.append(triangle_weights[q]*line_weights[k])
    return np.asarray(points).T,np.asarray(weights)


def _pyramid_quadrature(intorder):
    cube,cube_weights=_tensor_quadrature(3,intorder)
    xi,eta,z=cube
    scale=1.-z
    points=np.vstack((scale*xi,scale*eta,z))
    return points,cube_weights*scale**2


class Basis:
    def __init__(
        self,mesh:MeshTet,element:ElementVector,intorder=2,
        quadrature=None,elements=None,
    ):
        intorder=2 if intorder is None else int(intorder)
        self.intorder=intorder
        if isinstance(element,ElementComposite):
            self._init_composite(mesh,element,intorder,quadrature)
            self.tind=np.arange(mesh.nelements,dtype=np.int64)
            if elements is not None:
                self._restrict_elements(elements)
            return
        scalar=element.elem if isinstance(element,ElementVector) else None
        base=scalar.elem if isinstance(scalar,ElementDG) else scalar
        supported=(
            ElementTriP0,ElementTriP1,ElementTriP2,
            ElementQuad0,ElementQuad1,ElementQuad2,
            ElementTetP0,ElementTetP1,ElementTetP2,
            ElementWedge1,
            ElementPyramid1,
            ElementHex0,ElementHex1,ElementHex2,
        )
        if not isinstance(element,ElementVector) or not isinstance(base,supported):
            raise NotImplementedError(
                "independent Basis supports P0 and nodal H1/DG elements"
            )
        self.mesh, self.elem = mesh, element
        self._base_element=base
        self._discontinuous=isinstance(scalar,ElementDG) or isinstance(
            base,(ElementTriP0,ElementQuad0,ElementTetP0,ElementHex0)
        )
        self._constant=isinstance(
            base,(ElementTriP0,ElementQuad0,ElementTetP0,ElementHex0)
        )
        self._tri=isinstance(base,(ElementTriP0,ElementTriP1,ElementTriP2))
        self._quadratic_tri=isinstance(base,ElementTriP2)
        self._quad=isinstance(base,(ElementQuad0,ElementQuad1,ElementQuad2))
        self._quadratic_quad=isinstance(base,ElementQuad2)
        self._tet=isinstance(base,(ElementTetP0,ElementTetP1,ElementTetP2))
        self._quadratic_tet=isinstance(base,ElementTetP2)
        self._wedge=isinstance(base,ElementWedge1)
        self._pyramid=isinstance(base,ElementPyramid1)
        self._quadratic_hex=isinstance(base,ElementHex2)
        if quadrature is not None:
            self.X,self.W=_validate_quadrature(
                quadrature,mesh.dim()
            )
        elif self._wedge:
            self.X,self.W=_wedge_quadrature(intorder)
        elif self._pyramid:
            self.X,self.W=_pyramid_quadrature(intorder)
        elif intorder>4:
            self.X,self.W=(
                _simplex_quadrature(mesh.dim(),intorder)
                if self._tri or self._tet else
                _tensor_quadrature(mesh.dim(),intorder)
            )
        elif self._quadratic_tri or (self._tri and intorder>=4):
            a=.445948490915965;b=.091576213509771
            bary=np.array([
                [a,a,1-2*a],[a,1-2*a,a],[1-2*a,a,a],
                [b,b,1-2*b],[b,1-2*b,b],[1-2*b,b,b],
            ])
            self.X=bary[:,1:].T
            self.W=np.array(
                [.1116907948390055]*3+[.054975871827661]*3
            )
        elif self._tri:
            self.X=np.array([
                [1./6.,2./3.,1./6.],
                [1./6.,1./6.,2./3.],
            ])
            self.W=np.full(3,1./6.)
        elif self._quadratic_quad or (self._quad and intorder>=4):
            points,weights=np.polynomial.legendre.leggauss(3)
            points=(points+1.)/2.;weights=weights/2.
            entries=[
                (x,y,wx*wy)
                for y,wy in zip(points,weights)
                for x,wx in zip(points,weights)
            ]
            self.X=np.array([entry[:2] for entry in entries]).T
            self.W=np.array([entry[2] for entry in entries])
        elif self._quad:
            points,weights=np.polynomial.legendre.leggauss(2)
            points=(points+1.)/2.;weights=weights/2.
            entries=[
                (x,y,wx*wy)
                for y,wy in zip(points,weights)
                for x,wx in zip(points,weights)
            ]
            self.X=np.array([entry[:2] for entry in entries]).T
            self.W=np.array([entry[2] for entry in entries])
        elif self._quadratic_tet or (self._tet and intorder>=4):
            self.X=np.array(
                [[.25,.7857142857142857,.0714285714285714,.0714285714285714,.0714285714285714,
                  .1005964238332008,.3994035761667992,.3994035761667992,.3994035761667992,
                  .1005964238332008,.1005964238332008],
                 [.25,.0714285714285714,.0714285714285714,.0714285714285714,.7857142857142857,
                  .3994035761667992,.1005964238332008,.3994035761667992,.1005964238332008,
                  .3994035761667992,.1005964238332008],
                 [.25,.0714285714285714,.0714285714285714,.7857142857142857,.0714285714285714,
                  .3994035761667992,.3994035761667992,.1005964238332008,.1005964238332008,
                  .1005964238332008,.3994035761667992]])
            self.W=np.array([-.01315555555555555]+[.00762222222222222]*4+[.02488888888888888]*6)
        elif self._tet:
            self.X = np.array(
                [[.5854101966249685,.1381966011250105,.1381966011250105,.1381966011250105],
                 [.1381966011250105,.5854101966249685,.1381966011250105,.1381966011250105],
                 [.1381966011250105,.1381966011250105,.5854101966249685,.1381966011250105]]
            )
            self.W = np.full(4,1./24.)
        elif self._quadratic_hex or (
            not self._tri and not self._tet and intorder>=4
        ):
            points=np.array([.5-np.sqrt(3./5.)/2.,.5,.5+np.sqrt(3./5.)/2.])
            one_weights=np.array([5./18.,4./9.,5./18.])
            entries=[(x,y,z,wx*wy*wz) for z,wz in zip(points,one_weights)
                     for y,wy in zip(points,one_weights) for x,wx in zip(points,one_weights)]
            self.X=np.array([entry[:3] for entry in entries]).T
            self.W=np.array([entry[3] for entry in entries])
        else:
            gauss=(1.+np.array([-1.,1.])/np.sqrt(3.))/2.
            self.X=np.array([[a,b,c] for c in gauss for b in gauss for a in gauss]).T
            self.W=np.full(8,1./8.)
        self.quadrature=(self.X.copy(),self.W.copy())
        base_connectivity=self._field_connectivity(mesh,base)
        if self._discontinuous:
            nodes=base_connectivity.shape[0]
            connectivity=np.arange(
                mesh.nelements*nodes,dtype=np.int64
            ).reshape(mesh.nelements,nodes).T
        else:
            connectivity=base_connectivity
        nodes=connectivity.shape[0]
        components=element._dim
        if self._discontinuous:
            active_nodes=np.arange(
                connectivity.max(initial=-1)+1,dtype=np.int64
            )
        else:
            used_nodes=np.zeros(mesh.p.shape[1],dtype=bool)
            used_nodes[connectivity]=True
            active_nodes=np.flatnonzero(used_nodes)
        self.N=len(active_nodes)*components
        self.nodal_dofs = np.arange(self.N).reshape(-1, components).T
        if self._discontinuous:
            local_dofs=connectivity
        else:
            node_positions=np.full(mesh.p.shape[1],-1,dtype=np.int64)
            node_positions[active_nodes]=np.arange(len(active_nodes))
            local_dofs=node_positions[connectivity]
        self.element_dofs = self.nodal_dofs[:,local_dofs].transpose(2,1,0).reshape(
            mesh.nelements, nodes*components
        ).T
        if self._discontinuous:
            if self._constant:
                positions=mesh.p[:,mesh.t[:_corner_count(mesh)]].mean(axis=1)
            else:
                positions=mesh.p[:,base_connectivity].transpose(0,2,1)
            positions=(
                positions.reshape(mesh.dim(),-1)
                if positions.ndim==3 else positions
            )
            self.doflocs=np.repeat(positions,components,axis=1)
        else:
            self.doflocs=np.repeat(
                mesh.p[:,active_nodes],components,axis=1
            )
        self.active_nodes=active_nodes
        self.field_connectivity=connectivity
        (
            self.tabulated_shape,self.tabulated_gradients,self.dx,
            self.global_coordinates,
        )=self._geometry()
        self.normals=None
        self.basis = self._vector_fields()
        self.tind=np.arange(mesh.nelements,dtype=np.int64)
        if elements is not None:
            self._restrict_elements(elements)

    @property
    def nelems(self):
        return self.dx.shape[0]

    def _element_ids(self,elements):
        if isinstance(elements,str):
            try:
                selected=np.asarray(self.mesh.subdomains[elements])
            except KeyError as error:
                raise KeyError(
                    f"unknown subdomain {elements!r}"
                ) from error
        elif callable(elements):
            centers=self.mesh.p[
                :,self.mesh.t[:_corner_count(self.mesh)]
            ].mean(axis=1)
            selected=np.asarray(elements(centers))
        else:
            selected=np.asarray(elements)
        if selected.dtype==bool:
            if selected.shape!=(self.mesh.nelements,):
                raise ValueError(
                    "element mask must have shape (mesh.nelements,)"
                )
            selected=np.flatnonzero(selected)
        selected=np.asarray(selected,dtype=np.int64).reshape(-1)
        if np.any(selected<0) or np.any(selected>=self.mesh.nelements):
            raise IndexError("element index is out of bounds")
        if len(np.unique(selected))!=len(selected):
            raise ValueError("element selection contains duplicates")
        return selected

    def _restrict_elements(self,elements):
        selected=self._element_ids(elements)
        positions={int(entity):local for local,entity in enumerate(self.tind)}
        try:
            local=np.asarray(
                [positions[int(entity)] for entity in selected],dtype=np.int64
            )
        except KeyError as error:
            raise ValueError(
                "element selection is not contained in this Basis"
            ) from error
        self.tind=selected.copy()
        self.element_dofs=self.element_dofs[:,local]
        self.dx=self.dx[local]
        self.global_coordinates=self.global_coordinates[local]
        self._geometry_determinants=self._geometry_determinants[local]
        self._geometry_tolerances=self._geometry_tolerances[local]
        self._geometry_condition_numbers=(
            self._geometry_condition_numbers[local]
        )
        self._update_geometry_diagnostics()
        if hasattr(self,"subbases"):
            for subbasis in self.subbases:
                subbasis._restrict_elements(selected)
        else:
            self.field_connectivity=self.field_connectivity[:,local]
            self.tabulated_shape=self.tabulated_shape[local]
            self.tabulated_gradients=self.tabulated_gradients[local]
            self.basis=self._vector_fields()
        return self

    def with_elements(self,elements):
        return type(self)(
            self.mesh,self.elem,quadrature=self.quadrature,
            elements=elements,
        )

    def interpolate(self,coefficients):
        coefficients=np.asarray(coefficients,dtype=np.float64)
        if coefficients.shape!=(self.N,):
            raise ValueError(
                f"coefficients must have shape ({self.N},)"
            )
        if hasattr(self,"subbases"):
            return tuple(
                subbasis._interpolate(coefficients)
                for subbasis in self.subbases
            )
        return self._interpolate(coefficients)

    def _interpolate(self,coefficients):
        components=self.elem._dim
        nodes=self.tabulated_shape.shape[2]
        local=coefficients[self.element_dofs.T].reshape(
            self.dx.shape[0],nodes,components
        )
        value=np.einsum(
            "eqn,enc->ceq",self.tabulated_shape,local
        )
        gradient=np.einsum(
            "eqnd,enc->cdeq",self.tabulated_gradients,local
        )
        if components==1:
            value=value[0]
            gradient=gradient[0]
        return DiscreteField(value,gradient)

    def split_bases(self):
        if not hasattr(self,"subbases"):
            raise ValueError("split_bases() requires ElementComposite")
        return tuple(
            Basis(
                self.mesh,element,intorder=self.intorder,
                quadrature=self.quadrature,elements=self.tind,
            )
            for element in (
                field if isinstance(field,ElementVector)
                else ElementVector(field,dim=1)
                for field in self.elem.elems
            )
        )

    def split_indices(self):
        if not hasattr(self,"subbases"):
            raise ValueError("split_indices() requires ElementComposite")
        return tuple(
            subbasis.nodal_dofs.reshape(-1,order="F").copy()
            for subbasis in self.subbases
        )

    def get_dofs(
        self,facets=None,elements=None,nodes=None,skip=None,
        *,components=None,fields=None,
    ):
        if sum(value is not None for value in (facets,elements,nodes))>1:
            raise ValueError("select only one of facets, elements, or nodes")
        if nodes is not None:
            selected=(
                np.asarray(nodes(self.mesh.p))
                if callable(nodes) else np.asarray(nodes)
            )
            if selected.dtype==bool:
                selected=np.flatnonzero(selected)
            selected=np.asarray(selected,dtype=np.int64).reshape(-1)
        elif elements is not None:
            element_ids=self._element_ids(elements)
            selected=np.unique(self.mesh.t[:,element_ids])
        else:
            boundary_facets=self.mesh.boundary_facets()
            if facets is None:
                selected_facets=boundary_facets
            elif isinstance(facets,str):
                try:
                    selected_facets=self.mesh.boundaries[facets]
                except KeyError as error:
                    raise KeyError(
                        f"unknown boundary {facets!r}"
                    ) from error
            elif callable(facets):
                centers=self.mesh.p[
                    :,self.mesh.facets[:,boundary_facets]
                ].mean(axis=1)
                mask=np.asarray(facets(centers),dtype=bool)
                if mask.shape!=(len(boundary_facets),):
                    raise ValueError(
                        "facet predicate must return one boolean per "
                        "boundary facet"
                    )
                selected_facets=boundary_facets[mask]
            else:
                selected_facets=np.asarray(
                    facets,dtype=np.int64
                ).reshape(-1)
            selected=np.unique(
                self.mesh._facet_connectivity(
                    selected_facets,full=True
                )
            )
        if getattr(self,"_discontinuous",False):
            if elements is not None:
                positions={
                    int(entity):local
                    for local,entity in enumerate(self.tind)
                }
                local_elements=np.asarray([
                    positions[int(entity)] for entity in element_ids
                    if int(entity) in positions
                ],dtype=np.int64)
                dofs=np.unique(self.element_dofs[:,local_elements])
            else:
                dofs=np.empty(0,dtype=np.int64)
            count=self.elem._dim
            groups={
                f"u^{component+1}":dofs[dofs%count==component]
                for component in range(count)
            }
            return self._filtered_dofs_view(
                groups,skip=skip,components=components,fields=fields
            )
        selected_set=set(map(int,selected))
        if hasattr(self,"subbases"):
            groups={}
            for field,subbasis in enumerate(self.subbases):
                component_values=[[] for _ in range(subbasis.elem._dim)]
                for local,node in enumerate(subbasis.active_nodes):
                    if int(node) in selected_set:
                        for component,dof in enumerate(
                            subbasis.nodal_dofs[:,local]
                        ):
                            component_values[component].append(dof)
                for component,values in enumerate(component_values):
                    groups[f"field{field}^{component+1}"]=values
        else:
            component_values=[[] for _ in range(self.elem._dim)]
            for local,node in enumerate(self.active_nodes):
                if int(node) in selected_set:
                    for component,dof in enumerate(
                        self.nodal_dofs[:,local]
                    ):
                        component_values[component].append(dof)
            groups={
                f"u^{component+1}":values
                for component,values in enumerate(component_values)
            }
        return self._filtered_dofs_view(
            groups,skip=skip,components=components,fields=fields
        )

    @staticmethod
    def _indices(value,count,label):
        if value is None:
            return list(range(count))
        values=[value] if np.isscalar(value) else list(value)
        result=[]
        for item in values:
            index=int(item)
            if index<0 or index>=count:
                raise IndexError(f"{label} index {index} is out of range")
            if index not in result:
                result.append(index)
        return result

    def _filtered_dofs_view(
        self,groups,*,skip=None,components=None,fields=None
    ):
        if hasattr(self,"subbases"):
            field_ids=self._indices(fields,len(self.subbases),"field")
            if components is not None and not isinstance(components,dict):
                raise TypeError(
                    "composite components must be a {field: components} mapping"
                )
            selected_names=[]
            for field in field_ids:
                count=self.subbases[field].elem._dim
                requested=(
                    None if components is None
                    else components.get(field,None)
                )
                for component in self._indices(
                    requested,count,f"field {field} component"
                ):
                    selected_names.append(f"field{field}^{component+1}")
        else:
            if fields is not None:
                raise ValueError("fields is only valid for ElementComposite")
            selected_names=[
                f"u^{component+1}" for component in self._indices(
                    components,self.elem._dim,"component"
                )
            ]
        view=DofsView(
            np.concatenate([
                np.asarray(groups[name],dtype=np.int64)
                for name in selected_names
            ]) if selected_names else np.empty(0,dtype=np.int64),
            self.doflocs,
            {name:groups[name] for name in selected_names},
        )
        return view.drop(skip) if skip else view

    def _init_composite(self,mesh,element,intorder,quadrature=None):
        vector_elements=tuple(
            field if isinstance(field,ElementVector)
            else ElementVector(field,dim=1)
            for field in element.elems
        )
        subbases=[
            Basis(
                mesh,field,intorder=intorder,quadrature=quadrature
            )
            for field in vector_elements
        ]
        if any(basis._discontinuous for basis in subbases):
            raise NotImplementedError(
                "ElementComposite with discontinuous fields is not implemented"
            )
        quadrature_shapes={(basis.X.shape,basis.dx.shape) for basis in subbases}
        if len(quadrature_shapes)!=1 or any(
            not np.array_equal(subbases[0].X,basis.X)
            for basis in subbases[1:]
        ):
            raise ValueError(
                "composite fields require a common quadrature rule; "
                "increase intorder for mixed-order elements"
            )
        components=tuple(field._dim for field in vector_elements)
        next_dof=0
        field_node_dofs=[]
        active_sets=[set(map(int,basis.active_nodes)) for basis in subbases]
        for field,(active,count) in enumerate(zip(active_sets,components)):
            field_node_dofs.append({})
        for node in range(mesh.p.shape[1]):
            for field,(active,count) in enumerate(zip(active_sets,components)):
                if node in active:
                    field_node_dofs[field][node]=np.arange(
                        next_dof,next_dof+count,dtype=np.int64
                    )
                    next_dof+=count
        for field,(subbasis,count) in enumerate(zip(subbases,components)):
            nodal=np.stack([
                [field_node_dofs[field][int(node)][component]
                 for node in subbasis.active_nodes]
                for component in range(count)
            ])
            subbasis.nodal_dofs=nodal
            subbasis.N=next_dof
            positions=np.full(mesh.p.shape[1],-1,dtype=np.int64)
            positions[subbasis.active_nodes]=np.arange(len(subbasis.active_nodes))
            local=positions[subbasis.field_connectivity]
            subbasis.element_dofs=nodal[:,local].transpose(
                2,1,0
            ).reshape(mesh.nelements,local.shape[0]*count).T
        first=subbases[0]
        self.mesh,self.elem=mesh,element
        self.subbases=tuple(subbases)
        self.field_components=components
        self.N=next_dof
        common_nodes=sorted(set.intersection(*active_sets))
        self.nodal_dofs=np.concatenate([
            np.stack([
                field_node_dofs[field][node]
                for node in common_nodes
            ],axis=1)
            for field in range(len(subbases))
        ],axis=0)
        self.element_dofs=np.concatenate(
            [subbasis.element_dofs for subbasis in subbases],axis=0
        )
        doflocs=np.empty((mesh.p.shape[0],self.N))
        for field,subbasis in enumerate(subbases):
            for local_node,node in enumerate(subbasis.active_nodes):
                doflocs[:,subbasis.nodal_dofs[:,local_node]]=mesh.p[:,node,None]
        self.doflocs=doflocs
        self.X,self.W=first.X,first.W
        self.quadrature=(self.X.copy(),self.W.copy())
        self.dx=first.dx
        self.global_coordinates=first.global_coordinates
        self._geometry_determinants=first._geometry_determinants
        self._geometry_tolerances=first._geometry_tolerances
        self._geometry_condition_numbers=first._geometry_condition_numbers
        self.geometry_diagnostics=first.geometry_diagnostics
        self.normals=None
        self.basis=tuple()

    @staticmethod
    def _field_connectivity(mesh,scalar):
        if isinstance(
            scalar,(ElementTriP0,ElementQuad0,ElementTetP0,ElementHex0)
        ):
            return np.arange(mesh.nelements,dtype=np.int64)[None,:]
        if isinstance(scalar,ElementTriP1):
            return mesh.t[:3]
        if isinstance(scalar,ElementTriP2):
            return mesh.t[:6]
        if isinstance(scalar,ElementQuad1):
            return mesh.t[:4]
        if isinstance(scalar,ElementQuad2):
            return mesh.t[:9]
        if isinstance(scalar,ElementTetP1):
            return mesh.t[:4]
        if isinstance(scalar,ElementTetP2):
            return mesh.t[:10]
        if isinstance(scalar,ElementWedge1):
            return mesh.t[:6]
        if isinstance(scalar,ElementPyramid1):
            return mesh.t[:5]
        if isinstance(scalar,ElementHex1):
            if mesh.t.shape[0]==27:
                return mesh.t[[0,2,6,18,8,20,24,26]]
            return mesh.t[:8]
        if isinstance(scalar,ElementHex2):
            return mesh.t[:27]
        raise NotImplementedError("unsupported scalar element")

    def _geometry(self):
        if self._constant:
            shape=np.ones((self.X.shape[1],1))
            refgrad=np.zeros((
                self.X.shape[1],1,self.mesh.dim()
            ))
        elif self._quadratic_tri:
            bary=np.vstack((1.-self.X.sum(axis=0),self.X)).T
            pairs=((0,1),(1,2),(0,2))
            dl=np.array([[-1.,-1.],[1.,0.],[0.,1.]])
            nq=len(self.W);shape=np.empty((nq,6))
            refgrad=np.empty((nq,6,2))
            for q,L in enumerate(bary):
                for i in range(3):
                    shape[q,i]=L[i]*(2.*L[i]-1.)
                    refgrad[q,i]=(4.*L[i]-1.)*dl[i]
                for k,(i,j) in enumerate(pairs,start=3):
                    shape[q,k]=4.*L[i]*L[j]
                    refgrad[q,k]=4.*(L[j]*dl[i]+L[i]*dl[j])
        elif self._tri:
            shape=np.vstack((1.-self.X.sum(axis=0),self.X)).T
            reference=np.array([[-1.,-1.],[1.,0.],[0.,1.]])
            refgrad=np.broadcast_to(
                reference,(self.X.shape[1],3,2)
            )
        elif self._quad:
            shape,refgrad=_quad_shapes(self.X,self._quadratic_quad)
        elif self._quadratic_tet:
            bary=np.vstack((1.-self.X.sum(axis=0),self.X)).T
            pairs=((0,1),(1,2),(2,0),(0,3),(1,3),(2,3))
            nq=len(self.W);shape=np.empty((nq,10));refgrad=np.empty((nq,10,3))
            dl=np.array([[-1.,-1.,-1.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
            for q,L in enumerate(bary):
                for i in range(4):
                    shape[q,i]=L[i]*(2.*L[i]-1.)
                    refgrad[q,i]=(4.*L[i]-1.)*dl[i]
                for k,(i,j) in enumerate(pairs,start=4):
                    shape[q,k]=4.*L[i]*L[j]
                    refgrad[q,k]=4.*(L[j]*dl[i]+L[i]*dl[j])
        elif self._tet:
            reference=np.array([[-1.,-1.,-1.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
            shape=np.vstack((1.-self.X.sum(axis=0),self.X)).T
            refgrad=np.broadcast_to(reference,(self.X.shape[1],4,3))
        elif self._wedge:
            shape,refgrad=_wedge_shapes(self.X)
        elif self._pyramid:
            shape,refgrad=_pyramid_shapes(self.X)
        elif self._quadratic_hex:
            grid=(0.,.5,1.)
            def values(x):
                return np.array([2.*(x-.5)*(x-1.),4.*x*(1.-x),2.*x*(x-.5)])
            def derivatives(x):
                return np.array([4.*x-3.,4.-8.*x,4.*x-1.])
            nq=self.X.shape[1];shape=np.empty((nq,27));refgrad=np.empty((nq,27,3))
            for q,(x,y,z) in enumerate(self.X.T):
                v=(values(x),values(y),values(z));d=(derivatives(x),derivatives(y),derivatives(z))
                n=0
                for k in range(3):
                    for j in range(3):
                        for i in range(3):
                            shape[q,n]=v[0][i]*v[1][j]*v[2][k]
                            refgrad[q,n]=(
                                d[0][i]*v[1][j]*v[2][k],
                                v[0][i]*d[1][j]*v[2][k],
                                v[0][i]*v[1][j]*d[2][k],
                            );n+=1
        else:
            bits=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],
                           [1,1,0],[1,0,1],[0,1,1],[1,1,1]])
            shape=np.empty((self.X.shape[1],8));refgrad=np.empty((self.X.shape[1],8,3))
            for q,xi in enumerate(self.X.T):
                for n,bit in enumerate(bits):
                    factors=np.where(bit,xi,1.-xi);shape[q,n]=np.prod(factors)
                    for j in range(3):
                        refgrad[q,n,j]=(1. if bit[j] else -1.)*np.prod(np.delete(factors,j))
        geometry_shape=shape
        geometry_refgrad=refgrad
        geometry_nodes=self.mesh.t.shape[0]
        if geometry_nodes!=shape.shape[1]:
            if not self._constant and geometry_nodes<shape.shape[1]:
                raise ValueError(
                    "mesh geometry nodes are incompatible with the element"
                )
            geometry_shape,geometry_refgrad=_mesh_geometry_shapes(
                self.mesh,self.X
            )
        result=tabulate_basis_geometry(
            self.mesh.p,self.mesh.t,shape,refgrad,
            geometry_shape,geometry_refgrad,self.W,
        )
        geometry=result[:4]
        self._geometry_determinants=np.asarray(result[4])
        self._geometry_tolerances=np.asarray(result[5])
        self._geometry_condition_numbers=np.asarray(result[6])
        self._update_geometry_diagnostics()
        return geometry

    def _update_geometry_diagnostics(self):
        determinants=self._geometry_determinants
        tolerances=self._geometry_tolerances
        if determinants.size==0:
            self.geometry_diagnostics=GeometryDiagnostics(
                element_count=0,
                quadrature_points_per_element=(
                    determinants.shape[1] if determinants.ndim==2 else 0
                ),
                minimum_determinant=float("inf"),
                maximum_determinant=float("-inf"),
                minimum_scaled_determinant=float("inf"),
                worst_element=-1,
                worst_quadrature_point=-1,
                determinant_tolerance=0.,
                negative_orientation_elements=0,
                maximum_condition_number=0.,
                worst_condition_element=-1,
                worst_condition_quadrature_point=-1,
            )
            return
        scale_power=tolerances/(
            64.*np.finfo(np.float64).eps
        )
        scaled=np.abs(determinants)/scale_power
        flat=int(np.argmin(scaled))
        local_element,quadrature_point=np.unravel_index(
            flat,determinants.shape
        )
        element=(
            int(self.tind[local_element])
            if hasattr(self,"tind") and len(self.tind)==len(determinants)
            else int(local_element)
        )
        condition_flat=int(np.argmax(self._geometry_condition_numbers))
        condition_local,condition_quadrature=np.unravel_index(
            condition_flat,self._geometry_condition_numbers.shape
        )
        condition_element=(
            int(self.tind[condition_local])
            if hasattr(self,"tind") and len(self.tind)==len(determinants)
            else int(condition_local)
        )
        self.geometry_diagnostics=GeometryDiagnostics(
            element_count=int(determinants.shape[0]),
            quadrature_points_per_element=int(determinants.shape[1]),
            minimum_determinant=float(np.min(determinants)),
            maximum_determinant=float(np.max(determinants)),
            minimum_scaled_determinant=float(scaled.flat[flat]),
            worst_element=element,
            worst_quadrature_point=int(quadrature_point),
            determinant_tolerance=float(tolerances.flat[flat]),
            negative_orientation_elements=int(np.count_nonzero(
                determinants[:,0]<0.
            )),
            maximum_condition_number=float(
                self._geometry_condition_numbers.flat[condition_flat]
            ),
            worst_condition_element=condition_element,
            worst_condition_quadrature_point=int(condition_quadrature),
        )

    def _evaluate_reference(self, points):
        old_points = self.X
        self.X = np.asarray(points)
        try:
            if self._constant:
                return (
                    np.ones((self.X.shape[1],1)),
                    np.zeros((self.X.shape[1],1,self.mesh.dim())),
                )
            if self._quadratic_tri:
                bary=np.vstack((1.-self.X.sum(axis=0),self.X)).T
                pairs=((0,1),(1,2),(0,2))
                dl=np.array([[-1.,-1.],[1.,0.],[0.,1.]])
                shape=np.empty((len(bary),6))
                grad=np.empty((len(bary),6,2))
                for q,L in enumerate(bary):
                    for i in range(3):
                        shape[q,i]=L[i]*(2.*L[i]-1.)
                        grad[q,i]=(4.*L[i]-1.)*dl[i]
                    for k,(i,j) in enumerate(pairs,start=3):
                        shape[q,k]=4.*L[i]*L[j]
                        grad[q,k]=4.*(L[j]*dl[i]+L[i]*dl[j])
                return shape,grad
            if self._tri:
                shape=np.vstack((1.-self.X.sum(axis=0),self.X)).T
                grad=np.broadcast_to(
                    np.array([[-1.,-1.],[1.,0.],[0.,1.]]),
                    (shape.shape[0],3,2),
                )
                return shape,grad
            if self._quad:
                return _quad_shapes(self.X,self._quadratic_quad)
            if self._quadratic_tet:
                bary=np.vstack((1.-self.X.sum(axis=0),self.X)).T
                pairs=((0,1),(1,2),(2,0),(0,3),(1,3),(2,3))
                shape=np.empty((len(bary),10));grad=np.empty((len(bary),10,3))
                dl=np.array([[-1.,-1.,-1.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
                for q,L in enumerate(bary):
                    for i in range(4):
                        shape[q,i]=L[i]*(2*L[i]-1);grad[q,i]=(4*L[i]-1)*dl[i]
                    for k,(i,j) in enumerate(pairs,start=4):
                        shape[q,k]=4*L[i]*L[j];grad[q,k]=4*(L[j]*dl[i]+L[i]*dl[j])
                return shape,grad
            if self._tet:
                shape=np.vstack((1.-self.X.sum(axis=0),self.X)).T
                grad=np.broadcast_to(np.array([[-1.,-1.,-1.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]),
                                     (shape.shape[0],4,3))
                return shape,grad
            if self._wedge:
                return _wedge_shapes(self.X)
            if self._pyramid:
                return _pyramid_shapes(self.X)
            if self._quadratic_hex:
                values=lambda x:np.array([2*(x-.5)*(x-1),4*x*(1-x),2*x*(x-.5)])
                deriv=lambda x:np.array([4*x-3,4-8*x,4*x-1])
                shape=np.empty((self.X.shape[1],27));grad=np.empty((self.X.shape[1],27,3))
                for q,(x,y,z) in enumerate(self.X.T):
                    v=(values(x),values(y),values(z));d=(deriv(x),deriv(y),deriv(z));n=0
                    for k in range(3):
                        for j in range(3):
                            for i in range(3):
                                shape[q,n]=v[0][i]*v[1][j]*v[2][k]
                                grad[q,n]=(d[0][i]*v[1][j]*v[2][k],
                                           v[0][i]*d[1][j]*v[2][k],
                                           v[0][i]*v[1][j]*d[2][k]);n+=1
                return shape,grad
            bits=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],
                           [1,1,0],[1,0,1],[0,1,1],[1,1,1]])
            shape=np.empty((self.X.shape[1],8));grad=np.empty((self.X.shape[1],8,3))
            for q,xi in enumerate(self.X.T):
                for n,bit in enumerate(bits):
                    factors=np.where(bit,xi,1-xi);shape[q,n]=np.prod(factors)
                    for j in range(3):
                        grad[q,n,j]=(1 if bit[j] else -1)*np.prod(np.delete(factors,j))
            return shape,grad
        finally:
            self.X = old_points

    def _vector_fields(self):
        return _LazyVectorFields(self)


class FacetBasis:
    def __init__(
        self,mesh,element,facets=None,intorder=2,quadrature=None,
        *,_side=0,_interior=False
    ):
        intorder=2 if intorder is None else int(intorder)
        if isinstance(facets,str):
            try:
                facets=mesh.boundaries[facets]
            except KeyError as error:
                raise KeyError(f"unknown boundary {facets!r}") from error
        oriented=facets if isinstance(facets,FacetRegion) else None
        orientation_sides=(None if oriented is None else oriented.sides)
        normal_signs=(None if oriented is None else oriented.normal_signs)
        if mesh.dim()==2:
            self._init_2d(
                mesh,element,facets,intorder,
                side=_side,interior=_interior,
                quadrature=quadrature,
                orientation_sides=orientation_sides,
                normal_signs=normal_signs,
            )
            return
        if mesh.t.shape[0] in (5,6):
            self._init_mixed_3d(
                mesh,element,facets,intorder,
                side=_side,interior=_interior,quadrature=quadrature,
                orientation_sides=orientation_sides,
                normal_signs=normal_signs,
            )
            return
        volume = Basis(mesh, element, intorder=intorder)
        scalar=(
            element.elem.elem
            if isinstance(element.elem,ElementDG) else element.elem
        )
        is_tet=isinstance(
            scalar,(ElementTetP0,ElementTetP1,ElementTetP2)
        )
        quadratic_field=isinstance(
            scalar,(ElementTetP2,ElementHex2)
        )
        if _interior:
            facet_ids=(
                mesh.interior_facets() if facets is None else
                np.asarray(facets,dtype=np.int64)
            )
        else:
            facet_ids=(
                mesh.boundary_facets()
                if facets is None else np.asarray(facets)
            )
        id_selection=np.asarray(facet_ids).ndim==1
        selected_ids=(
            np.asarray(facet_ids,dtype=np.int64) if id_selection else None
        )
        if np.asarray(facet_ids).ndim==1:
            facets=mesh._facet_connectivity(facet_ids,full=True)
        else:
            facets=np.asarray(facet_ids,dtype=np.int64)
        quadratic_geometry=mesh.t.shape[0] in (10,27)
        expected_face_nodes=(
            6 if is_tet and quadratic_geometry else
            3 if is_tet else
            9 if quadratic_geometry else 4
        )
        if facets.shape[0] != expected_face_nodes:
            facets = facets.T
        if is_tet:
            local_faces=((0,1,2),(0,1,3),(0,2,3),(1,2,3))
            if mesh.t.shape[0]==10:
                edge_map={(0,1):4,(1,2):5,(0,2):6,(0,3):7,(1,3):8,(2,3):9}
                local_faces=tuple(tuple(face)+tuple(
                    edge_map[tuple(sorted((face[i],face[(i+1)%3])))] for i in range(3)
                ) for face in local_faces)
            corners=tuple(face[:3] for face in local_faces)
            if intorder>=4:
                a=.445948490915965;b=.091576213509771
                bary=np.array([[a,a,1-2*a],[a,1-2*a,a],[1-2*a,a,a],
                               [b,b,1-2*b],[b,1-2*b,b],[1-2*b,b,b]])
                face_weights=np.array([.1116907948390055]*3+[.054975871827661]*3)
            else:
                bary=np.array([[2/3,1/6,1/6],[1/6,2/3,1/6],[1/6,1/6,2/3]])
                face_weights=np.full(3,1/6)
            face_points=bary[:,1:]
        else:
            if mesh.t.shape[0]!=27:
                local_faces=((0,1,4,2),(0,2,6,3),(0,3,5,1),
                             (2,4,7,6),(1,5,7,4),(3,6,7,5))
                corners=local_faces
            else:
                index=lambda i,j,k:i+3*j+9*k
                local_faces=(
                    tuple(index(i,j,0) for j in range(3) for i in range(3)),
                    tuple(index(0,j,k) for k in range(3) for j in range(3)),
                    tuple(index(i,0,k) for k in range(3) for i in range(3)),
                    tuple(index(i,2,k) for k in range(3) for i in range(3)),
                    tuple(index(2,j,k) for k in range(3) for j in range(3)),
                    tuple(index(i,j,2) for j in range(3) for i in range(3)))
                corners=tuple((face[0],face[2],face[8],face[6]) for face in local_faces)
            order=3 if intorder>=4 else 2
            points,weights=np.polynomial.legendre.leggauss(order)
            points=(points+1.)/2.;weights=weights/2.
            face_points=np.array([[r,s] for s in points for r in points])
            face_weights=np.array([wr*ws for ws in weights for wr in weights])
        if quadrature is not None:
            facet_points,face_weights=_validate_quadrature(
                quadrature,2
            )
            face_points=facet_points.T
        elif intorder>4:
            facet_points,face_weights=(
                _simplex_quadrature(2,intorder) if is_tet else
                _tensor_quadrature(2,intorder)
            )
            face_points=facet_points.T
        lookup={}
        for e,nodes in enumerate(mesh.t.T):
            for local_index,(local,corner) in enumerate(
                zip(local_faces,corners)
            ):
                key=tuple(sorted(int(nodes[i]) for i in corner))
                lookup.setdefault(key,[]).append(
                    (local_index,e,local,corner)
                )
        for adjacent in lookup.values():
            adjacent.sort(key=lambda item:(item[0],item[1]))
        nq=len(face_weights);nodes_per_element=len(element.elem.doflocs)
        self.mesh, self.elem = mesh, element
        self.X=face_points.T;self.W=face_weights
        self.quadrature=(self.X.copy(),self.W.copy())
        self.N = volume.N
        self.doflocs = volume.doflocs
        components=element._dim
        self.element_dofs=np.empty((nodes_per_element*components,facets.shape[1]),dtype=np.int64)
        self.tabulated_shape=np.zeros((facets.shape[1],nq,nodes_per_element))
        self.tabulated_gradients=np.empty((facets.shape[1],nq,nodes_per_element,3))
        self.dx = np.empty((facets.shape[1], nq))
        self.global_coordinates=np.empty((facets.shape[1],nq,3))
        self.normals=np.empty((facets.shape[1],nq,3))
        self.parent_elements=np.empty(facets.shape[1],dtype=np.int64)
        self.facet_ids=(
            selected_ids.copy() if selected_ids is not None
            else np.full(facets.shape[1],-1,dtype=np.int64)
        )
        facet_sides=(
            np.full(facets.shape[1],_side,dtype=np.int8)
            if orientation_sides is None else np.asarray(orientation_sides)
        )
        facet_signs=(
            np.ones(facets.shape[1],dtype=np.int8)
            if normal_signs is None else np.asarray(normal_signs)
        )
        if facet_sides.shape!=(facets.shape[1],) or facet_signs.shape!=(facets.shape[1],):
            raise ValueError("facet orientation metadata has the wrong shape")
        self.facet_sides=facet_sides.copy()
        self.normal_signs=facet_signs.copy()
        self.local_faces=[]
        for f, face in enumerate(facets.T):
            face_corners=face[:3] if is_tet else (face[0],face[2] if len(face)==9 else face[1],
                face[8] if len(face)==9 else face[2],face[6] if len(face)==9 else face[3])
            adjacent=lookup[tuple(sorted(map(int,face_corners)))]
            local_side=int(facet_sides[f])
            if selected_ids is not None:
                parent=int(mesh.f2t[local_side,selected_ids[f]])
                matching=[item for item in adjacent if item[1]==parent]
            else:
                matching=(
                    [adjacent[local_side]] if local_side<len(adjacent) else []
                )
            if not matching:
                raise ValueError(
                    f"facet {tuple(face_corners)} does not have side {local_side}"
                )
            _,e,local,corner=matching[0]
            self.parent_elements[f]=e
            self.local_faces.append(tuple(local))
            self.element_dofs[:, f] = volume.element_dofs[:, e]
            reference_vertices=(
                ElementTetP2().doflocs
                if mesh.t.shape[0]==10 else
                ElementHex2().doflocs
                if mesh.t.shape[0]==27 else
                np.array([[0.,0.,0.],[1.,0.,0.],
                          [0.,1.,0.],[0.,0.,1.]])
                if is_tet else
                np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],
                          [0.,0.,1.],[1.,1.,0.],[1.,0.,1.],
                          [0.,1.,1.],[1.,1.,1.]])
            )
            local_corner_ids=np.asarray(corner)
            global_corners=mesh.t[local_corner_ids,e]
            order=[
                int(np.flatnonzero(global_corners==node)[0])
                for node in face_corners
            ]
            reference_corners=reference_vertices[
                local_corner_ids[np.asarray(order)]
            ]
            for q,(r,s) in enumerate(face_points):
                if is_tet:
                    reference=(1-r-s)*reference_corners[0]+r*reference_corners[1]+s*reference_corners[2]
                    derivatives=np.stack((reference_corners[1]-reference_corners[0],
                                          reference_corners[2]-reference_corners[0]),axis=1)
                else:
                    weights4=np.array([(1-r)*(1-s),r*(1-s),r*s,(1-r)*s])
                    reference=weights4@reference_corners
                    derivatives=np.stack((
                        (1-s)*(reference_corners[1]-reference_corners[0])+s*(reference_corners[2]-reference_corners[3]),
                        (1-r)*(reference_corners[3]-reference_corners[0])+r*(reference_corners[2]-reference_corners[1])),axis=1)
                shape,refgrad=volume._evaluate_reference(reference[None,:].T)
                if mesh.t.shape[0]==shape.shape[1]:
                    geometry_shape,geometry_refgrad=shape,refgrad
                else:
                    geometry_shape,geometry_refgrad=(
                        _mesh_geometry_shapes(
                            mesh,reference[None,:].T
                        )
                    )
                x=mesh.p[:,mesh.t[:,e]]
                jacobian=x@geometry_refgrad[0]
                physical=refgrad[0]@np.linalg.inv(jacobian)
                tangents=jacobian@derivatives
                point=geometry_shape[0]@x.T
                normal=np.cross(tangents[:,0],tangents[:,1])
                normal/=np.linalg.norm(normal)
                if np.dot(normal,point-x.mean(axis=1))<0.:
                    normal=-normal
                if _interior and local_side==1:
                    normal=-normal
                normal*=facet_signs[f]
                self.tabulated_shape[f,q]=shape[0]
                self.tabulated_gradients[f,q]=physical
                self.global_coordinates[f,q]=point
                self.normals[f,q]=normal
                self.dx[f,q]=np.linalg.norm(np.cross(tangents[:,0],tangents[:,1]))*face_weights[q]
        self.basis = self._vector_fields()
        self.volume_basis=volume

    def _init_mixed_3d(
        self,mesh,element,facets,intorder,*,side,interior,quadrature,
        orientation_sides=None,normal_signs=None,
    ):
        volume=Basis(mesh,element,intorder=intorder)
        if facets is None:
            facet_ids=(
                mesh.interior_facets() if interior else mesh.boundary_facets()
            )
        else:
            facet_ids=np.asarray(facets,dtype=np.int64)
        if facet_ids.ndim!=1:
            raise ValueError(
                "Wedge6/Pyramid5 FacetBasis expects one-dimensional facet IDs"
            )
        mesh._build_topology()
        if quadrature is None:
            parameter_points,parameter_weights=_tensor_quadrature(2,intorder)
        else:
            parameter_points,parameter_weights=_validate_quadrature(
                quadrature,2
            )
        face_points=parameter_points.T
        nq=len(parameter_weights)
        scalar=(
            element.elem.elem
            if isinstance(element.elem,ElementDG) else element.elem
        )
        if not isinstance(scalar,(ElementWedge1,ElementPyramid1)):
            raise ValueError("mixed-face mesh requires its matching nodal element")
        local_faces=mesh._local_facets()
        reference_vertices=scalar.doflocs
        nodes_per_element=len(scalar.doflocs)
        components=element._dim
        count=len(facet_ids)
        self.mesh,self.elem=mesh,element
        self.facet_ids=np.asarray(facet_ids,dtype=np.int64)
        facet_sides=(
            np.full(count,side,dtype=np.int8)
            if orientation_sides is None else np.asarray(orientation_sides)
        )
        facet_signs=(
            np.ones(count,dtype=np.int8)
            if normal_signs is None else np.asarray(normal_signs)
        )
        if facet_sides.shape!=(count,) or facet_signs.shape!=(count,):
            raise ValueError("facet orientation metadata has the wrong shape")
        self.facet_sides=facet_sides.copy()
        self.normal_signs=facet_signs.copy()
        self.X=parameter_points;self.W=parameter_weights
        self.quadrature=(self.X.copy(),self.W.copy())
        self.N=volume.N;self.doflocs=volume.doflocs
        self.element_dofs=np.empty(
            (nodes_per_element*components,count),dtype=np.int64
        )
        self.tabulated_shape=np.empty((count,nq,nodes_per_element))
        self.tabulated_gradients=np.empty((count,nq,nodes_per_element,3))
        self.dx=np.empty((count,nq))
        self.global_coordinates=np.empty((count,nq,3))
        self.normals=np.empty((count,nq,3))
        self.parent_elements=np.empty(count,dtype=np.int64)
        self.local_faces=[]
        for f,facet in enumerate(facet_ids):
            local_side=int(facet_sides[f])
            element_index=int(mesh.f2t[local_side,facet])
            if element_index<0:
                raise ValueError(
                    f"facet {facet} does not have side {local_side}"
                )
            local_index=int(np.flatnonzero(
                mesh.t2f[:,element_index]==facet
            )[0])
            local=local_faces[local_index]
            corners=reference_vertices[np.asarray(local)]
            self.parent_elements[f]=element_index
            self.local_faces.append(tuple(local))
            self.element_dofs[:,f]=volume.element_dofs[:,element_index]
            geometry_nodes=mesh.p[:,mesh.t[:,element_index]]
            for q,(u,v) in enumerate(face_points):
                if len(local)==3:
                    r=u;s=(1.-u)*v
                    reference=(1.-r-s)*corners[0]+r*corners[1]+s*corners[2]
                    derivatives=np.stack((
                        corners[1]-corners[0]-v*(corners[2]-corners[0]),
                        (1.-u)*(corners[2]-corners[0]),
                    ),axis=1)
                else:
                    weights4=np.array(
                        [(1.-u)*(1.-v),u*(1.-v),u*v,(1.-u)*v]
                    )
                    reference=weights4@corners
                    derivatives=np.stack((
                        (1.-v)*(corners[1]-corners[0])
                        +v*(corners[2]-corners[3]),
                        (1.-u)*(corners[3]-corners[0])
                        +u*(corners[2]-corners[1]),
                    ),axis=1)
                shape,refgrad=volume._evaluate_reference(reference[:,None])
                geometry_shape,geometry_refgrad=_mesh_geometry_shapes(
                    mesh,reference[:,None]
                )
                jacobian=geometry_nodes@geometry_refgrad[0]
                physical=refgrad[0]@np.linalg.inv(jacobian)
                tangents=jacobian@derivatives
                point=geometry_shape[0]@geometry_nodes.T
                cross=np.cross(tangents[:,0],tangents[:,1])
                measure=np.linalg.norm(cross)
                normal=cross/measure
                if np.dot(normal,point-geometry_nodes.mean(axis=1))<0.:
                    normal=-normal
                if interior and local_side==1:
                    normal=-normal
                normal*=facet_signs[f]
                self.tabulated_shape[f,q]=shape[0]
                self.tabulated_gradients[f,q]=physical
                self.global_coordinates[f,q]=point
                self.normals[f,q]=normal
                self.dx[f,q]=measure*parameter_weights[q]
        self.basis=self._vector_fields()
        self.volume_basis=volume

    def _init_2d(
        self,mesh,element,facets,intorder,*,side=0,interior=False,
        quadrature=None,orientation_sides=None,normal_signs=None,
    ):
        volume=Basis(mesh,element,intorder=intorder)
        if interior:
            facet_ids=(
                mesh.interior_facets() if facets is None else
                np.asarray(facets,dtype=np.int64)
            )
        else:
            facet_ids=(
                mesh.boundary_facets()
                if facets is None else np.asarray(facets)
            )
        if np.asarray(facet_ids).ndim==1:
            facets=mesh._facet_connectivity(facet_ids,full=True)
        else:
            facets=np.asarray(facet_ids,dtype=np.int64)
        quadratic_geometry=mesh.t.shape[0] in (6,9)
        expected=3 if quadratic_geometry else 2
        if facets.shape[0]!=expected:
            facets=facets.T
        if intorder>=4:
            points,weights=np.polynomial.legendre.leggauss(3)
        else:
            points,weights=np.polynomial.legendre.leggauss(2)
        points=(points+1.)/2.;weights=weights/2.
        if quadrature is not None:
            facet_points,weights=_validate_quadrature(quadrature,1)
            points=facet_points[0]
        elif intorder>4:
            facet_points,weights=_tensor_quadrature(1,intorder)
            points=facet_points[0]
        facet_scalar=(
            element.elem.elem
            if isinstance(element.elem,ElementDG) else element.elem
        )
        is_quad=isinstance(
            facet_scalar,(ElementQuad0,ElementQuad1,ElementQuad2)
        )
        if is_quad:
            local_edges=((0,1),(1,2),(2,3),(3,0))
            if isinstance(facet_scalar,ElementQuad2):
                local_edges=tuple(
                    edge+(midpoint,) for edge,midpoint in zip(
                        local_edges,(4,5,6,7)
                    )
                )
        else:
            edge_mid={(0,1):3,(1,2):4,(0,2):5}
            local_edges=((0,1),(1,2),(0,2))
            if isinstance(facet_scalar,ElementTriP2):
                local_edges=tuple(
                    edge+(edge_mid[tuple(sorted(edge))],)
                    for edge in local_edges
                )
        lookup={}
        for entity,nodes in enumerate(mesh.t.T):
            for local_index,edge in enumerate(local_edges):
                key=tuple(sorted((int(nodes[edge[0]]),int(nodes[edge[1]]))))
                lookup.setdefault(key,[]).append(
                    (local_index,entity,edge)
                )
        for adjacent in lookup.values():
            adjacent.sort(key=lambda item:(item[0],item[1]))
        entities=facets.shape[1];nq=len(points)
        selected_ids=(
            np.asarray(facet_ids,dtype=np.int64)
            if np.asarray(facet_ids).ndim==1 else None
        )
        facet_sides=(
            np.full(entities,side,dtype=np.int8)
            if orientation_sides is None else np.asarray(orientation_sides)
        )
        facet_signs=(
            np.ones(entities,dtype=np.int8)
            if normal_signs is None else np.asarray(normal_signs)
        )
        if facet_sides.shape!=(entities,) or facet_signs.shape!=(entities,):
            raise ValueError("facet orientation metadata has the wrong shape")
        nodes_per_element=len(element.elem.doflocs)
        components=element._dim
        self.mesh,self.elem=mesh,element
        self.X=points[None,:];self.W=weights
        self.quadrature=(self.X.copy(),self.W.copy())
        self.N=volume.N;self.doflocs=volume.doflocs
        self.element_dofs=np.empty(
            (nodes_per_element*components,entities),dtype=np.int64
        )
        self.tabulated_shape=np.empty((entities,nq,nodes_per_element))
        self.tabulated_gradients=np.empty(
            (entities,nq,nodes_per_element,2)
        )
        self.dx=np.empty((entities,nq))
        self.global_coordinates=np.empty((entities,nq,2))
        self.normals=np.empty((entities,nq,2))
        self.parent_elements=np.empty(entities,dtype=np.int64)
        self.facet_ids=(
            selected_ids.copy() if selected_ids is not None
            else np.full(entities,-1,dtype=np.int64)
        )
        self.facet_sides=facet_sides.copy()
        self.normal_signs=facet_signs.copy()
        self.local_faces=[]
        for facet,facet_nodes in enumerate(facets.T):
            key=tuple(sorted(map(int,facet_nodes[:2])))
            adjacent=lookup[key]
            local_side=int(facet_sides[facet])
            if selected_ids is not None:
                parent=int(mesh.f2t[local_side,selected_ids[facet]])
                matching=[item for item in adjacent if item[1]==parent]
            else:
                matching=(
                    [adjacent[local_side]]
                    if local_side<len(adjacent) else []
                )
            if not matching:
                raise ValueError(
                    f"facet {key} does not have side {local_side}"
                )
            _,entity,edge=matching[0]
            self.parent_elements[facet]=entity
            self.local_faces.append(tuple(edge))
            self.element_dofs[:,facet]=volume.element_dofs[:,entity]
            reference_vertices=(
                np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])
                if is_quad else
                np.array([[0.,0.],[1.,0.],[0.,1.]])
            )
            reference_corners=reference_vertices[np.asarray(edge[:2])]
            global_edge=mesh.t[np.asarray(edge[:2]),entity]
            if tuple(map(int,global_edge))!=tuple(
                map(int,facet_nodes[:2])
            ):
                reference_corners=reference_corners[::-1]
            x=mesh.p[:,mesh.t[:,entity]]
            centroid=x.mean(axis=1)
            for q,r in enumerate(points):
                reference=(
                    (1.-r)*reference_corners[0]
                    +r*reference_corners[1]
                )
                shape,refgrad=volume._evaluate_reference(
                    reference[:,None]
                )
                geometry_shape=shape
                geometry_refgrad=refgrad
                if x.shape[1]!=shape.shape[1]:
                    geometry_shape,geometry_refgrad=(
                        _mesh_geometry_shapes(
                            mesh,reference[:,None]
                        )
                    )
                jacobian=x@geometry_refgrad[0]
                physical=refgrad[0]@np.linalg.inv(jacobian)
                tangent=jacobian@(
                    reference_corners[1]-reference_corners[0]
                )
                point=geometry_shape[0]@x.T
                normal=np.array([tangent[1],-tangent[0]])
                length=np.linalg.norm(normal)
                normal/=length
                if np.dot(normal,point-centroid)<0.:
                    normal=-normal
                if interior and local_side==1:
                    normal=-normal
                normal*=facet_signs[facet]
                self.tabulated_shape[facet,q]=shape[0]
                self.tabulated_gradients[facet,q]=physical
                self.global_coordinates[facet,q]=point
                self.normals[facet,q]=normal
                self.dx[facet,q]=length*weights[q]
        self.basis=self._vector_fields()
        self.volume_basis=volume

    def interpolate(self,coefficients):
        coefficients=np.asarray(coefficients,dtype=np.float64)
        if coefficients.shape!=(self.N,):
            raise ValueError(
                f"coefficients must have shape ({self.N},)"
            )
        return Basis._interpolate(self,coefficients)

    def _vector_fields(self):
        return _LazyVectorFields(self)


class InteriorFacetBasis(FacetBasis):
    """Traces on one side of every selected interior facet."""

    def __init__(
        self,mesh,element,mapping=None,intorder=2,quadrature=None,
        facets=None,dofs=None,side=0,disable_doflocs=False,
    ):
        if mapping is not None or dofs is not None:
            raise NotImplementedError(
                "custom mapping and dofs are not implemented"
            )
        if side not in (0,1):
            raise ValueError("side must be 0 or 1")
        super().__init__(
            mesh,element,facets=facets,intorder=intorder,
            quadrature=quadrature,
            _side=side,_interior=True,
        )
        self.side=side
        if disable_doflocs:
            self.doflocs=None
