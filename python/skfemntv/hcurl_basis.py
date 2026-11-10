"""Minimal lowest-order H(curl) basis for affine triangular meshes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._hcurl_mapping import (
    covariant_piola,covariant_piola_curl,covariant_piola_vector_curl,
    tetrahedron_affine_jacobian,triangle_affine_jacobian,
)
from ._nedelec_reference import (
    tet_n1_basis,tet_n1_curl,tri_n1_basis,tri_n1_curl,
    tet_n2_basis,tet_n2_curl,tet_n3_basis,tet_n3_curl,
    tri_n2_basis,tri_n2_curl,
    tri_n3_basis,tri_n3_curl,
)
from ._tet_n2_coefficients import (
    TET_N2_INTERPOLATION_MATRIX,TET_N2_INTERPOLATION_POINTS,
    TET_N3_INTERPOLATION_MATRIX,TET_N3_INTERPOLATION_POINTS,
)
from .edge_dof_map import build_oriented_edge_dof_map


@dataclass(frozen=True)
class OrientedTriN2DofMap:
    topology: object
    element_dofs: np.ndarray
    basis_signs: np.ndarray
    ndofs: int


class AffineTriN2Basis:
    """Second-order Nedelec basis on affine linear triangular meshes."""

    def __init__(self,mesh,*,intorder=4,quadrature=None):
        if mesh.dim()!=2 or mesh.t.shape[0]!=3:
            raise TypeError("AffineTriN2Basis requires a linear triangular mesh")
        self.mesh=mesh
        edge_map=build_oriented_edge_dof_map(mesh)
        topology=edge_map.topology
        cells=mesh.nelements
        facets=mesh.facets.shape[1]
        self.N=2*facets+2*cells
        local_facets=((0,1),(1,2),(0,2))
        element_dofs=np.empty((8,cells),dtype=np.int64)
        signs=np.ones((8,cells),dtype=np.float64)
        for local,facet_row in enumerate(local_facets):
            facet_ids=mesh.t2f[local]
            element_dofs[2*local]=2*facet_ids
            element_dofs[2*local+1]=2*facet_ids+1
            first,second=local_facets[local]
            orientation=np.where(mesh.t[first]<mesh.t[second],1.,-1.)
            signs[2*local:2*local+2]=orientation[None,:]
        interior_start=2*facets
        element_dofs[6]=interior_start+2*np.arange(cells)
        element_dofs[7]=interior_start+2*np.arange(cells)+1
        self.element_dofs=np.ascontiguousarray(element_dofs)
        self.basis_signs=np.ascontiguousarray(signs)
        self.dof_map=OrientedTriN2DofMap(
            topology,self.element_dofs,self.basis_signs,self.N
        )
        self.X,self.W=(
            _triangle_quadrature(intorder) if quadrature is None else
            (np.asarray(quadrature[0],dtype=np.float64),
             np.asarray(quadrature[1],dtype=np.float64))
        )
        if (self.X.ndim!=2 or self.X.shape[0]!=2
                or self.W.shape!=(self.X.shape[1],)):
            raise ValueError(
                "quadrature must contain points shaped (2, nq) and weights shaped (nq,)"
            )
        vertices=mesh.p[:,mesh.t[:3]]
        self.jacobians=triangle_affine_jacobian(vertices)
        self.detJ=(self.jacobians[0,0]*self.jacobians[1,1]
                   -self.jacobians[0,1]*self.jacobians[1,0])
        if np.any(np.isclose(self.detJ,0.)):
            raise ValueError("H(curl) basis requires nonsingular triangles")
        self.dx=np.abs(self.detJ)[:,None]*self.W[None,:]
        values=covariant_piola(
            tri_n2_basis(self.X)[:,:,None,:],self.jacobians[...,None]
        ).transpose(2,0,1,3)
        curls=covariant_piola_curl(
            tri_n2_curl(self.X)[:,None,:],self.jacobians[...,None]
        ).transpose(1,0,2)
        self._element_values=np.ascontiguousarray(
            values*self.basis_signs.T[:,:,None,None]
        )
        self._element_curls=np.ascontiguousarray(
            curls*self.basis_signs.T[:,:,None]
        )
        doflocs=np.empty((2,self.N),dtype=np.float64)
        doflocs[:,0:2*facets:2]=mesh.p[:,mesh.facets[0]]
        doflocs[:,1:2*facets:2]=mesh.p[:,mesh.facets[1]]
        centers=vertices.mean(axis=1)
        doflocs[:,2*facets:]=np.repeat(centers,2,axis=1)
        self.doflocs=doflocs

    @property
    def global_coordinates(self):
        vertices=self.mesh.p[:,self.mesh.t[0]]
        return vertices[:,:,None]+np.einsum(
            "ire,rq->ieq",self.jacobians,self.X
        )

    def evaluate(self,coefficients):
        coefficients=np.asarray(coefficients,dtype=np.float64)
        if coefficients.shape!=(self.N,):
            raise ValueError(f"coefficients must have shape ({self.N},)")
        return np.einsum(
            "eb,ebiq->ieq",coefficients[self.element_dofs.T],
            self._element_values,
        )

    def evaluate_curl(self,coefficients):
        coefficients=np.asarray(coefficients,dtype=np.float64)
        if coefficients.shape!=(self.N,):
            raise ValueError(f"coefficients must have shape ({self.N},)")
        return np.einsum(
            "eb,ebq->eq",coefficients[self.element_dofs.T],
            self._element_curls,
        )

    def boundary_dofs(self,boundary=None):
        edge_ids=_boundary_edge_dofs(self,boundary)
        return np.column_stack((2*edge_ids,2*edge_ids+1)).reshape(-1)


@dataclass(frozen=True)
class OrientedTriN3DofMap:
    topology: object
    element_dofs: np.ndarray
    basis_signs: np.ndarray
    basis_permutation: np.ndarray
    ndofs: int


class AffineTriN3Basis:
    """Third-order Nedelec basis on affine linear triangular meshes."""

    def __init__(self,mesh,*,intorder=6,quadrature=None):
        if mesh.dim()!=2 or mesh.t.shape[0]!=3:
            raise TypeError("AffineTriN3Basis requires a linear triangular mesh")
        self.mesh=mesh
        edge_map=build_oriented_edge_dof_map(mesh)
        topology=edge_map.topology
        cells=mesh.nelements
        facets=mesh.facets.shape[1]
        self.N=3*facets+6*cells
        local_facets=((0,1),(1,2),(0,2))
        element_dofs=np.empty((15,cells),dtype=np.int64)
        signs=np.ones((15,cells),dtype=np.float64)
        permutation=np.tile(np.arange(15,dtype=np.int64)[:,None],(1,cells))
        for local,(first,second) in enumerate(local_facets):
            facet_ids=mesh.t2f[local]
            element_dofs[3*local:3*local+3]=(
                3*facet_ids[None,:]+np.arange(3,dtype=np.int64)[:,None]
            )
            orientation=np.where(mesh.t[first]<mesh.t[second],1.,-1.)
            for subindex in range(3):
                local_dof=3*local+subindex
                target=local_dof+(2 if subindex==0 else -2 if subindex==2 else 0)
                swap_condition=-1 if local<2 else 1
                choose_local=(orientation>0.) if swap_condition==-1 else (orientation<0.)
                permutation[local_dof]=np.where(
                    choose_local,local_dof,target
                )
                fixed_sign=np.where(
                    (permutation[local_dof]>=6)&(permutation[local_dof]<=8),
                    -1.,1.,
                )
                signs[local_dof]=orientation*fixed_sign
        interior_start=3*facets
        element_dofs[9:15]=(
            interior_start+6*np.arange(cells,dtype=np.int64)[None,:]
            +np.arange(6,dtype=np.int64)[:,None]
        )
        self.element_dofs=np.ascontiguousarray(element_dofs)
        self.basis_signs=np.ascontiguousarray(signs)
        self.basis_permutation=np.ascontiguousarray(permutation)
        self.dof_map=OrientedTriN3DofMap(
            topology,self.element_dofs,self.basis_signs,
            self.basis_permutation,self.N,
        )
        self.X,self.W=(
            _triangle_quadrature(intorder) if quadrature is None else
            (np.asarray(quadrature[0],dtype=np.float64),
             np.asarray(quadrature[1],dtype=np.float64))
        )
        if (self.X.ndim!=2 or self.X.shape[0]!=2
                or self.W.shape!=(self.X.shape[1],)):
            raise ValueError(
                "quadrature must contain points shaped (2, nq) and weights shaped (nq,)"
            )
        vertices=mesh.p[:,mesh.t[:3]]
        self.jacobians=triangle_affine_jacobian(vertices)
        self.detJ=(self.jacobians[0,0]*self.jacobians[1,1]
                   -self.jacobians[0,1]*self.jacobians[1,0])
        if np.any(np.isclose(self.detJ,0.)):
            raise ValueError("H(curl) basis requires nonsingular triangles")
        self.dx=np.abs(self.detJ)[:,None]*self.W[None,:]
        reference_values=tri_n3_basis(self.X)
        reference_curls=tri_n3_curl(self.X)
        values=np.empty((cells,15,2,len(self.W)),dtype=np.float64)
        curls=np.empty((cells,15,len(self.W)),dtype=np.float64)
        for cell in range(cells):
            selected=reference_values[self.basis_permutation[:,cell]]
            selected_curl=reference_curls[self.basis_permutation[:,cell]]
            mapped=covariant_piola(
                selected[:,:,None,:],self.jacobians[:,:,cell,None]
            ).transpose(2,0,1,3)[0]
            mapped_curl=covariant_piola_curl(
                selected_curl[:,None,:],self.jacobians[:,:,cell,None]
            ).transpose(1,0,2)[0]
            values[cell]=mapped*self.basis_signs[:,cell,None,None]
            curls[cell]=mapped_curl*self.basis_signs[:,cell,None]
        self._element_values=np.ascontiguousarray(values)
        self._element_curls=np.ascontiguousarray(curls)
        reference_doflocs=np.array([
            [.25,0.],[.5,0.],[.75,0.],
            [.75,.25],[.5,.5],[.25,.75],
            [0.,.75],[0.,.5],[0.,.25],
            [.25,.25],[.25,.25],[.5,.25],[.5,.25],[.25,.5],[.25,.5],
        ])
        mapped_doflocs=(
            vertices[:,0,None,:]+np.einsum(
                "ire,br->ibe",self.jacobians,reference_doflocs
            )
        ).transpose(2,1,0)
        doflocs=np.empty((2,self.N),dtype=np.float64)
        assigned=np.zeros(self.N,dtype=bool)
        for cell in range(cells):
            for local_dof,global_dof in enumerate(self.element_dofs[:,cell]):
                if not assigned[global_dof]:
                    doflocs[:,global_dof]=mapped_doflocs[cell,local_dof]
                    assigned[global_dof]=True
        self.doflocs=doflocs

    @property
    def global_coordinates(self):
        vertices=self.mesh.p[:,self.mesh.t[0]]
        return vertices[:,:,None]+np.einsum(
            "ire,rq->ieq",self.jacobians,self.X
        )

    def evaluate(self,coefficients):
        coefficients=np.asarray(coefficients,dtype=np.float64)
        if coefficients.shape!=(self.N,):
            raise ValueError(f"coefficients must have shape ({self.N},)")
        return np.einsum(
            "eb,ebiq->ieq",coefficients[self.element_dofs.T],
            self._element_values,
        )

    def evaluate_curl(self,coefficients):
        coefficients=np.asarray(coefficients,dtype=np.float64)
        if coefficients.shape!=(self.N,):
            raise ValueError(f"coefficients must have shape ({self.N},)")
        return np.einsum(
            "eb,ebq->eq",coefficients[self.element_dofs.T],
            self._element_curls,
        )

    def boundary_dofs(self,boundary=None):
        edge_ids=_boundary_edge_dofs(self,boundary)
        return (3*edge_ids[:,None]+np.arange(3,dtype=np.int64)).reshape(-1)


_TET_N2_EDGES=((2,3),(1,3),(1,2),(0,3),(0,2),(0,1))
_TET_N2_FACES=((1,2,3),(0,2,3),(0,1,3),(0,1,2))


def _tet_cell_transforms(
    mesh,*,edge_reflection,face_rotation,face_reflection,
    edge_dofs,face_dofs,cell_dofs,
):
    """Build per-cell DOF transforms from globally ordered entities."""
    ndofs=6*edge_dofs+4*face_dofs+cell_dofs
    transforms=np.broadcast_to(np.eye(ndofs),(mesh.nelements,ndofs,ndofs)).copy()
    for cell in range(mesh.nelements):
        vertices=mesh.t[:4,cell]
        matrix=transforms[cell]
        for edge,(first,second) in enumerate(_TET_N2_EDGES):
            if vertices[first]>vertices[second]:
                start=edge_dofs*edge
                matrix[start:start+edge_dofs,start:start+edge_dofs]=edge_reflection
        for face,local_face in enumerate(_TET_N2_FACES):
            face_vertices=vertices[list(local_face)]
            order=np.argsort(face_vertices)
            entity_local=np.asarray(local_face,dtype=np.int64)[order]
            entity_global=face_vertices[order]
            min_local=int(np.argmin(entity_local))
            pre=int(entity_local[(min_local+2)%3])
            post=int(entity_local[(min_local+1)%3])
            min_global=int(np.argmin(entity_global))
            g_pre=int(entity_global[(min_global+2)%3])
            g_post=int(entity_global[(min_global+1)%3])
            rotations=(
                (min_global+3-min_local)%3 if g_post>g_pre else
                (min_local+3-min_global)%3
            )
            reflected=(post>pre)==(g_post<g_pre)
            block=np.eye(face_dofs)
            if reflected:
                block=face_reflection@block
            for _ in range(rotations):
                block=face_rotation@block
            start=6*edge_dofs+face_dofs*face
            matrix[start:start+face_dofs,start:start+face_dofs]=block
    return transforms


def _tet_n2_cell_transforms(mesh):
    return _tet_cell_transforms(
        mesh,
        edge_reflection=np.array(((0.,-1.),(-1.,0.))),
        face_rotation=np.array(((-1.,-1.),(1.,0.))),
        face_reflection=np.array(((0.,1.),(1.,0.))),
        edge_dofs=2,face_dofs=2,cell_dofs=0,
    )


def _tet_n3_cell_transforms(mesh):
    return _tet_cell_transforms(
        mesh,
        edge_reflection=np.array(((0.,-1.,0.),(-1.,0.,0.),(0.,0.,-1.))),
        face_rotation=np.array((
            (0.,0.,0.,0.,-1.,-1.),
            (0.,0.,0.,0.,1.,0.),
            (-1.,-1.,0.,0.,0.,0.),
            (1.,0.,0.,0.,0.,0.),
            (0.,0.,-1.,-1.,0.,0.),
            (0.,0.,1.,0.,0.,0.),
        )),
        face_reflection=np.array((
            (0.,1.,0.,0.,0.,0.),
            (1.,0.,0.,0.,0.,0.),
            (0.,0.,0.,0.,0.,1.),
            (0.,0.,0.,0.,1.,0.),
            (0.,0.,0.,1.,0.,0.),
            (0.,0.,1.,0.,0.,0.),
        )),
        edge_dofs=3,face_dofs=6,cell_dofs=3,
    )


@dataclass(frozen=True)
class OrientedTetN2DofMap:
    topology: object
    element_dofs: np.ndarray
    transformations: np.ndarray
    ndofs: int


class AffineTetN2Basis:
    """Second-order first-family Nedelec basis on affine tetrahedra."""

    def __init__(self,mesh,*,intorder=6,quadrature=None):
        if mesh.dim()!=3 or mesh.t.shape[0]!=4:
            raise TypeError("AffineTetN2Basis requires a linear tetrahedral mesh")
        self.mesh=mesh
        edge_map=build_oriented_edge_dof_map(mesh)
        edges=edge_map.topology.edges
        edge_lookup={tuple(edge):i for i,edge in enumerate(edges.T)}
        facets=mesh.facets
        face_lookup={tuple(sorted(map(int,face))):i for i,face in enumerate(facets.T)}
        dofs=np.empty((20,mesh.nelements),dtype=np.int64)
        for cell in range(mesh.nelements):
            vertices=mesh.t[:4,cell]
            for local,(first,second) in enumerate(_TET_N2_EDGES):
                key=tuple(sorted((int(vertices[first]),int(vertices[second]))))
                edge_id=edge_lookup[key]
                dofs[2*local:2*local+2,cell]=2*edge_id+np.arange(2)
            for local,face in enumerate(_TET_N2_FACES):
                key=tuple(sorted(int(vertices[i]) for i in face))
                face_id=face_lookup[key]
                dofs[12+2*local:14+2*local,cell]=2*edges.shape[1]+2*face_id+np.arange(2)
        self.N=2*edges.shape[1]+2*facets.shape[1]
        self.element_dofs=np.ascontiguousarray(dofs)
        self.basis_signs=np.ones_like(self.element_dofs,dtype=np.float64)
        self.transformations=_tet_n2_cell_transforms(mesh)
        self.dof_map=OrientedTetN2DofMap(
            edge_map.topology,self.element_dofs,self.transformations,self.N
        )
        self.X,self.W=(
            _tetrahedron_quadrature(intorder) if quadrature is None else
            (np.asarray(quadrature[0],dtype=np.float64),
             np.asarray(quadrature[1],dtype=np.float64))
        )
        if self.X.ndim!=2 or self.X.shape[0]!=3 or self.W.shape!=(self.X.shape[1],):
            raise ValueError("quadrature must contain points shaped (3, nq) and weights shaped (nq,)")
        vertices=mesh.p[:,mesh.t[:4]]
        self.jacobians=tetrahedron_affine_jacobian(vertices)
        self.detJ=np.linalg.det(self.jacobians.transpose(2,0,1))
        if np.any(np.isclose(self.detJ,0.)):
            raise ValueError("H(curl) basis requires nonsingular tetrahedra")
        self.dx=np.abs(self.detJ)[:,None]*self.W[None,:]
        ref_values=tet_n2_basis(self.X)
        ref_curls=tet_n2_curl(self.X)
        values=np.empty((mesh.nelements,20,3,len(self.W)))
        curls=np.empty_like(values)
        for cell in range(mesh.nelements):
            transform=self.transformations[cell]
            local_values=np.einsum("ab,biq->aiq",transform,ref_values)
            local_curls=np.einsum("ab,biq->aiq",transform,ref_curls)
            values[cell]=covariant_piola(
                local_values[:,:,None,:],self.jacobians[:,:,cell,None]
            ).transpose(2,0,1,3)[0]
            curls[cell]=covariant_piola_vector_curl(
                local_curls[:,:,None,:],self.jacobians[:,:,cell,None]
            ).transpose(2,0,1,3)[0]
        self._element_values=np.ascontiguousarray(values)
        self._element_curls=np.ascontiguousarray(curls)
        doflocs=np.empty((3,self.N),dtype=np.float64)
        for entity,(first,second) in enumerate(edges.T):
            doflocs[:,2*entity:2*entity+2]=(.5*(mesh.p[:,first]+mesh.p[:,second]))[:,None]
        face_offset=2*edges.shape[1]
        for entity,face in enumerate(facets.T):
            doflocs[:,face_offset+2*entity:face_offset+2*entity+2]=mesh.p[:,face].mean(axis=1)[:,None]
        self.doflocs=doflocs

    @property
    def global_coordinates(self):
        vertices=self.mesh.p[:,self.mesh.t[0]]
        return vertices[:,:,None]+np.einsum("ire,rq->ieq",self.jacobians,self.X)

    @property
    def values(self):
        return self._element_values.transpose(1,2,0,3)

    @property
    def curls(self):
        return self._element_curls.transpose(1,2,0,3)

    def evaluate(self,coefficients):
        coefficients=np.asarray(coefficients,dtype=np.float64)
        if coefficients.shape!=(self.N,):
            raise ValueError(f"coefficients must have shape ({self.N},)")
        return np.einsum("eb,ebiq->ieq",coefficients[self.element_dofs.T],self._element_values)

    def evaluate_curl(self,coefficients):
        coefficients=np.asarray(coefficients,dtype=np.float64)
        if coefficients.shape!=(self.N,):
            raise ValueError(f"coefficients must have shape ({self.N},)")
        return np.einsum("eb,ebiq->ieq",coefficients[self.element_dofs.T],self._element_curls)

    def element_mass_matrices(self):
        return np.einsum("ebiq,eciq,eq->ebc",self._element_values,self._element_values,self.dx)

    def element_curl_curl_matrices(self):
        return np.einsum("ebiq,eciq,eq->ebc",self._element_curls,self._element_curls,self.dx)

    def interpolate_moments(self,field):
        """Interpolate a physical vector field using TetN2 edge and face moments.

        The dual functionals use the reference N1E degree-2 interpolation
        points and matrix. Cell-local moments are pulled back by ``J.T`` and
        transformed to the globally oriented edge/face basis.
        """
        if not callable(field):
            raise TypeError("field must be callable")
        points=TET_N2_INTERPOLATION_POINTS.T
        accumulated=np.zeros(self.N,dtype=np.float64)
        counts=np.zeros(self.N,dtype=np.int64)
        for cell in range(self.mesh.nelements):
            jacobian=self.jacobians[:,:,cell]
            origin=self.mesh.p[:,self.mesh.t[0,cell]]
            physical_points=origin[:,None]+jacobian@points
            physical_values=np.asarray(field(physical_points),dtype=np.float64)
            try:
                physical_values=np.broadcast_to(physical_values,physical_points.shape)
            except ValueError as error:
                raise ValueError(
                    "field must return vectors broadcastable to (3, 24)"
                ) from error
            reference_values=jacobian.T@physical_values
            local=TET_N2_INTERPOLATION_MATRIX@reference_values.reshape(-1)
            global_oriented=np.linalg.solve(self.transformations[cell].T,local)
            dofs=self.element_dofs[:,cell]
            np.add.at(accumulated,dofs,global_oriented)
            np.add.at(counts,dofs,1)
        if np.any(counts==0):
            raise RuntimeError("TetN2 interpolation left unassigned global DOFs")
        return accumulated/counts

    def boundary_dofs(self,boundary=None):
        """Return both edge and face DOFs on selected boundary facets."""
        edge_ids=_boundary_edge_dofs(self,boundary)
        edge_dofs=(2*edge_ids[:,None]+np.arange(2,dtype=np.int64)).reshape(-1)
        if boundary is None:
            facets=self.mesh.boundary_facets()
        elif isinstance(boundary,str):
            facets=np.asarray(self.mesh.boundaries[boundary],dtype=np.int64)
        elif callable(boundary):
            candidates=self.mesh.boundary_facets()
            centers=self.mesh.p[:,self.mesh.facets[:,candidates]].mean(axis=1)
            mask=np.asarray(boundary(centers),dtype=bool)
            if mask.shape!=(len(candidates),):
                raise ValueError("boundary predicate must return one boolean per boundary facet")
            facets=candidates[mask]
        elif isinstance(boundary,tuple) and all(isinstance(name,str) for name in boundary):
            facets=np.concatenate([self.mesh.boundaries[name] for name in boundary]) if boundary else np.empty(0,dtype=np.int64)
        else:
            facets=np.asarray(boundary)
            if facets.dtype==bool:
                candidates=self.mesh.boundary_facets()
                if facets.shape!=(len(candidates),):
                    raise ValueError("boundary mask must contain one boolean per boundary facet")
                facets=candidates[facets]
            facets=np.asarray(facets,dtype=np.int64).reshape(-1)
        face_lookup={tuple(sorted(map(int,face))):i for i,face in enumerate(self.mesh.facets.T)}
        face_offset=2*self.dof_map.topology.edges.shape[1]
        selected_faces=[face_lookup[tuple(sorted(map(int,self.mesh.facets[:,facet])))] for facet in facets]
        face_dofs=(face_offset+2*np.asarray(selected_faces)[:,None]+np.arange(2,dtype=np.int64)).reshape(-1)
        return np.unique(np.r_[edge_dofs,face_dofs])


@dataclass(frozen=True)
class OrientedTetN3DofMap:
    topology: object
    element_dofs: np.ndarray
    transformations: np.ndarray
    ndofs: int


class AffineTetN3Basis:
    """Third-order first-family Nedelec basis on affine tetrahedra."""

    def __init__(self,mesh,*,intorder=8,quadrature=None):
        if mesh.dim()!=3 or mesh.t.shape[0]!=4:
            raise TypeError("AffineTetN3Basis requires a linear tetrahedral mesh")
        self.mesh=mesh
        edge_map=build_oriented_edge_dof_map(mesh)
        edges=edge_map.topology.edges
        edge_lookup={tuple(edge):i for i,edge in enumerate(edges.T)}
        facets=mesh.facets
        face_lookup={tuple(sorted(map(int,face))):i for i,face in enumerate(facets.T)}
        dofs=np.empty((45,mesh.nelements),dtype=np.int64)
        for cell in range(mesh.nelements):
            vertices=mesh.t[:4,cell]
            for local,(first,second) in enumerate(_TET_N2_EDGES):
                key=tuple(sorted((int(vertices[first]),int(vertices[second]))))
                edge_id=edge_lookup[key]
                start=3*local
                dofs[start:start+3,cell]=3*edge_id+np.arange(3)
            for local,face in enumerate(_TET_N2_FACES):
                key=tuple(sorted(int(vertices[i]) for i in face))
                face_id=face_lookup[key]
                start=18+6*local
                dofs[start:start+6,cell]=3*edges.shape[1]+6*face_id+np.arange(6)
            dofs[42:45,cell]=(3*edges.shape[1]+6*facets.shape[1]
                               +3*cell+np.arange(3))
        self.N=3*edges.shape[1]+6*facets.shape[1]+3*mesh.nelements
        self.element_dofs=np.ascontiguousarray(dofs)
        self.basis_signs=np.ones_like(self.element_dofs,dtype=np.float64)
        self.transformations=_tet_n3_cell_transforms(mesh)
        self.dof_map=OrientedTetN3DofMap(
            edge_map.topology,self.element_dofs,self.transformations,self.N
        )
        self.X,self.W=(
            _tetrahedron_quadrature(intorder) if quadrature is None else
            (np.asarray(quadrature[0],dtype=np.float64),
             np.asarray(quadrature[1],dtype=np.float64))
        )
        if self.X.ndim!=2 or self.X.shape[0]!=3 or self.W.shape!=(self.X.shape[1],):
            raise ValueError("quadrature must contain points shaped (3, nq) and weights shaped (nq,)")
        vertices=mesh.p[:,mesh.t[:4]]
        self.jacobians=tetrahedron_affine_jacobian(vertices)
        self.detJ=np.linalg.det(self.jacobians.transpose(2,0,1))
        if np.any(np.isclose(self.detJ,0.)):
            raise ValueError("H(curl) basis requires nonsingular tetrahedra")
        self.dx=np.abs(self.detJ)[:,None]*self.W[None,:]
        ref_values=tet_n3_basis(self.X)
        ref_curls=tet_n3_curl(self.X)
        values=np.empty((mesh.nelements,45,3,len(self.W)))
        curls=np.empty_like(values)
        for cell in range(mesh.nelements):
            transform=self.transformations[cell]
            local_values=np.einsum("ab,biq->aiq",transform,ref_values)
            local_curls=np.einsum("ab,biq->aiq",transform,ref_curls)
            values[cell]=covariant_piola(
                local_values[:,:,None,:],self.jacobians[:,:,cell,None]
            ).transpose(2,0,1,3)[0]
            curls[cell]=covariant_piola_vector_curl(
                local_curls[:,:,None,:],self.jacobians[:,:,cell,None]
            ).transpose(2,0,1,3)[0]
        self._element_values=np.ascontiguousarray(values)
        self._element_curls=np.ascontiguousarray(curls)
        doflocs=np.empty((3,self.N),dtype=np.float64)
        for entity,(first,second) in enumerate(edges.T):
            doflocs[:,3*entity:3*entity+3]=(.5*(mesh.p[:,first]+mesh.p[:,second]))[:,None]
        face_offset=3*edges.shape[1]
        for entity,face in enumerate(facets.T):
            doflocs[:,face_offset+6*entity:face_offset+6*entity+6]=mesh.p[:,face].mean(axis=1)[:,None]
        cell_offset=face_offset+6*facets.shape[1]
        for cell in range(mesh.nelements):
            doflocs[:,cell_offset+3*cell:cell_offset+3*cell+3]=vertices[:,:,cell].mean(axis=1)[:,None]
        self.doflocs=doflocs

    @property
    def global_coordinates(self):
        vertices=self.mesh.p[:,self.mesh.t[0]]
        return vertices[:,:,None]+np.einsum("ire,rq->ieq",self.jacobians,self.X)

    @property
    def values(self):
        return self._element_values.transpose(1,2,0,3)

    @property
    def curls(self):
        return self._element_curls.transpose(1,2,0,3)

    def evaluate(self,coefficients):
        coefficients=np.asarray(coefficients,dtype=np.float64)
        if coefficients.shape!=(self.N,):
            raise ValueError(f"coefficients must have shape ({self.N},)")
        return np.einsum("eb,ebiq->ieq",coefficients[self.element_dofs.T],self._element_values)

    def evaluate_curl(self,coefficients):
        coefficients=np.asarray(coefficients,dtype=np.float64)
        if coefficients.shape!=(self.N,):
            raise ValueError(f"coefficients must have shape ({self.N},)")
        return np.einsum("eb,ebiq->ieq",coefficients[self.element_dofs.T],self._element_curls)

    def element_mass_matrices(self):
        return np.einsum("ebiq,eciq,eq->ebc",self._element_values,self._element_values,self.dx)

    def element_curl_curl_matrices(self):
        return np.einsum("ebiq,eciq,eq->ebc",self._element_curls,self._element_curls,self.dx)

    def interpolate_moments(self,field):
        """Interpolate a vector field using TetN3 edge, face, and cell moments."""
        if not callable(field):
            raise TypeError("field must be callable")
        points=TET_N3_INTERPOLATION_POINTS.T
        accumulated=np.zeros(self.N,dtype=np.float64)
        counts=np.zeros(self.N,dtype=np.int64)
        for cell in range(self.mesh.nelements):
            jacobian=self.jacobians[:,:,cell]
            origin=self.mesh.p[:,self.mesh.t[0,cell]]
            physical_points=origin[:,None]+jacobian@points
            physical_values=np.asarray(field(physical_points),dtype=np.float64)
            try:
                physical_values=np.broadcast_to(physical_values,physical_points.shape)
            except ValueError as error:
                raise ValueError(
                    "field must return vectors broadcastable to (3, 47)"
                ) from error
            reference_values=jacobian.T@physical_values
            local=TET_N3_INTERPOLATION_MATRIX@reference_values.reshape(-1)
            global_oriented=np.linalg.solve(self.transformations[cell].T,local)
            dofs=self.element_dofs[:,cell]
            np.add.at(accumulated,dofs,global_oriented)
            np.add.at(counts,dofs,1)
        if np.any(counts==0):
            raise RuntimeError("TetN3 interpolation left unassigned global DOFs")
        return accumulated/counts

    def boundary_dofs(self,boundary=None):
        edge_ids=_boundary_edge_dofs(self,boundary)
        edge_dofs=(3*edge_ids[:,None]+np.arange(3,dtype=np.int64)).reshape(-1)
        if boundary is None:
            selected_facets=self.mesh.boundary_facets()
        elif isinstance(boundary,str):
            selected_facets=np.asarray(self.mesh.boundaries[boundary],dtype=np.int64)
        elif callable(boundary):
            candidates=self.mesh.boundary_facets()
            centers=self.mesh.p[:,self.mesh.facets[:,candidates]].mean(axis=1)
            mask=np.asarray(boundary(centers),dtype=bool)
            if mask.shape!=(len(candidates),):
                raise ValueError("boundary predicate must return one boolean per boundary facet")
            selected_facets=candidates[mask]
        elif isinstance(boundary,tuple) and all(isinstance(name,str) for name in boundary):
            selected_facets=np.concatenate([self.mesh.boundaries[name] for name in boundary]) if boundary else np.empty(0,dtype=np.int64)
        else:
            selected_facets=np.asarray(boundary)
            if selected_facets.dtype==bool:
                candidates=self.mesh.boundary_facets()
                if selected_facets.shape!=(len(candidates),):
                    raise ValueError("boundary mask must contain one boolean per boundary facet")
                selected_facets=candidates[selected_facets]
            selected_facets=np.asarray(selected_facets,dtype=np.int64).reshape(-1)
        face_lookup={tuple(sorted(map(int,face))):i for i,face in enumerate(self.mesh.facets.T)}
        face_offset=3*self.dof_map.topology.edges.shape[1]
        selected_faces=[face_lookup[tuple(sorted(map(int,self.mesh.facets[:,facet])))] for facet in selected_facets]
        face_dofs=(face_offset+6*np.asarray(selected_faces)[:,None]+np.arange(6,dtype=np.int64)).reshape(-1)
        return np.unique(np.r_[edge_dofs,face_dofs])


__all__=[
    "AffineTriN2Basis",
    "OrientedTriN2DofMap",
]


def _triangle_quadrature(order):
    count=max(1,(int(order)+3)//2)
    points,weights=np.polynomial.legendre.leggauss(count)
    points=(points+1.)/2.
    weights=weights/2.
    coordinates=[]
    result_weights=[]
    for first in range(count):
        for second in range(count):
            x=points[first]
            y=(1.-x)*points[second]
            coordinates.append((x,y))
            result_weights.append(
                (1.-x)*weights[first]*weights[second]
            )
    return np.asarray(coordinates).T,np.asarray(result_weights)


def _tetrahedron_quadrature(order):
    """Duffy-product quadrature on the unit tetrahedron."""
    count=max(1,(int(order)+4)//2)
    points,weights=np.polynomial.legendre.leggauss(count)
    points=(points+1.)/2.
    weights=weights/2.
    coordinates=[]
    result_weights=[]
    for first in range(count):
        for second in range(count):
            for third in range(count):
                x=points[first]
                y=(1.-x)*points[second]
                z=(1.-x)*(1.-points[second])*points[third]
                coordinates.append((x,y,z))
                result_weights.append(
                    (1.-x)**2*(1.-points[second])
                    *weights[first]*weights[second]*weights[third]
                )
    return np.asarray(coordinates).T,np.asarray(result_weights)


def _boundary_edge_dofs(basis,boundary):
    """Resolve boundary facets to their globally owned edge DOFs."""
    if isinstance(boundary,tuple) and all(
        isinstance(name,str) for name in boundary
    ):
        selected=np.concatenate([
            _boundary_edge_dofs(basis,name) for name in boundary
        ]) if boundary else np.empty(0,dtype=np.int64)
        return np.unique(selected)
    if boundary is None:
        facets=basis.mesh.boundary_facets()
    elif isinstance(boundary,str):
        try:
            facets=np.asarray(basis.mesh.boundaries[boundary],dtype=np.int64)
        except KeyError as error:
            raise KeyError(f"unknown boundary {boundary!r}") from error
    elif callable(boundary):
        candidates=basis.mesh.boundary_facets()
        centers=basis.mesh.p[:,basis.mesh.facets[:,candidates]].mean(axis=1)
        mask=np.asarray(boundary(centers),dtype=bool)
        if mask.shape!=(len(candidates),):
            raise ValueError(
                "boundary predicate must return one boolean per boundary facet"
            )
        facets=candidates[mask]
    else:
        facets=np.asarray(boundary)
        if facets.dtype==bool:
            candidates=basis.mesh.boundary_facets()
            if facets.shape!=(len(candidates),):
                raise ValueError(
                    "boundary mask must contain one value per boundary facet"
                )
            facets=candidates[facets]
        facets=np.asarray(facets,dtype=np.int64).reshape(-1)
    boundary_set=set(map(int,basis.mesh.boundary_facets()))
    if any(int(facet) not in boundary_set for facet in facets):
        raise ValueError("H(curl) boundary DOFs require boundary facet IDs")
    edge_lookup={
        tuple(edge):index
        for index,edge in enumerate(basis.dof_map.topology.edges.T)
    }
    selected=set()
    for facet in facets:
        nodes=tuple(map(int,basis.mesh.facets[:,facet]))
        for first_index,first in enumerate(nodes):
            for second in nodes[first_index+1:]:
                edge_id=edge_lookup.get(tuple(sorted((first,second))))
                if edge_id is not None:
                    selected.add(edge_id)
    return np.asarray(sorted(selected),dtype=np.int64)


@dataclass(frozen=True)
class HcurlGeometryDiagnostics:
    minimum_signed_determinant: float
    minimum_absolute_determinant: float
    minimum_area: float
    maximum_aspect_ratio: float
    inverted_cell_count: int


@dataclass(frozen=True)
class HcurlTetGeometryDiagnostics:
    minimum_signed_determinant: float
    minimum_absolute_determinant: float
    minimum_volume: float
    maximum_aspect_ratio: float
    inverted_cell_count: int


class AffineTriN1Basis:
    """Reference tabulation, Piola mapping, orientation and integration data."""

    def __init__(self,mesh,*,intorder=2,quadrature=None,max_aspect_ratio=None):
        if mesh.dim()!=2 or mesh.t.shape[0] not in (3,6):
            raise TypeError("AffineTriN1Basis requires a triangular mesh")
        self.mesh=mesh
        self.dof_map=build_oriented_edge_dof_map(mesh)
        self.N=self.dof_map.ndofs
        self.element_dofs=self.dof_map.element_dofs
        self.basis_signs=self.dof_map.basis_signs
        self.X,self.W=(
            _triangle_quadrature(intorder) if quadrature is None else
            (np.asarray(quadrature[0],dtype=np.float64),
             np.asarray(quadrature[1],dtype=np.float64))
        )
        if self.X.ndim!=2 or self.X.shape[0]!=2 or self.W.shape!=(self.X.shape[1],):
            raise ValueError("quadrature must contain points shaped (2, nq) and weights shaped (nq,)")

        # Advanced indexing already produces (component, local vertex, cell),
        # which is the component-first Jacobian input contract.
        vertices=mesh.p[:,mesh.t[:3]]
        self.jacobians=triangle_affine_jacobian(vertices)
        determinants=(
            self.jacobians[0,0]*self.jacobians[1,1]
            -self.jacobians[0,1]*self.jacobians[1,0]
        )
        if np.any(np.isclose(determinants,0.)):
            raise ValueError("H(curl) basis requires nonsingular triangles")
        self.detJ=determinants
        edge_lengths=[]
        for first,second in ((0,1),(1,2),(2,0)):
            edge_lengths.append(np.linalg.norm(
                vertices[:,second]-vertices[:,first],axis=0
            ))
        longest=np.max(edge_lengths,axis=0)
        aspect=longest**2/np.abs(determinants)
        self.geometry_diagnostics=HcurlGeometryDiagnostics(
            minimum_signed_determinant=float(np.min(determinants)),
            minimum_absolute_determinant=float(np.min(np.abs(determinants))),
            minimum_area=float(.5*np.min(np.abs(determinants))),
            maximum_aspect_ratio=float(np.max(aspect)),
            inverted_cell_count=int(np.count_nonzero(determinants<0.)),
        )
        if max_aspect_ratio is not None:
            if (
                isinstance(max_aspect_ratio,bool)
                or not np.isscalar(max_aspect_ratio)
                or not np.isfinite(max_aspect_ratio)
                or max_aspect_ratio<=0.
            ):
                raise ValueError("max_aspect_ratio must be a positive finite scalar")
            if self.geometry_diagnostics.maximum_aspect_ratio>max_aspect_ratio:
                raise ValueError(
                    "H(curl) triangle aspect ratio exceeds "
                    f"max_aspect_ratio={max_aspect_ratio}"
                )
        self.dx=np.abs(determinants)[:,None]*self.W[None,:]

        reference_values=tri_n1_basis(self.X)
        reference_curls=tri_n1_curl(self.X)
        # Map every cell in one NumPy call.  The mapping helpers use a
        # component-first convention, so their batched outputs are
        # (basis, component, cell, quadrature) and (basis, cell, quadrature).
        # The private assembler layout remains entity-first to keep each
        # element's local tensors contiguous in the repeated assembly path.
        values=covariant_piola(
            reference_values[:,:,None,:],self.jacobians[...,None]
        ).transpose(2,0,1,3)
        curls=covariant_piola_curl(
            reference_curls[:,None,:],self.jacobians[...,None]
        ).transpose(1,0,2)
        self._element_values=np.ascontiguousarray(
            values*self.basis_signs.T[:,:,None,None]
        )
        self._element_curls=np.ascontiguousarray(
            curls*self.basis_signs.T[:,:,None]
        )

    @property
    def values(self):
        """Mapped basis values as ``(basis, component, cell, quadrature)``."""
        return self._element_values.transpose(1,2,0,3)

    @property
    def curls(self):
        """Mapped scalar curls as ``(basis, cell, quadrature)``."""
        return self._element_curls.transpose(1,0,2)

    def element_mass_matrices(self):
        return np.einsum(
            "ebiq,eciq,eq->ebc",
            self._element_values,self._element_values,self.dx
        )

    def element_curl_curl_matrices(self):
        return np.einsum(
            "ebq,ecq,eq->ebc",
            self._element_curls,self._element_curls,self.dx,
        )

    def interpolate_edge_moments(self,field,*,quadrature_order=4):
        """Interpolate a vector field by globally oriented tangential moments.

        ``field(x)`` receives physical points with shape ``(2, points)`` and
        must return a vector array broadcastable to the same shape.
        """
        if not callable(field):
            raise TypeError("field must be callable")
        if (
            isinstance(quadrature_order,bool)
            or not isinstance(quadrature_order,(int,np.integer))
            or quadrature_order<1
        ):
            raise ValueError("quadrature_order must be a positive integer")
        nodes,weights=np.polynomial.legendre.leggauss(int(quadrature_order))
        parameter=.5*(nodes+1.)
        weights=.5*weights
        coefficients=np.empty(self.N,dtype=np.float64)
        for edge_id,(start,end) in enumerate(self.dof_map.topology.edges.T):
            first=self.mesh.p[:,start]
            tangent=self.mesh.p[:,end]-first
            points=first[:,None]+tangent[:,None]*parameter[None,:]
            values=np.asarray(field(points),dtype=np.float64)
            try:
                values=np.broadcast_to(values,points.shape)
            except ValueError as error:
                raise ValueError(
                    "field must return vectors broadcastable to (2, points)"
                ) from error
            coefficients[edge_id]=np.einsum(
                "iq,i,q->",values,tangent,weights
            )
        return coefficients

    def _local_coefficients(self,coefficients):
        values=np.asarray(coefficients,dtype=np.float64)
        if values.shape!=(self.N,):
            raise ValueError(f"coefficients must have shape ({self.N},)")
        return values[self.element_dofs].T

    def evaluate(self,coefficients):
        """Evaluate a global edge field as ``(component, cell, quadrature)``."""
        local=self._local_coefficients(coefficients)
        return np.einsum("eb,ebiq->ieq",local,self._element_values)

    def evaluate_curl(self,coefficients):
        """Evaluate physical scalar curl as ``(cell, quadrature)``."""
        local=self._local_coefficients(coefficients)
        return np.einsum("eb,ebq->eq",local,self._element_curls)

    @property
    def global_coordinates(self):
        """Physical quadrature coordinates as ``(component, cell, quadrature)``."""
        vertices=self.mesh.p[:,self.mesh.t[0]]
        return vertices[:,:,None]+np.einsum(
            "ire,rq->ieq",self.jacobians,self.X
        )

    def boundary_dofs(self,boundary=None):
        """Select unique edge DOFs on all, named, or predicate boundaries.

        A tuple of boundary names forms their union.  Because a global edge is
        one DOF, overlaps at named-boundary intersections are removed.
        """
        return _boundary_edge_dofs(self,boundary)


class AffineTetN1Basis:
    """Lowest-order H(curl) basis on affine tetrahedral meshes."""

    def __init__(self,mesh,*,intorder=2,quadrature=None,max_aspect_ratio=None):
        if mesh.dim()!=3 or mesh.t.shape[0] not in (4,10):
            raise TypeError("AffineTetN1Basis requires a tetrahedral mesh")
        self.mesh=mesh
        self.dof_map=build_oriented_edge_dof_map(mesh)
        self.N=self.dof_map.ndofs
        self.element_dofs=self.dof_map.element_dofs
        self.basis_signs=self.dof_map.basis_signs
        self.X,self.W=(
            _tetrahedron_quadrature(intorder) if quadrature is None else
            (np.asarray(quadrature[0],dtype=np.float64),
             np.asarray(quadrature[1],dtype=np.float64))
        )
        if self.X.ndim!=2 or self.X.shape[0]!=3 or self.W.shape!=(self.X.shape[1],):
            raise ValueError("quadrature must contain points shaped (3, nq) and weights shaped (nq,)")

        vertices=mesh.p[:,mesh.t[:4]]
        self.jacobians=tetrahedron_affine_jacobian(vertices)
        self.detJ=(
            self.jacobians[0,0]*(
                self.jacobians[1,1]*self.jacobians[2,2]
                -self.jacobians[1,2]*self.jacobians[2,1]
            )
            -self.jacobians[0,1]*(
                self.jacobians[1,0]*self.jacobians[2,2]
                -self.jacobians[1,2]*self.jacobians[2,0]
            )
            +self.jacobians[0,2]*(
                self.jacobians[1,0]*self.jacobians[2,1]
                -self.jacobians[1,1]*self.jacobians[2,0]
            )
        )
        if np.any(np.isclose(self.detJ,0.)):
            raise ValueError("H(curl) basis requires nonsingular tetrahedra")
        edge_lengths=[
            np.linalg.norm(vertices[:,second]-vertices[:,first],axis=0)
            for first,second in ((0,1),(1,2),(2,0),(0,3),(1,3),(2,3))
        ]
        longest=np.max(edge_lengths,axis=0)
        aspect=longest**3/np.abs(self.detJ)
        self.geometry_diagnostics=HcurlTetGeometryDiagnostics(
            minimum_signed_determinant=float(np.min(self.detJ)),
            minimum_absolute_determinant=float(np.min(np.abs(self.detJ))),
            minimum_volume=float(np.min(np.abs(self.detJ))/6.),
            maximum_aspect_ratio=float(np.max(aspect)),
            inverted_cell_count=int(np.count_nonzero(self.detJ<0.)),
        )
        if max_aspect_ratio is not None:
            if (
                isinstance(max_aspect_ratio,bool)
                or not np.isscalar(max_aspect_ratio)
                or not np.isfinite(max_aspect_ratio)
                or max_aspect_ratio<=0.
            ):
                raise ValueError("max_aspect_ratio must be a positive finite scalar")
            if self.geometry_diagnostics.maximum_aspect_ratio>max_aspect_ratio:
                raise ValueError(
                    "H(curl) tetrahedron aspect ratio exceeds "
                    f"max_aspect_ratio={max_aspect_ratio}"
                )
        self.dx=np.abs(self.detJ)[:,None]*self.W[None,:]

        # Mapping helpers return component-first batches; the private arrays
        # are transposed once to entity-first assembler layout.
        values=covariant_piola(
            tet_n1_basis(self.X)[:,:,None,:],self.jacobians[...,None]
        ).transpose(2,0,1,3)
        curls=covariant_piola_vector_curl(
            tet_n1_curl(self.X)[:,:,None,:],self.jacobians[...,None]
        ).transpose(2,0,1,3)
        self._element_values=np.ascontiguousarray(
            values*self.basis_signs.T[:,:,None,None]
        )
        self._element_curls=np.ascontiguousarray(
            curls*self.basis_signs.T[:,:,None,None]
        )

    @property
    def values(self):
        """Mapped values as ``(basis, component, cell, quadrature)``."""
        return self._element_values.transpose(1,2,0,3)

    @property
    def curls(self):
        """Mapped vector curls as ``(basis, component, cell, quadrature)``."""
        return self._element_curls.transpose(1,2,0,3)

    def element_mass_matrices(self):
        return np.einsum(
            "ebiq,eciq,eq->ebc",
            self._element_values,self._element_values,self.dx,
        )

    def element_curl_curl_matrices(self):
        return np.einsum(
            "ebiq,eciq,eq->ebc",
            self._element_curls,self._element_curls,self.dx,
        )

    def interpolate_edge_moments(self,field,*,quadrature_order=4):
        """Interpolate a vector field by ascending-global-edge moments."""
        if not callable(field):
            raise TypeError("field must be callable")
        if (
            isinstance(quadrature_order,bool)
            or not isinstance(quadrature_order,(int,np.integer))
            or quadrature_order<1
        ):
            raise ValueError("quadrature_order must be a positive integer")
        nodes,weights=np.polynomial.legendre.leggauss(int(quadrature_order))
        parameter=.5*(nodes+1.)
        weights=.5*weights
        coefficients=np.empty(self.N,dtype=np.float64)
        for edge_id,(start,end) in enumerate(self.dof_map.topology.edges.T):
            first=self.mesh.p[:,start]
            tangent=self.mesh.p[:,end]-first
            points=first[:,None]+tangent[:,None]*parameter[None,:]
            values=np.asarray(field(points),dtype=np.float64)
            try:
                values=np.broadcast_to(values,points.shape)
            except ValueError as error:
                raise ValueError(
                    "field must return vectors broadcastable to (3, points)"
                ) from error
            coefficients[edge_id]=np.einsum(
                "iq,i,q->",values,tangent,weights
            )
        return coefficients

    def _local_coefficients(self,coefficients):
        values=np.asarray(coefficients,dtype=np.float64)
        if values.shape!=(self.N,):
            raise ValueError(f"coefficients must have shape ({self.N},)")
        return values[self.element_dofs].T

    def evaluate(self,coefficients):
        """Evaluate a global edge field as ``(component, cell, point)``."""
        local=self._local_coefficients(coefficients)
        return np.einsum("eb,ebiq->ieq",local,self._element_values)

    def evaluate_curl(self,coefficients):
        """Evaluate vector curl as ``(component, cell, point)``."""
        local=self._local_coefficients(coefficients)
        return np.einsum("eb,ebiq->ieq",local,self._element_curls)

    @property
    def global_coordinates(self):
        """Physical quadrature coordinates as ``(component, cell, point)``."""
        vertices=self.mesh.p[:,self.mesh.t[0]]
        return vertices[:,:,None]+np.einsum(
            "ire,rq->ieq",self.jacobians,self.X
        )

    def boundary_dofs(self,boundary=None):
        """Select edge DOFs owned by all or selected boundary facets."""
        return _boundary_edge_dofs(self,boundary)


__all__=[
    "AffineTetN1Basis","AffineTetN2Basis","AffineTetN3Basis",
    "AffineTriN1Basis","AffineTriN2Basis",
    "AffineTriN3Basis","OrientedTriN3DofMap",
    "HcurlGeometryDiagnostics","HcurlTetGeometryDiagnostics",
]
