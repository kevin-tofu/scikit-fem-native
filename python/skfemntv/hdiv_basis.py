"""Affine simplex Raviart--Thomas RT0 bases with oriented facet fluxes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._hcurl_mapping import jacobian_determinant,tetrahedron_affine_jacobian,triangle_affine_jacobian
from ._raviart_thomas import tet_rt0_basis,tri_rt0_basis
from .hcurl_basis import _tetrahedron_quadrature,_triangle_quadrature


@dataclass(frozen=True)
class OrientedFacetDofMap:
    element_dofs: np.ndarray
    basis_signs: np.ndarray
    ndofs: int


class AffineSimplexRT0Basis:
    """Lowest-order H(div) basis on affine triangles or tetrahedra."""

    def __init__(self,mesh,*,intorder=2,quadrature=None):
        dimension=mesh.dim()
        expected_nodes=3 if dimension==2 else 4 if dimension==3 else 0
        if expected_nodes==0 or mesh.t.shape[0]!=expected_nodes:
            raise TypeError("RT0 basis requires a linear triangle or tetrahedron mesh")
        self.mesh=mesh
        if dimension==2:
            default_quadrature=_triangle_quadrature(intorder)
            selected=default_quadrature if quadrature is None else quadrature
            self.X=np.asarray(selected[0],dtype=np.float64)
            self.W=np.asarray(selected[1],dtype=np.float64)
            jacobian=triangle_affine_jacobian(mesh.p[:,mesh.t[:3]])
            reference_values,reference_divergences=tri_rt0_basis(self.X)
            local_facet_rows=(0,1,2)
        else:
            default_quadrature=_tetrahedron_quadrature(intorder)
            selected=default_quadrature if quadrature is None else quadrature
            self.X=np.asarray(selected[0],dtype=np.float64)
            self.W=np.asarray(selected[1],dtype=np.float64)
            jacobian=tetrahedron_affine_jacobian(mesh.p[:,mesh.t[:4]])
            reference_values,reference_divergences=tet_rt0_basis(self.X)
            local_facet_rows=(0,1,2,3)
        if (
            self.X.ndim!=2 or self.X.shape[0]!=dimension
            or self.W.ndim!=1 or self.W.shape[0]!=self.X.shape[1]
        ):
            raise ValueError(
                "quadrature must contain points shaped (dimension, points) "
                "and matching one-dimensional weights"
            )
        determinant=jacobian_determinant(jacobian)
        if np.any(np.isclose(determinant,0.)):
            raise ValueError("RT0 basis requires nonsingular cells")
        self.jacobians=jacobian
        self.detJ=determinant
        self.dx=np.abs(determinant)[:,None]*self.W[None,:]
        raw_dofs=mesh.t2f[np.asarray(local_facet_rows)]
        owner=mesh.f2t[0,raw_dofs]
        signs=np.where(owner==np.arange(mesh.nelements)[None,:],1.,-1.)
        self.element_dofs=np.ascontiguousarray(raw_dofs,dtype=np.int64)
        self.basis_signs=np.ascontiguousarray(signs,dtype=np.float64)
        self.dof_map=OrientedFacetDofMap(
            self.element_dofs,self.basis_signs,int(mesh.facets.shape[1])
        )
        self.N=self.dof_map.ndofs
        reference_doflocs=(
            np.array([[.5,0.],[.5,.5],[0.,.5]]) if dimension==2 else
            np.array([[.5,.5,0.],[.5,0.,.5],[0.,.5,.5],[.5,.5,.5]])
        )
        self.doflocs=np.empty((dimension,self.N),dtype=np.float64)
        for facet in range(self.N):
            owner=int(mesh.f2t[0,facet])
            local=int(np.flatnonzero(self.element_dofs[:,owner]==facet)[0])
            origin=mesh.p[:,mesh.t[0,owner]]
            self.doflocs[:,facet]=origin+jacobian[:,:,owner]@reference_doflocs[local]
        self._element_values=np.ascontiguousarray(
            self._map_values(jacobian,reference_values,determinant,signs)
        )
        self._element_divergences=(np.ascontiguousarray(
            reference_divergences[:,None,:]/np.abs(determinant)[None,:,None]
            *signs[:,:,None]
        ).transpose(1,0,2).copy(order="C"))

    @staticmethod
    def _map_values(jacobian,reference_values,determinant,signs):
        mapped=np.einsum("ijc,bjq->bicq",jacobian,reference_values)
        mapped/=np.abs(determinant)[None,None,:,None]
        return mapped.transpose(2,0,1,3)*signs.T[:,:,None,None]

    @property
    def values(self):
        return self._element_values.transpose(1,2,0,3)

    @property
    def divergences(self):
        return self._element_divergences

    @property
    def global_coordinates(self):
        dimension=self.mesh.dim()
        vertices=self.mesh.p[:,self.mesh.t[0]]
        return vertices[:,:,None]+np.einsum(
            "ire,rq->ieq",self.jacobians,self.X
        )

    def evaluate(self,coefficients):
        coefficients=np.asarray(coefficients,dtype=np.float64)
        if coefficients.shape!=(self.N,):
            raise ValueError(f"coefficients must have shape ({self.N},)")
        local=coefficients[self.element_dofs.T]
        return np.einsum("eb,ebiq->ieq",local,self._element_values)

    def evaluate_divergence(self,coefficients):
        coefficients=np.asarray(coefficients,dtype=np.float64)
        if coefficients.shape!=(self.N,):
            raise ValueError(f"coefficients must have shape ({self.N},)")
        local=coefficients[self.element_dofs.T]
        return np.einsum("eb,ebq->eq",local,self._element_divergences)

    def boundary_dofs(self,boundary=None):
        if boundary is None:
            selected=self.mesh.boundary_facets()
        elif isinstance(boundary,str):
            try:
                selected=np.asarray(self.mesh.boundaries[boundary],dtype=np.int64)
            except KeyError as error:
                raise KeyError(f"unknown boundary {boundary!r}") from error
        elif callable(boundary):
            candidates=self.mesh.boundary_facets()
            centers=self.mesh.p[:,self.mesh.facets[:self.mesh.dim(),candidates]].mean(axis=1)
            mask=np.asarray(boundary(centers),dtype=bool)
            if mask.shape!=(len(candidates),):
                raise ValueError("boundary predicate must return one boolean per facet")
            selected=candidates[mask]
        else:
            selected=np.asarray(boundary,dtype=np.int64).reshape(-1)
        valid=set(map(int,self.mesh.boundary_facets()))
        if any(int(facet) not in valid for facet in selected):
            raise ValueError("H(div) boundary DOFs require boundary facet IDs")
        return np.unique(selected)


class AffineTriRT0Basis(AffineSimplexRT0Basis):
    def __init__(self,mesh,*,intorder=2,quadrature=None):
        if mesh.dim()!=2:
            raise TypeError("AffineTriRT0Basis requires a triangular mesh")
        super().__init__(mesh,intorder=intorder,quadrature=quadrature)


class AffineTetRT0Basis(AffineSimplexRT0Basis):
    def __init__(self,mesh,*,intorder=2,quadrature=None):
        if mesh.dim()!=3:
            raise TypeError("AffineTetRT0Basis requires a tetrahedral mesh")
        super().__init__(mesh,intorder=intorder,quadrature=quadrature)


__all__=["AffineSimplexRT0Basis","AffineTetRT0Basis","AffineTriRT0Basis"]
