"""Minimal lowest-order H(curl) basis for affine triangular meshes."""

from __future__ import annotations

import numpy as np

from ._hcurl_mapping import (
    covariant_piola,covariant_piola_curl,triangle_affine_jacobian,
)
from ._nedelec_reference import tri_n1_basis,tri_n1_curl
from .edge_dof_map import build_oriented_edge_dof_map


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


class AffineTriN1Basis:
    """Reference tabulation, Piola mapping, orientation and integration data."""

    def __init__(self,mesh,*,intorder=2):
        if mesh.dim()!=2 or mesh.t.shape[0] not in (3,6):
            raise TypeError("AffineTriN1Basis requires a triangular mesh")
        self.mesh=mesh
        self.dof_map=build_oriented_edge_dof_map(mesh)
        self.N=self.dof_map.ndofs
        self.element_dofs=self.dof_map.element_dofs
        self.basis_signs=self.dof_map.basis_signs
        self.X,self.W=_triangle_quadrature(intorder)

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
        self.dx=np.abs(determinants)[:,None]*self.W[None,:]

        reference_values=tri_n1_basis(self.X)
        reference_curls=tri_n1_curl(self.X)
        values=np.empty((mesh.nelements,3,2,len(self.W)))
        curls=np.empty((mesh.nelements,3,len(self.W)))
        for cell in range(mesh.nelements):
            values[cell]=covariant_piola(
                reference_values,self.jacobians[...,cell,None]
            )*self.basis_signs[:,cell,None,None]
            curls[cell]=covariant_piola_curl(
                reference_curls,self.jacobians[...,cell,None]
            )*self.basis_signs[:,cell,None]
        self._element_values=np.ascontiguousarray(values)
        self._element_curls=np.ascontiguousarray(curls)

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
        if isinstance(boundary,tuple) and all(
            isinstance(name,str) for name in boundary
        ):
            selected=np.concatenate([
                self.boundary_dofs(name) for name in boundary
            ]) if boundary else np.empty(0,dtype=np.int64)
            return np.unique(selected)
        if boundary is None:
            facets=self.mesh.boundary_facets()
        elif isinstance(boundary,str):
            try:
                facets=np.asarray(self.mesh.boundaries[boundary],dtype=np.int64)
            except KeyError as error:
                raise KeyError(f"unknown boundary {boundary!r}") from error
        elif callable(boundary):
            candidates=self.mesh.boundary_facets()
            centers=self.mesh.p[:,self.mesh.facets[:,candidates]].mean(axis=1)
            mask=np.asarray(boundary(centers),dtype=bool)
            if mask.shape!=(len(candidates),):
                raise ValueError(
                    "boundary predicate must return one boolean per boundary facet"
                )
            facets=candidates[mask]
        else:
            facets=np.asarray(boundary)
            if facets.dtype==bool:
                candidates=self.mesh.boundary_facets()
                if facets.shape!=(len(candidates),):
                    raise ValueError(
                        "boundary mask must contain one value per boundary facet"
                    )
                facets=candidates[facets]
            facets=np.asarray(facets,dtype=np.int64).reshape(-1)
        boundary_set=set(map(int,self.mesh.boundary_facets()))
        if any(int(facet) not in boundary_set for facet in facets):
            raise ValueError("H(curl) boundary DOFs require boundary facet IDs")
        edge_lookup={
            tuple(edge):index
            for index,edge in enumerate(self.dof_map.topology.edges.T)
        }
        return np.asarray(sorted({
            edge_lookup[tuple(sorted(map(int,self.mesh.facets[:,facet])))]
            for facet in facets
        }),dtype=np.int64)


__all__=["AffineTriN1Basis"]
