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
)
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

    def __init__(self,mesh,*,intorder=2,max_aspect_ratio=None):
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

    def __init__(self,mesh,*,intorder=2,max_aspect_ratio=None):
        if mesh.dim()!=3 or mesh.t.shape[0] not in (4,10):
            raise TypeError("AffineTetN1Basis requires a tetrahedral mesh")
        self.mesh=mesh
        self.dof_map=build_oriented_edge_dof_map(mesh)
        self.N=self.dof_map.ndofs
        self.element_dofs=self.dof_map.element_dofs
        self.basis_signs=self.dof_map.basis_signs
        self.X,self.W=_tetrahedron_quadrature(intorder)

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
    "AffineTetN1Basis","AffineTriN1Basis",
    "HcurlGeometryDiagnostics","HcurlTetGeometryDiagnostics",
]
