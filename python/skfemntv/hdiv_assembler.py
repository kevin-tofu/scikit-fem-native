"""Sparse assembly for affine simplex RT0 bases using native cell kernels."""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix

from ._skfn import integrate_rt0,integrate_rt0_vector_load
from .hdiv_basis import AffineSimplexRT0Basis


class SimplexRT0Assembler:
    """Assemble RT0 mass, divergence, and vector-load terms.

    Cell quadrature is evaluated by the C++ backend. Sparse facet-DOF
    accumulation uses SciPy COO and returns a new CSR matrix per call.
    """

    def __init__(self,basis):
        if not isinstance(basis,AffineSimplexRT0Basis):
            raise TypeError("basis must be an affine simplex RT0 basis")
        self.basis=basis
        self._local=np.empty(
            (basis.mesh.nelements,basis.element_dofs.shape[0],
             basis.element_dofs.shape[0]),dtype=np.float64
        )
        self._local_load=np.empty(
            (basis.mesh.nelements,basis.element_dofs.shape[0]),dtype=np.float64
        )

    def _scalar_coefficient(self,coefficient,name):
        if coefficient is None:
            return np.ones(self.basis.dx.shape,dtype=np.float64)
        values=np.asarray(coefficient,dtype=np.float64)
        try:
            return np.ascontiguousarray(np.broadcast_to(values,self.basis.dx.shape))
        except ValueError as error:
            raise ValueError(
                f"{name} coefficient must broadcast to {self.basis.dx.shape}"
            ) from error

    def assemble(self,*,value=None,divergence=None):
        basis=self.basis
        if value is None and divergence is None:
            raise ValueError("at least one RT0 bilinear coefficient is required")
        value_coefficient=self._scalar_coefficient(value,"value")
        divergence_coefficient=self._scalar_coefficient(divergence,"divergence")
        integrate_rt0(
            basis._element_values,basis._element_divergences,basis.dx,
            value_coefficient,divergence_coefficient,value is not None,
            divergence is not None,self._local,
        )
        dofs=basis.element_dofs.T
        rows=np.broadcast_to(dofs[:,:,None],self._local.shape).reshape(-1)
        columns=np.broadcast_to(dofs[:,None,:],self._local.shape).reshape(-1)
        return coo_matrix(
            (self._local.reshape(-1),(rows,columns)),shape=(basis.N,basis.N)
        ).tocsr()

    def assemble_vector_load(self,field):
        basis=self.basis
        values=field(basis.global_coordinates) if callable(field) else field
        values=np.asarray(values,dtype=np.float64)
        dimension=basis.mesh.dim()
        expected=(dimension,)+basis.dx.shape
        if values.shape==(dimension,):
            values=values[:,None,None]
        elif values.shape==(dimension,basis.dx.shape[0]):
            values=values[:,:,None]
        try:
            values=np.ascontiguousarray(np.broadcast_to(values,expected))
        except ValueError as error:
            raise ValueError(f"vector load must broadcast to {expected}") from error
        integrate_rt0_vector_load(
            basis._element_values,values,basis.dx,self._local_load
        )
        result=np.zeros(basis.N,dtype=np.float64)
        np.add.at(result,basis.element_dofs.T.reshape(-1),self._local_load.reshape(-1))
        return result


__all__=["SimplexRT0Assembler"]
