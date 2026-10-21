"""Sparse assembly for affine simplex RT0 bases using native cell kernels."""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix

from ._skfn import (
    integrate_rt0,integrate_rt0_dg0_divergence,
    integrate_rt0_vector_load,
)
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


class RT0DG0MixedAssembler:
    """Assemble RT0-to-DG0 divergence blocks on aligned simplex bases."""

    def __init__(self,rt0_basis,scalar_basis):
        if not isinstance(rt0_basis,AffineSimplexRT0Basis):
            raise TypeError("rt0_basis must be an affine simplex RT0 basis")
        if rt0_basis.mesh.dim()!=scalar_basis.mesh.dim():
            raise ValueError("RT0 and DG0 bases must have the same dimension")
        if rt0_basis.dx.shape!=scalar_basis.dx.shape or not np.allclose(
            rt0_basis.dx,scalar_basis.dx
        ):
            raise ValueError("RT0 and DG0 quadrature weights must match")
        if not np.allclose(
            rt0_basis.global_coordinates,
            np.moveaxis(scalar_basis.global_coordinates,-1,0),
        ):
            raise ValueError("RT0 and DG0 quadrature coordinates must match")
        if scalar_basis.tabulated_shape.shape[2]!=1:
            raise ValueError("scalar basis must have one DG0 DOF per cell")
        self.rt0_basis=rt0_basis
        self.scalar_basis=scalar_basis
        self._scalar_shapes=np.ascontiguousarray(
            scalar_basis.tabulated_shape.transpose(0,2,1)
        )

    def assemble_divergence(self,coefficient=1.,*,rt0_is_test=True):
        rt0=self.rt0_basis
        scalar=self.scalar_basis
        weights=np.asarray(coefficient,dtype=np.float64)
        try:
            weights=np.ascontiguousarray(np.broadcast_to(weights,rt0.dx.shape))
        except ValueError as error:
            raise ValueError(
                f"coefficient must broadcast to {rt0.dx.shape}"
            ) from error
        rows=rt0.element_dofs.T if rt0_is_test else scalar.element_dofs.T
        columns=scalar.element_dofs.T if rt0_is_test else rt0.element_dofs.T
        row_local=rows.shape[1]
        column_local=columns.shape[1]
        local=np.empty(
            (rt0.mesh.nelements,row_local,column_local),dtype=np.float64
        )
        integrate_rt0_dg0_divergence(
            rt0._element_divergences,self._scalar_shapes,rt0.dx,weights,
            rt0_is_test,local,
        )
        row_ids=np.broadcast_to(rows[:,:,None],local.shape).reshape(-1)
        column_ids=np.broadcast_to(
            columns[:,None,:],local.shape
        ).reshape(-1)
        return coo_matrix(
            (local.reshape(-1),(row_ids,column_ids)),
            shape=(
                self.rt0_basis.N if rt0_is_test else self.scalar_basis.N,
                self.scalar_basis.N if rt0_is_test else self.rt0_basis.N,
            ),
        ).tocsr()


__all__=["RT0DG0MixedAssembler","SimplexRT0Assembler"]
