from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from ._skfn import BilinearFormAssembler,CrossBilinearAssembler


class NativeBilinearForm:
    """Native component-wise value and gradient bilinear assembly."""

    def __init__(self, basis):
        scalar = basis.elem.elem
        components = basis.elem._dim
        nodes = len(scalar.doflocs)
        entities, quadrature = basis.dx.shape
        dofs = basis.element_dofs.T.reshape(entities, nodes, components)
        self._native = BilinearFormAssembler(
            np.asarray(dofs, dtype=np.int64, order="C"),
            np.asarray(basis.tabulated_shape, dtype=np.float64, order="C"),
            np.asarray(
                basis.tabulated_gradients, dtype=np.float64, order="C"
            ),
            np.asarray(basis.dx, dtype=np.float64, order="C"),
        )
        self._coefficient_shape = (entities, quadrature)
        self._matrix = csr_matrix(
            (
                self._native.values,
                self._native.indices,
                self._native.indptr,
            ),
            shape=(self._native.ndofs, self._native.ndofs),
            copy=False,
        )

    def assemble(self, *, value=None, gradient=None):
        value = self._coefficient("value", value)
        gradient = self._coefficient("gradient", gradient)
        self._native.assemble(value, gradient)
        return self._matrix

    def _coefficient(self, name, coefficient):
        if coefficient is None:
            return None
        coefficient = np.asarray(coefficient)
        if coefficient.dtype != np.float64:
            coefficient = coefficient.astype(np.float64)
        try:
            coefficient = np.broadcast_to(
                coefficient, self._coefficient_shape
            )
        except ValueError as error:
            raise ValueError(
                f"{name} coefficient must broadcast to "
                f"{self._coefficient_shape}"
            ) from error
        return np.ascontiguousarray(coefficient)


class NativeCompositeBilinearForm:
    """Reusable rectangular native blocks for a composite H1 basis."""

    def __init__(self,basis):
        self.basis=basis
        self._assemblers={}

    def assemble(
        self,row_field,column_field,*,kind,coefficient,
    ):
        row=self.basis.subbases[row_field]
        column=self.basis.subbases[column_field]
        row_components=row.elem._dim
        column_components=column.elem._dim
        if row_components!=column_components:
            raise ValueError(
                "component-mismatched composite coupling requires "
                "an explicit tensor contraction"
            )
        key=(row_field,column_field,kind)
        native=self._assemblers.get(key)
        if native is None:
            row_nodes=len(row.elem.elem.doflocs)
            column_nodes=len(column.elem.elem.doflocs)
            entities,quadrature=self.basis.dx.shape
            row_dofs=row.element_dofs.T.reshape(
                entities,row_nodes,row_components
            )
            column_dofs=column.element_dofs.T.reshape(
                entities,column_nodes,column_components
            )
            native=CrossBilinearAssembler(
                np.ascontiguousarray(row_dofs,dtype=np.int64),
                np.ascontiguousarray(column_dofs,dtype=np.int64),
                np.ascontiguousarray(row.tabulated_shape),
                np.ascontiguousarray(column.tabulated_shape),
                np.ascontiguousarray(self.basis.dx),
                np.ascontiguousarray(row.tabulated_gradients),
                np.ascontiguousarray(column.tabulated_gradients),
            )
            self._assemblers[key]=native
        target=self.basis.dx.shape
        coefficient=np.ascontiguousarray(np.broadcast_to(
            np.asarray(coefficient,dtype=np.float64),target
        ))
        native.assemble(coefficient,kind,kind)
        matrix=csr_matrix(
            (native.values,native.indices,native.indptr),
            shape=(native.rows,native.columns),copy=False,
        )
        matrix.resize((self.basis.N,self.basis.N))
        return matrix.copy()
