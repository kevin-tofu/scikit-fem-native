from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from ._skfn import BilinearFormAssembler


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
