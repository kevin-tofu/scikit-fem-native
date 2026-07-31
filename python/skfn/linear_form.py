from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._skfn import LinearFormAssembler


@dataclass(frozen=True)
class LinearFormDiagnostics:
    entity_count: int
    quadrature_point_count: int
    assembly_seconds: float


class NativeLinearForm:
    """Native assembly of ``f·v + G:grad(v)`` on Basis or FacetBasis."""

    def __init__(self, basis) -> None:
        scalar_element = getattr(basis.elem, "elem", None)
        components = getattr(basis.elem, "_dim", None)
        if scalar_element is None or components is None:
            raise ValueError("basis must use ElementVector over a nodal H1 element")
        nodes = len(scalar_element.doflocs)
        if basis.element_dofs.shape[0] != nodes * components:
            raise ValueError("only nodal vector H1 elements are supported")
        entity_count, quadrature_count = basis.dx.shape
        shape = np.empty((entity_count, quadrature_count, nodes))
        gradients = np.empty(
            (entity_count, quadrature_count, nodes, basis.mesh.dim())
        )
        for node in range(nodes):
            # CellBasis and FacetBasis already contain correctly mapped vector
            # basis fields; using them also handles facet reference coordinates.
            field = basis.basis[node * components][0]
            shape[:, :, node] = np.asarray(field)[0]
            gradients[:, :, node, :] = field.grad[0].transpose(1, 2, 0)
        dofs = basis.element_dofs.T.reshape(
            entity_count, nodes, components
        )
        self._native = LinearFormAssembler(
            np.asarray(dofs, dtype=np.int64, order="C"),
            np.asarray(shape, dtype=np.float64, order="C"),
            np.asarray(gradients, dtype=np.float64, order="C"),
            np.asarray(basis.dx, dtype=np.float64, order="C"),
        )
        self._shape = (entity_count, quadrature_count, components)
        self._gradient_shape = (
            entity_count,
            quadrature_count,
            components,
            basis.mesh.dim(),
        )

    @property
    def ndofs(self) -> int:
        return self._native.ndofs

    def assemble(
        self,
        *,
        value: np.ndarray | None = None,
        gradient: np.ndarray | None = None,
    ) -> tuple[np.ndarray, LinearFormDiagnostics]:
        value = self._coefficient("value", value, self._shape)
        gradient = self._coefficient(
            "gradient", gradient, self._gradient_shape
        )
        result, seconds = self._native.assemble(value, gradient)
        return result, LinearFormDiagnostics(
            entity_count=self._native.entity_count,
            quadrature_point_count=self._native.quadrature_point_count,
            assembly_seconds=seconds,
        )

    @staticmethod
    def _coefficient(name, coefficient, shape):
        if coefficient is None:
            return None
        coefficient = np.asarray(coefficient)
        if coefficient.dtype != np.float64:
            raise TypeError(f"{name} coefficient must have dtype float64")
        try:
            coefficient = np.broadcast_to(coefficient, shape)
        except ValueError as error:
            raise ValueError(
                f"{name} coefficient must broadcast to {shape}"
            ) from error
        return np.ascontiguousarray(coefficient)


class NativeCompositeLinearForm:
    """Reusable native linear assemblers for composite H1 subfields."""

    def __init__(self,basis) -> None:
        self.basis=basis
        self._assemblers={}

    def assembler(self,field):
        native=self._assemblers.get(field)
        if native is None:
            native=NativeLinearForm(self.basis.subbases[field])
            self._assemblers[field]=native
        return native

    def assemble(self,field,*,value=None,gradient=None):
        vector,_=self.assembler(field).assemble(
            value=value,gradient=gradient
        )
        if vector.shape[0]==self.basis.N:
            return vector
        result=np.zeros(self.basis.N,dtype=np.float64)
        result[:vector.shape[0]]=vector
        return result
