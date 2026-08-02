from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._skfn import CutLinearFormAssembler,LinearFormAssembler


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
        shape=np.asarray(basis.tabulated_shape,dtype=np.float64,order="C")
        gradients=np.asarray(
            basis.tabulated_gradients,dtype=np.float64,order="C"
        )
        self._cut=hasattr(basis,"cell_offsets")
        if self._cut:
            self._native=CutLinearFormAssembler(
                np.ascontiguousarray(basis.cell_dofs,dtype=np.int64),
                np.ascontiguousarray(basis.cell_offsets,dtype=np.int64),
                np.ascontiguousarray(basis.shape,dtype=np.float64),
                np.ascontiguousarray(basis.gradients,dtype=np.float64),
                np.ascontiguousarray(basis.weights,dtype=np.float64),
            )
        else:
            dofs = basis.element_dofs.T.reshape(
                entity_count, nodes, components
            )
            self._native = LinearFormAssembler(
                np.asarray(dofs, dtype=np.int64, order="C"),shape,gradients,
                np.asarray(basis.dx,dtype=np.float64,order="C"),
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
        num_threads: int = 0,
    ) -> tuple[np.ndarray, LinearFormDiagnostics]:
        value = self._coefficient("value", value, self._shape)
        gradient = self._coefficient(
            "gradient", gradient, self._gradient_shape
        )
        if self._cut:
            if value is not None and value.ndim==3:
                value=value[:,0,:]
            if gradient is not None and gradient.ndim==4:
                gradient=gradient[:,0,:,:]
            result,seconds=self._native.assemble(value,gradient,num_threads)
            entity_count=self._native.cell_count
            point_count=self._native.point_count
        else:
            result,seconds=self._native.assemble(value,gradient,num_threads)
            entity_count=self._native.entity_count
            point_count=self._native.quadrature_point_count
        return result, LinearFormDiagnostics(
            entity_count=entity_count,
            quadrature_point_count=point_count,
            assembly_seconds=seconds,
        )

    @staticmethod
    def _coefficient(name, coefficient, shape):
        if coefficient is None:
            return None
        coefficient = np.asarray(coefficient)
        if coefficient.dtype != np.float64:
            raise TypeError(f"{name} coefficient must have dtype float64")
        constant_shape = shape[2:]
        if coefficient.shape == constant_shape:
            return np.ascontiguousarray(coefficient)
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
