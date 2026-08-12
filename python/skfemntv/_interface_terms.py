"""Typed trace nodes produced while compiling interface forms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._coefficients import is_symbolic_coefficient
from ._errors import UnsupportedNativeForm


@dataclass(frozen=True)
class InterfaceTrace:
    role: str
    weights: tuple[float, float] | None = None
    kind: str = "value"

    def _interface_transform(self, operation, value):
        if operation == "weights":
            return InterfaceTrace(self.role, tuple(value), self.kind)
        return InterfaceTrace(self.role, self.weights, value)


@dataclass(frozen=True)
class InterfaceCoefficientTrace:
    trace: InterfaceTrace
    coefficient: Any


@dataclass(frozen=True)
class InterfaceLinearTerm:
    trace: InterfaceTrace
    coefficient: Any
    factor: float = 1.0

    def __neg__(self):
        return InterfaceLinearTerm(self.trace, self.coefficient, -self.factor)

    def __mul__(self, value):
        if np.isscalar(value):
            return InterfaceLinearTerm(
                self.trace, self.coefficient, self.factor * value
            )
        return NotImplemented

    __rmul__ = __mul__

    def __add__(self, other):
        if isinstance(other, InterfaceLinearTerm):
            return InterfaceLinearSum((self, other))
        if isinstance(other, InterfaceLinearSum):
            return InterfaceLinearSum((self,) + other.terms)
        return NotImplemented


@dataclass(frozen=True)
class InterfaceLinearSum:
    terms: tuple[InterfaceLinearTerm, ...]

    def __add__(self, other):
        if isinstance(other, InterfaceLinearTerm):
            return InterfaceLinearSum(self.terms + (other,))
        if isinstance(other, InterfaceLinearSum):
            return InterfaceLinearSum(self.terms + other.terms)
        return NotImplemented


@dataclass(frozen=True)
class InterfaceBilinearTerm:
    row: InterfaceTrace
    column: InterfaceTrace
    coefficient: Any = None
    factor: float = 1.0

    def __mul__(self, other):
        if np.isscalar(other):
            return InterfaceBilinearTerm(
                self.row, self.column, self.coefficient, self.factor * other
            )
        if is_symbolic_coefficient(other):
            return InterfaceBilinearTerm(
                self.row, self.column, other, self.factor
            )
        if isinstance(other, np.ndarray):
            if self.coefficient is not None:
                raise UnsupportedNativeForm(
                    "multiple interface coefficients are not supported"
                )
            return InterfaceBilinearTerm(
                self.row, self.column, other, self.factor
            )
        return NotImplemented

    __rmul__ = __mul__

    def __neg__(self):
        return InterfaceBilinearTerm(
            self.row, self.column, self.coefficient, -self.factor
        )

    def __add__(self, other):
        if isinstance(other, InterfaceBilinearTerm):
            return InterfaceSum((self, other))
        return NotImplemented

    __radd__ = __add__

    def __sub__(self, other):
        return self + (-other)


@dataclass(frozen=True)
class InterfaceSum:
    terms: tuple[InterfaceBilinearTerm, ...]

    def __add__(self, other):
        if isinstance(other, InterfaceBilinearTerm):
            return InterfaceSum(self.terms + (other,))
        if isinstance(other, InterfaceSum):
            return InterfaceSum(self.terms + other.terms)
        return NotImplemented

    __radd__ = __add__

    def __neg__(self):
        return InterfaceSum(tuple(-term for term in self.terms))

    def __sub__(self, other):
        return self + (-other)


__all__ = [
    "InterfaceBilinearTerm",
    "InterfaceCoefficientTrace",
    "InterfaceLinearSum",
    "InterfaceLinearTerm",
    "InterfaceSum",
    "InterfaceTrace",
]
