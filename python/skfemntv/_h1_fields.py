"""Symbolic trial/test nodes for the typed H1 form subset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from ._errors import UnsupportedNativeForm
from ._form_terms import BilinearTerm, LinearTerm


@dataclass(frozen=True)
class TestValue:
    __array_priority__: ClassVar[int] = 1000
    factor: float = 1.0

    def __mul__(self, value):
        if np.isscalar(value):
            return TestValue(self.factor * value)
        if isinstance(value, TrialValue):
            return BilinearTerm(
                "value", coefficient=value.coefficient,
                factor=self.factor * value.factor,
            )
        if isinstance(value, np.ndarray) or hasattr(value, "value"):
            return LinearTerm("value", np.asarray(value), self.factor)
        return NotImplemented

    __rmul__ = __mul__

    def __getitem__(self, component):
        if isinstance(component, bool) or not isinstance(component, (int, np.integer)):
            raise IndexError("test-field component index must be an integer")
        return TestComponent(int(component), self.factor)


@dataclass(frozen=True)
class TestComponent:
    __array_priority__: ClassVar[int] = 1000
    component: int
    factor: float = 1.0

    def __mul__(self, value):
        if np.isscalar(value):
            return LinearTerm("value", value, self.factor, self.component)
        if isinstance(value, np.ndarray) or hasattr(value, "value"):
            return LinearTerm(
                "value", np.asarray(value), self.factor, self.component
            )
        return NotImplemented

    __rmul__ = __mul__


@dataclass(frozen=True)
class TestGradient:
    factor: float = 1.0


@dataclass(frozen=True)
class TrialValue:
    __array_priority__: ClassVar[int] = 1000
    factor: float = 1.0
    coefficient: Any = None

    def __mul__(self, value):
        if np.isscalar(value):
            return TrialValue(
                self.factor * value, self.coefficient
            )
        if isinstance(value, TestValue):
            return BilinearTerm(
                "value", coefficient=self.coefficient,
                factor=self.factor * value.factor,
            )
        if isinstance(value, np.ndarray) or hasattr(value, "value"):
            if self.coefficient is not None:
                raise UnsupportedNativeForm(
                    "multiple trial-value coefficients are not supported"
                )
            return TrialValue(self.factor, np.asarray(value))
        return NotImplemented

    __rmul__ = __mul__

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if ufunc is np.multiply and method == "__call__":
            other = inputs[0] if inputs[1] is self else inputs[1]
            return self.__mul__(other)
        return NotImplemented


@dataclass(frozen=True)
class TrialGradient:
    factor: float = 1.0


@dataclass(frozen=True)
class TensorGradient:
    gradient: TrialGradient | TestGradient
    coefficient: Any


@dataclass(frozen=True)
class SymmetricGradient:
    role: str
    factor: float = 1.0


@dataclass(frozen=True)
class Divergence:
    __array_priority__: ClassVar[int] = 1000
    role: str
    factor: float = 1.0
    coefficient: Any = None

    def __mul__(self, other):
        if np.isscalar(other):
            return Divergence(
                self.role, self.factor * other, self.coefficient
            )
        if isinstance(other, np.ndarray) or hasattr(other, "value"):
            if self.coefficient is not None:
                raise UnsupportedNativeForm(
                    "multiple divergence coefficients are not supported"
                )
            return Divergence(
                self.role, self.factor, np.asarray(other)
            )
        if isinstance(other, Divergence) and self.role != other.role:
            if (
                self.coefficient is not None
                and other.coefficient is not None
            ):
                raise UnsupportedNativeForm(
                    "multiple divergence coefficients are not supported"
                )
            return BilinearTerm(
                "divergence",
                coefficient=(
                    self.coefficient
                    if self.coefficient is not None
                    else other.coefficient
                ),
                factor=self.factor * other.factor,
            )
        if self.role == "trial" and isinstance(other, TestValue):
            return BilinearTerm(
                "column_divergence",
                coefficient=self.coefficient,
                factor=self.factor * other.factor,
            )
        if self.role == "test" and isinstance(other, TrialValue):
            return BilinearTerm(
                "row_divergence",
                coefficient=self.coefficient,
                factor=self.factor * other.factor,
            )
        return NotImplemented

    __rmul__ = __mul__

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if ufunc is np.multiply and method == "__call__":
            other = inputs[0] if inputs[1] is self else inputs[1]
            return self.__mul__(other)
        return NotImplemented


@dataclass(frozen=True)
class Curl:
    role: str
    factor: float = 1.0

    def __mul__(self,other):
        if np.isscalar(other):
            return Curl(self.role,self.factor*other)
        if isinstance(other,Curl) and self.role!=other.role:
            return BilinearTerm("curl",factor=self.factor*other.factor)
        return NotImplemented

    __rmul__=__mul__


__all__ = [
    "Curl",
    "Divergence",
    "SymmetricGradient",
    "TensorGradient",
    "TestGradient",
    "TestComponent",
    "TestValue",
    "TrialGradient",
    "TrialValue",
]
