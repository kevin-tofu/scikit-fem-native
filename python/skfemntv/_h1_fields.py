"""Symbolic trial/test nodes for the typed H1 form subset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._errors import UnsupportedNativeForm
from ._coefficients import is_symbolic_coefficient
from ._form_terms import BilinearTerm, LinearTerm


@dataclass(frozen=True)
class TestValue:
    __array_priority__ = 1000
    factor: float = 1.0
    component: int | None = None

    def __getitem__(self, index):
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise ValueError("test-value component index must be an integer")
        if index < 0:
            raise ValueError("test-value component index must be nonnegative")
        return TestValue(self.factor, int(index))

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if ufunc is np.multiply and method == "__call__" and len(inputs) == 2:
            other = inputs[0] if inputs[1] is self else inputs[1]
            return self * other
        return NotImplemented

    def __mul__(self, value):
        if np.isscalar(value):
            return TestValue(self.factor * value, self.component)
        if isinstance(value, TrialValue):
            return BilinearTerm("value", factor=self.factor * value.factor)
        if is_symbolic_coefficient(value) or isinstance(value, np.ndarray):
            return LinearTerm(
                "value", value, self.factor, self.component
            )
        if hasattr(value, "value") and hasattr(value, "grad"):
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
    __array_priority__ = 1000
    factor: float = 1.0

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if ufunc is np.multiply and method == "__call__" and len(inputs) == 2:
            other = inputs[0] if inputs[1] is self else inputs[1]
            return self * other
        return NotImplemented

    def __mul__(self, value):
        if np.isscalar(value):
            return TrialValue(self.factor * value)
        if isinstance(value, TestValue):
            return BilinearTerm("value", factor=self.factor * value.factor)
        if is_symbolic_coefficient(value):
            return TensorValue(self, value)
        if isinstance(value, np.ndarray):
            return TensorValue(self, value)
        return NotImplemented

    __rmul__ = __mul__


@dataclass(frozen=True)
class TrialGradient:
    factor: float = 1.0


@dataclass(frozen=True)
class TensorGradient:
    gradient: TrialGradient | TestGradient
    coefficient: Any


@dataclass(frozen=True)
class TensorValue:
    value: TrialValue | TestValue
    coefficient: Any

    def __mul__(self, other):
        if isinstance(other, (TrialValue, TestValue)):
            if self.value.__class__ is other.__class__:
                raise UnsupportedNativeForm(
                    "value contraction requires trial and test fields"
                )
            return BilinearTerm(
                "value",
                self.coefficient,
                self.value.factor * other.factor,
            )
        return NotImplemented

@dataclass(frozen=True)
class SymmetricGradient:
    role: str
    factor: float = 1.0


@dataclass(frozen=True)
class Divergence:
    __array_priority__ = 1000

    role: str
    factor: float = 1.0
    coefficient: Any = None

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if ufunc is np.multiply and method == "__call__" and len(inputs) == 2:
            other = inputs[0] if inputs[1] is self else inputs[1]
            return self * other
        return NotImplemented

    def __mul__(self, other):
        if np.isscalar(other):
            return Divergence(
                self.role, self.factor * other, self.coefficient
            )
        if is_symbolic_coefficient(other) or isinstance(other, np.ndarray):
            coefficient = (
                other
                if self.coefficient is None
                else self.coefficient * other
            )
            return Divergence(self.role, self.factor, coefficient)
        if isinstance(other, Divergence) and self.role != other.role:
            coefficients = [
                value
                for value in (self.coefficient, other.coefficient)
                if value is not None
            ]
            if len(coefficients) > 1:
                raise UnsupportedNativeForm(
                    "multiple divergence coefficients are not supported"
                )
            return BilinearTerm(
                "divergence",
                coefficient=coefficients[0] if coefficients else None,
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


@dataclass(frozen=True)
class Curl:
    role: str
    factor: float = 1.0
    coefficient: object = None

    def __mul__(self,other):
        if np.isscalar(other):
            return Curl(self.role,self.factor*other,self.coefficient)
        if isinstance(other,np.ndarray):
            coefficient=(other if self.coefficient is None else
                         self.coefficient*other)
            return Curl(self.role,self.factor,coefficient)
        if isinstance(other,Curl) and self.role!=other.role:
            coefficient=self.coefficient
            if other.coefficient is not None:
                coefficient=(other.coefficient if coefficient is None else
                             coefficient*other.coefficient)
            return BilinearTerm(
                "curl",coefficient,self.factor*other.factor
            )
        return NotImplemented

    __rmul__=__mul__


__all__ = [
    "Curl",
    "Divergence",
    "SymmetricGradient",
    "TensorGradient",
    "TestGradient",
    "TestValue",
    "TrialGradient",
    "TrialValue",
]
