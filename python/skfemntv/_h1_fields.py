"""Symbolic trial/test nodes for the typed H1 form subset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._errors import UnsupportedNativeForm
from ._form_terms import BilinearTerm


@dataclass(frozen=True)
class TestValue:
    factor: float = 1.0

    def __mul__(self, value):
        return TestValue(self.factor * value) if np.isscalar(value) else NotImplemented

    __rmul__ = __mul__


@dataclass(frozen=True)
class TestGradient:
    factor: float = 1.0


@dataclass(frozen=True)
class TrialValue:
    factor: float = 1.0

    def __mul__(self, value):
        return TrialValue(self.factor * value) if np.isscalar(value) else NotImplemented

    __rmul__ = __mul__


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
    role: str
    factor: float = 1.0
    coefficient: Any = None

    def __mul__(self, other):
        if np.isscalar(other):
            return Divergence(
                self.role, self.factor * other, self.coefficient
            )
        if isinstance(other, Divergence) and self.role != other.role:
            if self.coefficient is not None or other.coefficient is not None:
                raise UnsupportedNativeForm(
                    "weighted divergence products are not supported"
                )
            return BilinearTerm(
                "divergence", factor=self.factor * other.factor
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


__all__ = [
    "Divergence",
    "SymmetricGradient",
    "TensorGradient",
    "TestGradient",
    "TestValue",
    "TrialGradient",
    "TrialValue",
]
