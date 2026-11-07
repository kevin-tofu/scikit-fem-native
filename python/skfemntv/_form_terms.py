"""Assembly-independent result nodes produced by typed form expressions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._coefficients import is_symbolic_coefficient
from ._errors import UnsupportedNativeForm


@dataclass(frozen=True)
class LinearTerm:
    kind: str
    coefficient: Any
    factor: float = 1.0

    def __neg__(self):
        return LinearTerm(self.kind, self.coefficient, -self.factor)

    def __mul__(self, value):
        if np.isscalar(value):
            return LinearTerm(self.kind, self.coefficient, self.factor * value)
        return NotImplemented

    __rmul__ = __mul__

    def __add__(self, other):
        return LinearSum((self,)) + other

    def __sub__(self, other):
        return self + (-other)


@dataclass(frozen=True)
class LinearSum:
    terms: tuple[LinearTerm, ...]

    def __add__(self, other):
        if isinstance(other, LinearTerm):
            return LinearSum(self.terms + (other,))
        if isinstance(other, LinearSum):
            return LinearSum(self.terms + other.terms)
        raise UnsupportedNativeForm

    __radd__ = __add__

    def __neg__(self):
        return LinearSum(tuple(-term for term in self.terms))

    def __sub__(self, other):
        return self + (-other)


@dataclass(frozen=True)
class BilinearTerm:
    kind: str
    coefficient: Any = None
    factor: float = 1.0

    def __mul__(self, other):
        if np.isscalar(other):
            return BilinearTerm(
                self.kind, self.coefficient, self.factor * other
            )
        if is_symbolic_coefficient(other):
            return BilinearTerm(self.kind, other, self.factor)
        if isinstance(other, np.ndarray) or (
            hasattr(other, "value") and hasattr(other, "grad")
        ):
            if self.coefficient is not None:
                raise UnsupportedNativeForm(
                    "multiple bilinear coefficients are not supported"
                )
            return BilinearTerm(self.kind, np.asarray(other), self.factor)
        return NotImplemented

    __rmul__ = __mul__

    def __neg__(self):
        return BilinearTerm(self.kind, self.coefficient, -self.factor)

    def __add__(self, other):
        if isinstance(other, BilinearTerm):
            return BilinearSum((self, other))
        if isinstance(other, BilinearSum):
            return BilinearSum((self,) + other.terms)
        return NotImplemented

    def __sub__(self, other):
        return self + (-other)


@dataclass(frozen=True)
class BilinearSum:
    terms: tuple[BilinearTerm, ...]

    def __add__(self, other):
        if isinstance(other, BilinearTerm):
            return BilinearSum(self.terms + (other,))
        if isinstance(other, BilinearSum):
            return BilinearSum(self.terms + other.terms)
        return NotImplemented

    __radd__ = __add__

    def __neg__(self):
        return BilinearSum(tuple(-term for term in self.terms))

    def __sub__(self, other):
        return self + (-other)


@dataclass(frozen=True)
class CompositeBilinearTerm:
    row_field: int
    column_field: int
    kind: str
    coefficient: Any = None
    factor: float = 1.0

    def __mul__(self, other):
        if np.isscalar(other):
            return CompositeBilinearTerm(
                self.row_field,
                self.column_field,
                self.kind,
                self.coefficient,
                self.factor * other,
            )
        if is_symbolic_coefficient(other):
            return CompositeBilinearTerm(
                self.row_field,
                self.column_field,
                self.kind,
                other,
                self.factor,
            )
        if isinstance(other, np.ndarray) or (
            hasattr(other, "value") and hasattr(other, "grad")
        ):
            if self.coefficient is not None:
                raise UnsupportedNativeForm(
                    "multiple composite coefficients are not supported"
                )
            return CompositeBilinearTerm(
                self.row_field,
                self.column_field,
                self.kind,
                np.asarray(other),
                self.factor,
            )
        return NotImplemented

    __rmul__ = __mul__

    def __neg__(self):
        return CompositeBilinearTerm(
            self.row_field,
            self.column_field,
            self.kind,
            self.coefficient,
            -self.factor,
        )

    def __add__(self, other):
        if isinstance(other, CompositeBilinearTerm):
            return CompositeBilinearSum((self, other))
        if isinstance(other, CompositeBilinearSum):
            return CompositeBilinearSum((self,) + other.terms)
        return NotImplemented

    def __sub__(self, other):
        return self + (-other)


@dataclass(frozen=True)
class CompositeBilinearSum:
    terms: tuple[CompositeBilinearTerm, ...]

    def __add__(self, other):
        if isinstance(other, CompositeBilinearTerm):
            return CompositeBilinearSum(self.terms + (other,))
        if isinstance(other, CompositeBilinearSum):
            return CompositeBilinearSum(self.terms + other.terms)
        return NotImplemented

    __radd__ = __add__

    def __neg__(self):
        return CompositeBilinearSum(tuple(-term for term in self.terms))

    def __sub__(self, other):
        return self + (-other)


__all__ = [
    "BilinearSum",
    "BilinearTerm",
    "CompositeBilinearSum",
    "CompositeBilinearTerm",
    "LinearSum",
    "LinearTerm",
]
