"""Symbolic subfield nodes for composite H1 forms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._coefficients import is_symbolic_coefficient
from ._errors import UnsupportedNativeForm
from ._form_terms import CompositeBilinearTerm


def _contraction(left, right, kind):
    if not isinstance(left, CompositeField) or not isinstance(right, CompositeField):
        raise UnsupportedNativeForm("composite contraction requires two subfields")
    if left.role == right.role:
        raise UnsupportedNativeForm(
            "composite contraction requires trial and test subfields"
        )
    row = left if left.role == "test" else right
    column = right if left.role == "test" else left
    return CompositeBilinearTerm(row.field, column.field, kind)


def _divergence_contraction(value, divergence):
    gradient = divergence.field
    if value.role == gradient.role:
        raise UnsupportedNativeForm(
            "divergence coupling requires trial and test subfields"
        )
    if gradient.role == "test":
        return CompositeBilinearTerm(
            gradient.field, value.field, "row_divergence"
        )
    return CompositeBilinearTerm(
        value.field, gradient.field, "column_divergence"
    )


@dataclass(frozen=True)
class CompositeField:
    role: str
    field: int
    kind: str = "value"

    def __mul__(self, other):
        if isinstance(other, CompositeDivergence):
            return _divergence_contraction(self, other)
        if isinstance(other, CompositeField):
            if self.kind != "value" or other.kind != "value":
                return NotImplemented
            return _contraction(self, other, "value")
        if is_symbolic_coefficient(other):
            return CompositeWeightedField(self, other)
        if np.isscalar(other) or hasattr(other, "__array__") or (
            hasattr(other, "value") and hasattr(other, "grad")
        ):
            return CompositeWeightedField(self, np.asarray(other))
        return NotImplemented

    __rmul__ = __mul__

    def __neg__(self):
        return CompositeWeightedField(self, -1.0)


@dataclass(frozen=True)
class CompositeWeightedField:
    field: CompositeField
    coefficient: Any

    def __mul__(self, other):
        if isinstance(other, CompositeDivergence):
            term = _divergence_contraction(self.field, other)
            return CompositeBilinearTerm(
                term.row_field,
                term.column_field,
                term.kind,
                self.coefficient,
                term.factor,
            )
        if not isinstance(other, CompositeField):
            return NotImplemented
        term = _contraction(self.field, other, "value")
        return CompositeBilinearTerm(
            term.row_field,
            term.column_field,
            term.kind,
            self.coefficient,
            term.factor,
        )

    __rmul__ = __mul__

    def _linear_term(self):
        if self.field.role != "test":
            raise UnsupportedNativeForm(
                "composite LinearForm requires test subfields"
            )
        return CompositeLinearTerm(
            self.field.field, self.field.kind, self.coefficient
        )

    def __add__(self, other):
        return self._linear_term() + other

    __radd__ = __add__

    def __neg__(self):
        if isinstance(self.coefficient, str):
            return CompositeLinearTerm(
                self.field.field, self.field.kind, self.coefficient, -1.0
            )
        return CompositeWeightedField(
            self.field, -np.asarray(self.coefficient)
        )

    def __sub__(self, other):
        return self + (-other)


@dataclass(frozen=True)
class CompositeDivergence:
    field: CompositeField

    def __mul__(self, other):
        if isinstance(other, (CompositeField, CompositeWeightedField)):
            return other * self
        return NotImplemented

    __rmul__ = __mul__


@dataclass(frozen=True)
class CompositeLinearTerm:
    field: int
    kind: str
    coefficient: Any
    factor: float = 1.0

    def __mul__(self, other):
        if np.isscalar(other):
            return CompositeLinearTerm(
                self.field, self.kind, self.coefficient, self.factor * other
            )
        return NotImplemented

    __rmul__ = __mul__

    def __neg__(self):
        return CompositeLinearTerm(
            self.field, self.kind, self.coefficient, -self.factor
        )

    def __add__(self, other):
        if isinstance(other, CompositeWeightedField):
            other = other._linear_term()
        if isinstance(other, CompositeLinearTerm):
            return CompositeLinearSum((self, other))
        if isinstance(other, CompositeLinearSum):
            return CompositeLinearSum((self,) + other.terms)
        return NotImplemented

    __radd__ = __add__

    def __sub__(self, other):
        return self + (-other)


@dataclass(frozen=True)
class CompositeLinearSum:
    terms: tuple[CompositeLinearTerm, ...]

    def __add__(self, other):
        if isinstance(other, CompositeWeightedField):
            other = other._linear_term()
        if isinstance(other, CompositeLinearTerm):
            return CompositeLinearSum(self.terms + (other,))
        if isinstance(other, CompositeLinearSum):
            return CompositeLinearSum(self.terms + other.terms)
        return NotImplemented

    __radd__ = __add__

    def __neg__(self):
        return CompositeLinearSum(tuple(-term for term in self.terms))

    def __sub__(self, other):
        return self + (-other)


composite_contraction = _contraction
composite_divergence_contraction = _divergence_contraction


__all__ = [
    "CompositeDivergence",
    "CompositeField",
    "CompositeLinearSum",
    "CompositeLinearTerm",
    "CompositeWeightedField",
    "composite_contraction",
    "composite_divergence_contraction",
]
