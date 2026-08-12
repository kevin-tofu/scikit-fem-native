"""Coefficient descriptors and resolution for typed native forms.

This module deliberately knows nothing about trial/test expression nodes or
assemblers.  Keeping coefficient lookup here gives every form path identical
missing-field and component-index diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class Coefficient:
    """A named form parameter which has not been supplied yet."""

    name: str

    def __getitem__(self, index: int) -> "CoefficientComponent":
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise ValueError("coefficient component index must be an integer")
        return CoefficientComponent(self.name, int(index))


@dataclass(frozen=True)
class CoefficientComponent:
    """First-axis component of a named form parameter."""

    name: str
    index: int


SymbolicCoefficient = Coefficient | CoefficientComponent


def is_symbolic_coefficient(value: Any) -> bool:
    return isinstance(value, (Coefficient, CoefficientComponent))


def resolve_coefficient(value: Any, parameters: Mapping[str, Any]) -> Any:
    """Resolve symbolic and legacy named coefficients in one place."""
    if isinstance(value, Coefficient):
        name = value.name
        index = None
    elif isinstance(value, CoefficientComponent):
        name = value.name
        index = value.index
    elif isinstance(value, str):
        name = value
        index = None
    else:
        return value

    if name not in parameters:
        raise ValueError(f"missing form parameter {name!r}")
    resolved = parameters[name]
    if index is None:
        return resolved
    array = np.asarray(resolved)
    try:
        return array[index]
    except IndexError as error:
        raise ValueError(
            f"form parameter {name!r} has no component {index}"
        ) from error


def evaluate_coefficient(
    value: Any,
    parameters: Mapping[str, Any],
    *,
    factor: float = 1.0,
    squeeze: bool = False,
) -> np.ndarray | float:
    """Resolve a coefficient and convert it to the assembly numeric form."""
    if value is None:
        return factor
    evaluated = factor * np.asarray(
        resolve_coefficient(value, parameters), dtype=np.float64
    )
    if squeeze and evaluated.ndim > 2:
        evaluated = np.squeeze(evaluated)
    return evaluated


__all__ = [
    "Coefficient",
    "CoefficientComponent",
    "SymbolicCoefficient",
    "evaluate_coefficient",
    "is_symbolic_coefficient",
    "resolve_coefficient",
]
