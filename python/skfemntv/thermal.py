"""Physical material data and element-local scalar thermal kernels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IsotropicThermalMaterial:
    conductivity: float
    density: float
    specific_heat: float

    def __post_init__(self) -> None:
        for name in ("conductivity", "density", "specific_heat"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class ScalarThermalElementContributions:
    conductivity: np.ndarray
    capacity: np.ndarray
    heat_source: np.ndarray


class ScalarThermalElementKernel:
    """Integrate scalar temperature element matrices and source vectors."""

    def __init__(self, material: IsotropicThermalMaterial):
        if not isinstance(material, IsotropicThermalMaterial):
            raise TypeError("material must be an IsotropicThermalMaterial")
        self.material = material

    def integrate(
        self,
        shape_values: np.ndarray,
        physical_shape_gradients: np.ndarray,
        quadrature_weights: np.ndarray,
        *,
        volumetric_heat_source: float = 0.0,
    ) -> ScalarThermalElementContributions:
        shape = np.asarray(shape_values, dtype=np.float64)
        gradients = np.asarray(physical_shape_gradients, dtype=np.float64)
        weights = np.asarray(quadrature_weights, dtype=np.float64)
        source = float(volumetric_heat_source)
        if shape.ndim != 3:
            raise ValueError("shape_values must have shape (elements, quadrature, nodes)")
        elements, quadrature, nodes = shape.shape
        if gradients.ndim != 4 or gradients.shape[:3] != (elements, quadrature, nodes):
            raise ValueError(
                "physical_shape_gradients must have shape (elements, quadrature, nodes, dimension)"
            )
        if weights.shape != (elements, quadrature):
            raise ValueError("quadrature_weights must have shape (elements, quadrature)")
        if not np.isfinite(source) or np.any(weights <= 0.0):
            raise ValueError("source must be finite and quadrature weights positive")
        if not all(
            np.all(np.isfinite(values)) for values in (shape, gradients, weights)
        ):
            raise ValueError("thermal integration data must be finite")
        conductivity = self.material.conductivity * np.einsum(
            "eqid,eqjd,eq->eij", gradients, gradients, weights, optimize=True
        )
        volumetric_capacity = self.material.density * self.material.specific_heat
        capacity = volumetric_capacity * np.einsum(
            "eqi,eqj,eq->eij", shape, shape, weights, optimize=True
        )
        heat_source_values = source * np.einsum(
            "eqi,eq->ei", shape, weights, optimize=True
        )
        return ScalarThermalElementContributions(
            conductivity=np.ascontiguousarray(conductivity),
            capacity=np.ascontiguousarray(capacity),
            heat_source=np.ascontiguousarray(heat_source_values),
        )


__all__ = [
    "IsotropicThermalMaterial",
    "ScalarThermalElementContributions",
    "ScalarThermalElementKernel",
]
