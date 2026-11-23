"""Physical material data and element-local scalar thermal kernels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fields import ScalarFieldEvaluation


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
class AffineTemperatureConductivity:
    """Conductivity ``k(T)=k_ref*(1+slope*(T-T_ref))`` for nonlinear heat flow."""

    reference_conductivity: float
    reference_temperature: float
    relative_slope: float

    def __post_init__(self) -> None:
        conductivity = float(self.reference_conductivity)
        temperature = float(self.reference_temperature)
        slope = float(self.relative_slope)
        if not np.isfinite(conductivity) or conductivity <= 0.0:
            raise ValueError("reference_conductivity must be finite and positive")
        if not np.isfinite(temperature):
            raise ValueError("reference_temperature must be finite")
        if not np.isfinite(slope):
            raise ValueError("relative_slope must be finite")
        object.__setattr__(self, "reference_conductivity", conductivity)
        object.__setattr__(self, "reference_temperature", temperature)
        object.__setattr__(self, "relative_slope", slope)

    @property
    def derivative(self) -> float:
        return self.reference_conductivity * self.relative_slope

    def evaluate(self, temperature: np.ndarray) -> np.ndarray:
        values = np.asarray(temperature, dtype=np.float64)
        conductivity = self.reference_conductivity * (
            1.0 + self.relative_slope * (values - self.reference_temperature)
        )
        if not np.all(np.isfinite(conductivity)) or np.any(conductivity <= 0.0):
            raise ValueError("temperature-dependent conductivity became nonpositive")
        return conductivity


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


@dataclass(frozen=True)
class NonlinearThermalElementContributions:
    residual: np.ndarray
    tangent: np.ndarray


@dataclass(frozen=True)
class KelvinVoigtDissipationContributions:
    """Element heat source and optional derivative with respect to velocity."""

    heat_source: np.ndarray
    velocity_tangent: np.ndarray | None
    total_power: float


@dataclass(frozen=True)
class ThermalFluxEvaluation:
    temperature: np.ndarray
    temperature_gradient: np.ndarray
    conductivity: np.ndarray
    heat_flux: np.ndarray


def evaluate_thermal_flux(
    field: ScalarFieldEvaluation,
    conductivity: float | AffineTemperatureConductivity,
) -> ThermalFluxEvaluation:
    """Evaluate Fourier heat flux ``q = -k(T) grad(T)`` at quadrature points."""
    if not isinstance(field, ScalarFieldEvaluation):
        raise TypeError("field must be a ScalarFieldEvaluation")
    if isinstance(conductivity, AffineTemperatureConductivity):
        conductivity_values = conductivity.evaluate(field.values)
    else:
        coefficient = float(conductivity)
        if not np.isfinite(coefficient) or coefficient <= 0.0:
            raise ValueError("conductivity must be finite and positive")
        conductivity_values = np.full_like(field.values, coefficient)
    heat_flux = -conductivity_values[..., None] * field.gradients
    return ThermalFluxEvaluation(
        temperature=field.values,
        temperature_gradient=field.gradients,
        conductivity=conductivity_values,
        heat_flux=np.ascontiguousarray(heat_flux),
    )


class AffineTemperatureConductivityKernel:
    """Element residual and consistent tangent for affine ``k(T)``."""

    def __init__(self, law: AffineTemperatureConductivity):
        if not isinstance(law, AffineTemperatureConductivity):
            raise TypeError("law must be an AffineTemperatureConductivity")
        self.law = law

    def integrate(
        self,
        shape_values: np.ndarray,
        physical_shape_gradients: np.ndarray,
        quadrature_weights: np.ndarray,
        nodal_temperature: np.ndarray,
    ) -> NonlinearThermalElementContributions:
        shape = np.asarray(shape_values, dtype=np.float64)
        gradients = np.asarray(physical_shape_gradients, dtype=np.float64)
        weights = np.asarray(quadrature_weights, dtype=np.float64)
        temperature = np.asarray(nodal_temperature, dtype=np.float64)
        if shape.ndim != 3:
            raise ValueError("shape_values must have shape (elements, quadrature, nodes)")
        elements, quadrature, nodes = shape.shape
        if gradients.ndim != 4 or gradients.shape[:3] != shape.shape:
            raise ValueError(
                "physical_shape_gradients must have shape (elements, quadrature, nodes, dimension)"
            )
        if weights.shape != (elements, quadrature):
            raise ValueError("quadrature_weights must have shape (elements, quadrature)")
        if temperature.shape != (elements, nodes):
            raise ValueError("nodal_temperature must have shape (elements, nodes)")
        if not all(
            np.all(np.isfinite(value)) for value in (shape, gradients, weights, temperature)
        ) or np.any(weights <= 0.0):
            raise ValueError("thermal element data must be finite with positive weights")

        point_temperature = np.einsum("eqn,en->eq", shape, temperature, optimize=True)
        point_gradient = np.einsum(
            "eqnd,en->eqd", gradients, temperature, optimize=True
        )
        conductivity = self.law.evaluate(point_temperature)
        test_gradient_contraction = np.einsum(
            "eqd,eqid->eqi", point_gradient, gradients, optimize=True
        )
        residual = np.einsum(
            "eq,eqi,eq->ei", conductivity, test_gradient_contraction, weights,
            optimize=True,
        )
        gradient_products = np.einsum(
            "eqid,eqjd->eqij", gradients, gradients, optimize=True
        )
        tangent = np.einsum(
            "eq,eqij,eq->eij", conductivity, gradient_products, weights,
            optimize=True,
        )
        tangent += self.law.derivative * np.einsum(
            "eqj,eqi,eq->eij", shape, test_gradient_contraction, weights,
            optimize=True,
        )
        return NonlinearThermalElementContributions(
            residual=np.ascontiguousarray(residual),
            tangent=np.ascontiguousarray(tangent),
        )

    def integrate_relative_slope_residual_derivative(
        self,
        shape_values: np.ndarray,
        physical_shape_gradients: np.ndarray,
        quadrature_weights: np.ndarray,
        nodal_temperature: np.ndarray,
    ) -> np.ndarray:
        r"""Return element ``dR/d(relative_slope)`` at fixed temperature.

        For ``k(T, p) = k_ref * (1 + p * (T - T_ref))``, this evaluates
        ``integral k_ref * (T - T_ref) * grad(T) . grad(N_i) dV``.
        """
        shape = np.asarray(shape_values, dtype=np.float64)
        gradients = np.asarray(physical_shape_gradients, dtype=np.float64)
        weights = np.asarray(quadrature_weights, dtype=np.float64)
        temperature = np.asarray(nodal_temperature, dtype=np.float64)
        if shape.ndim != 3:
            raise ValueError("shape_values must have shape (elements, quadrature, nodes)")
        elements, quadrature, nodes = shape.shape
        if gradients.ndim != 4 or gradients.shape[:3] != shape.shape:
            raise ValueError(
                "physical_shape_gradients must have shape (elements, quadrature, nodes, dimension)"
            )
        if weights.shape != (elements, quadrature):
            raise ValueError("quadrature_weights must have shape (elements, quadrature)")
        if temperature.shape != (elements, nodes):
            raise ValueError("nodal_temperature must have shape (elements, nodes)")
        if not all(
            np.all(np.isfinite(value))
            for value in (shape, gradients, weights, temperature)
        ) or np.any(weights <= 0.0):
            raise ValueError("thermal element data must be finite with positive weights")

        point_temperature = np.einsum("eqn,en->eq", shape, temperature, optimize=True)
        point_gradient = np.einsum(
            "eqnd,en->eqd", gradients, temperature, optimize=True
        )
        test_gradient_contraction = np.einsum(
            "eqd,eqid->eqi", point_gradient, gradients, optimize=True
        )
        derivative = self.law.reference_conductivity * (
            point_temperature - self.law.reference_temperature
        )
        return np.ascontiguousarray(
            np.einsum(
                "eq,eqi,eq->ei",
                derivative,
                test_gradient_contraction,
                weights,
                optimize=True,
            )
        )


class KelvinVoigtDissipationKernel:
    """Element viscous dissipation heat source for isotropic Kelvin-Voigt flow."""

    def __init__(self, shear_viscosity: float, bulk_viscosity: float):
        shear = float(shear_viscosity)
        bulk = float(bulk_viscosity)
        if not np.isfinite(shear) or shear < 0.0:
            raise ValueError("shear_viscosity must be finite and nonnegative")
        if not np.isfinite(bulk) or bulk < 0.0:
            raise ValueError("bulk_viscosity must be finite and nonnegative")
        self.shear_viscosity = shear
        self.bulk_viscosity = bulk

    def integrate(
        self,
        shape_values: np.ndarray,
        physical_shape_gradients: np.ndarray,
        quadrature_weights: np.ndarray,
        nodal_velocity: np.ndarray,
        *,
        compute_velocity_tangent: bool = False,
    ) -> KelvinVoigtDissipationContributions:
        """Integrate dissipation and optionally ``d(heat source)/d(velocity)``.

        The power density is ``2*eta_s*dev(eps_dot):dev(eps_dot) +
        eta_b*tr(eps_dot)**2``. The optional tangent differentiates the
        element nodal heat-source vector while holding geometry fixed.
        """
        shape = np.asarray(shape_values, dtype=np.float64)
        gradients = np.asarray(physical_shape_gradients, dtype=np.float64)
        weights = np.asarray(quadrature_weights, dtype=np.float64)
        velocity = np.asarray(nodal_velocity, dtype=np.float64)
        if shape.ndim != 3:
            raise ValueError("shape_values must have shape (elements, quadrature, nodes)")
        elements, quadrature, nodes = shape.shape
        if gradients.shape != (elements, quadrature, nodes, 3):
            raise ValueError(
                "physical_shape_gradients must have shape (elements, quadrature, nodes, 3)"
            )
        if weights.shape != (elements, quadrature):
            raise ValueError("quadrature_weights must have shape (elements, quadrature)")
        if velocity.shape != (elements, nodes, 3):
            raise ValueError("nodal_velocity must have shape (elements, nodes, 3)")
        if not all(
            np.all(np.isfinite(value)) for value in (shape, gradients, weights, velocity)
        ) or np.any(weights <= 0.0):
            raise ValueError("element data must be finite with positive quadrature weights")

        velocity_gradient = np.einsum(
            "eai,eqaj->eqij", velocity, gradients, optimize=True
        )
        strain_rate = 0.5 * (
            velocity_gradient + np.swapaxes(velocity_gradient, -1, -2)
        )
        trace = np.trace(strain_rate, axis1=-2, axis2=-1)
        deviator = strain_rate - (
            trace[..., None, None] * np.eye(3) / 3.0
        )
        power_density = (
            2.0 * self.shear_viscosity
            * np.sum(deviator * deviator, axis=(-2, -1))
            + self.bulk_viscosity * trace * trace
        )
        heat_source = np.einsum(
            "eqa,eq,eq->ea", shape, power_density, weights, optimize=True
        )
        tangent = None
        if compute_velocity_tangent:
            identity = np.eye(3)
            strain_direction = 0.5 * (
                np.einsum(
                    "ik,eqbj->eqbkij", identity, gradients, optimize=True
                )
                + np.einsum(
                    "jk,eqbi->eqbkij", identity, gradients, optimize=True
                )
            )
            trace_direction = np.trace(strain_direction, axis1=-2, axis2=-1)
            power_direction = (
                4.0 * self.shear_viscosity
                * np.einsum(
                    "eqij,eqbkij->eqbk", deviator, strain_direction, optimize=True
                )
                + 2.0 * self.bulk_viscosity
                * trace[..., None, None] * trace_direction
            )
            tangent = np.ascontiguousarray(
                np.einsum(
                    "eqa,eqbk,eq->eabk",
                    shape,
                    power_direction,
                    weights,
                    optimize=True,
                )
            )
        integrated_heat = np.ascontiguousarray(heat_source)
        return KelvinVoigtDissipationContributions(
            heat_source=integrated_heat,
            velocity_tangent=tangent,
            total_power=float(np.sum(integrated_heat)),
        )


__all__ = [
    "AffineTemperatureConductivityKernel",
    "AffineTemperatureConductivity",
    "KelvinVoigtDissipationContributions",
    "KelvinVoigtDissipationKernel",
    "NonlinearThermalElementContributions",
    "IsotropicThermalMaterial",
    "ScalarThermalElementContributions",
    "ScalarThermalElementKernel",
]
