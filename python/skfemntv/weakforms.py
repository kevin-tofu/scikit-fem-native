"""Backend-neutral weak-form kernels for common physical models.

These callables operate on the trial/test fields supplied by an assembly
backend.  They contain the physical weak form, while quadrature and global
matrix assembly remain the backend's responsibility.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .helpers import ddot, div, dot, grad, sym_grad
from .linear_elasticity import IsotropicLinearElasticity
from .thermal import IsotropicThermalMaterial


def isotropic_linear_elasticity_weak_form(
    material: IsotropicLinearElasticity,
) -> Callable:
    """Return the small-strain isotropic elasticity bilinear form.

    The returned ``(trial, test, parameters)`` callable is compatible with
    both scikit-fem's ``BilinearForm`` and skfemntv's native ``BilinearForm``.
    It evaluates

    ``2 mu eps(trial):eps(test) + lambda div(trial) div(test)``.
    """
    if not isinstance(material, IsotropicLinearElasticity):
        raise TypeError("material must be an IsotropicLinearElasticity")
    lame_lambda = material.lame_lambda
    lame_mu = material.lame_mu

    def weak_form(trial, test, _parameters):
        return (
            2.0 * lame_mu * ddot(sym_grad(trial), sym_grad(test))
            + lame_lambda * div(trial) * div(test)
        )

    return weak_form


def scalar_thermal_conductivity_weak_form(
    material: IsotropicThermalMaterial,
) -> Callable:
    """Return ``integral k grad(T) . grad(test_T)`` for heat conduction."""
    if not isinstance(material, IsotropicThermalMaterial):
        raise TypeError("material must be an IsotropicThermalMaterial")

    def weak_form(trial, test, _parameters):
        return material.conductivity * dot(grad(trial), grad(test))

    return weak_form


def scalar_thermal_capacity_weak_form(
    material: IsotropicThermalMaterial,
) -> Callable:
    """Return ``integral rho c T test_T`` for transient heat capacity."""
    if not isinstance(material, IsotropicThermalMaterial):
        raise TypeError("material must be an IsotropicThermalMaterial")
    volumetric_capacity = material.density * material.specific_heat

    def weak_form(trial, test, _parameters):
        return volumetric_capacity * trial * test

    return weak_form


def isotropic_mass_weak_form(density: float) -> Callable:
    """Return the consistent mass form ``integral density * u . v``."""
    value = float(density)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("density must be finite and positive")

    def weak_form(trial, test, _parameters):
        return value * dot(trial, test)

    return weak_form


def body_force_weak_form(force_density: np.ndarray) -> Callable:
    """Return the body-force load form ``integral force_density . test``."""
    force = np.asarray(force_density, dtype=np.float64)
    if force.ndim != 1 or not np.all(np.isfinite(force)):
        raise ValueError("force_density must be a finite vector")
    force = force.copy()
    force.setflags(write=False)

    def weak_form(test, _parameters):
        return dot(test, force)

    return weak_form


def surface_traction_weak_form(traction: np.ndarray) -> Callable:
    """Return the surface traction load form ``integral traction . test``."""
    value = np.asarray(traction, dtype=np.float64)
    if value.ndim != 1 or not np.all(np.isfinite(value)):
        raise ValueError("traction must be a finite vector")
    value = value.copy()
    value.setflags(write=False)

    def weak_form(test, _parameters):
        return dot(test, value)

    return weak_form


def normal_pressure_weak_form(pressure: float) -> Callable:
    """Return the normal-pressure load ``integral pressure * n . test``."""
    value = float(pressure)
    if not np.isfinite(value):
        raise ValueError("pressure must be finite")

    def weak_form(test, parameters):
        return value * dot(test, parameters.n)

    return weak_form


def volumetric_heat_source_weak_form(source: float | np.ndarray) -> Callable:
    """Return the heat-source load ``integral source * test_T``.

    ``source`` may be a scalar or a backend-shaped quadrature field.
    """
    value = np.asarray(source, dtype=np.float64)
    if not np.all(np.isfinite(value)):
        raise ValueError("volumetric heat source must be finite")
    if value.ndim == 0:
        value = float(value)
    else:
        value = value.copy()
        value.setflags(write=False)

    def weak_form(test, _parameters):
        return value * test

    return weak_form


def surface_heat_flux_weak_form(flux_into_domain: float) -> Callable:
    """Return the boundary load ``integral flux_into_domain * test_T``.

    Positive flux is defined as heat entering the modeled body.
    """
    value = float(flux_into_domain)
    if not np.isfinite(value):
        raise ValueError("surface heat flux must be finite")

    def weak_form(test, _parameters):
        return value * test

    return weak_form


def surface_convection_weak_form(heat_transfer_coefficient: float) -> Callable:
    """Return the Robin boundary matrix ``integral h * T * test_T``."""
    value = float(heat_transfer_coefficient)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("heat_transfer_coefficient must be finite and nonnegative")

    def weak_form(trial, test, _parameters):
        return value * trial * test

    return weak_form


def thermal_expansion_coupling_weak_form(coefficient: float) -> Callable:
    """Return ``integral coefficient * T * div(test_u)`` coupling."""
    value = float(coefficient)
    if not np.isfinite(value):
        raise ValueError("thermal expansion coupling coefficient must be finite")

    def weak_form(trial_temperature, test_displacement, _parameters):
        return value * trial_temperature * div(test_displacement)

    return weak_form


def isotropic_viscous_weak_form(
    shear_viscosity: float,
    bulk_viscosity: float,
) -> Callable:
    """Return an isotropic Kelvin--Voigt rate bilinear form."""
    shear = float(shear_viscosity)
    bulk = float(bulk_viscosity)
    if not np.isfinite(shear) or shear < 0.0:
        raise ValueError("shear_viscosity must be finite and nonnegative")
    if not np.isfinite(bulk) or bulk < 0.0:
        raise ValueError("bulk_viscosity must be finite and nonnegative")

    def weak_form(trial_velocity, test_displacement, _parameters):
        return (
            2.0 * shear
            * ddot(sym_grad(trial_velocity), sym_grad(test_displacement))
            + (bulk - (2.0 / 3.0) * shear)
            * div(trial_velocity)
            * div(test_displacement)
        )

    return weak_form


__all__ = [
    "body_force_weak_form",
    "isotropic_mass_weak_form",
    "isotropic_linear_elasticity_weak_form",
    "isotropic_viscous_weak_form",
    "normal_pressure_weak_form",
    "scalar_thermal_capacity_weak_form",
    "scalar_thermal_conductivity_weak_form",
    "surface_traction_weak_form",
    "surface_convection_weak_form",
    "surface_heat_flux_weak_form",
    "thermal_expansion_coupling_weak_form",
    "volumetric_heat_source_weak_form",
]
