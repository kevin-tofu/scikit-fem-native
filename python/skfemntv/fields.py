"""Backend-neutral evaluation of finite-element fields at quadrature points."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScalarFieldEvaluation:
    """Scalar field value and physical gradient, shaped ``(element, qp, ...)``."""

    values: np.ndarray
    gradients: np.ndarray


@dataclass(frozen=True)
class VectorFieldEvaluation:
    """Vector field kinematics at quadrature points.

    ``gradients[..., i, j]`` means derivative of component ``i`` with respect
    to spatial coordinate ``j``. ``small_strain`` is the symmetric gradient.
    """

    values: np.ndarray
    gradients: np.ndarray
    small_strain: np.ndarray

    @property
    def deformation_gradient(self) -> np.ndarray:
        dimension = self.gradients.shape[-1]
        if self.gradients.shape[-2] != dimension:
            raise ValueError("deformation gradient requires equal field/spatial dimensions")
        identity = np.eye(dimension, dtype=float)
        return identity + self.gradients


def _validated_basis_arrays(
    shape_values: np.ndarray,
    physical_shape_gradients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    shape = np.asarray(shape_values, dtype=float)
    gradients = np.asarray(physical_shape_gradients, dtype=float)
    if shape.ndim != 3:
        raise ValueError("shape_values must have shape (elements, quadrature, nodes)")
    if gradients.ndim != 4 or gradients.shape[:3] != shape.shape:
        raise ValueError(
            "physical_shape_gradients must have shape "
            "(elements, quadrature, nodes, dimension)"
        )
    if not np.all(np.isfinite(shape)) or not np.all(np.isfinite(gradients)):
        raise ValueError("basis values and gradients must be finite")
    return shape, gradients


def evaluate_scalar_field(
    shape_values: np.ndarray,
    physical_shape_gradients: np.ndarray,
    element_values: np.ndarray,
) -> ScalarFieldEvaluation:
    """Interpolate scalar nodal coefficients and their spatial gradients."""
    shape, gradients = _validated_basis_arrays(
        shape_values, physical_shape_gradients
    )
    coefficients = np.asarray(element_values, dtype=float)
    expected = (shape.shape[0], shape.shape[2])
    if coefficients.shape != expected or not np.all(np.isfinite(coefficients)):
        raise ValueError(f"element_values must be finite with shape {expected}")
    values = np.einsum("eqn,en->eq", shape, coefficients, optimize=True)
    field_gradients = np.einsum(
        "eqnd,en->eqd", gradients, coefficients, optimize=True
    )
    return ScalarFieldEvaluation(
        values=np.ascontiguousarray(values),
        gradients=np.ascontiguousarray(field_gradients),
    )


def evaluate_vector_field(
    shape_values: np.ndarray,
    physical_shape_gradients: np.ndarray,
    element_values: np.ndarray,
) -> VectorFieldEvaluation:
    """Interpolate vector nodal coefficients, gradient, and small strain.

    ``element_values`` uses the canonical shape ``(element, node, component)``.
    """
    shape, basis_gradients = _validated_basis_arrays(
        shape_values, physical_shape_gradients
    )
    coefficients = np.asarray(element_values, dtype=float)
    expected_prefix = (shape.shape[0], shape.shape[2])
    if (
        coefficients.ndim != 3
        or coefficients.shape[:2] != expected_prefix
        or not np.all(np.isfinite(coefficients))
    ):
        raise ValueError(
            "element_values must be finite with shape "
            f"(elements, nodes, components), prefix={expected_prefix}"
        )
    values = np.einsum("eqn,enc->eqc", shape, coefficients, optimize=True)
    gradients = np.einsum(
        "eqnd,enc->eqcd", basis_gradients, coefficients, optimize=True
    )
    if gradients.shape[-2] != gradients.shape[-1]:
        raise ValueError("small strain requires equal field/spatial dimensions")
    strain = 0.5 * (gradients + np.swapaxes(gradients, -1, -2))
    return VectorFieldEvaluation(
        values=np.ascontiguousarray(values),
        gradients=np.ascontiguousarray(gradients),
        small_strain=np.ascontiguousarray(strain),
    )


def evaluate_vector_dof_field(
    local_shape_values: np.ndarray,
    local_shape_gradients: np.ndarray,
    local_dof_values: np.ndarray,
) -> VectorFieldEvaluation:
    """Evaluate a vector FE field from per-local-DOF vector basis values.

    This form supports vector elements whose DOFs are not represented by a
    shared scalar nodal connectivity (for example, higher-order H1 elements).
    Shapes are ``(local_dof, component, element, quadrature)`` and
    ``(local_dof, component, dimension, element, quadrature)``.
    """
    values_basis = np.asarray(local_shape_values, dtype=float)
    gradients_basis = np.asarray(local_shape_gradients, dtype=float)
    coefficients = np.asarray(local_dof_values, dtype=float)
    if values_basis.ndim != 4:
        raise ValueError(
            "local_shape_values must have shape "
            "(local_dof, component, elements, quadrature)"
        )
    local_dofs, components, elements, quadrature = values_basis.shape
    expected_gradient_shape = (
        local_dofs, components, gradients_basis.shape[2], elements, quadrature
    ) if gradients_basis.ndim == 5 else ()
    if gradients_basis.shape != expected_gradient_shape:
        raise ValueError("local_shape_gradients have incompatible dimensions")
    if coefficients.shape != (local_dofs, elements):
        raise ValueError("local_dof_values must have shape (local_dof, elements)")
    if not all(
        np.all(np.isfinite(value))
        for value in (values_basis, gradients_basis, coefficients)
    ):
        raise ValueError("local vector basis data and coefficients must be finite")
    values = np.einsum("le,lceq->eqc", coefficients, values_basis, optimize=True)
    gradients = np.einsum(
        "le,lcdeq->eqcd", coefficients, gradients_basis, optimize=True
    )
    if gradients.shape[-2] != gradients.shape[-1]:
        raise ValueError("small strain requires equal field/spatial dimensions")
    strain = 0.5 * (gradients + np.swapaxes(gradients, -1, -2))
    return VectorFieldEvaluation(
        np.ascontiguousarray(values),
        np.ascontiguousarray(gradients),
        np.ascontiguousarray(strain),
    )


__all__ = [
    "ScalarFieldEvaluation",
    "VectorFieldEvaluation",
    "evaluate_scalar_field",
    "evaluate_vector_field",
    "evaluate_vector_dof_field",
]
