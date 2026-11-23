"""Linear and total-Lagrangian Tet10 local elasticity kernels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .basis import Basis, ElementTetP2, ElementVector, MeshTet2
from .linear_elasticity import IsotropicLinearElasticity
from .nonlinear import StVenantKirchhoffMaterial


@dataclass(frozen=True)
class Tet10IntegrationData:
    """Geometry, quadrature, and linear stiffness data for a Tet10 mesh."""

    reference_points: NDArray[np.float64]
    quadrature_weights: NDArray[np.float64]
    shape_values: NDArray[np.float64]
    physical_shape_gradients: NDArray[np.float64]
    jacobian_determinants: NDArray[np.float64]
    integration_weights: NDArray[np.float64]
    strain_displacement_matrices: NDArray[np.float64]
    element_stiffness: NDArray[np.float64]


@dataclass(frozen=True)
class Tet10LinearElasticityResponse:
    strain: NDArray[np.float64]
    stress: NDArray[np.float64]
    residual: NDArray[np.float64]


@dataclass(frozen=True)
class Tet10StVenantKirchhoffResponse:
    deformation_gradient: NDArray[np.float64]
    green_lagrange_strain: NDArray[np.float64]
    second_piola_kirchhoff_stress: NDArray[np.float64]
    first_piola_kirchhoff_stress: NDArray[np.float64]
    cauchy_stress: NDArray[np.float64]
    deformation_jacobian: NDArray[np.float64]
    residual: NDArray[np.float64]
    tangent: NDArray[np.float64]


class Tet10LinearElasticityKernel:
    """Build per-element Tet10 linear-elasticity kernels from a quadratic mesh."""

    def __init__(
        self,
        points: NDArray[np.float64],
        elements: NDArray[np.integer],
        material: IsotropicLinearElasticity,
        *,
        integration_order: int = 4,
    ) -> None:
        coordinates = np.asarray(points, dtype=np.float64)
        connectivity = np.asarray(elements, dtype=np.int64)
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise ValueError("Tet10 points must have shape (nodes, 3)")
        if connectivity.ndim != 2 or connectivity.shape[1] != 10:
            raise ValueError("Tet10 elements must have shape (elements, 10)")
        if connectivity.shape[0] == 0:
            raise ValueError("Tet10 elements must contain at least one element")
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("Tet10 points must be finite")
        if np.any(connectivity < 0) or np.any(connectivity >= coordinates.shape[0]):
            raise ValueError("Tet10 elements contain an out-of-range node index")
        if isinstance(integration_order, bool) or not isinstance(
            integration_order, (int, np.integer)
        ) or integration_order < 2:
            raise ValueError("Tet10 integration_order must be an integer >= 2")
        if not isinstance(material, IsotropicLinearElasticity):
            raise TypeError("material must be an IsotropicLinearElasticity")
        self.points = coordinates
        self.elements = connectivity
        self.material = material
        self.integration_order = int(integration_order)
        self.mesh = MeshTet2(coordinates.T, connectivity.T)
        self.basis = Basis(
            self.mesh,
            ElementVector(ElementTetP2(), dim=3),
            intorder=self.integration_order,
        )
        self.integration = self._build_integration_data()

    @property
    def element_count(self) -> int:
        return int(self.elements.shape[0])

    @property
    def integration_point_count(self) -> int:
        return int(self.integration.quadrature_weights.size)

    def evaluate_element(
        self,
        element_index: int,
        displacement: NDArray[np.float64],
    ) -> Tet10LinearElasticityResponse:
        """Evaluate the local linear strain, stress, and internal force."""
        local_values = _local_displacement(displacement, node_count=10)
        strain = np.einsum(
            "qij,j->qi",
            self.integration.strain_displacement_matrices[element_index],
            local_values,
        )
        strain_tensor = _engineering_voigt_to_tensor(strain)
        stress_tensor = self.material.stress(strain_tensor)
        stress = _tensor_to_voigt_stress(stress_tensor)
        residual = self.integration.element_stiffness[element_index] @ local_values
        return Tet10LinearElasticityResponse(strain, stress, residual)

    def evaluate_at_reference_points(
        self,
        displacement: NDArray[np.float64],
        reference_points: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        points = _reference_points(reference_points)
        custom_basis = Basis(
            self.mesh,
            ElementVector(ElementTetP2(), dim=3),
            quadrature=(points.T, np.ones(points.shape[0], dtype=np.float64)),
        )
        values = np.asarray(displacement, dtype=np.float64).reshape(-1)
        expected_dofs = 3 * self.points.shape[0]
        if values.shape != (expected_dofs,) or not np.all(np.isfinite(values)):
            raise ValueError(
                f"displacement must be finite with shape ({expected_dofs},)"
            )
        local_values = values.reshape((-1, 3))[self.elements].reshape(
            self.element_count, 30
        )
        b_matrices = _strain_displacement_matrices(
            custom_basis.tabulated_gradients
        )
        strain = np.einsum("eqij,ej->eqi", b_matrices, local_values)
        return strain, _tensor_to_voigt_stress(
            self.material.stress(_engineering_voigt_to_tensor(strain))
        )

    def _build_integration_data(self) -> Tet10IntegrationData:
        basis = self.basis
        gradients = np.asarray(basis.tabulated_gradients, dtype=np.float64)
        shapes = np.asarray(basis.tabulated_shape, dtype=np.float64)
        determinants = np.asarray(basis._geometry_determinants, dtype=np.float64)
        if np.any(determinants <= basis.geometry_diagnostics.determinant_tolerance):
            diagnostics = basis.geometry_diagnostics
            raise ValueError(
                "Tet10 has a non-positive or singular Jacobian: "
                f"element={diagnostics.worst_element} "
                f"quadrature_point={diagnostics.worst_quadrature_point} "
                f"determinant={diagnostics.minimum_determinant:.17g}"
            )
        integration_weights = np.asarray(basis.dx, dtype=np.float64)
        b_matrices = _strain_displacement_matrices(gradients)
        constitutive = self.material.voigt_stiffness
        stiffness = np.einsum(
            "eqia,ij,eqjb,eq->eab",
            b_matrices,
            constitutive,
            b_matrices,
            integration_weights,
            optimize=True,
        )
        return Tet10IntegrationData(
            reference_points=np.asarray(basis.X.T, dtype=np.float64).copy(),
            quadrature_weights=np.asarray(basis.W, dtype=np.float64).copy(),
            shape_values=shapes,
            physical_shape_gradients=gradients,
            jacobian_determinants=determinants,
            integration_weights=integration_weights,
            strain_displacement_matrices=b_matrices,
            element_stiffness=stiffness,
        )


class Tet10StVenantKirchhoffKernel:
    """Evaluate local total-Lagrangian Tet10 force and consistent tangent."""

    def __init__(self, material: StVenantKirchhoffMaterial) -> None:
        if not isinstance(material, StVenantKirchhoffMaterial):
            raise TypeError("material must be a StVenantKirchhoffMaterial")
        self.material = material

    def evaluate(
        self,
        integration: Tet10IntegrationData,
        element_index: int,
        displacement: NDArray[np.float64],
    ) -> Tet10StVenantKirchhoffResponse:
        local_values = _local_displacement(displacement, node_count=10)
        nodal_values = local_values.reshape((10, 3))
        gradients = integration.physical_shape_gradients[element_index]
        weights = integration.integration_weights[element_index]
        deformation = np.eye(3) + np.einsum(
            "ni,qnj->qij", nodal_values, gradients
        )
        material_response = self.material.evaluate(deformation)
        residual = np.einsum(
            "qnj,qij,q->ni",
            gradients,
            material_response.first_piola_kirchhoff_stress,
            weights,
            optimize=True,
        ).reshape(30)
        tangent = np.einsum(
            "qaj,qijkL,qbL,q->aibk",
            gradients,
            material_response.tangent,
            gradients,
            weights,
            optimize=True,
        ).reshape((30, 30))
        return Tet10StVenantKirchhoffResponse(
            deformation_gradient=material_response.deformation_gradient,
            green_lagrange_strain=material_response.green_lagrange_strain,
            second_piola_kirchhoff_stress=(
                material_response.second_piola_kirchhoff_stress
            ),
            first_piola_kirchhoff_stress=(
                material_response.first_piola_kirchhoff_stress
            ),
            cauchy_stress=material_response.cauchy_stress,
            deformation_jacobian=np.linalg.det(deformation),
            residual=residual,
            tangent=tangent,
        )


def _strain_displacement_matrices(
    gradients: NDArray[np.float64],
) -> NDArray[np.float64]:
    squeeze_element = gradients.ndim == 3
    if squeeze_element:
        gradients = gradients[None, ...]
    if gradients.ndim != 4 or gradients.shape[-2:] != (10, 3):
        raise ValueError("Tet10 gradients must have shape (E, Q, 10, 3) or (Q, 10, 3)")
    matrices = np.zeros(gradients.shape[:2] + (6, 30), dtype=np.float64)
    for node in range(10):
        dx, dy, dz = np.moveaxis(gradients[:, :, node, :], -1, 0)
        column = 3 * node
        matrices[:, :, 0, column] = dx
        matrices[:, :, 1, column + 1] = dy
        matrices[:, :, 2, column + 2] = dz
        matrices[:, :, 3, column] = dy
        matrices[:, :, 3, column + 1] = dx
        matrices[:, :, 4, column + 1] = dz
        matrices[:, :, 4, column + 2] = dy
        matrices[:, :, 5, column] = dz
        matrices[:, :, 5, column + 2] = dx
    return matrices[0] if squeeze_element else matrices


def _local_displacement(
    displacement: NDArray[np.float64],
    *,
    node_count: int,
) -> NDArray[np.float64]:
    values = np.asarray(displacement, dtype=np.float64)
    if values.shape == (node_count, 3):
        values = values.reshape(3 * node_count)
    if values.shape != (3 * node_count,) or not np.all(np.isfinite(values)):
        raise ValueError(
            f"local displacement must be finite with shape ({3 * node_count},) "
            f"or ({node_count}, 3)"
        )
    return values


def _reference_points(points: NDArray[np.float64]) -> NDArray[np.float64]:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not np.all(np.isfinite(values)):
        raise ValueError("reference_points must be finite with shape (points, 3)")
    return values


def _engineering_voigt_to_tensor(
    values: NDArray[np.float64],
) -> NDArray[np.float64]:
    result = np.empty(values.shape[:-1] + (3, 3), dtype=np.float64)
    result[..., 0, 0] = values[..., 0]
    result[..., 1, 1] = values[..., 1]
    result[..., 2, 2] = values[..., 2]
    result[..., 0, 1] = result[..., 1, 0] = 0.5 * values[..., 3]
    result[..., 1, 2] = result[..., 2, 1] = 0.5 * values[..., 4]
    result[..., 0, 2] = result[..., 2, 0] = 0.5 * values[..., 5]
    return result


def _tensor_to_voigt_stress(values: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.stack(
        (
            values[..., 0, 0],
            values[..., 1, 1],
            values[..., 2, 2],
            values[..., 0, 1],
            values[..., 1, 2],
            values[..., 0, 2],
        ),
        axis=-1,
    )


__all__ = [
    "Tet10IntegrationData",
    "Tet10LinearElasticityKernel",
    "Tet10LinearElasticityResponse",
    "Tet10StVenantKirchhoffKernel",
    "Tet10StVenantKirchhoffResponse",
]
