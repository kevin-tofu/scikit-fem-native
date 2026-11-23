"""Local Hex8 kernels for small-strain isotropic elasticity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .basis import ElementHex1
from .linear_elasticity import IsotropicLinearElasticity
from .nonlinear import StVenantKirchhoffMaterial


@dataclass(frozen=True)
class Hex8LinearElasticityIntegration:
    """Reference-to-physical data and element stiffness for one Hex8 cell."""

    gauss_point_coordinates: NDArray[np.float64]
    gauss_weights: NDArray[np.float64]
    strain_displacement_matrices: NDArray[np.float64]
    physical_shape_gradients: NDArray[np.float64]
    jacobian_determinants: NDArray[np.float64]
    stiffness: NDArray[np.float64]


@dataclass(frozen=True)
class Hex8LinearElasticityResponse:
    """Integration-point fields and local force for one Hex8 cell."""

    strain: NDArray[np.float64]
    stress: NDArray[np.float64]
    residual: NDArray[np.float64]


@dataclass(frozen=True)
class Hex8StVenantKirchhoffResponse:
    """Local total-Lagrangian response for one Hex8 cell."""

    deformation_gradient: NDArray[np.float64]
    green_lagrange_strain: NDArray[np.float64]
    second_piola_kirchhoff_stress: NDArray[np.float64]
    first_piola_kirchhoff_stress: NDArray[np.float64]
    cauchy_stress: NDArray[np.float64]
    deformation_jacobian: NDArray[np.float64]
    residual: NDArray[np.float64]
    tangent: NDArray[np.float64]


class Hex8LinearElasticityKernel:
    """Evaluate local Hex8 elasticity terms in scikit-fem local node order.

    This kernel computes geometry, quadrature, strain, stress, and element
    matrices. It does not assign global DOFs or scatter into a global matrix.
    The supplied coordinates must correspond row-for-row to
    ``ElementHex1().doflocs`` and displacement must use the same local order.
    """

    def __init__(
        self,
        material: IsotropicLinearElasticity,
        *,
        integration_order: int = 2,
    ) -> None:
        if not isinstance(material, IsotropicLinearElasticity):
            raise TypeError("material must be an IsotropicLinearElasticity")
        if (
            isinstance(integration_order, bool)
            or not isinstance(integration_order, (int, np.integer))
            or integration_order < 2
        ):
            raise ValueError(
                "integration_order must be an integer of at least 2; "
                "one-point Hex8 integration requires hourglass control"
            )
        self.material = material
        self.integration_order = int(integration_order)
        self._reference_nodes = 2.0 * np.asarray(
            ElementHex1().doflocs, dtype=np.float64
        ) - 1.0
        self._gauss_points, self._gauss_weights = self._gauss_rule(
            self.integration_order
        )

    @property
    def gauss_point_coordinates(self) -> NDArray[np.float64]:
        return self._gauss_points.copy()

    def integrate(
        self,
        coordinates: NDArray[np.float64],
    ) -> Hex8LinearElasticityIntegration:
        """Compute a local stiffness and its reusable quadrature data."""
        points = np.asarray(coordinates, dtype=np.float64)
        if points.shape != (8, 3) or not np.all(np.isfinite(points)):
            raise ValueError("coordinates must be finite with shape (8, 3)")
        point_count = self._gauss_points.shape[0]
        gradients = np.empty((point_count, 8, 3), dtype=np.float64)
        jacobians = np.empty(point_count, dtype=np.float64)
        b_matrices = np.empty((point_count, 6, 24), dtype=np.float64)
        constitutive = self.material.voigt_stiffness
        stiffness = np.zeros((24, 24), dtype=np.float64)
        for point_index, (natural, weight) in enumerate(
            zip(self._gauss_points, self._gauss_weights, strict=True)
        ):
            natural_gradients = self._shape_gradients(natural)
            jacobian = points.T @ natural_gradients
            determinant = float(np.linalg.det(jacobian))
            if not np.isfinite(determinant) or abs(determinant) <= 1.0e-14:
                raise ValueError(
                    f"Hex8 has a singular Jacobian at Gauss point {point_index}: "
                    f"{determinant:.6e}"
                )
            physical_gradients = natural_gradients @ np.linalg.inv(jacobian)
            b_matrix = self._strain_displacement_matrix(physical_gradients)
            gradients[point_index] = physical_gradients
            jacobians[point_index] = abs(determinant)
            b_matrices[point_index] = b_matrix
            stiffness += (
                b_matrix.T
                @ constitutive
                @ b_matrix
                * abs(determinant)
                * weight
            )
        return Hex8LinearElasticityIntegration(
            gauss_point_coordinates=self._gauss_points.copy(),
            gauss_weights=self._gauss_weights.copy(),
            strain_displacement_matrices=b_matrices,
            physical_shape_gradients=gradients,
            jacobian_determinants=jacobians,
            stiffness=stiffness,
        )

    def evaluate(
        self,
        integration: Hex8LinearElasticityIntegration,
        displacement: NDArray[np.float64],
    ) -> Hex8LinearElasticityResponse:
        """Evaluate local strain, stress, and internal force."""
        values = np.asarray(displacement, dtype=np.float64)
        if values.shape == (8, 3):
            values = values.reshape(24)
        if values.shape != (24,) or not np.all(np.isfinite(values)):
            raise ValueError("displacement must be finite with shape (24,) or (8, 3)")
        strain_vectors = np.einsum(
            "qij,j->qi", integration.strain_displacement_matrices, values
        )
        strain = np.asarray(
            [self._strain_tensor(vector) for vector in strain_vectors]
        )
        stress = self.material.stress(strain)
        return Hex8LinearElasticityResponse(
            strain=strain,
            stress=stress,
            residual=integration.stiffness @ values,
        )

    def _shape_gradients(self, point: NDArray[np.float64]) -> NDArray[np.float64]:
        nodes = self._reference_nodes
        gradients = np.empty((8, 3), dtype=np.float64)
        coordinates = tuple(float(value) for value in point)
        for node_index, node in enumerate(nodes):
            for axis in range(3):
                derivative = node[axis]
                for other in range(3):
                    if other == axis:
                        continue
                    derivative *= 1.0 + node[other] * coordinates[other]
                derivative *= 0.125
                gradients[node_index, axis] = derivative
        return gradients

    @staticmethod
    def _strain_displacement_matrix(
        gradients: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        matrix = np.zeros((6, 24), dtype=np.float64)
        for node, (dx, dy, dz) in enumerate(gradients):
            column = 3 * node
            matrix[0, column] = dx
            matrix[1, column + 1] = dy
            matrix[2, column + 2] = dz
            matrix[3, column] = dy
            matrix[3, column + 1] = dx
            matrix[4, column + 1] = dz
            matrix[4, column + 2] = dy
            matrix[5, column] = dz
            matrix[5, column + 2] = dx
        return matrix

    @staticmethod
    def _strain_tensor(vector: NDArray[np.float64]) -> NDArray[np.float64]:
        xx, yy, zz, xy, yz, zx = vector
        return np.asarray(
            [
                [xx, 0.5 * xy, 0.5 * zx],
                [0.5 * xy, yy, 0.5 * yz],
                [0.5 * zx, 0.5 * yz, zz],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _gauss_rule(
        integration_order: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        coordinates, weights = np.polynomial.legendre.leggauss(integration_order)
        source_points = np.asarray(
            [
                (float(x), float(y), float(z))
                for z in coordinates
                for x in coordinates
                for y in coordinates
            ],
            dtype=np.float64,
        )
        # Preserve the existing CalculiX/Gmsh quadrature order while storing
        # points in the native ElementHex1 reference axes.
        points = np.column_stack(
            (
                -source_points[:, 2],
                -source_points[:, 0],
                -source_points[:, 1],
            )
        )
        product_weights = np.asarray(
            [
                float(weights[x] * weights[y] * weights[z])
                for z in range(integration_order)
                for x in range(integration_order)
                for y in range(integration_order)
            ],
            dtype=np.float64,
        )
        return points, product_weights


class Hex8StVenantKirchhoffKernel:
    """Total-Lagrangian Hex8 local residual and consistent tangent kernel."""

    def __init__(self, material: StVenantKirchhoffMaterial) -> None:
        if not isinstance(material, StVenantKirchhoffMaterial):
            raise TypeError("material must be a StVenantKirchhoffMaterial")
        self.material = material

    def evaluate(
        self,
        integration: Hex8LinearElasticityIntegration,
        displacement: NDArray[np.float64],
    ) -> Hex8StVenantKirchhoffResponse:
        """Return element internal force, tangent, and integration-point fields."""
        values = np.asarray(displacement, dtype=np.float64)
        if values.shape == (8, 3):
            values = values.reshape(24)
        if values.shape != (24,) or not np.all(np.isfinite(values)):
            raise ValueError("displacement must be finite with shape (24,) or (8, 3)")
        element_values = values.reshape((8, 3))
        point_count = integration.gauss_weights.size
        deformation_gradients = np.empty((point_count, 3, 3), dtype=np.float64)
        green_strains = np.empty_like(deformation_gradients)
        second_piola = np.empty_like(deformation_gradients)
        first_piola = np.empty_like(deformation_gradients)
        cauchy = np.empty_like(deformation_gradients)
        deformation_jacobians = np.empty(point_count, dtype=np.float64)
        residual = np.zeros(24, dtype=np.float64)
        tangent = np.zeros((24, 24), dtype=np.float64)
        for point_index, (gradients, determinant, weight) in enumerate(
            zip(
                integration.physical_shape_gradients,
                integration.jacobian_determinants,
                integration.gauss_weights,
                strict=True,
            )
        ):
            deformation_gradient = np.eye(3) + element_values.T @ gradients
            try:
                response = self.material.evaluate(deformation_gradient)
            except ValueError as error:
                raise ValueError(
                    "invalid Hex8 deformation gradient at Gauss point "
                    f"{point_index}"
                ) from error
            deformation_gradients[point_index] = deformation_gradient
            green_strains[point_index] = response.green_lagrange_strain
            second_piola[point_index] = response.second_piola_kirchhoff_stress
            first_piola[point_index] = response.first_piola_kirchhoff_stress
            cauchy[point_index] = response.cauchy_stress
            deformation_jacobians[point_index] = np.linalg.det(
                deformation_gradient
            )
            integration_weight = determinant * weight
            residual.reshape((8, 3))[:] += (
                integration_weight * gradients @ response.first_piola_kirchhoff_stress.T
            )
            local_tangent = np.einsum(
                "aJ,iJkL,bL->aibk",
                gradients,
                response.tangent,
                gradients,
                optimize=True,
            ).reshape((24, 24))
            tangent += integration_weight * local_tangent
        return Hex8StVenantKirchhoffResponse(
            deformation_gradient=deformation_gradients,
            green_lagrange_strain=green_strains,
            second_piola_kirchhoff_stress=second_piola,
            first_piola_kirchhoff_stress=first_piola,
            cauchy_stress=cauchy,
            deformation_jacobian=deformation_jacobians,
            residual=residual,
            tangent=tangent,
        )


__all__ = [
    "Hex8LinearElasticityIntegration",
    "Hex8LinearElasticityKernel",
    "Hex8LinearElasticityResponse",
    "Hex8StVenantKirchhoffKernel",
    "Hex8StVenantKirchhoffResponse",
]
