"""Element-local total-Lagrangian StVK kernel for linear Tet4 geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Tet4StVenantKirchhoffResponse:
    deformation_gradient: FloatArray
    green_lagrange_strain: FloatArray
    second_piola_kirchhoff_stress: FloatArray
    first_piola_kirchhoff_stress: FloatArray
    cauchy_stress: FloatArray
    deformation_jacobian: float
    internal_force: FloatArray
    tangent: FloatArray


class Tet4StVenantKirchhoffKernel:
    """Compute reference geometry and local residual/tangent per Tet4 cell."""

    def __init__(self, material: object) -> None:
        self.material = material

    def integrate(self, coordinates: FloatArray) -> tuple[FloatArray, float]:
        points = np.asarray(coordinates, dtype=np.float64)
        if points.shape != (4, 3) or not np.all(np.isfinite(points)):
            raise ValueError("Tet4 coordinates must be finite with shape (4, 3)")
        reference = np.column_stack((np.ones(4), points))
        determinant = float(np.linalg.det(reference))
        volume = abs(determinant) / 6.0
        if not np.isfinite(volume) or volume <= np.finfo(float).eps:
            raise ValueError("Tet4 element has zero or invalid reference volume")
        gradients = np.linalg.inv(reference)[1:, :].T
        return gradients, volume

    def evaluate(
        self,
        gradients: FloatArray,
        volume: float,
        displacement: FloatArray,
    ) -> Tet4StVenantKirchhoffResponse:
        grad = np.asarray(gradients, dtype=np.float64)
        values = np.asarray(displacement, dtype=np.float64)
        if grad.shape != (4, 3):
            raise ValueError("Tet4 shape gradients must have shape (4, 3)")
        if values.shape == (12,):
            values = values.reshape((4, 3))
        if values.shape != (4, 3) or not np.all(np.isfinite(values)):
            raise ValueError("Tet4 displacement must be finite with shape (4, 3)")
        deformation_gradient = np.eye(3) + values.T @ grad
        material = self.material.evaluate(deformation_gradient)
        first_piola = material.first_piola_kirchhoff_stress
        internal_force = float(volume) * grad @ first_piola.T
        tangent = np.empty((12, 12), dtype=np.float64)
        for local_node in range(4):
            for direction in range(3):
                column = 3 * local_node + direction
                delta_first_piola = np.einsum(
                    "iJL,L->iJ",
                    material.tangent[:, :, direction, :],
                    grad[local_node],
                )
                tangent[:, column] = (
                    float(volume) * grad @ delta_first_piola.T
                ).reshape(-1)
        return Tet4StVenantKirchhoffResponse(
            deformation_gradient=deformation_gradient,
            green_lagrange_strain=material.green_lagrange_strain,
            second_piola_kirchhoff_stress=material.second_piola_kirchhoff_stress,
            first_piola_kirchhoff_stress=first_piola,
            cauchy_stress=material.cauchy_stress,
            deformation_jacobian=float(np.linalg.det(deformation_gradient)),
            internal_force=internal_force,
            tangent=tangent,
        )


__all__ = ["Tet4StVenantKirchhoffKernel", "Tet4StVenantKirchhoffResponse"]
