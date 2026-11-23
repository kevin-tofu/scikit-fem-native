"""Finite-deformation thermoelastic element and global assembly."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix

from .basis import Basis, ElementVector
from .nonlinear import ThermoelasticStVenantKirchhoffMaterial


@dataclass(frozen=True)
class ThermoelasticAssemblyEvaluation:
    """Global mechanical and reversible thermal residual/tangent blocks.

    The thermal residual is the reference-temperature-scaled entropy rate.
    Conduction and ordinary heat capacity remain separate thermal operators.
    """

    mechanical_residual: NDArray[np.float64]
    mechanical_displacement_tangent: csr_matrix
    mechanical_temperature_tangent: csr_matrix
    thermal_entropy_residual: NDArray[np.float64]
    thermal_displacement_tangent: csr_matrix
    thermal_temperature_tangent: csr_matrix
    strain_energy: float


class ThermoelasticStVenantKirchhoffAssembler:
    """Assemble finite-strain StVK mechanics and its entropy coupling.

    The mechanical basis must be a three-component vector H1 Tet basis. The
    temperature basis must be a scalar H1 Tet basis on the same mesh, with the
    same quadrature rule. Thermal conduction and heat capacity are assembled
    independently and added to the returned thermal blocks by the caller.
    """

    def __init__(
        self,
        mechanical_basis: Basis,
        temperature_basis: Basis,
        material: ThermoelasticStVenantKirchhoffMaterial,
    ) -> None:
        if (
            not isinstance(mechanical_basis, Basis)
            or not isinstance(mechanical_basis.elem, ElementVector)
            or mechanical_basis.elem._dim != 3
        ):
            raise TypeError("mechanical_basis must be a vector skfemntv Basis")
        if (
            not isinstance(temperature_basis, Basis)
            or not isinstance(temperature_basis.elem, ElementVector)
            or temperature_basis.elem._dim != 1
        ):
            raise TypeError("temperature_basis must be a scalar skfemntv Basis")
        if not isinstance(material, ThermoelasticStVenantKirchhoffMaterial):
            raise TypeError("material must be ThermoelasticStVenantKirchhoffMaterial")
        if not np.array_equal(mechanical_basis.mesh.t, temperature_basis.mesh.t):
            raise ValueError("mechanical and temperature bases must share mesh cells")
        if not np.allclose(mechanical_basis.mesh.p, temperature_basis.mesh.p):
            raise ValueError("mechanical and temperature bases must share mesh points")
        if mechanical_basis.dx.shape != temperature_basis.dx.shape:
            raise ValueError("mechanical and temperature quadrature shapes must match")
        if not np.allclose(mechanical_basis.dx, temperature_basis.dx):
            raise ValueError("mechanical and temperature quadrature weights must match")
        if not np.allclose(
            mechanical_basis.global_coordinates,
            temperature_basis.global_coordinates,
        ):
            raise ValueError("mechanical and temperature quadrature points must match")

        mechanical_local_dofs = mechanical_basis.element_dofs.shape[0]
        if mechanical_local_dofs % 3 != 0:
            raise ValueError("mechanical element DOFs must have three components")
        self.mechanical_basis = mechanical_basis
        self.temperature_basis = temperature_basis
        self.material = material
        self._mechanical_dofs = np.asarray(
            mechanical_basis.element_dofs.T, dtype=np.int64
        )
        self._temperature_dofs = np.asarray(
            temperature_basis.element_dofs.T, dtype=np.int64
        )
        node_count = mechanical_local_dofs // 3
        self._mechanical_node_dofs = self._mechanical_dofs.reshape(
            mechanical_basis.nelems, node_count, 3
        )
        self._mechanical_gradients = np.asarray(
            mechanical_basis.tabulated_gradients, dtype=np.float64
        )
        self._temperature_shapes = np.asarray(
            temperature_basis.tabulated_shape, dtype=np.float64
        )
        self._weights = np.asarray(mechanical_basis.dx, dtype=np.float64)

    @property
    def mechanical_dof_count(self) -> int:
        return self.mechanical_basis.N

    @property
    def temperature_dof_count(self) -> int:
        return self.temperature_basis.N

    def evaluate(
        self,
        displacement: NDArray[np.float64],
        temperature_change: NDArray[np.float64],
        previous_displacement: NDArray[np.float64],
        previous_temperature_change: NDArray[np.float64],
        *,
        time_step: float,
        reference_temperature: float,
    ) -> ThermoelasticAssemblyEvaluation:
        current_displacement = self._field(
            displacement, self.mechanical_dof_count, "displacement"
        )
        current_temperature = self._field(
            temperature_change, self.temperature_dof_count, "temperature_change"
        )
        old_displacement = self._field(
            previous_displacement,
            self.mechanical_dof_count,
            "previous_displacement",
        )
        old_temperature = self._field(
            previous_temperature_change,
            self.temperature_dof_count,
            "previous_temperature_change",
        )
        dt = float(time_step)
        reference = float(reference_temperature)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("time_step must be positive and finite")
        if not np.isfinite(reference) or reference <= 0.0:
            raise ValueError("reference_temperature must be positive and finite")

        element_count = self._mechanical_dofs.shape[0]
        mechanical_local_count = self._mechanical_dofs.shape[1]
        thermal_local_count = self._temperature_dofs.shape[1]
        mechanical_residual = np.zeros(self.mechanical_dof_count)
        thermal_residual = np.zeros(self.temperature_dof_count)
        local_uu = np.zeros(
            (element_count, mechanical_local_count, mechanical_local_count)
        )
        local_u_temperature = np.zeros(
            (element_count, mechanical_local_count, thermal_local_count)
        )
        local_temperature_u = np.zeros(
            (element_count, thermal_local_count, mechanical_local_count)
        )
        local_temperature_temperature = np.zeros(
            (element_count, thermal_local_count, thermal_local_count)
        )
        strain_energy = 0.0
        entropy_rate_factor = reference / dt

        for element in range(element_count):
            mechanical_dofs = self._mechanical_node_dofs[element]
            current_u_element = current_displacement[mechanical_dofs]
            previous_u_element = old_displacement[mechanical_dofs]
            current_t_element = current_temperature[
                self._temperature_dofs[element]
            ]
            previous_t_element = old_temperature[
                self._temperature_dofs[element]
            ]
            gradients = self._mechanical_gradients[element]
            shapes = self._temperature_shapes[element]
            weights = self._weights[element]
            current_gradient = np.einsum(
                "ai,qaj->qij", current_u_element, gradients
            )
            previous_gradient = np.einsum(
                "ai,qaj->qij", previous_u_element, gradients
            )
            identity = np.eye(3)
            current_deformation = current_gradient + identity
            previous_deformation = previous_gradient + identity
            current_temperature_q = shapes @ current_t_element
            previous_temperature_q = shapes @ previous_t_element
            current_response = self.material.evaluate(
                current_deformation, current_temperature_q
            )
            previous_response = self.material.evaluate(
                previous_deformation, previous_temperature_q
            )

            local_residual = np.einsum(
                "qij,qaj,q->ai",
                current_response.first_piola_kirchhoff_stress,
                gradients,
                weights,
            )
            mechanical_residual[self._mechanical_dofs[element]] += (
                local_residual.reshape(-1)
            )
            local_tangent_uu = np.einsum(
                "qaj,qijkl,qbl,q->aibk",
                gradients,
                current_response.tangent_deformation_gradient,
                gradients,
                weights,
            )
            local_uu[element] = local_tangent_uu.reshape(
                mechanical_local_count, mechanical_local_count
            )
            local_tangent_u_temperature = np.einsum(
                "qaj,qij,qb,q->aib",
                gradients,
                current_response.tangent_temperature_change,
                shapes,
                weights,
            )
            local_u_temperature[element] = local_tangent_u_temperature.reshape(
                mechanical_local_count, thermal_local_count
            )

            entropy_difference = (
                current_response.entropy_density - previous_response.entropy_density
            )
            local_entropy_residual = entropy_rate_factor * np.einsum(
                "qb,q,q->b", shapes, entropy_difference, weights
            )
            thermal_residual[self._temperature_dofs[element]] += (
                local_entropy_residual
            )
            local_tangent_temperature_u = entropy_rate_factor * np.einsum(
                "qb,qkL,qaL,q->bak",
                shapes,
                current_response.entropy_deformation_gradient,
                gradients,
                weights,
            )
            local_temperature_u[element] = local_tangent_temperature_u.reshape(
                thermal_local_count, mechanical_local_count
            )
            local_tangent_temperature_temperature = entropy_rate_factor * np.einsum(
                "qb,q,qc,q->bc",
                shapes,
                current_response.entropy_temperature_change,
                shapes,
                weights,
            )
            local_temperature_temperature[element] = (
                local_tangent_temperature_temperature
            )
            strain_energy += float(
                np.dot(current_response.strain_energy_density, weights)
            )

        return ThermoelasticAssemblyEvaluation(
            mechanical_residual=mechanical_residual,
            mechanical_displacement_tangent=self._assemble_matrix(
                local_uu,
                self._mechanical_dofs,
                self._mechanical_dofs,
                (self.mechanical_dof_count, self.mechanical_dof_count),
            ),
            mechanical_temperature_tangent=self._assemble_matrix(
                local_u_temperature,
                self._mechanical_dofs,
                self._temperature_dofs,
                (self.mechanical_dof_count, self.temperature_dof_count),
            ),
            thermal_entropy_residual=thermal_residual,
            thermal_displacement_tangent=self._assemble_matrix(
                local_temperature_u,
                self._temperature_dofs,
                self._mechanical_dofs,
                (self.temperature_dof_count, self.mechanical_dof_count),
            ),
            thermal_temperature_tangent=self._assemble_matrix(
                local_temperature_temperature,
                self._temperature_dofs,
                self._temperature_dofs,
                (self.temperature_dof_count, self.temperature_dof_count),
            ),
            strain_energy=strain_energy,
        )

    @staticmethod
    def _field(
        values: NDArray[np.float64], size: int, name: str
    ) -> NDArray[np.float64]:
        field = np.asarray(values, dtype=np.float64)
        if field.shape != (size,) or not np.all(np.isfinite(field)):
            raise ValueError(f"{name} must be a finite vector with shape ({size},)")
        return field

    @staticmethod
    def _assemble_matrix(
        local_matrices: NDArray[np.float64],
        row_dofs: NDArray[np.int64],
        column_dofs: NDArray[np.int64],
        shape: tuple[int, int],
    ) -> csr_matrix:
        rows = np.broadcast_to(
            row_dofs[:, :, None], local_matrices.shape
        ).reshape(-1)
        columns = np.broadcast_to(
            column_dofs[:, None, :], local_matrices.shape
        ).reshape(-1)
        return coo_matrix(
            (local_matrices.reshape(-1), (rows, columns)), shape=shape
        ).tocsr()


__all__ = ["ThermoelasticAssemblyEvaluation", "ThermoelasticStVenantKirchhoffAssembler"]
