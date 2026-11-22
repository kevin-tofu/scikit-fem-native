"""Element-local linear elasticity kernel for serendipity Hex20 cells."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .basis import Basis, ElementHex20, ElementVector, FacetBasis, MeshHex20


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class Hex20LinearElasticityResponse:
    strain: FloatArray
    stress: FloatArray


@dataclass(frozen=True)
class Hex20FacetLoadContributions:
    global_dofs: IntArray
    local_force: FloatArray


class Hex20LinearElasticityKernel:
    """Return per-cell stiffness and integration-point material responses."""

    def __init__(
        self,
        points: FloatArray,
        elements: IntArray,
        material: object,
        *,
        integration_order: int = 4,
    ) -> None:
        coordinates = np.asarray(points, dtype=np.float64)
        cells = np.asarray(elements, dtype=np.int64)
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise ValueError("Hex20 points must have shape (nodes, 3)")
        if cells.ndim != 2 or cells.shape[1] != 20:
            raise ValueError("Hex20 elements must have shape (elements, 20)")
        if cells.shape[0] == 0:
            raise ValueError("Hex20 elements must contain at least one cell")
        if np.any(cells < 0) or np.any(cells >= coordinates.shape[0]):
            raise ValueError("Hex20 elements contain an out-of-range node index")
        self.points = coordinates
        self.elements = cells
        self.material = material
        self.integration_order = int(integration_order)
        self.mesh = MeshHex20(coordinates.T, cells.T)
        self.basis = Basis(
            self.mesh,
            ElementVector(ElementHex20(), dim=3),
            intorder=self.integration_order,
        )
        self.physical_shape_gradients = self.basis.tabulated_gradients.copy()
        self.integration_weights = self.basis.dx.copy()
        self.element_stiffness = self._integrate_stiffness()

    @property
    def element_count(self) -> int:
        return int(self.elements.shape[0])

    @property
    def integration_point_count(self) -> int:
        return int(self.integration_weights.shape[1])

    @property
    def local_dof_count(self) -> int:
        return 60

    @property
    def minimum_jacobian(self) -> float:
        return float(self.basis.geometry_diagnostics.minimum_determinant)

    def _integrate_stiffness(self) -> FloatArray:
        stiffness_tensor = np.asarray(
            self.material.stiffness_tensor, dtype=np.float64
        )
        expected = (3, 3, 3, 3)
        if stiffness_tensor.shape != expected:
            raise ValueError(f"material stiffness tensor must have shape {expected}")
        local = np.einsum(
            "eqaj,ijkl,eqbl,eq->eaibk",
            self.physical_shape_gradients,
            stiffness_tensor,
            self.physical_shape_gradients,
            self.integration_weights,
            optimize=True,
        )
        return np.ascontiguousarray(local.reshape((self.element_count, 60, 60)))

    def evaluate_element(
        self,
        element_index: int,
        displacement: FloatArray,
    ) -> Hex20LinearElasticityResponse:
        if element_index < 0 or element_index >= self.element_count:
            raise IndexError("Hex20 element index is out of range")
        values = np.asarray(displacement, dtype=np.float64)
        if values.shape == (60,):
            nodal = values.reshape((20, 3))
        elif values.shape == (20, 3):
            nodal = values
        else:
            raise ValueError("element displacement must have shape (60,) or (20, 3)")
        gradient = np.einsum(
            "ai,qaj->qij",
            nodal,
            self.physical_shape_gradients[element_index],
            optimize=True,
        )
        strain = 0.5 * (gradient + np.swapaxes(gradient, -1, -2))
        return Hex20LinearElasticityResponse(
            strain=strain,
            stress=self.material.stress(strain),
        )

    def integrate_surface_total_force(
        self,
        facets: IntArray,
        total_force: FloatArray,
    ) -> Hex20FacetLoadContributions:
        facet_cells = np.asarray(facets, dtype=np.int64)
        force = np.asarray(total_force, dtype=np.float64)
        if facet_cells.ndim != 2 or facet_cells.shape[1] != 8:
            raise ValueError("Hex20 facets must have shape (facets, 8)")
        if not np.all(np.isfinite(force)) or force.shape != (3,):
            raise ValueError("total_force must be a finite vector with shape (3,)")
        if np.any(facet_cells < 0) or np.any(facet_cells >= self.points.shape[0]):
            raise ValueError("Hex20 facets contain an out-of-range node index")
        facet_basis = FacetBasis(
            self.mesh,
            ElementVector(ElementHex20(), dim=3),
            facets=facet_cells.T,
            intorder=self.integration_order,
        )
        areas = np.sum(facet_basis.dx, axis=1)
        total_area = float(np.sum(areas))
        if not np.isfinite(total_area) or total_area <= 0.0:
            raise ValueError("Hex20 load surface must have positive area")
        nodal_weights = np.einsum(
            "fqn,fq->fn", facet_basis.tabulated_shape, facet_basis.dx
        )
        local_force = (
            nodal_weights[:, :, None] * (force / total_area)[None, None, :]
        ).reshape((facet_cells.shape[0], 60))
        return Hex20FacetLoadContributions(
            global_dofs=facet_basis.element_dofs.T.copy(),
            local_force=np.ascontiguousarray(local_force),
        )


__all__ = [
    "Hex20FacetLoadContributions",
    "Hex20LinearElasticityKernel",
    "Hex20LinearElasticityResponse",
]
