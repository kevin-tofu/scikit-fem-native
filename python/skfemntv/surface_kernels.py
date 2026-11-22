"""Local quadrature and load kernels for linear and quadratic surface cells."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .basis import Basis, ElementQuad1, ElementTriP2, MeshQuad, MeshTri2


@dataclass(frozen=True)
class Quad4SurfaceIntegrationData:
    natural_coordinates: NDArray[np.float64]
    physical_coordinates: NDArray[np.float64]
    shape_values: NDArray[np.float64]
    surface_jacobian: NDArray[np.float64]
    normals: NDArray[np.float64]
    weights: NDArray[np.float64]


@dataclass(frozen=True)
class Tri6SurfaceQuadratureData:
    coordinates: NDArray[np.float64]
    shape_values: NDArray[np.float64]
    normals: NDArray[np.float64]
    weights: NDArray[np.float64]


class Quad4SurfaceKernel:
    """Integrate linear quadrilateral surface cells embedded in 3D."""

    def __init__(
        self,
        points: NDArray[np.float64],
        quadrilaterals: NDArray[np.integer],
        *,
        integration_order: int = 2,
    ) -> None:
        coordinates = np.asarray(points, dtype=np.float64)
        cells = np.asarray(quadrilaterals, dtype=np.int64)
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise ValueError("Quad4 points must have shape (nodes, 3)")
        if cells.ndim != 2 or cells.shape[1] != 4:
            raise ValueError("Quad4 cells must have shape (elements, 4)")
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("Quad4 points must be finite")
        if np.any(cells < 0) or np.any(cells >= coordinates.shape[0]):
            raise ValueError("Quad4 cells contain an out-of-range node index")
        if (
            isinstance(integration_order, bool)
            or not isinstance(integration_order, (int, np.integer))
            or integration_order < 2
        ):
            raise ValueError("Quad4 integration_order must be an integer >= 2")
        self.points = coordinates
        self.quadrilaterals = cells
        self.integration_order = int(integration_order)
        reference_nodes = np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        )
        reference_mesh = MeshQuad(
            reference_nodes.T,
            np.arange(4, dtype=np.int64)[:, None],
        )
        gauss_points, gauss_weights = np.polynomial.legendre.leggauss(
            self.integration_order
        )
        unit_points = 0.5 * (gauss_points + 1.0)
        unit_weights = 0.5 * gauss_weights
        quadrature_points = np.asarray(
            [(unit_points[i], unit_points[j])
             for i in range(self.integration_order)
             for j in range(self.integration_order)],
            dtype=np.float64,
        ).T
        quadrature_weights = np.asarray(
            [unit_weights[i] * unit_weights[j]
             for i in range(self.integration_order)
             for j in range(self.integration_order)],
            dtype=np.float64,
        )
        basis = Basis(
            reference_mesh,
            ElementQuad1(),
            quadrature=(quadrature_points, quadrature_weights),
        )
        self._reference_points = basis.X.T.copy()
        self._reference_weights = basis.W.copy()
        self._shape_values = basis.tabulated_shape[0].copy()
        self._shape_gradients = basis.tabulated_gradients[0].copy()
        self.data = self._integrate_all()

    @property
    def gauss_point_count(self) -> int:
        return int(self._reference_weights.size)

    def integrate_scalar(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        scalar_values = np.asarray(values, dtype=np.float64)
        expected = (self.quadrilaterals.shape[0], self.gauss_point_count)
        if scalar_values.shape != expected:
            raise ValueError(f"scalar Gauss values must have shape {expected}")
        return np.sum(scalar_values * self.data.weights, axis=1)

    def element_nodal_areas(self) -> NDArray[np.float64]:
        return np.einsum(
            "eqn,eq->en",
            self.data.shape_values,
            self.data.weights,
        )

    def _integrate_all(self) -> Quad4SurfaceIntegrationData:
        element_count = self.quadrilaterals.shape[0]
        quadrature_count = self.gauss_point_count
        physical = np.empty((element_count, quadrature_count, 3))
        jacobians = np.empty((element_count, quadrature_count))
        normals = np.empty((element_count, quadrature_count, 3))
        weights = np.empty((element_count, quadrature_count))
        for element_index, nodes in enumerate(self.quadrilaterals):
            coordinates = self.points[nodes]
            for quadrature_index, (shape, gradients, reference_weight) in enumerate(
                zip(
                    self._shape_values,
                    self._shape_gradients,
                    self._reference_weights,
                    strict=True,
                )
            ):
                tangents = gradients.T @ coordinates
                normal_vector = np.cross(tangents[0], tangents[1])
                jacobian = float(np.linalg.norm(normal_vector))
                if not np.isfinite(jacobian) or jacobian <= np.finfo(float).eps:
                    raise ValueError(
                        f"Quad4 element {element_index} has zero surface "
                        f"Jacobian at Gauss point {quadrature_index}"
                    )
                physical[element_index, quadrature_index] = shape @ coordinates
                # Keep the public Quad4 Jacobian in the conventional
                # [-1, 1]^2 coordinates used by the previous kktkit API.
                jacobians[element_index, quadrature_index] = jacobian
                normals[element_index, quadrature_index] = normal_vector / jacobian
                weights[element_index, quadrature_index] = jacobian * reference_weight
        natural_coordinates = 2.0 * self._reference_points - 1.0
        return Quad4SurfaceIntegrationData(
            natural_coordinates=np.broadcast_to(
                natural_coordinates, (element_count, quadrature_count, 2)
            ).copy(),
            physical_coordinates=physical,
            shape_values=np.broadcast_to(
                self._shape_values, (element_count, quadrature_count, 4)
            ).copy(),
            surface_jacobian=jacobians,
            normals=normals,
            weights=weights,
        )


class Tri6SurfaceKernel:
    """Quadratic triangular surface quadrature and local traction kernels."""

    def __init__(self, *, integration_order: int = 4) -> None:
        if (
            isinstance(integration_order, bool)
            or not isinstance(integration_order, (int, np.integer))
            or integration_order < 2
        ):
            raise ValueError("Tri6 integration_order must be an integer >= 2")
        self.integration_order = int(integration_order)
        reference_nodes = np.asarray(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.5, 0.0],
                [0.5, 0.5],
                [0.0, 0.5],
            ]
        )
        reference_mesh = MeshTri2(
            reference_nodes.T,
            np.arange(6, dtype=np.int64)[:, None],
        )
        basis = Basis(
            reference_mesh,
            ElementTriP2(),
            intorder=self.integration_order,
        )
        self.integration_points = basis.X.T.copy()
        self.integration_weights = basis.W.copy()
        self._shape_values = basis.tabulated_shape[0].copy()
        self._shape_gradients = basis.tabulated_gradients[0].copy()

    def quadrature(self, coordinates: NDArray[np.float64]) -> Tri6SurfaceQuadratureData:
        nodes = np.asarray(coordinates, dtype=np.float64)
        if nodes.shape != (6, 3) or not np.all(np.isfinite(nodes)):
            raise ValueError("Tri6 coordinates must be finite with shape (6, 3)")
        tangents = np.einsum("qni,nj->qij", self._shape_gradients, nodes)
        area_vectors = np.cross(tangents[:, 0], tangents[:, 1])
        jacobians = np.linalg.norm(area_vectors, axis=1)
        if np.any(~np.isfinite(jacobians)) or np.any(jacobians <= np.finfo(float).eps):
            raise ValueError("Tri6 surface has a zero or invalid Jacobian")
        return Tri6SurfaceQuadratureData(
            coordinates=self._shape_values @ nodes,
            shape_values=self._shape_values.copy(),
            normals=area_vectors / jacobians[:, None],
            weights=jacobians * self.integration_weights,
        )

    def area(self, coordinates: NDArray[np.float64]) -> float:
        return float(np.sum(self.quadrature(coordinates).weights))

    def nodal_tributary_areas(
        self,
        coordinates: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        data = self.quadrature(coordinates)
        return data.shape_values.T @ data.weights

    def integrate_total_force(
        self,
        coordinates: NDArray[np.float64],
        total_force: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        force = np.asarray(total_force, dtype=np.float64)
        if force.shape != (3,) or not np.all(np.isfinite(force)):
            raise ValueError("total_force must be a finite vector with shape (3,)")
        data = self.quadrature(coordinates)
        total_area = float(np.sum(data.weights))
        if total_area <= 0.0:
            raise ValueError("Tri6 surface area must be positive")
        return (data.shape_values.T @ data.weights / total_area)[:, None] * force

    def integrate_traction(
        self,
        coordinates: NDArray[np.float64],
        traction: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        vector = np.asarray(traction, dtype=np.float64)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise ValueError("traction must be a finite vector with shape (3,)")
        data = self.quadrature(coordinates)
        return (data.shape_values.T @ data.weights)[:, None] * vector


__all__ = [
    "Quad4SurfaceIntegrationData",
    "Quad4SurfaceKernel",
    "Tri6SurfaceKernel",
    "Tri6SurfaceQuadratureData",
]
