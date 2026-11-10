"""Two-dimensional piecewise-constant mortar contact against rigid profiles.

The multiplier is constant on each P0 slave subfacet. The gap equation may
use the quadrature-average signed distance, the minimum quadrature distance,
or a differentiable conservative soft minimum. In each case the contact force
and geometric tangent differentiate the same selected gap. Sample gaps are
retained separately so callers can check pointwise nonpenetration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse

from .rigid_profile import AnalyticalRigidProfile2D


@dataclass(frozen=True)
class MortarContactEvaluation2D:
    """Reduced mortar gaps and their consistent geometric derivatives."""

    gaps: NDArray[np.float64]
    jacobian: sparse.csr_matrix
    sample_gaps: NDArray[np.float64]
    minimum_sample_gaps: NDArray[np.float64]
    sample_points: NDArray[np.float64]
    sample_normals: NDArray[np.float64]
    facet_points: NDArray[np.float64]
    facet_normals: NDArray[np.float64]
    facet_lengths: NDArray[np.float64]
    facet_dofs: NDArray[np.int64]
    facet_hessians: tuple[NDArray[np.float64], ...]


class RigidProfileMortarContact2D:
    """Build P0-multiplier mortar rows on linear 2D boundary facets.

    Parameters
    ----------
    facet_nodes:
        Integer array of shape ``(n_facets, 2)``. Each row gives the two
        displacement nodes of one straight slave boundary facet.
    reference_coordinates:
        Nodal coordinates with shape ``(n_nodes, 2)``.
    dof_x, dof_y:
        Global displacement DOF index for each node and coordinate.
    variable_count:
        Total number of primal variables. Defaults to one past the largest
        displacement DOF. It may include a rigid-tool translation variable.
    tool_translation_dof, tool_translation_direction:
        Optional primal variable that translates the rigid profile. The
        direction is a two-vector; its gap derivative is ``-normal·direction``.
    quadrature_order:
        Gauss-Legendre order used to integrate each facet's mortar gap.
    subfacets_per_facet:
        Number of independent P0 contact resultants on each finite-element
        boundary facet. Subfacets partition the parent edge parametrically.
    gap_reduction:
        ``"average"`` enforces the quadrature-average gap. ``"minimum"``
        enforces the smallest quadrature gap on each subfacet and uses that
        sample's consistent normal and curvature. ``"softmin"`` uses a
        differentiable, quadrature-weighted reduction while retaining exact
        sample gaps for pointwise penetration checks. Because its normalized
        quadrature weights can make the reduced gap larger than the smallest
        sample gap, it is not a conservative nonpenetration constraint.
        ``"conservative-softmin"`` adds a lower-bound correction, so its gap
        never exceeds the smallest sample gap.
    gap_smoothing_temperature:
        Positive length scale for either soft-minimum reduction.
    """

    def __init__(
        self,
        facet_nodes: ArrayLike,
        reference_coordinates: ArrayLike,
        dof_x: ArrayLike,
        dof_y: ArrayLike,
        *,
        variable_count: int | None = None,
        tool_translation_dof: int | None = None,
        tool_translation_direction: ArrayLike = (0.0, 1.0),
        quadrature_order: int = 4,
        subfacets_per_facet: int = 1,
        gap_reduction: str = "average",
        gap_smoothing_temperature: float = 1e-5,
    ) -> None:
        nodes = np.asarray(facet_nodes, dtype=np.int64)
        coordinates = np.asarray(reference_coordinates, dtype=float)
        x_dofs = np.asarray(dof_x, dtype=np.int64)
        y_dofs = np.asarray(dof_y, dtype=np.int64)
        direction = np.asarray(tool_translation_direction, dtype=float)
        if nodes.ndim != 2 or nodes.shape[1] != 2 or nodes.shape[0] == 0:
            raise ValueError("facet_nodes must have shape (n_facets, 2)")
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError("reference_coordinates must have shape (n_nodes, 2)")
        if x_dofs.shape != (coordinates.shape[0],) or y_dofs.shape != x_dofs.shape:
            raise ValueError("dof_x and dof_y must contain one DOF per node")
        if (np.any(nodes < 0) or np.any(nodes >= coordinates.shape[0])
                or np.any(x_dofs < 0) or np.any(y_dofs < 0)
                or not np.all(np.isfinite(coordinates))):
            raise ValueError("facet connectivity or nodal data are invalid")
        if direction.shape != (2,) or not np.all(np.isfinite(direction)):
            raise ValueError("tool_translation_direction must be a finite 2-vector")
        if quadrature_order < 1:
            raise ValueError("quadrature_order must be positive")
        if subfacets_per_facet < 1:
            raise ValueError("subfacets_per_facet must be positive")
        if gap_reduction not in {
            "average", "minimum", "softmin", "conservative-softmin"
        }:
            raise ValueError(
                "gap_reduction must be 'average', 'minimum', 'softmin', "
                "or 'conservative-softmin'"
            )
        if (not np.isfinite(gap_smoothing_temperature)
                or gap_smoothing_temperature <= 0.):
            raise ValueError(
                "gap_smoothing_temperature must be finite and positive"
            )
        inferred_count = int(max(np.max(x_dofs), np.max(y_dofs))) + 1
        count = inferred_count if variable_count is None else int(variable_count)
        if count < inferred_count:
            raise ValueError("variable_count does not cover all displacement DOFs")
        if (tool_translation_dof is not None
                and not 0 <= int(tool_translation_dof) < count):
            raise ValueError("tool_translation_dof is outside variable_count")
        if tool_translation_dof is not None and int(tool_translation_dof) in set(
                np.r_[x_dofs, y_dofs].tolist()):
            raise ValueError("tool translation DOF must be distinct from displacement DOFs")
        lengths = np.linalg.norm(
            coordinates[nodes[:, 1]] - coordinates[nodes[:, 0]], axis=1
        )
        if np.any(lengths <= np.finfo(float).eps):
            raise ValueError("mortar facets must have positive length")

        points, gauss_weights = np.polynomial.legendre.leggauss(
            int(quadrature_order)
        )
        self.facet_nodes = np.ascontiguousarray(nodes)
        self.reference_coordinates = np.ascontiguousarray(coordinates)
        self.dof_x = np.ascontiguousarray(x_dofs)
        self.dof_y = np.ascontiguousarray(y_dofs)
        self.variable_count = count
        self.tool_translation_dof = (
            None if tool_translation_dof is None
            else int(tool_translation_dof)
        )
        self.tool_translation_direction = np.ascontiguousarray(direction)
        self.quadrature_order = int(quadrature_order)
        self.quadrature_points = np.ascontiguousarray(points)
        self.subfacets_per_facet = int(subfacets_per_facet)
        self.gap_reduction = gap_reduction
        self.gap_smoothing_temperature = float(gap_smoothing_temperature)
        self.parent_facet_indices = np.repeat(
            np.arange(nodes.shape[0], dtype=np.int64),
            self.subfacets_per_facet,
        )
        self.subfacet_indices = np.tile(
            np.arange(self.subfacets_per_facet, dtype=np.int64),
            nodes.shape[0],
        )
        parent_coordinates = (
            -1.0
            + (2.0 * self.subfacet_indices[:, None]
               + points[None, :] + 1.0) / self.subfacets_per_facet
        )
        self.subfacet_shape_values = np.ascontiguousarray(np.stack((
            0.5 * (1.0 - parent_coordinates),
            0.5 * (1.0 + parent_coordinates),
        ), axis=-1))
        # Normalized weights define the average-gap option and average
        # geometric diagnostics. Multipliers are subfacet resultants, not
        # pressure values.
        self.quadrature_weights = np.ascontiguousarray(0.5 * gauss_weights)
        self.shape_values = np.ascontiguousarray(np.column_stack((
            0.5 * (1.0 - points), 0.5 * (1.0 + points),
        )))
        self.facet_lengths = np.ascontiguousarray(lengths)

    @property
    def facet_count(self) -> int:
        """Number of P0 contact rows, including subdivisions."""
        return int(self.parent_facet_indices.size)

    @property
    def parent_facet_count(self) -> int:
        """Number of finite-element boundary facets before subdivision."""
        return int(self.facet_nodes.shape[0])

    def evaluate(
        self,
        displacement: ArrayLike,
        profile: AnalyticalRigidProfile2D,
        *,
        translation: ArrayLike = (0.0, 0.0),
        curvature_blend_length: float = 0.0,
        feature_blend_temperature: float = 0.0,
    ) -> MortarContactEvaluation2D:
        """Evaluate facet-average gaps, Jacobian rows, and gap Hessians."""
        if not isinstance(profile, AnalyticalRigidProfile2D):
            raise TypeError("profile must be an AnalyticalRigidProfile2D")
        u = np.asarray(displacement, dtype=float)
        if u.shape != (self.variable_count,) or not np.all(np.isfinite(u)):
            raise ValueError(
                f"displacement must be a finite vector of shape ({self.variable_count},)"
            )
        shift = np.asarray(translation, dtype=float)
        if shift.shape != (2,) or not np.all(np.isfinite(shift)):
            raise ValueError("translation must be a finite 2-vector")

        row_nodes = self.facet_nodes[self.parent_facet_indices]
        facet_dofs = np.column_stack((
            self.dof_x[row_nodes[:, 0]],
            self.dof_y[row_nodes[:, 0]],
            self.dof_x[row_nodes[:, 1]],
            self.dof_y[row_nodes[:, 1]],
            np.full(self.facet_count, -1, dtype=np.int64)
            if self.tool_translation_dof is None
            else np.full(self.facet_count, self.tool_translation_dof, dtype=np.int64),
        ))
        node_coordinates = self.reference_coordinates + np.column_stack((
            u[self.dof_x], u[self.dof_y],
        ))
        endpoint_coordinates = node_coordinates[row_nodes]
        sample_points = np.einsum(
            "rqa,rad->rqd", self.subfacet_shape_values, endpoint_coordinates
        ).reshape(-1, 2)
        projection = profile.translated(shift).project(
            sample_points,
            curvature_blend_length=curvature_blend_length,
            feature_blend_temperature=feature_blend_temperature,
        )
        sample_gaps = np.asarray(projection.gaps, dtype=float)
        sample_normals = np.asarray(projection.normals, dtype=float)
        sample_hessians = np.asarray(projection.gap_hessians, dtype=float)

        sample_gap_matrix = sample_gaps.reshape(
            self.facet_count, self.quadrature_order
        )
        minimum_sample_indices = np.argmin(sample_gap_matrix, axis=1)
        if self.gap_reduction == "minimum":
            reduction_weights = np.zeros_like(sample_gap_matrix)
            reduction_weights[
                np.arange(self.facet_count), minimum_sample_indices
            ] = 1.
            gaps = sample_gap_matrix[
                np.arange(self.facet_count), minimum_sample_indices
            ]
            sample_points_matrix = sample_points.reshape(
                self.facet_count, self.quadrature_order, 2
            )
            sample_normals_matrix = sample_normals.reshape(
                self.facet_count, self.quadrature_order, 2
            )
            facet_points = sample_points_matrix[
                np.arange(self.facet_count), minimum_sample_indices
            ]
            facet_normals = sample_normals_matrix[
                np.arange(self.facet_count), minimum_sample_indices
            ]
        else:
            if self.gap_reduction == "average":
                reduction_weights = np.broadcast_to(
                    self.quadrature_weights,
                    (self.facet_count, self.quadrature_order),
                ).copy()
                gaps = np.einsum(
                    "rq,rq->r", reduction_weights, sample_gap_matrix
                )
            else:
                minimum_gaps = sample_gap_matrix.min(axis=1)
                shifted = (
                    sample_gap_matrix - minimum_gaps[:, None]
                ) / self.gap_smoothing_temperature
                exponential_weights = self.quadrature_weights[None, :] * np.exp(
                    -shifted
                )
                weight_sums = exponential_weights.sum(axis=1)
                reduction_weights = exponential_weights / weight_sums[:, None]
                gaps = minimum_gaps - self.gap_smoothing_temperature * np.log(
                    weight_sums
                )
                if self.gap_reduction == "conservative-softmin":
                    # The normalized weighted soft minimum above is shifted
                    # above the true minimum. Dividing by the smallest positive
                    # quadrature weight makes this a lower bound and leaves
                    # its first and second derivatives unchanged.
                    gaps += self.gap_smoothing_temperature * np.log(
                        float(np.min(self.quadrature_weights))
                    )
            facet_points = np.einsum(
                "rq,rqd->rd", reduction_weights,
                sample_points.reshape(self.facet_count, self.quadrature_order, 2),
            )
            raw_normals = np.einsum(
                "rq,rqd->rd", reduction_weights,
                sample_normals.reshape(self.facet_count, self.quadrature_order, 2),
            )
            normal_lengths = np.linalg.norm(raw_normals, axis=1)
            facet_normals = raw_normals / np.maximum(normal_lengths[:, None], 1e-30)
        current_facet_lengths = np.linalg.norm(
            endpoint_coordinates[:, 1] - endpoint_coordinates[:, 0], axis=1
        ) / self.subfacets_per_facet

        rows: list[int] = []
        columns: list[int] = []
        values: list[float] = []
        facet_hessians: list[NDArray[np.float64]] = []
        for row in range(self.facet_count):
            local_dofs = facet_dofs[row]
            local_jacobian = np.zeros(5, dtype=float)
            local_hessian = np.zeros((5, 5), dtype=float)
            quadrature_indices = (
                (int(minimum_sample_indices[row]),)
                if self.gap_reduction == "minimum"
                else range(self.quadrature_order)
            )
            sample_jacobians = np.zeros(
                (self.quadrature_order, 5), dtype=float
            )
            for q in quadrature_indices:
                sample = row * self.quadrature_order + q
                shape = self.subfacet_shape_values[row, q]
                normal = sample_normals[sample]
                mapping = np.zeros((2, 5), dtype=float)
                mapping[0, 0] = shape[0]
                mapping[1, 1] = shape[0]
                mapping[0, 2] = shape[1]
                mapping[1, 3] = shape[1]
                if self.tool_translation_dof is not None:
                    mapping[:, 4] = -self.tool_translation_direction
                sample_jacobians[q] = mapping.T @ normal
                weight = reduction_weights[row, q]
                local_jacobian += weight * sample_jacobians[q]
                local_hessian += weight * (
                    mapping.T @ sample_hessians[sample] @ mapping
                )
            if self.gap_reduction in {"softmin", "conservative-softmin"}:
                for q in range(self.quadrature_order):
                    difference = sample_jacobians[q] - local_jacobian
                    local_hessian -= (
                        reduction_weights[row, q]
                        / self.gap_smoothing_temperature
                    ) * np.outer(difference, difference)
            for local_col, global_dof in enumerate(local_dofs):
                if global_dof < 0:
                    continue
                value = local_jacobian[local_col]
                if value != 0.0:
                    rows.append(row)
                    columns.append(int(global_dof))
                    values.append(float(value))
            facet_hessians.append(local_hessian)

        jacobian = sparse.coo_matrix(
            (values, (rows, columns)),
            shape=(self.facet_count, self.variable_count),
        ).tocsr()
        return MortarContactEvaluation2D(
            gaps=np.asarray(gaps, dtype=float),
            jacobian=jacobian,
            sample_gaps=sample_gaps,
            minimum_sample_gaps=sample_gaps.reshape(
                self.facet_count, self.quadrature_order
            ).min(axis=1),
            sample_points=np.asarray(sample_points, dtype=float),
            sample_normals=sample_normals,
            facet_points=np.asarray(facet_points, dtype=float),
            facet_normals=np.asarray(facet_normals, dtype=float),
            facet_lengths=np.asarray(current_facet_lengths, dtype=float),
            facet_dofs=facet_dofs,
            facet_hessians=tuple(facet_hessians),
        )

    def contact_force(
        self, evaluation: MortarContactEvaluation2D, multipliers: ArrayLike
    ) -> NDArray[np.float64]:
        """Return the integrated weak contact force for facet resultants."""
        self._validate_evaluation(evaluation)
        lam = np.asarray(multipliers, dtype=float)
        if lam.shape != (self.facet_count,) or not np.all(np.isfinite(lam)):
            raise ValueError("multipliers must be a finite value per facet")
        return np.asarray(evaluation.jacobian.T @ lam).reshape(-1)

    def geometric_tangent(
        self, evaluation: MortarContactEvaluation2D, multipliers: ArrayLike
    ) -> sparse.csr_matrix:
        """Assemble the consistent signed-distance curvature tangent."""
        self._validate_evaluation(evaluation)
        lam = np.asarray(multipliers, dtype=float)
        if lam.shape != (self.facet_count,) or not np.all(np.isfinite(lam)):
            raise ValueError("multipliers must be a finite value per facet")
        rows: list[int] = []
        columns: list[int] = []
        values: list[float] = []
        for facet, multiplier in enumerate(lam):
            if abs(multiplier) <= 1e-14:
                continue
            dofs = evaluation.facet_dofs[facet]
            local = multiplier * evaluation.facet_hessians[facet]
            for i, row in enumerate(dofs):
                if row < 0:
                    continue
                for j, column in enumerate(dofs):
                    if column < 0:
                        continue
                    value = local[i, j]
                    if value != 0.0:
                        rows.append(int(row))
                        columns.append(int(column))
                        values.append(float(value))
        return sparse.coo_matrix(
            (values, (rows, columns)),
            shape=(self.variable_count, self.variable_count),
        ).tocsr()

    def facet_pressure(
        self, evaluation: MortarContactEvaluation2D, multipliers: ArrayLike
    ) -> NDArray[np.float64]:
        """Convert facet resultants to signed normal pressure per unit width."""
        self._validate_evaluation(evaluation)
        lam = np.asarray(multipliers, dtype=float)
        if lam.shape != (self.facet_count,) or not np.all(np.isfinite(lam)):
            raise ValueError("multipliers must be a finite value per facet")
        return lam / evaluation.facet_lengths

    def _validate_evaluation(self, evaluation: MortarContactEvaluation2D) -> None:
        if (not isinstance(evaluation, MortarContactEvaluation2D)
                or evaluation.gaps.shape != (self.facet_count,)
                or evaluation.minimum_sample_gaps.shape != (self.facet_count,)
                or evaluation.facet_lengths.shape != (self.facet_count,)
                or evaluation.jacobian.shape
                != (self.facet_count, self.variable_count)):
            raise ValueError("evaluation does not belong to this mortar contact")


__all__ = ["MortarContactEvaluation2D", "RigidProfileMortarContact2D"]
