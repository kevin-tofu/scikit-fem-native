"""Experimental shifted-Heaviside and crack-tip XFEM bases for TriP1.

The basis enriches nodes belonging to level-set cut cells with
``N_i (H - H_i)``.  It is intended as a small, explicit starting point for
strong jumps.  An optional four-function Mode-I crack-tip enrichment and a
component-wise vector wrapper support small 2D linear-elastic crack problems.
The implementation remains limited to affine triangular P1 backgrounds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .basis import Basis, DiscreteField, ElementTriP1, ElementVector
from .cut_basis import CutCellBasis, ImplicitFacetBasis
from .implicit_interface import ImplicitInterfacePair
from .levelset import (
    CellClassificationResult,
    ImplicitInterfaceQuadrature,
    LevelSet,
)


@dataclass(frozen=True)
class _EnrichedScalarElement:
    doflocs: np.ndarray


@dataclass(frozen=True)
class _EnrichedVectorElement:
    elem: _EnrichedScalarElement
    _dim: int = 1


class ShiftedHeavisideCutBasis:
    """Cut-quadrature basis with shifted Heaviside enrichment for scalar TriP1.

    The global vector contains the original continuous P1 DOFs followed by
    one enrichment DOF for every node incident to a cut cell.  Instantiate
    once for each side of the same level set; both instances share the same
    global DOF numbering and can be assembled into one system.
    """

    def __init__(
        self,
        basis: Basis,
        level_set: LevelSet,
        *,
        side: str,
        intorder: int = 2,
        classification: CellClassificationResult | None = None,
        enrichment_nodes: np.ndarray | None = None,
    ) -> None:
        if not isinstance(basis, Basis):
            raise TypeError("basis must be a skfemntv Basis")
        if not isinstance(level_set, LevelSet):
            raise TypeError("level_set must be a LevelSet")
        if side not in {"inside", "outside"}:
            raise ValueError("side must be 'inside' or 'outside'")
        if not isinstance(basis.elem, ElementVector) or basis.elem._dim != 1:
            raise NotImplementedError("XFEM currently supports scalar fields only")
        if not isinstance(basis.elem.elem, ElementTriP1):
            raise NotImplementedError("XFEM currently supports triangular P1 only")
        if basis.mesh.t.shape[0] != 3:
            raise NotImplementedError("XFEM currently supports linear triangles only")

        self.mesh = basis.mesh
        self.parent_basis = basis
        self.side = side
        self.level_set = level_set
        self.classification = (
            level_set.classify(basis.mesh)
            if classification is None else classification
        )
        if len(self.classification.labels) != basis.mesh.nelements:
            raise ValueError("classification and basis mesh differ")
        self.quadrature = level_set.cut_quadrature(
            basis.mesh,
            side=side,
            intorder=intorder,
            classification=self.classification,
        )
        self._cut_basis = CutCellBasis(basis, self.quadrature)
        values = level_set.values(basis.mesh)
        nodal_sign = np.where(values < 0.0, -1.0, 1.0)
        nodal_sign.flags.writeable = False
        cut_cells = np.concatenate((
            np.asarray(self.classification.cut, dtype=np.int64),
            np.asarray(self.classification.touching, dtype=np.int64),
        ))
        cut_enrichment_nodes = (
            np.unique(basis.mesh.t[:, cut_cells]).astype(np.int64)
            if len(cut_cells) else np.empty(0, dtype=np.int64)
        )
        if enrichment_nodes is None:
            enriched_nodes = cut_enrichment_nodes
        else:
            enriched_nodes = np.unique(
                np.asarray(enrichment_nodes, dtype=np.int64)
            )
            if np.any(enriched_nodes < 0) or np.any(
                enriched_nodes >= basis.mesh.p.shape[1]
            ):
                raise ValueError("enrichment_nodes contains an invalid node id")
            if not np.isin(enriched_nodes, cut_enrichment_nodes).all():
                raise ValueError(
                    "enrichment_nodes must be incident to cut or "
                    "interface-touching cells"
                )
        enriched_dof = np.full(basis.mesh.p.shape[1], -1, dtype=np.int64)
        enriched_dof[enriched_nodes] = basis.N + np.arange(
            len(enriched_nodes), dtype=np.int64
        )

        quadrature_cells = self.quadrature.cells
        local_nodes = basis.mesh.t[:, quadrature_cells].T
        local_enriched_dofs = enriched_dof[local_nodes]
        active_enrichment = local_enriched_dofs >= 0
        # Inactive enrichment slots have exactly zero shape and use a harmless
        # valid global index so the native sparse assembler can keep a fixed
        # local element width.
        local_enriched_dofs = np.where(local_enriched_dofs >= 0,
                                       local_enriched_dofs, 0)
        side_sign = -1.0 if side == "inside" else 1.0
        shifted_factor = side_sign - nodal_sign[local_nodes]
        self.shape = np.concatenate((
            self._cut_basis.shape,
            self._cut_basis.shape * shifted_factor * active_enrichment,
        ), axis=1)
        self.gradients = np.concatenate((
            self._cut_basis.gradients,
            self._cut_basis.gradients
            * shifted_factor[:, :, None]
            * active_enrichment[:, :, None],
        ), axis=1)
        standard_dofs = self._cut_basis.quadrature_dofs
        self.quadrature_dofs = np.concatenate(
            (standard_dofs, local_enriched_dofs), axis=1
        )

        active_cells = self._cut_basis.tind
        cell_nodes = basis.mesh.t[:, active_cells].T
        cell_enriched_dofs = enriched_dof[cell_nodes]
        cell_enriched_dofs = np.where(cell_enriched_dofs >= 0,
                                      cell_enriched_dofs, 0)
        self.active_cell_dofs = np.concatenate((
            self._cut_basis.active_cell_dofs[:, :, 0],
            cell_enriched_dofs,
        ), axis=1)[:, :, None]
        counts = np.diff(self.quadrature.cell_offsets)[active_cells]
        self.active_cell_offsets = np.concatenate((
            np.array([0], dtype=np.int64),
            np.cumsum(counts, dtype=np.int64),
        ))

        self.N = basis.N + len(enriched_nodes)
        self.elem = _EnrichedVectorElement(_EnrichedScalarElement(
            np.concatenate((basis.elem.elem.doflocs, basis.elem.elem.doflocs))
        ))
        self.cells = self.quadrature.cells
        self.points = self.quadrature.points
        self.weights = self.quadrature.weights
        self.normal_vectors = self.quadrature.normals
        self.tind = active_cells
        self.dx = self.weights[:, None]
        self.tabulated_shape = self.shape[:, None, :]
        self.tabulated_gradients = self.gradients[:, None, :, :]
        self.global_coordinates = self.points[:, None, :]
        self.normals = self.normal_vectors[:, None, :]
        self.element_dofs = self.quadrature_dofs.T
        self.basis = ()
        self.enriched_nodes = enriched_nodes
        self.enriched_dof = enriched_dof
        self._nodal_sign = nodal_sign
        for array in (
            self.shape, self.gradients, self.quadrature_dofs,
            self.active_cell_dofs, self.active_cell_offsets, self.dx,
            self.tabulated_shape, self.tabulated_gradients,
            self.global_coordinates, self.normals,
        ):
            array.flags.writeable = False

    @property
    def npoints(self) -> int:
        return len(self.weights)

    def interpolate(self, coefficients: np.ndarray) -> DiscreteField:
        coefficients = np.asarray(coefficients, dtype=np.float64)
        if coefficients.shape != (self.N,):
            raise ValueError(f"coefficients must have shape ({self.N},)")
        local = coefficients[self.quadrature_dofs]
        value = np.einsum("qn,qn->q", self.shape, local)
        gradient = np.einsum("qnd,qn->dq", self.gradients, local)
        return DiscreteField(value, gradient)

    def integrate(self, values: np.ndarray) -> float:
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (self.npoints,):
            raise ValueError(f"values must have shape ({self.npoints},)")
        return float(self.weights @ values)

    def interface_trace(
        self, quadrature: ImplicitInterfaceQuadrature
    ) -> ShiftedHeavisideTraceBasis:
        """Tabulate this enriched field on a reconstructed level-set interface."""
        return ShiftedHeavisideTraceBasis(self, quadrature)


def _mode_i_tip_functions(
    points: np.ndarray, tip: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the four square-root crack-tip functions and their gradients."""
    relative = np.asarray(points, dtype=np.float64) - tip
    x, y = relative[:, 0], relative[:, 1]
    radius = np.hypot(x, y)
    at_tip = radius == 0.0
    safe_radius = np.where(at_tip, 1.0, radius)
    theta = np.arctan2(y, x)
    root_radius = np.sqrt(radius)
    safe_root_radius = np.sqrt(safe_radius)
    half = 0.5 * theta
    sine, cosine = np.sin(half), np.cos(half)
    sine_theta, cosine_theta = np.sin(theta), np.cos(theta)

    values = np.column_stack((
        root_radius * sine,
        root_radius * cosine,
        root_radius * sine * sine_theta,
        root_radius * cosine * sine_theta,
    ))
    radial_derivatives = 0.5 / safe_root_radius[:, None] * np.column_stack((
        sine,
        cosine,
        sine * sine_theta,
        cosine * sine_theta,
    ))
    angular_derivatives = safe_root_radius[:, None] * np.column_stack((
        0.5 * cosine,
        -0.5 * sine,
        0.5 * cosine * sine_theta + sine * cosine_theta,
        -0.5 * sine * sine_theta + cosine * cosine_theta,
    ))
    gradients = np.empty((len(points), 4, 2), dtype=np.float64)
    gradients[:, :, 0] = (
        cosine_theta[:, None] * radial_derivatives
        - sine_theta[:, None] / safe_radius[:, None] * angular_derivatives
    )
    gradients[:, :, 1] = (
        sine_theta[:, None] * radial_derivatives
        + cosine_theta[:, None] / safe_radius[:, None] * angular_derivatives
    )
    gradients[at_tip] = 0.0
    return values, gradients


class ModeICrackTipCutBasis(ShiftedHeavisideCutBasis):
    """Scalar TriP1 XFEM basis with local Mode-I crack-tip enrichment.

    Four branch functions ``sqrt(r) * (sin(theta/2), cos(theta/2),
    sin(theta/2) sin(theta), cos(theta/2) sin(theta))`` are shifted at each
    enriched node.  The Heaviside enrichment may be restricted to nodes along
    a finite crack while the tip functions occupy a circular nodal patch.
    This basis is scalar; wrap it in ``ShiftedHeavisideVectorCutBasis`` for
    elastic displacement fields.
    """

    def __init__(
        self,
        basis: Basis,
        level_set: LevelSet,
        *,
        side: str,
        tip: np.ndarray,
        tip_radius: float,
        intorder: int = 2,
        classification: CellClassificationResult | None = None,
        enrichment_nodes: np.ndarray | None = None,
    ) -> None:
        super().__init__(
            basis,
            level_set,
            side=side,
            intorder=intorder,
            classification=classification,
            enrichment_nodes=enrichment_nodes,
        )
        tip = np.asarray(tip, dtype=np.float64)
        if tip.shape != (2,) or not np.all(np.isfinite(tip)):
            raise ValueError("tip must be a finite 2D coordinate")
        if not np.isfinite(tip_radius) or tip_radius <= 0.0:
            raise ValueError("tip_radius must be finite and positive")
        self.tip = tip.copy()
        self.tip_radius = float(tip_radius)
        nodal_distance = np.linalg.norm(basis.mesh.p.T - tip, axis=1)
        tip_nodes = np.flatnonzero(nodal_distance <= self.tip_radius)
        self.tip_enriched_nodes = tip_nodes.astype(np.int64)
        tip_dof = np.full((basis.mesh.p.shape[1], 4), -1, dtype=np.int64)
        tip_dof[tip_nodes] = self.N + np.arange(
            4 * len(tip_nodes), dtype=np.int64
        ).reshape(len(tip_nodes), 4)

        quadrature_cells = self.quadrature.cells
        local_nodes = basis.mesh.t[:, quadrature_cells].T
        local_tip_dofs = tip_dof[local_nodes]
        active_tip = local_tip_dofs >= 0
        local_tip_dofs = np.where(local_tip_dofs >= 0, local_tip_dofs, 0)
        nodal_functions, _ = _mode_i_tip_functions(
            basis.mesh.p[:, local_nodes.reshape(-1)].T, tip
        )
        nodal_functions = nodal_functions.reshape(len(local_nodes), 3, 4)
        point_functions, point_gradients = _mode_i_tip_functions(
            self.points, tip
        )
        standard_shape = self._cut_basis.shape
        standard_gradients = self._cut_basis.gradients
        tip_shape = standard_shape[:, :, None] * (
            point_functions[:, None, :] - nodal_functions
        )
        tip_gradients = (
            standard_gradients[:, :, :, None]
            * (point_functions[:, None, None, :]
               - nodal_functions[:, :, None, :])
            + standard_shape[:, :, None, None]
            * point_gradients[:, None, :, :].transpose(0, 1, 3, 2)
        )
        tip_shape *= active_tip.reshape(len(quadrature_cells), 3, 4)
        tip_gradients *= active_tip.reshape(
            len(quadrature_cells), 3, 4
        )[:, :, None, :]
        tip_shape = tip_shape.reshape(len(self.weights), 12)
        tip_gradients = tip_gradients.reshape(len(self.weights), 12, 2)
        self.shape = np.concatenate((self.shape, tip_shape), axis=1)
        self.gradients = np.concatenate((self.gradients, tip_gradients), axis=1)
        self.quadrature_dofs = np.concatenate((
            self.quadrature_dofs,
            local_tip_dofs.reshape(len(self.weights), 12),
        ), axis=1)

        active_cells = self._cut_basis.tind
        active_nodes = basis.mesh.t[:, active_cells].T
        active_tip_dofs = tip_dof[active_nodes]
        active_tip_dofs = np.where(active_tip_dofs >= 0, active_tip_dofs, 0)
        self.active_cell_dofs = np.concatenate((
            self.active_cell_dofs[:, :, 0],
            active_tip_dofs.reshape(len(active_cells), 12),
        ), axis=1)[:, :, None]
        self.N += 4 * len(tip_nodes)
        self.tip_dof = tip_dof
        self.elem = _EnrichedVectorElement(_EnrichedScalarElement(
            np.concatenate((
                self.elem.elem.doflocs,
                np.repeat(basis.elem.elem.doflocs, 4, axis=0),
            ))
        ))
        self.tabulated_shape = self.shape[:, None, :]
        self.tabulated_gradients = self.gradients[:, None, :, :]
        self.element_dofs = self.quadrature_dofs.T
        for array in (
            self.shape,
            self.gradients,
            self.quadrature_dofs,
            self.active_cell_dofs,
            self.tabulated_shape,
            self.tabulated_gradients,
            self.element_dofs,
            self.tip,
            self.tip_enriched_nodes,
            self.tip_dof,
        ):
            array.flags.writeable = False


class ShiftedHeavisideVectorCutBasis:
    """Component-wise vector wrapper for a scalar shifted-Heaviside cut basis.

    The local scalar shape functions are shared by each vector component, as
    for a standard ``ElementVector``.  Global vector DOFs are interleaved by
    scalar enriched DOF and component.  This allows the native CutCell
    assembler to apply its symmetric-gradient and divergence contractions to
    XFEM displacement fields.
    """

    def __init__(
        self, scalar_basis: ShiftedHeavisideCutBasis, *, components: int = 2
    ) -> None:
        if not isinstance(scalar_basis, ShiftedHeavisideCutBasis):
            raise TypeError("scalar_basis must be a ShiftedHeavisideCutBasis")
        if isinstance(components, bool) or int(components) < 1:
            raise ValueError("components must be a positive integer")
        self.scalar_basis = scalar_basis
        self.components = int(components)
        self.mesh = scalar_basis.mesh
        self.parent_basis = scalar_basis.parent_basis
        self.side = scalar_basis.side
        self.level_set = scalar_basis.level_set
        self.classification = scalar_basis.classification
        self.quadrature = scalar_basis.quadrature
        self.shape = scalar_basis.shape
        self.gradients = scalar_basis.gradients
        component_ids = np.arange(self.components, dtype=np.int64)
        self.quadrature_dofs = (
            scalar_basis.quadrature_dofs[:, :, None] * self.components
            + component_ids[None, None, :]
        ).reshape(len(scalar_basis.weights), -1)
        self.active_cell_dofs = (
            scalar_basis.active_cell_dofs[:, :, 0, None] * self.components
            + component_ids[None, None, :]
        )
        self.active_cell_offsets = scalar_basis.active_cell_offsets
        self.N = scalar_basis.N * self.components
        self.elem = _EnrichedVectorElement(
            scalar_basis.elem.elem, _dim=self.components
        )
        self.cells = scalar_basis.cells
        self.points = scalar_basis.points
        self.weights = scalar_basis.weights
        self.normal_vectors = scalar_basis.normal_vectors
        self.tind = scalar_basis.tind
        self.dx = scalar_basis.dx
        self.tabulated_shape = scalar_basis.tabulated_shape
        self.tabulated_gradients = scalar_basis.tabulated_gradients
        self.global_coordinates = scalar_basis.global_coordinates
        self.normals = scalar_basis.normals
        self.element_dofs = self.quadrature_dofs.T
        self.basis = ()
        self.enriched_nodes = scalar_basis.enriched_nodes
        self.enriched_dof = scalar_basis.enriched_dof
        for array in (
            self.quadrature_dofs,
            self.active_cell_dofs,
            self.element_dofs,
        ):
            array.flags.writeable = False

    @property
    def npoints(self) -> int:
        return len(self.weights)

    def interpolate(self, coefficients: np.ndarray) -> DiscreteField:
        coefficients = np.asarray(coefficients, dtype=np.float64)
        if coefficients.shape != (self.N,):
            raise ValueError(f"coefficients must have shape ({self.N},)")
        scalar_coefficients = coefficients.reshape(
            self.scalar_basis.N, self.components
        )
        local = scalar_coefficients[self.scalar_basis.quadrature_dofs]
        value = np.einsum("qn,qnc->cq", self.shape, local)
        gradient = np.einsum("qnd,qnc->cdq", self.gradients, local)
        return DiscreteField(value, gradient)

    def integrate(self, values: np.ndarray) -> float:
        values = np.asarray(values, dtype=np.float64)
        if values.shape == (self.npoints,):
            return float(self.weights @ values)
        if values.shape == (self.components, self.npoints):
            return float(np.sum(values @ self.weights))
        raise ValueError(
            "integrand must have shape (quadrature_points,) or "
            "(components, quadrature_points)"
        )


class ShiftedHeavisideTraceBasis(ImplicitFacetBasis):
    """One-sided interface trace of a shifted-Heaviside XFEM basis."""

    def __init__(
        self,
        enriched_basis: ShiftedHeavisideCutBasis,
        quadrature: ImplicitInterfaceQuadrature,
    ) -> None:
        if not isinstance(enriched_basis, ShiftedHeavisideCutBasis):
            raise TypeError(
                "enriched_basis must be a ShiftedHeavisideCutBasis"
            )
        if not isinstance(quadrature, ImplicitInterfaceQuadrature):
            raise TypeError(
                "quadrature must be an ImplicitInterfaceQuadrature"
            )
        if quadrature.side != "interface":
            raise ValueError("trace quadrature must describe the zero interface")
        if len(quadrature.cell_offsets) != enriched_basis.mesh.nelements + 1:
            raise ValueError("interface quadrature and XFEM mesh differ")

        interface_side = (
            "negative" if enriched_basis.side == "inside" else "positive"
        )
        super().__init__(
            enriched_basis.parent_basis, quadrature, side=interface_side
        )
        local_nodes = self.mesh.t[:, quadrature.cells].T
        local_enriched_dofs = enriched_basis.enriched_dof[local_nodes]
        active_enrichment = local_enriched_dofs >= 0
        local_enriched_dofs = np.where(local_enriched_dofs >= 0,
                                       local_enriched_dofs, 0)
        side_sign = -1.0 if enriched_basis.side == "inside" else 1.0
        shifted_factor = (
            side_sign - enriched_basis._nodal_sign[local_nodes]
        )

        self.shape = np.concatenate((
            self.shape,
            self.shape * shifted_factor * active_enrichment,
        ), axis=1)
        self.gradients = np.concatenate((
            self.gradients,
            self.gradients
            * shifted_factor[:, :, None]
            * active_enrichment[:, :, None],
        ), axis=1)
        self.quadrature_dofs = np.concatenate((
            self.quadrature_dofs, local_enriched_dofs,
        ), axis=1)

        active_cells = self.tind
        active_nodes = self.mesh.t[:, active_cells].T
        active_enriched_dofs = enriched_basis.enriched_dof[active_nodes]
        active_enriched_dofs = np.where(
            active_enriched_dofs >= 0, active_enriched_dofs, 0
        )
        all_nodes = self.mesh.t.T
        all_enriched_dofs = enriched_basis.enriched_dof[all_nodes]
        all_enriched_dofs = np.where(
            all_enriched_dofs >= 0, all_enriched_dofs, 0
        )
        self.active_cell_dofs = np.concatenate((
            self.active_cell_dofs[:, :, 0], active_enriched_dofs,
        ), axis=1)[:, :, None]
        self.cell_dofs = np.concatenate((
            self.cell_dofs, all_enriched_dofs[:, :, None],
        ), axis=1)
        self.N = enriched_basis.N
        self.elem = enriched_basis.elem
        self.element_dofs = self.quadrature_dofs.T
        self.tabulated_shape = self.shape[:, None, :]
        self.tabulated_gradients = self.gradients[:, None, :, :]
        self.xfem_basis = enriched_basis
        for array in (
            self.shape,
            self.gradients,
            self.quadrature_dofs,
            self.active_cell_dofs,
            self.cell_dofs,
            self.element_dofs,
            self.tabulated_shape,
            self.tabulated_gradients,
        ):
            array.flags.writeable = False


class ShiftedHeavisideInterfacePair:
    """Interface integrator that assembles both traces into shared XFEM DOFs."""

    def __init__(
        self,
        negative: ShiftedHeavisideTraceBasis,
        positive: ShiftedHeavisideTraceBasis,
    ) -> None:
        if negative.side != "negative" or positive.side != "positive":
            raise ValueError("XFEM traces must be negative then positive")
        if negative.N != positive.N:
            raise ValueError("XFEM interface traces have different global sizes")
        if not np.array_equal(
            negative.xfem_basis.enriched_dof,
            positive.xfem_basis.enriched_dof,
        ):
            raise ValueError("XFEM interface traces use different DOF maps")
        negative_parent = negative.xfem_basis.parent_basis
        positive_parent = positive.xfem_basis.parent_basis
        if not (
            np.array_equal(negative_parent.mesh.p, positive_parent.mesh.p)
            and np.array_equal(negative_parent.mesh.t, positive_parent.mesh.t)
            and np.array_equal(
                negative_parent.element_dofs, positive_parent.element_dofs
            )
        ):
            raise ValueError("XFEM interface traces use different parent bases")
        self._pair = ImplicitInterfacePair(negative, positive)
        self.size = negative.N
        self.global_coordinates = negative.global_coordinates
        self.master_normals = negative.normals
        self.slave_normals = positive.normals
        self._weights = negative.dx
        self._coefficient_shape = negative.dx.shape
        self.gap = np.zeros(self._coefficient_shape, dtype=np.float64)
        self.gap.flags.writeable = False

    def assemble_traces(
        self,
        row_weights,
        column_weights,
        *,
        row_kind="value",
        column_kind="value",
        coefficient=1.0,
        num_threads=None,
    ):
        """Assemble a two-sided form after identifying the shared XFEM DOFs."""
        matrix = self._pair.assemble_traces(
            row_weights,
            column_weights,
            row_kind=row_kind,
            column_kind=column_kind,
            coefficient=coefficient,
            num_threads=num_threads,
        )
        size = self.size
        return (
            matrix[:size, :size]
            + matrix[:size, size:]
            + matrix[size:, :size]
            + matrix[size:, size:]
        ).tocsr()

    def assemble_linear_trace(
        self,
        trace_weights,
        *,
        trace_kind="value",
        coefficient=1.0,
        num_threads=None,
    ):
        """Assemble a two-sided load into the shared XFEM coefficient vector."""
        vector = self._pair.assemble_linear_trace(
            trace_weights,
            trace_kind=trace_kind,
            coefficient=coefficient,
            num_threads=num_threads,
        )
        return vector[: self.size] + vector[self.size :]
