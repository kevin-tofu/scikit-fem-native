"""Physical-point evaluation for affine simplex H1 bases."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class PointEvaluation:
    """Field values, physical gradients, and point-location diagnostics."""

    value: np.ndarray
    grad: np.ndarray
    cell_ids: np.ndarray
    reference_coordinates: np.ndarray
    on_boundary: np.ndarray
    ambiguous: np.ndarray
    outside: np.ndarray


def _shape_and_gradient(basis, reference):
    """Evaluate one local H1 shape at a reference simplex point."""
    base = basis._base_element
    dim = basis.mesh.dim()
    if basis._constant:
        return np.ones(1), np.zeros((1, dim))

    if dim == 2:
        xi, eta = reference
        bary = np.array((1.0 - xi - eta, xi, eta))
        bary_gradient = np.array(((-1.0, -1.0), (1.0, 0.0), (0.0, 1.0)))
        if not basis._quadratic_tri:
            return bary, bary_gradient
        pairs = ((0, 1), (1, 2), (0, 2))
        shape = [value * (2.0 * value - 1.0) for value in bary]
        gradient = [
            (4.0 * value - 1.0) * bary_gradient[i]
            for i, value in enumerate(bary)
        ]
        for i, j in pairs:
            shape.append(4.0 * bary[i] * bary[j])
            gradient.append(
                4.0 * (bary[j] * bary_gradient[i] + bary[i] * bary_gradient[j])
            )
        return np.asarray(shape), np.asarray(gradient)

    xi, eta, zeta = reference
    bary = np.array((1.0 - xi - eta - zeta, xi, eta, zeta))
    bary_gradient = np.array(
        ((-1.0, -1.0, -1.0), (1.0, 0.0, 0.0),
         (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    )
    if not basis._quadratic_tet:
        return bary, bary_gradient
    pairs = ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))
    shape = [value * (2.0 * value - 1.0) for value in bary]
    gradient = [
        (4.0 * value - 1.0) * bary_gradient[i]
        for i, value in enumerate(bary)
    ]
    for i, j in pairs:
        shape.append(4.0 * bary[i] * bary[j])
        gradient.append(
            4.0 * (bary[j] * bary_gradient[i] + bary[i] * bary_gradient[j])
        )
    return np.asarray(shape), np.asarray(gradient)


def evaluate_basis_at(basis, points, coefficients, *, outside="error"):
    """Evaluate an affine triangle/tetrahedron H1 field at physical points."""
    if outside not in ("error", "nan"):
        raise ValueError("outside must be 'error' or 'nan'")
    if hasattr(basis, "subbases"):
        raise NotImplementedError("point evaluation of composite bases is not implemented")
    if not (basis._tri or basis._tet):
        raise NotImplementedError(
            "point evaluation currently supports triangle and tetrahedron bases"
        )
    dim = basis.mesh.dim()
    if not (
        basis._constant
        or (basis._tri and not basis._quadratic_tri)
        or basis._quadratic_tri
        or (basis._tet and not basis._quadratic_tet)
        or basis._quadratic_tet
    ):
        raise NotImplementedError("point evaluation supports simplex P0, P1, and P2")

    points = np.asarray(points, dtype=np.float64)
    if points.ndim == 1:
        points = points[:, None]
    if points.ndim != 2 or points.shape[0] != dim:
        raise ValueError(f"points must have shape ({dim}, npoints)")
    if not np.all(np.isfinite(points)):
        raise ValueError("points must contain only finite coordinates")
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if coefficients.shape != (basis.N,):
        raise ValueError(f"coefficients must have shape ({basis.N},)")

    cell_ids = np.full(points.shape[1], -1, dtype=np.int64)
    references = np.full((dim, points.shape[1]), np.nan)
    boundary = np.zeros(points.shape[1], dtype=bool)
    ambiguous = np.zeros(points.shape[1], dtype=bool)
    outside_mask = np.zeros(points.shape[1], dtype=bool)
    components = basis.elem._dim
    local_nodes = basis.element_dofs.shape[0] // components
    value = np.full((components, points.shape[1]), np.nan)
    gradient = np.full((components, dim, points.shape[1]), np.nan)

    mesh_cells = basis._mesh.t[: dim + 1, basis.tind]
    vertices = basis._mesh.p[:, mesh_cells]
    geometry_cells = basis._mesh.t[:, basis.tind]
    centers = vertices.mean(axis=1)
    radii = np.linalg.norm(vertices - centers[:, None, :], axis=0).max(axis=0)
    scale = float(np.max(np.ptp(vertices.reshape(dim, -1), axis=1)))
    coordinate_scale = max(scale, float(np.max(np.abs(vertices))))
    physical_tolerance = 64.0 * np.finfo(float).eps * coordinate_scale
    affine_geometry = geometry_cells.shape[0] == dim + 1
    if dim == 2 and geometry_cells.shape[0] == 6:
        pairs = ((0, 1), (1, 2), (0, 2))
        expected = np.stack([
            0.5 * (vertices[:, i] + vertices[:, j]) for i, j in pairs
        ], axis=1)
        affine_geometry = np.all(
            np.abs(basis._mesh.p[:, geometry_cells[3:6]] - expected)
            <= physical_tolerance
        )
    elif dim == 3 and geometry_cells.shape[0] == 10:
        pairs = ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))
        expected = np.stack([
            0.5 * (vertices[:, i] + vertices[:, j]) for i, j in pairs
        ], axis=1)
        affine_geometry = np.all(
            np.abs(basis._mesh.p[:, geometry_cells[4:10]] - expected)
            <= physical_tolerance
        )
    if not affine_geometry:
        raise NotImplementedError(
            "point evaluation currently requires affine simplex geometry"
        )
    tree = cKDTree(centers.T)
    reference_tolerance = 1.0e-10
    max_radius = float(radii.max(initial=0.0)) + physical_tolerance

    for point_id, point in enumerate(points.T):
        candidates = tree.query_ball_point(point, max_radius)
        hits = []
        for local_cell in candidates:
            cell_vertices = vertices[:, :, local_cell]
            if np.any(point < cell_vertices.min(axis=1) - physical_tolerance):
                continue
            if np.any(point > cell_vertices.max(axis=1) + physical_tolerance):
                continue
            jacobian = cell_vertices[:, 1:] - cell_vertices[:, :1]
            reference = np.linalg.solve(jacobian, point - cell_vertices[:, 0])
            bary = np.concatenate(([1.0 - reference.sum()], reference))
            if np.min(bary) < -reference_tolerance or np.max(bary) > 1.0 + reference_tolerance:
                continue
            hits.append((int(basis.tind[local_cell]), local_cell, reference, bary))

        if not hits:
            outside_mask[point_id] = True
            continue
        hits.sort(key=lambda hit: hit[0])
        global_cell, local_cell, reference, bary = hits[0]
        cell_ids[point_id] = global_cell
        references[:, point_id] = reference
        boundary[point_id] = bool(np.any(np.abs(bary) <= reference_tolerance))
        ambiguous[point_id] = len(hits) > 1

        shape, reference_gradient = _shape_and_gradient(basis, reference)
        if len(shape) != local_nodes:
            raise RuntimeError("element shape and DOF counts do not match")
        jacobian = vertices[:, 1:, local_cell] - vertices[:, :1, local_cell]
        physical_gradient = reference_gradient @ np.linalg.inv(jacobian)
        local_coefficients = coefficients[
            basis.element_dofs[:, local_cell]
        ].reshape(local_nodes, components)
        value[:, point_id] = shape @ local_coefficients
        gradient[:, :, point_id] = np.einsum(
            "nd,nc->cd", physical_gradient, local_coefficients
        )

    if np.any(outside_mask) and outside == "error":
        point_id = int(np.flatnonzero(outside_mask)[0])
        raise ValueError(
            f"point {point_id} {points[:, point_id].tolist()} lies outside the basis mesh"
        )
    if components == 1:
        value = value[0]
        gradient = gradient[0]
    return PointEvaluation(
        value=value,
        grad=gradient,
        cell_ids=cell_ids,
        reference_coordinates=references,
        on_boundary=boundary,
        ambiguous=ambiguous,
        outside=outside_mask,
    )


__all__ = ["PointEvaluation", "evaluate_basis_at"]
