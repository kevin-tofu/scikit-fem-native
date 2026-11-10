#!/usr/bin/env python3
"""Transfer a forming checkpoint to a refined or graded contact mesh.

Nested uniform refinement preserves each parent's piecewise-constant
quadrature history in its children. A change in the graded x partition maps
history from the source triangle containing each target centroid. Both modes
interpolate the continuous P1 displacement field and remap contact reactions.
This is intended for same-model continuation studies, not for comparing
independent mesh-converged solutions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import skfemntv


def _base_mesh(
    nx: int, nz: int, half_length: float, mesh_mode: str,
) -> skfemntv.MeshTri:
    if mesh_mode == "default":
        n_punch_die = max(1, int(round(0.45 * nx)))
        n_holder = max(1, int(round(0.35 * nx)))
        n_flange = nx - n_punch_die - n_holder
        if n_flange < 1:
            n_flange = 1
            n_punch_die = max(1, int(round(0.4 * (nx - 1))))
            n_holder = nx - n_punch_die - n_flange
        x_nodes = np.r_[
            np.linspace(0., 36., n_punch_die + 1),
            np.linspace(36., 91., n_holder + 1)[1:],
            np.linspace(91., half_length, n_flange + 1)[1:],
        ]
    else:
        if mesh_mode in {
            "refine-holder-return", "refine-holder-return-punch-arc",
        }:
            bounds = np.array([
                0., 18., 25., 29.5, 30.4, 32.2, 33.5, 36., 40., 48.,
                60., 69.5, 80., 91., half_length,
            ])
            weights = np.array([
                2, 1, 0, 0, 1, 0, 0, 1, 2, 2, 1, 1, 1, 9,
            ], dtype=float)
            weights /= float(np.sum(weights))
        elif mesh_mode == "refine-tool-contacts":
            bounds = np.array([
                0., 18., 25., 29.5, 30.4, 32.2, 33.5, 36., 40., 48., 91.,
                half_length,
            ])
            weights = np.array([
                0.12, 0.16, 0.08, 0.07, 0.12, 0.07,
                0.06, 0.12, 0.09, 0.07, 0.04,
            ])
        else:
            raise ValueError(f"unsupported mesh mode: {mesh_mode}")
        exact = (nx - len(weights)) * weights
        counts = 1 + np.floor(exact).astype(int)
        remainder = nx - int(np.sum(counts))
        counts[np.argsort(-(exact - np.floor(exact)))[:remainder]] += 1
        x_nodes = np.concatenate([
            np.linspace(left, right, int(cells) + 1)[:-1]
            for left, right, cells in zip(bounds[:-1], bounds[1:], counts)
        ] + [bounds[-1:]])
        if mesh_mode == "refine-holder-return-punch-arc":
            left = x_nodes[:-1]
            right = x_nodes[1:]
            forming_contact_edges = (
                (left >= 18. - 1e-12) & (right <= 36. + 1e-12)
            )
            x_nodes = np.sort(np.r_[
                x_nodes,
                0.5 * (left[forming_contact_edges]
                       + right[forming_contact_edges]),
            ])
    material_file = Path(__file__).with_name("numisheet93_draw_bending_reference.json")
    reference = json.loads(material_file.read_text(encoding="utf-8"))
    thickness = float(reference["blank"]["thickness"])
    z_nodes = np.linspace(-0.5 * thickness, 0.5 * thickness, nz + 1)
    return skfemntv.MeshTri.init_tensor(x_nodes, z_nodes)


def _signature(mesh: skfemntv.MeshTri) -> str:
    digest = hashlib.sha256(b"numisheet93-tri-mesh-v1")
    for value in (mesh.p, mesh.t):
        contiguous = np.ascontiguousarray(value)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def _boundary_samples(
    mesh: skfemntv.MeshTri,
    displacement: np.ndarray,
    z_value: float,
    x_min: float,
    x_max: float,
    points_per_facet: int,
) -> np.ndarray:
    """Return deformed x locations in the contact-row order."""
    x_reference, z_reference = mesh.p
    nodes = np.flatnonzero(np.isclose(z_reference, z_value))
    nodes = nodes[np.argsort(x_reference[nodes])]
    x_current = x_reference + displacement[0::2]
    gauss_points, _ = np.polynomial.legendre.leggauss(points_per_facet)
    locations: list[float] = []
    for left, right in zip(nodes[:-1], nodes[1:]):
        xa, xb = x_reference[[left, right]]
        if xa < x_min - 1e-10 or xb > x_max + 1e-10:
            continue
        for point in gauss_points:
            shape_left = 0.5 * (1. - point)
            shape_right = 0.5 * (1. + point)
            location = (
                shape_left * x_current[left] + shape_right * x_current[right]
            )
            locations.append(float(location))
    return np.asarray(locations)


def _remap_contact_force_shares(
    source_multipliers: np.ndarray,
    source_locations: np.ndarray,
    target_locations: np.ndarray,
) -> np.ndarray:
    """Move each integrated contact reaction to its nearest new sample."""
    if source_multipliers.shape != source_locations.shape:
        raise ValueError("source contact multipliers do not match source rows")
    if source_locations.size == 0 or target_locations.size == 0:
        return np.zeros(target_locations.size)
    mapped = np.zeros(target_locations.size)
    active_source = np.flatnonzero(np.abs(source_multipliers) > 1e-12)
    for source_row in active_source:
        target_row = int(np.argmin(np.abs(
            target_locations - source_locations[source_row]
        )))
        mapped[target_row] += source_multipliers[source_row]
    return mapped


def transfer_checkpoint(
    source: Path,
    destination: Path,
    level: int,
    *,
    contact_collocation_points_per_facet: int | None = None,
    holder_contact_collocation_points_per_facet: int | None = None,
    target_mesh_mode: str | None = None,
) -> dict[str, object]:
    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    metadata = json.loads(str(arrays["metadata_json"].item()))
    old_level = int(metadata.get("mesh_refinement_level", 0))
    if (metadata.get("refine_holder_return", False)
            and metadata.get("refine_punch_arc", False)):
        source_mode = "refine-holder-return-punch-arc"
    elif metadata.get("refine_holder_return", False):
        source_mode = "refine-holder-return"
    elif metadata.get("refine_tool_contacts", False):
        source_mode = "refine-tool-contacts"
    else:
        source_mode = "default"
    target_mode = target_mesh_mode or source_mode
    if target_mode not in {
        "default", "refine-holder-return", "refine-tool-contacts",
        "refine-holder-return-punch-arc",
    }:
        raise ValueError(f"unsupported target mesh mode: {target_mode}")
    if level < old_level or (level == old_level and target_mode == source_mode):
        raise ValueError("target mesh must refine or change the source mesh mode")
    if (metadata.get("refine_punch_arc", False)
            and not metadata.get("refine_holder_return", False)):
        raise ValueError(
            "transferring a standalone punch-arc mesh is not supported"
        )
    if int(metadata.get("nx", 0)) < 14 or int(metadata.get("nz", 0)) < 2:
        raise ValueError("checkpoint mesh metadata is invalid")
    if int(arrays.get("springback_active", np.asarray(0)).item()):
        raise ValueError("transfer the formed checkpoint before starting springback")

    reference_file = Path(__file__).with_name("numisheet93_draw_bending_reference.json")
    reference = json.loads(reference_file.read_text(encoding="utf-8"))
    half_length = float(reference["blank"]["half_model_length"])
    old_base = _base_mesh(
        int(metadata["nx"]), int(metadata["nz"]), half_length, source_mode,
    )
    new_base = _base_mesh(
        int(metadata["nx"]), int(metadata["nz"]), half_length, target_mode,
    )
    old_mesh = old_base.refined(old_level)
    new_mesh = new_base.refined(level)
    source_signature = _signature(old_mesh)
    if metadata.get("mesh_signature") != source_signature:
        raise ValueError(
            "source checkpoint mesh signature does not match its declared mode"
        )
    old_contact_points = int(metadata.get("contact_collocation_points_per_facet", 2))
    old_holder_points = int(metadata.get(
        "holder_contact_collocation_points_per_facet", old_contact_points
    ))
    mortar_subfacets = int(metadata.get("mortar_subfacets_per_facet", 1))
    new_contact_points = int(
        contact_collocation_points_per_facet or old_contact_points
    )
    new_holder_points = int(
        holder_contact_collocation_points_per_facet or old_holder_points
    )
    if min(old_contact_points, old_holder_points,
           new_contact_points, new_holder_points) < 1:
        raise ValueError("contact collocation counts must be positive")
    # Mortar multiplier rows are per subfacet, while collocation counts set
    # the quadrature order used to evaluate each subfacet gap. Keep these
    # independent: conflating them changes the model on checkpoint transfer.
    mortar_contact = metadata.get("contact_method") == "mortar-p0"
    old_contact_rows_per_facet = (
        mortar_subfacets if mortar_contact else old_contact_points
    )
    old_holder_rows_per_facet = (
        mortar_subfacets if mortar_contact else old_holder_points
    )
    new_contact_rows_per_facet = old_contact_rows_per_facet
    new_holder_rows_per_facet = old_holder_rows_per_facet
    centroid_rule = (np.array([[1. / 3.], [1. / 3.]]), np.array([0.5]))
    element = skfemntv.ElementVector(skfemntv.ElementTriP1(), dim=2)
    old_basis = skfemntv.Basis(
        old_mesh, element, intorder=2, quadrature=centroid_rule,
    )
    new_basis = skfemntv.Basis(
        new_mesh, element, intorder=2, quadrature=centroid_rule,
    )

    old_displacement = np.asarray(arrays["displacement"], dtype=float)
    if old_displacement.shape != (old_basis.N,):
        raise ValueError("checkpoint displacement does not match its declared mesh")
    new_x_dofs = 2 * np.arange(new_basis.N // 2)
    new_z_dofs = new_x_dofs + 1
    interpolated = old_basis.interpolator(old_displacement)(
        new_basis.doflocs[:, new_x_dofs]
    )
    new_displacement = np.zeros(new_basis.N, dtype=float)
    new_displacement[new_x_dofs] = interpolated[0]
    new_displacement[new_z_dofs] = interpolated[1]

    old_history = np.asarray(arrays["committed"], dtype=float)
    if old_history.shape[0] != old_mesh.nelements:
        raise ValueError("checkpoint plastic history does not match its declared mesh")
    if source_mode == target_mode:
        parent_elements = np.arange(old_mesh.nelements, dtype=np.int64)
        for _ in range(level - old_level):
            parent_elements = np.tile(parent_elements, 4)
        if parent_elements.size != new_mesh.nelements:
            raise RuntimeError("uniform refinement parent map has the wrong size")
        new_history = old_history[parent_elements].copy()
    else:
        # The refined return mesh changes the x partition. Transfer the
        # piecewise-constant quadrature history from the source triangle that
        # contains each target centroid; retain nearest-cell fallback only
        # for roundoff at the outer boundary.
        source_triangles = np.asarray(old_mesh.p[:, old_mesh.t]).transpose(2, 1, 0)
        source_centroids = source_triangles.mean(axis=1)
        tree = cKDTree(source_centroids)
        target_centroids = np.asarray(new_mesh.p[:, new_mesh.t]).mean(axis=1).T
        _, candidates = tree.query(
            target_centroids, k=min(12, source_triangles.shape[0])
        )
        candidates = np.asarray(candidates)
        if candidates.ndim == 1:
            candidates = candidates[:, None]
        source_elements = np.empty(new_mesh.nelements, dtype=np.int64)
        for target_index, point in enumerate(target_centroids):
            found = None
            for candidate in candidates[target_index]:
                vertices = source_triangles[int(candidate)]
                edges = np.column_stack((vertices[1] - vertices[0], vertices[2] - vertices[0]))
                coordinates = np.linalg.solve(edges, point - vertices[0])
                if (coordinates[0] >= -1e-10 and coordinates[1] >= -1e-10
                        and coordinates.sum() <= 1. + 1e-10):
                    found = int(candidate)
                    break
            source_elements[target_index] = (
                int(candidates[target_index, 0]) if found is None else found
            )
        new_history = old_history[source_elements].copy()
    arrays["displacement"] = new_displacement
    arrays["committed"] = new_history
    arrays.pop("previous_displacement", None)
    arrays.pop("previous_holder_z", None)
    arrays.pop("previous_stroke", None)
    old_thickness = float(reference["blank"]["thickness"])
    contact_regions = (
        (0.5 * old_thickness, 0., 27., old_contact_rows_per_facet),
        (-0.5 * old_thickness, 25., half_length, old_contact_rows_per_facet),
        (0.5 * old_thickness, 30., 88., old_holder_rows_per_facet),
    )
    target_contact_regions = (
        (0.5 * old_thickness, 0., 27., new_contact_rows_per_facet),
        (-0.5 * old_thickness, 25., half_length, new_contact_rows_per_facet),
        (0.5 * old_thickness, 30., 88., new_holder_rows_per_facet),
    )
    old_samples = [
        _boundary_samples(old_mesh, old_displacement, z, x0, x1, count)
        for z, x0, x1, count in contact_regions
    ]
    new_samples = [
        _boundary_samples(new_mesh, new_displacement, z, x0, x1, count)
        for z, x0, x1, count in target_contact_regions
    ]
    saved_contact = np.asarray(arrays.get("contact_multipliers", np.zeros(0)))
    saved_friction = np.asarray(arrays.get("friction_multipliers", np.zeros(0)))
    old_contact_count = sum(locations.size for locations in old_samples)
    new_contact_count = sum(locations.size for locations in new_samples)
    old_equality_count = saved_contact.size - old_contact_count
    new_symmetry_nodes = np.count_nonzero(np.isclose(new_mesh.p[0], 0.))
    new_equality_count = int(new_symmetry_nodes) + int(
        metadata.get("holder_position_control_mm") is not None
    )
    if old_equality_count < 0:
        raise ValueError("checkpoint contact multiplier count is inconsistent")

    def remap_rows(values: np.ndarray) -> np.ndarray:
        if values.shape != (saved_contact.size,):
            return np.zeros(new_equality_count + new_contact_count)
        mapped = np.zeros(new_equality_count + new_contact_count)
        old_symmetry_nodes = np.flatnonzero(np.isclose(old_mesh.p[0], 0.))
        new_symmetry_nodes = np.flatnonzero(np.isclose(new_mesh.p[0], 0.))
        old_symmetry_nodes = old_symmetry_nodes[
            np.argsort(old_mesh.p[1, old_symmetry_nodes])
        ]
        new_symmetry_nodes = new_symmetry_nodes[
            np.argsort(new_mesh.p[1, new_symmetry_nodes])
        ]
        old_symmetry_count = old_symmetry_nodes.size
        new_symmetry_count = new_symmetry_nodes.size
        has_holder_position_equality = (
            metadata.get("holder_position_control_mm") is not None
        )
        if old_equality_count != old_symmetry_count + int(
            has_holder_position_equality
        ):
            raise ValueError("checkpoint equality multiplier layout is inconsistent")
        # Transfer the existing symmetry reactions to the coincident nodes on
        # the nested refined edge. Mid-edge nodes start at zero and are
        # equilibrated by the first Newton solve.
        for source_row, source_node in enumerate(old_symmetry_nodes):
            target_row = int(np.argmin(np.abs(
                new_mesh.p[1, new_symmetry_nodes] - old_mesh.p[1, source_node]
            )))
            mapped[target_row] = values[source_row]
        if has_holder_position_equality:
            mapped[new_symmetry_count] = values[old_symmetry_count]
        target_offset = new_equality_count
        source_offset = old_equality_count
        for old_x, new_x in zip(old_samples, new_samples):
            old_slice = values[source_offset:source_offset + old_x.size]
            mapped[target_offset:target_offset + new_x.size] = (
                _remap_contact_force_shares(old_slice, old_x, new_x)
            )
            source_offset += old_x.size
            target_offset += new_x.size
        return mapped

    arrays["contact_multipliers"] = remap_rows(saved_contact)
    if saved_friction.shape == saved_contact.shape:
        arrays["friction_multipliers"] = remap_rows(saved_friction)
    else:
        arrays["friction_multipliers"] = np.zeros(
            new_equality_count + new_contact_count
        )
    metadata["contact_collocation_points_per_facet"] = new_contact_points
    metadata["holder_contact_collocation_points_per_facet"] = new_holder_points
    metadata["contact_row_count"] = new_equality_count + new_contact_count
    metadata["contact_multiplier_mesh_refinement_level"] = level
    metadata["mesh_refinement_level"] = level
    metadata["refine_holder_return"] = target_mode in {
        "refine-holder-return", "refine-holder-return-punch-arc",
    }
    metadata["refine_punch_arc"] = (
        target_mode == "refine-holder-return-punch-arc"
    )
    metadata["refine_tool_contacts"] = target_mode == "refine-tool-contacts"
    metadata["mesh_signature"] = _signature(new_mesh)
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(destination)
    return {
        "source": str(source),
        "destination": str(destination),
        "old_elements": int(old_mesh.nelements),
        "new_elements": int(new_mesh.nelements),
        "old_dofs": int(old_basis.N),
        "new_dofs": int(new_basis.N),
        "refinement_level": level,
        "source_mesh_mode": source_mode,
        "target_mesh_mode": target_mode,
        "history_transfer": (
            "nested-parent-copy" if source_mode == target_mode
            else "containing-source-element-at-target-centroid"
        ),
        "accepted_stroke_mm": float(arrays["accepted_stroke"].item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--contact-collocation-points-per-facet", type=int)
    parser.add_argument("--holder-contact-collocation-points-per-facet", type=int)
    parser.add_argument(
        "--target-mesh-mode",
        choices=(
            "default", "refine-holder-return", "refine-tool-contacts",
            "refine-holder-return-punch-arc",
        ),
        help="change the graded x partition while projecting displacement/history",
    )
    args = parser.parse_args()
    print(json.dumps(transfer_checkpoint(
        args.source,
        args.destination,
        args.level,
        contact_collocation_points_per_facet=(
            args.contact_collocation_points_per_facet
        ),
        holder_contact_collocation_points_per_facet=(
            args.holder_contact_collocation_points_per_facet
        ),
        target_mesh_mode=args.target_mesh_mode,
    ), indent=2))


if __name__ == "__main__":
    main()
