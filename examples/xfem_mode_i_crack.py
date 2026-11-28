"""Solve a fixed edge crack in Mode I with shifted-Heaviside XFEM.

Run from the repository root with::

    PYTHONPATH=packages/skfem-native/python python \
      packages/skfem-native/examples/xfem_mode_i_crack.py \
      --output-dir packages/skfem-native/tutorials/mode-i/results

The square contains a horizontal edge crack from the left boundary to an
interior tip deliberately offset from the background-mesh rows.  The crack
does not split or conform the mesh.
The vector displacement field uses shifted-Heaviside enrichment along the
crack and four square-root branch functions around its tip.  Exact Mode-I
Williams displacement is imposed on the outer boundary; the computed field is
compared with that solution and the stress field is used to recover K_I.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

import skfemntv
from skfemntv.helpers import ddot, div, sym_grad


@dataclass(frozen=True)
class ModeIResult:
    resolution: int
    triangles: int
    scalar_dofs: int
    vector_dofs: int
    crack_enriched_nodes: int
    tip_enriched_nodes: int
    displacement_l2_error: float
    displacement_l2_relative_error: float
    imposed_stress_intensity: float
    recovered_stress_intensity: float
    stress_intensity_relative_error: float


@dataclass(frozen=True)
class ModeIFields:
    points: np.ndarray
    triangles: np.ndarray
    solution: np.ndarray
    enriched_dof: np.ndarray
    crack_level_set: np.ndarray
    classification: np.ndarray
    crack_nodes: np.ndarray
    boundary_nodes: np.ndarray
    tip: np.ndarray


def _exact_mode_i_displacement(
    points: np.ndarray,
    *,
    stress_intensity: float,
    lame_mu: float,
    poisson_ratio: float,
    tip: np.ndarray,
    theta: np.ndarray | None = None,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    relative = points - np.asarray(tip, dtype=np.float64)
    x, y = relative[:, 0], relative[:, 1]
    radius = np.hypot(x, y)
    theta = np.arctan2(y, x) if theta is None else np.asarray(theta)
    kappa = 3.0 - 4.0 * poisson_ratio  # plane strain
    amplitude = (
        stress_intensity
        / (2.0 * lame_mu)
        * np.sqrt(radius / (2.0 * np.pi))
        * (kappa - np.cos(theta))
    )
    return np.column_stack((
        amplitude * np.cos(0.5 * theta),
        amplitude * np.sin(0.5 * theta),
    ))


def _exact_mode_i_sigma_yy(
    points: np.ndarray, *, stress_intensity: float, tip: np.ndarray
) -> np.ndarray:
    relative = points - np.asarray(tip, dtype=np.float64)
    x, y = relative[:, 0], relative[:, 1]
    radius = np.hypot(x, y)
    theta = np.arctan2(y, x)
    return (
        stress_intensity
        / np.sqrt(2.0 * np.pi * radius)
        * np.cos(0.5 * theta)
        * (1.0 + np.sin(0.5 * theta) * np.sin(1.5 * theta))
    )


def solve_mode_i(resolution: int) -> tuple[ModeIResult, ModeIFields]:
    if resolution < 8 or resolution % 2:
        raise ValueError("resolution must be an even integer >= 8")
    coordinates = np.linspace(-1.0, 1.0, resolution + 1)
    mesh = skfemntv.MeshTri.init_tensor(coordinates, coordinates)
    # A mild, piecewise-linear vertical warp keeps the tip at a mesh node but
    # makes the horizontal crack cross element interiors behind the tip.
    mesh.p[1] += 0.037 * (1.0 - np.abs(mesh.p[0]))
    scalar_parent = skfemntv.Basis(
        mesh,
        skfemntv.ElementVector(skfemntv.ElementTriP1(), dim=1),
        intorder=4,
    )
    tip = np.array([0.0, 0.037])
    crack_level_set = skfemntv.LevelSet(
        lambda x: x[1] - tip[1], tolerance=1.0e-12
    )
    classification = crack_level_set.classify(mesh)
    spacing = 2.0 / resolution
    cell_centers = mesh.p[:, mesh.t].mean(axis=1)
    crack_cells = np.asarray(classification.cut)[
        cell_centers[0, classification.cut] < tip[0]
    ]
    crack_nodes = np.unique(mesh.t[:, crack_cells])
    tip_radius = 0.55
    inside = skfemntv.ModeICrackTipCutBasis(
        scalar_parent,
        crack_level_set,
        side="inside",
        tip=tip,
        tip_radius=tip_radius,
        intorder=4,
        classification=classification,
        enrichment_nodes=crack_nodes,
    )
    outside = skfemntv.ModeICrackTipCutBasis(
        scalar_parent,
        crack_level_set,
        side="outside",
        tip=tip,
        tip_radius=tip_radius,
        intorder=4,
        classification=classification,
        enrichment_nodes=crack_nodes,
    )
    inside_vector = skfemntv.ShiftedHeavisideVectorCutBasis(
        inside, components=2
    )
    outside_vector = skfemntv.ShiftedHeavisideVectorCutBasis(
        outside, components=2
    )
    if inside.N != outside.N or not np.array_equal(
        inside.enriched_dof, outside.enriched_dof
    ) or not np.array_equal(inside.tip_dof, outside.tip_dof):
        raise RuntimeError("the two crack-side bases do not share a DOF map")

    young_modulus = 1.0
    poisson_ratio = 0.3
    lame_mu = young_modulus / (2.0 * (1.0 + poisson_ratio))
    lame_lambda = (
        young_modulus * poisson_ratio
        / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    )
    stress_intensity = 1.0

    @skfemntv.BilinearForm
    def plane_strain_elasticity(u, v, w):
        return (
            2.0 * lame_mu * ddot(sym_grad(u), sym_grad(v))
            + lame_lambda * div(u) * div(v)
        )

    stiffness = csr_matrix(skfemntv.asm(plane_strain_elasticity, inside_vector))
    stiffness += skfemntv.asm(plane_strain_elasticity, outside_vector)
    rhs = np.zeros(inside_vector.N)

    boundary_nodes = mesh.boundary_nodes()
    nodal_level_set = crack_level_set.values(mesh)
    nodal_sign = np.where(nodal_level_set < 0.0, -1.0, 1.0)
    coordinates_by_node = mesh.p.T
    local_coordinates = coordinates_by_node - tip
    nodal_theta = np.arctan2(
        local_coordinates[:, 1], local_coordinates[:, 0]
    )
    positive_theta = nodal_theta.copy()
    positive_theta[
        (local_coordinates[:, 0] < 0.0)
        & (local_coordinates[:, 1] < 0.0)
    ] += 2.0 * np.pi
    positive_theta[
        (local_coordinates[:, 0] < 0.0)
        & (local_coordinates[:, 1] == 0.0)
    ] = np.pi
    negative_theta = nodal_theta.copy()
    negative_theta[
        (local_coordinates[:, 0] < 0.0)
        & (local_coordinates[:, 1] > 0.0)
    ] -= 2.0 * np.pi
    negative_theta[
        (local_coordinates[:, 0] < 0.0)
        & (local_coordinates[:, 1] == 0.0)
    ] = -np.pi
    nodal_positive = _exact_mode_i_displacement(
        coordinates_by_node,
        stress_intensity=stress_intensity,
        lame_mu=lame_mu,
        poisson_ratio=poisson_ratio,
        tip=tip,
        theta=positive_theta,
    )
    nodal_negative = _exact_mode_i_displacement(
        coordinates_by_node,
        stress_intensity=stress_intensity,
        lame_mu=lame_mu,
        poisson_ratio=poisson_ratio,
        tip=tip,
        theta=negative_theta,
    )
    enriched_mask = inside.enriched_dof >= 0
    standard_boundary_values = np.where(
        (nodal_sign < 0.0)[:, None], nodal_negative, nodal_positive
    )
    fixed_dofs: list[int] = []
    fixed_values: list[float] = []
    for node in boundary_nodes:
        for component in range(2):
            fixed_dofs.append(2 * int(node) + component)
            fixed_values.append(float(standard_boundary_values[node, component]))
        if enriched_mask[node]:
            enrichment = int(inside.enriched_dof[node])
            jump_coefficient = 0.5 * (
                nodal_positive[node] - nodal_negative[node]
            )
            for component in range(2):
                fixed_dofs.append(2 * enrichment + component)
                fixed_values.append(float(jump_coefficient[component]))

    fixed = np.asarray(fixed_dofs, dtype=np.int64)
    fixed_value_array = np.asarray(fixed_values, dtype=np.float64)
    active = np.flatnonzero(stiffness.diagonal() > 0.0)
    free = np.setdiff1d(active, fixed, assume_unique=False)
    solution = np.zeros(inside_vector.N, dtype=np.float64)
    solution[fixed] = fixed_value_array
    reduced_rhs = rhs[free] - stiffness[free][:, fixed] @ solution[fixed]
    solution[free] = spsolve(stiffness[free][:, free], reduced_rhs)
    if not np.isfinite(solution).all():
        raise FloatingPointError("Mode-I XFEM solve returned non-finite values")

    stress_points: list[np.ndarray] = []
    stress_values: list[np.ndarray] = []
    stress_weights: list[np.ndarray] = []
    displacement_error_squared = 0.0
    displacement_reference_squared = 0.0
    for vector_basis in (inside_vector, outside_vector):
        field = vector_basis.interpolate(solution)
        exact_displacement = _exact_mode_i_displacement(
            vector_basis.points,
            stress_intensity=stress_intensity,
            lame_mu=lame_mu,
            poisson_ratio=poisson_ratio,
            tip=tip,
        )
        difference = field.value.T - exact_displacement
        displacement_error_squared += float(
            vector_basis.weights @ np.sum(difference**2, axis=1)
        )
        displacement_reference_squared += float(
            vector_basis.weights
            @ np.sum(exact_displacement**2, axis=1)
        )
        sigma_yy = lame_lambda * (
            field.grad[0, 0] + field.grad[1, 1]
        ) + 2.0 * lame_mu * field.grad[1, 1]
        stress_points.append(vector_basis.points)
        stress_values.append(sigma_yy)
        stress_weights.append(vector_basis.weights)

    all_stress_points = np.concatenate(stress_points)
    all_stress_values = np.concatenate(stress_values)
    all_stress_weights = np.concatenate(stress_weights)
    radius = np.linalg.norm(all_stress_points - tip, axis=1)
    fit_mask = (radius >= 0.20) & (radius <= 0.45)
    exact_unit_sigma = _exact_mode_i_sigma_yy(
        all_stress_points[fit_mask], stress_intensity=1.0, tip=tip
    )
    fit_weights = all_stress_weights[fit_mask]
    recovered_stress_intensity = float(
        (fit_weights @ (exact_unit_sigma * all_stress_values[fit_mask]))
        / (fit_weights @ (exact_unit_sigma**2))
    )
    result = ModeIResult(
        resolution=resolution,
        triangles=int(mesh.nelements),
        scalar_dofs=int(inside.N),
        vector_dofs=int(inside_vector.N),
        crack_enriched_nodes=len(inside.enriched_nodes),
        tip_enriched_nodes=len(inside.tip_enriched_nodes),
        displacement_l2_error=float(np.sqrt(displacement_error_squared)),
        displacement_l2_relative_error=float(
            np.sqrt(displacement_error_squared)
            / np.sqrt(displacement_reference_squared)
        ),
        imposed_stress_intensity=stress_intensity,
        recovered_stress_intensity=recovered_stress_intensity,
        stress_intensity_relative_error=abs(
            recovered_stress_intensity - stress_intensity
        ) / stress_intensity,
    )
    fields = ModeIFields(
        points=mesh.p.T,
        triangles=mesh.t.T,
        solution=solution,
        enriched_dof=inside.enriched_dof,
        crack_level_set=nodal_level_set,
        classification=classification.labels,
        crack_nodes=crack_nodes,
        boundary_nodes=boundary_nodes,
        tip=tip,
    )
    return result, fields


def write_results(
    output_dir: Path, results: list[ModeIResult], fields: ModeIFields
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    import meshio

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "mode-i-convergence.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=tuple(ModeIResult.__dataclass_fields__),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)

    mesh = meshio.Mesh(
        points=np.column_stack((fields.points, np.zeros(len(fields.points)))),
        cells=[("triangle", fields.triangles)],
        point_data={
            "level_set": fields.crack_level_set,
            "mode_i_crack_segment": (
                (np.abs(fields.crack_level_set) < 1.0e-12)
                & (fields.points[:, 0] <= fields.tip[0])
            ).astype(np.int8),
        },
        cell_data={
            "cut_cell_classification": [fields.classification.astype(np.int32)]
        },
    )
    scalar_dofs = len(fields.points)
    vector_solution = fields.solution.reshape(-1, 2)
    positive = vector_solution[:scalar_dofs].copy()
    negative = positive.copy()
    for node, enriched_dof in enumerate(fields.enriched_dof):
        if enriched_dof >= 0:
            coefficient = vector_solution[enriched_dof]
            if fields.crack_level_set[node] >= 0.0:
                negative[node] -= 2.0 * coefficient
            else:
                positive[node] += 2.0 * coefficient
    mesh.point_data["displacement_positive_trace"] = np.column_stack(
        (positive, np.zeros(len(positive)))
    )
    mesh.point_data["displacement_negative_trace"] = np.column_stack(
        (negative, np.zeros(len(negative)))
    )
    mesh.point_data["displacement"] = mesh.point_data[
        "displacement_positive_trace"
    ]
    meshio.write(output_dir / "mode-i-crack.vtu", mesh, file_format="vtu")

    points = fields.points
    triangles = fields.triangles
    displacement = vector_solution[:scalar_dofs]
    triangle_centers_y = points[triangles, 1].mean(axis=1)
    triangulation = mtri.Triangulation(points[:, 0], points[:, 1], triangles)
    scale = 3.0
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for axis, side, values, mask in (
        (axes[0], "upper trace", positive, triangle_centers_y < fields.tip[1]),
        (axes[1], "lower trace", negative, triangle_centers_y > fields.tip[1]),
    ):
        deformed = points + scale * values
        side_plot = mtri.Triangulation(
            deformed[:, 0], deformed[:, 1], triangles
        )
        side_plot.set_mask(mask)
        image = axis.tripcolor(
            side_plot,
            values[:, 1],
            shading="gouraud",
            cmap="coolwarm",
        )
        axis.triplot(side_plot, color="#202830", linewidth=0.25, alpha=0.55)
        axis.plot(
            (-1.0, fields.tip[0]),
            (fields.tip[1], fields.tip[1]),
            color="black",
            linewidth=2.0,
        )
        axis.set_aspect("equal")
        axis.set_title(f"Mode I displacement, {side} (scale {scale:g})")
        axis.set_xlabel("deformed x")
        axis.set_ylabel("deformed y")
        figure.colorbar(image, ax=axis, label="$u_y$")
    figure.savefig(output_dir / "mode-i-displacement.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    resolution = np.asarray([result.resolution for result in results])
    displacement_error = np.asarray(
        [result.displacement_l2_error for result in results]
    )
    intensity_error = np.asarray(
        [result.stress_intensity_relative_error for result in results]
    )
    axes[0].loglog(2.0 / resolution, displacement_error, "o-")
    axes[0].set(xlabel="mesh size $h$", ylabel="$L^2$ displacement error")
    axes[1].semilogx(2.0 / resolution, intensity_error, "s-")
    axes[1].set(xlabel="mesh size $h$", ylabel="$|K_I^{FE}-K_I|/K_I$")
    for axis in axes:
        axis.invert_xaxis()
        axis.grid(True, which="both", alpha=0.25)
    figure.savefig(output_dir / "mode-i-convergence.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resolutions", type=int, nargs="+", default=(12, 20, 28)
    )
    parser.add_argument("--output-dir", type=Path)
    arguments = parser.parse_args()
    results: list[ModeIResult] = []
    final_fields: ModeIFields | None = None
    print("2D edge-crack Mode-I XFEM / plane strain")
    print(
        "resolution  triangles  scalar_dofs  vector_dofs  "
        "L2_displacement_error  relative_L2_error  recovered_KI  relative_KI_error"
    )
    for resolution in arguments.resolutions:
        result, fields = solve_mode_i(resolution)
        results.append(result)
        final_fields = fields
        print(
            f"{result.resolution:10d}  {result.triangles:9d}  "
            f"{result.scalar_dofs:11d}  {result.vector_dofs:11d}  "
            f"{result.displacement_l2_error:21.8e}  "
            f"{result.displacement_l2_relative_error:16.6e}  "
            f"{result.recovered_stress_intensity:11.6f}  "
            f"{result.stress_intensity_relative_error:18.6e}"
        )
    if arguments.output_dir is not None and final_fields is not None:
        write_results(arguments.output_dir, results, final_fields)


if __name__ == "__main__":
    main()
