"""Load-controlled 2D Mode-I edge-crack growth with shifted-Heaviside XFEM.

The tutorial deliberately restricts propagation to a straight, symmetric
Mode-I path.  Each increment solves equilibrium on the unchanged background
mesh, estimates K_I from the near-tip stress field, and advances the level-set
tip by ``growth_step`` when K_I reaches the supplied toughness.

Run from the repository root with::

    PYTHONPATH=python python tutorials/fracture_mode_i/grow_edge_crack.py \
      --output-dir tutorials/fracture_mode_i/results
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
class GrowthStep:
    step: int
    crack_tip_x: float
    stress_intensity_k1: float
    toughness: float
    criterion_met: bool
    advanced: bool
    active_dofs: int
    enriched_nodes: int
    equilibrium_residual_l2: float


@dataclass(frozen=True)
class GrowthFields:
    points: np.ndarray
    triangles: np.ndarray
    displacement: np.ndarray
    crack_tip_x: float
    crack_y: float
    enriched_node_count: int
    level_set: np.ndarray
    classification: np.ndarray


def solve_growth_state(
    resolution: int,
    tip_x: float,
    *,
    crack_y: float,
    growth_step: float,
    young_modulus: float,
    poisson_ratio: float,
    applied_strain: float,
) -> tuple[float, GrowthFields, int, float]:
    """Solve one load-controlled state and recover K_I by stress fitting."""
    axis = np.linspace(-1.0, 1.0, resolution + 1)
    mesh = skfemntv.MeshTri.init_tensor(axis, axis)
    mesh.p[1] += 0.037 * (1.0 - np.abs(mesh.p[0]))
    parent = skfemntv.Basis(
        mesh,
        skfemntv.ElementVector(skfemntv.ElementTriP1(), dim=1),
        intorder=4,
    )
    level_set = skfemntv.LevelSet(
        lambda x: x[1] - crack_y, tolerance=1.0e-12
    )
    classification = level_set.classify(mesh)
    centers_x = mesh.p[0, mesh.t].mean(axis=0)
    crack_cells = np.asarray(classification.cut)[centers_x[classification.cut] < tip_x]
    touching_cells = np.asarray(classification.touching)[
        centers_x[classification.touching] < tip_x
    ]
    crack_nodes = np.unique(mesh.t[:, np.concatenate((crack_cells, touching_cells))])
    tip = np.array([tip_x, crack_y], dtype=np.float64)
    radius = max(2.5 * growth_step, 0.24)
    bases = tuple(
        skfemntv.ModeICrackTipCutBasis(
            parent,
            level_set,
            side=side,
            tip=tip,
            tip_radius=radius,
            intorder=4,
            classification=classification,
            enrichment_nodes=crack_nodes,
        )
        for side in ("inside", "outside")
    )
    vector_bases = tuple(
        skfemntv.ShiftedHeavisideVectorCutBasis(basis, components=2)
        for basis in bases
    )
    inside_vector, outside_vector = vector_bases
    if inside_vector.N != outside_vector.N:
        raise RuntimeError("crack-side bases have different global dimensions")

    lame_mu = young_modulus / (2.0 * (1.0 + poisson_ratio))
    lame_lambda = (
        young_modulus * poisson_ratio
        / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    )

    @skfemntv.BilinearForm
    def elasticity(u, v, w):
        return (
            2.0 * lame_mu * ddot(sym_grad(u), sym_grad(v))
            + lame_lambda * div(u) * div(v)
        )

    stiffness = csr_matrix(skfemntv.asm(elasticity, inside_vector))
    stiffness += skfemntv.asm(elasticity, outside_vector)
    rhs = np.zeros(inside_vector.N, dtype=np.float64)

    # Displacement-controlled remote tension on the horizontal boundaries.
    boundary = mesh.boundary_nodes()
    y = mesh.p[1, boundary]
    top = boundary[y > 0.95]
    bottom = boundary[y < -0.95]
    fixed_dofs = [2 * int(node) + 1 for node in top]
    fixed_values = [applied_strain for _ in top]
    fixed_dofs.extend(2 * int(node) + 1 for node in bottom)
    fixed_values.extend(-applied_strain for _ in bottom)
    # Remove horizontal rigid translation without constraining the crack mouth.
    anchor = int(np.argmin(np.sum((mesh.p.T - np.array([-1.0, -1.0])) ** 2, axis=1)))
    fixed_dofs.append(2 * anchor)
    fixed_values.append(0.0)
    fixed = np.asarray(fixed_dofs, dtype=np.int64)
    values = np.asarray(fixed_values, dtype=np.float64)
    if len(np.unique(fixed)) != len(fixed):
        raise RuntimeError("duplicate constrained DOFs")
    active = np.flatnonzero(stiffness.diagonal() > 0.0)
    free = np.setdiff1d(active, fixed, assume_unique=False)
    displacement = np.zeros(inside_vector.N, dtype=np.float64)
    displacement[fixed] = values
    reduced_rhs = rhs[free] - stiffness[free][:, fixed] @ values
    displacement[free] = spsolve(stiffness[free][:, free], reduced_rhs)
    residual = stiffness @ displacement - rhs
    free_residual = float(np.linalg.norm(residual[free]))
    if not np.isfinite(displacement).all() or not np.isfinite(free_residual):
        raise FloatingPointError("XFEM growth solve returned non-finite values")

    sample_points: list[np.ndarray] = []
    sample_stress: list[np.ndarray] = []
    sample_weights: list[np.ndarray] = []
    for basis in vector_bases:
        field = basis.interpolate(displacement)
        sigma_yy = lame_lambda * (field.grad[0, 0] + field.grad[1, 1])
        sigma_yy += 2.0 * lame_mu * field.grad[1, 1]
        sample_points.append(basis.points)
        sample_stress.append(sigma_yy)
        sample_weights.append(basis.weights)
    points = np.concatenate(sample_points)
    sigma = np.concatenate(sample_stress)
    weights = np.concatenate(sample_weights)
    relative = points - tip
    radial_distance = np.linalg.norm(relative, axis=1)
    theta = np.arctan2(relative[:, 1], relative[:, 0])
    fit = (radial_distance >= 1.5 * growth_step) & (radial_distance <= radius)
    fit &= relative[:, 0] > 0.0
    if np.count_nonzero(fit) < 8:
        raise RuntimeError("too few integration points for the K_I fit")
    unit_k_sigma = (
        np.cos(0.5 * theta[fit])
        * (1.0 + np.sin(0.5 * theta[fit]) * np.sin(1.5 * theta[fit]))
        / np.sqrt(2.0 * np.pi * radial_distance[fit])
    )
    denominator = float(weights[fit] @ (unit_k_sigma**2))
    k1 = float(weights[fit] @ (unit_k_sigma * sigma[fit]) / denominator)
    if not np.isfinite(k1):
        raise FloatingPointError("stress fit returned non-finite K_I")

    fields = GrowthFields(
        points=mesh.p.T.copy(),
        triangles=mesh.t.T.copy(),
        displacement=displacement.copy(),
        crack_tip_x=float(tip_x),
        crack_y=float(crack_y),
        enriched_node_count=len(crack_nodes),
        level_set=level_set.values(mesh).copy(),
        classification=np.asarray(classification.labels).copy(),
    )
    return k1, fields, len(active), free_residual


def run_growth(
    *,
    resolution: int,
    initial_tip_x: float,
    growth_step: float,
    toughness: float,
    maximum_steps: int,
    applied_strain: float,
) -> tuple[list[GrowthStep], list[GrowthFields]]:
    crack_y = 0.037
    records: list[GrowthStep] = []
    states: list[GrowthFields] = []
    tip_x = initial_tip_x
    for step in range(maximum_steps + 1):
        k1, fields, active_dofs, residual = solve_growth_state(
            resolution,
            tip_x,
            crack_y=crack_y,
            growth_step=growth_step,
            young_modulus=1.0,
            poisson_ratio=0.3,
            applied_strain=applied_strain,
        )
        criterion_met = k1 >= toughness
        advanced = criterion_met and step < maximum_steps
        records.append(GrowthStep(
            step=step,
            crack_tip_x=tip_x,
            stress_intensity_k1=k1,
            toughness=toughness,
            criterion_met=criterion_met,
            advanced=advanced,
            active_dofs=active_dofs,
            enriched_nodes=fields.enriched_node_count,
            equilibrium_residual_l2=residual,
        ))
        states.append(fields)
        print(
            f"step={step:02d} tip_x={tip_x:+.4f} K_I={k1:.6e} "
            f"K_IC={toughness:.6e} advanced={str(advanced).lower()} "
            f"free_residual={residual:.3e}",
            flush=True,
        )
        if not advanced:
            break
        tip_x += growth_step
    return records, states


def write_results(output_dir: Path, records: list[GrowthStep], states: list[GrowthFields]) -> None:
    import matplotlib.pyplot as plt
    import meshio

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "crack-growth.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=tuple(GrowthStep.__dataclass_fields__),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)
    for index, fields in enumerate(states):
        vector_solution = fields.displacement.reshape(-1, 2)
        n_nodes = len(fields.points)
        output_points = np.vstack((
            np.column_stack((fields.points, np.zeros(n_nodes))),
            np.array([
                [-1.0, fields.crack_y, 0.0],
                [fields.crack_tip_x, fields.crack_y, 0.0],
            ]),
        ))
        output_mesh = meshio.Mesh(
            points=output_points,
            cells=[
                ("triangle", fields.triangles),
                ("line", np.array([[n_nodes, n_nodes + 1]], dtype=np.int64)),
            ],
            point_data={
                "displacement": np.vstack((
                    np.column_stack((vector_solution[:n_nodes], np.zeros(n_nodes))),
                    np.zeros((2, 3)),
                )),
                "level_set": np.concatenate((fields.level_set, np.zeros(2))),
            },
            cell_data={
                "cut_cell_classification": [
                    fields.classification.astype(np.int32),
                    np.array([0], dtype=np.int32),
                ]
            },
        )
        meshio.write(
            output_dir / f"crack-growth-step-{index:03d}.vtu",
            output_mesh,
            file_format="vtu",
        )
    meshio.write(
        output_dir / "crack-growth-final.vtu",
        output_mesh,
        file_format="vtu",
    )

    steps = np.asarray([record.step for record in records])
    k1_values = np.asarray([record.stress_intensity_k1 for record in records])
    tips = np.asarray([record.crack_tip_x for record in records])
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(steps, k1_values, "o-", label="$K_I$")
    axes[0].axhline(records[0].toughness, color="tab:red", linestyle="--", label="$K_{IC}$")
    axes[0].set(xlabel="growth step", ylabel="$K_I$", title="Propagation criterion")
    axes[0].legend()
    axes[1].plot(steps, tips, "o-")
    axes[1].set(xlabel="growth step", ylabel="crack tip $x$", title="Crack advance")
    for axis in axes:
        axis.grid(True, alpha=0.25)
    figure.savefig(output_dir / "crack-growth-history.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=24)
    parser.add_argument("--initial-tip-x", type=float, default=-0.15)
    parser.add_argument("--growth-step", type=float, default=0.12)
    parser.add_argument("--toughness", type=float, default=0.012)
    parser.add_argument("--maximum-steps", type=int, default=4)
    parser.add_argument("--applied-strain", type=float, default=0.01)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.resolution < 12 or args.resolution % 2:
        parser.error("--resolution must be an even integer >= 12")
    if args.growth_step <= 0.0 or args.toughness <= 0.0 or args.maximum_steps < 0:
        parser.error("growth step/toughness must be positive and maximum steps nonnegative")
    results, fields = run_growth(
        resolution=args.resolution,
        initial_tip_x=args.initial_tip_x,
        growth_step=args.growth_step,
        toughness=args.toughness,
        maximum_steps=args.maximum_steps,
        applied_strain=args.applied_strain,
    )
    if args.output_dir:
        write_results(args.output_dir, results, fields)


if __name__ == "__main__":
    main()
