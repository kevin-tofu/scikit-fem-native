"""Paris-law fatigue growth for the tutorial's straight Mode-I edge crack.

Run from the repository root with::

    PYTHONPATH=python python tutorials/fracture_mode_i/grow_edge_crack_fatigue.py \
      --output-dir tutorials/fracture_mode_i/results/fatigue

The equilibrium and XFEM basis construction are shared with
``grow_edge_crack.py``.  Only the propagation rule changes: a linear-elastic
load range gives ``Delta K = (1 - R) K_max`` and Paris' law converts the crack
increment into a cycle increment.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from tutorials.fracture_mode_i.grow_edge_crack import solve_growth_state


@dataclass(frozen=True)
class FatigueStep:
    step: int
    crack_tip_x: float
    k_max: float
    delta_k: float
    paris_da_d_cycles: float
    cycles_this_increment: float
    cumulative_cycles: float
    threshold_met: bool
    advanced: bool
    equilibrium_residual_l2: float


def run_fatigue(
    *,
    resolution: int,
    initial_tip_x: float,
    growth_step: float,
    maximum_steps: int,
    applied_peak_strain: float,
    load_ratio: float,
    paris_c: float,
    paris_m: float,
    threshold_delta_k: float,
):
    """Run a quasistatic Paris-law sequence on one fixed background mesh."""
    if not 0.0 <= load_ratio < 1.0:
        raise ValueError("load_ratio must be in [0, 1)")
    if paris_c <= 0.0 or paris_m <= 0.0 or threshold_delta_k < 0.0:
        raise ValueError("Paris parameters must be positive and threshold nonnegative")

    tip_x = initial_tip_x
    cumulative_cycles = 0.0
    records: list[FatigueStep] = []
    states = []
    for step in range(maximum_steps + 1):
        k_max, fields, _, residual = solve_growth_state(
            resolution,
            tip_x,
            crack_y=0.037,
            growth_step=growth_step,
            young_modulus=1.0,
            poisson_ratio=0.3,
            applied_strain=applied_peak_strain,
        )
        delta_k = max(0.0, (1.0 - load_ratio) * k_max)
        threshold_met = delta_k > threshold_delta_k
        growth_rate = paris_c * delta_k**paris_m if threshold_met else 0.0
        cycles = growth_step / growth_rate if growth_rate > 0.0 else 0.0
        advanced = threshold_met and step < maximum_steps
        # The row's cycle count belongs to this solved crack length. The
        # increment is accumulated only for the next state.
        records.append(FatigueStep(
            step=step,
            crack_tip_x=tip_x,
            k_max=float(k_max),
            delta_k=float(delta_k),
            paris_da_d_cycles=float(growth_rate),
            cycles_this_increment=float(cycles if advanced else 0.0),
            cumulative_cycles=float(cumulative_cycles),
            threshold_met=threshold_met,
            advanced=advanced,
            equilibrium_residual_l2=float(residual),
        ))
        states.append(fields)
        print(
            f"step={step:02d} tip_x={tip_x:+.4f} Kmax={k_max:.6e} "
            f"deltaK={delta_k:.6e} da/dN={growth_rate:.6e} "
            f"dN={records[-1].cycles_this_increment:.6e} "
            f"N={cumulative_cycles:.6e} advanced={str(advanced).lower()} "
            f"free_residual={residual:.3e}",
            flush=True,
        )
        if not advanced:
            break
        cumulative_cycles += cycles
        tip_x += growth_step
    return records, states


def write_results(output_dir: Path, records: list[FatigueStep], states) -> None:
    import matplotlib.pyplot as plt
    import meshio

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "fatigue-growth.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=tuple(FatigueStep.__dataclass_fields__),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)

    final_mesh = None
    for index, fields in enumerate(states):
        n_nodes = len(fields.points)
        vector_solution = fields.displacement.reshape(-1, 2)
        points = np.vstack((
            np.column_stack((fields.points, np.zeros(n_nodes))),
            np.array([
                [-1.0, fields.crack_y, 0.0],
                [fields.crack_tip_x, fields.crack_y, 0.0],
            ]),
        ))
        final_mesh = meshio.Mesh(
            points=points,
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
            output_dir / f"fatigue-step-{index:03d}.vtu",
            final_mesh,
            file_format="vtu",
        )
    if final_mesh is not None:
        meshio.write(output_dir / "fatigue-final.vtu", final_mesh, file_format="vtu")

    tip = np.asarray([record.crack_tip_x for record in records])
    delta_k = np.asarray([record.delta_k for record in records])
    cycles = np.asarray([record.cumulative_cycles for record in records])
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(tip, delta_k, "o-")
    axes[0].set(xlabel="crack tip $x$", ylabel="$\\Delta K$", title="Load-range driving force")
    axes[1].plot(cycles / 1.0e5, tip, "o-")
    axes[1].set(
        xlabel="cycles $N$ [$10^5$ cycles]",
        ylabel="crack tip $x$",
        title="Paris-law growth",
    )
    for axis in axes:
        axis.grid(True, which="both", alpha=0.25)
    figure.savefig(output_dir / "fatigue-growth-history.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=24)
    parser.add_argument("--initial-tip-x", type=float, default=-0.15)
    parser.add_argument("--growth-step", type=float, default=0.12)
    parser.add_argument("--maximum-steps", type=int, default=4)
    parser.add_argument("--applied-peak-strain", type=float, default=0.01)
    parser.add_argument("--load-ratio", type=float, default=0.1)
    parser.add_argument("--paris-c", type=float, default=1.0)
    parser.add_argument("--paris-m", type=float, default=3.0)
    parser.add_argument("--threshold-delta-k", type=float, default=0.005)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.resolution < 12 or args.resolution % 2:
        parser.error("--resolution must be an even integer >= 12")
    if args.growth_step <= 0.0 or args.maximum_steps < 0:
        parser.error("growth step must be positive and maximum steps nonnegative")
    records, states = run_fatigue(
        resolution=args.resolution,
        initial_tip_x=args.initial_tip_x,
        growth_step=args.growth_step,
        maximum_steps=args.maximum_steps,
        applied_peak_strain=args.applied_peak_strain,
        load_ratio=args.load_ratio,
        paris_c=args.paris_c,
        paris_m=args.paris_m,
        threshold_delta_k=args.threshold_delta_k,
    )
    if args.output_dir:
        write_results(args.output_dir, records, states)


if __name__ == "__main__":
    main()
