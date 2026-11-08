"""Export and plot the accepted load--stroke history in a forming checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skfemntv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output-prefix", type=Path, default=None)
    parser.add_argument(
        "--reference", type=Path,
        default=Path(__file__).with_name("numisheet93_draw_bending_reference.json"),
    )
    args = parser.parse_args()

    with np.load(args.checkpoint, allow_pickle=True) as archive:
        metadata = json.loads(str(archive["metadata_json"]))
        records = json.loads(str(archive["records_json"]))
        displacement = np.asarray(archive["displacement"], dtype=float)

    if not records:
        raise ValueError("checkpoint contains no accepted forming records")
    modeled_width = float(metadata["modeled_width_mm"])
    full_blank_width = 35.0
    # Contact force is integrated over the half-length 2D model and its
    # modeled out-of-plane width. Scale by length symmetry and blank width.
    full_blank_scale = 2.0 * full_blank_width / modeled_width
    output_prefix = args.output_prefix or args.checkpoint.with_name(
        args.checkpoint.stem + "-load-stroke-progress"
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_suffix(".csv")
    png_path = output_prefix.with_suffix(".png")
    thickness_path = output_prefix.with_name(output_prefix.name + "-thickness").with_suffix(".csv")

    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    blank = reference["blank"]
    initial_thickness = float(blank["thickness"])
    half_length = float(blank["half_model_length"])
    nx = int(metadata["nx"])
    nz = int(metadata["nz"])
    if metadata.get("refine_holder_return", False):
        bounds = np.array([
            0., 18., 25., 29.5, 30.4, 32.2, 33.5, 36., 40., 48.,
            60., 69.5, 80., 91., half_length,
        ])
        extra_cells = np.array([2, 1, 0, 0, 1, 0, 0, 1, 2, 2, 1, 1, 1, 9])
        weights = extra_cells / float(np.sum(extra_cells))
    elif metadata.get("refine_tool_contacts", False):
        bounds = np.array([
            0., 18., 25., 29.5, 30.4, 32.2, 33.5, 36., 40., 48., 91.,
            half_length,
        ])
        weights = np.array([
            0.12, 0.16, 0.08, 0.07, 0.12, 0.07,
            0.06, 0.12, 0.09, 0.07, 0.04,
        ])
    elif metadata.get("refine_holder_corner", False):
        bounds = np.array([
            0., 18., 25., 29.5, 30.4, 32.2, 33.5, 36., 91., half_length,
        ])
        weights = np.array([0.14, 0.18, 0.10, 0.08, 0.15, 0.08, 0.07, 0.14, 0.06])
    elif metadata.get("refine_punch_arc", False):
        bounds = np.array([0., 18., 25., 36., 91., half_length])
        weights = np.array([0.15, 0.24, 0.12, 0.24, 0.25])
    else:
        punch_cells = max(1, int(round(0.45 * nx)))
        holder_cells = max(1, int(round(0.35 * nx)))
        flange_cells = nx - punch_cells - holder_cells
        if flange_cells < 1:
            flange_cells = 1
            punch_cells = max(1, int(round(0.4 * (nx - 1))))
            holder_cells = nx - punch_cells - flange_cells
        regions = [(0., 36., punch_cells), (36., 91., holder_cells),
                   (91., half_length, flange_cells)]
        x_nodes = np.concatenate([
            np.linspace(left, right, cells + 1)[:-1]
            for left, right, cells in regions
        ] + [np.array([half_length])])
        bounds = None

    if bounds is not None:
        exact = (nx - (len(bounds) - 1)) * weights
        cells = 1 + np.floor(exact).astype(int)
        remainder = nx - int(np.sum(cells))
        cells[np.argsort(-(exact - np.floor(exact)))[:remainder]] += 1
        x_nodes = np.concatenate([
            np.linspace(left, right, int(count) + 1)[:-1]
            for left, right, count in zip(bounds[:-1], bounds[1:], cells)
        ] + [bounds[-1:]])

    refinement = int(metadata["mesh_refinement_level"])
    z_nodes = np.linspace(-0.5 * initial_thickness, 0.5 * initial_thickness, nz + 1)
    mesh = skfemntv.MeshTri.init_tensor(x_nodes, z_nodes)
    for _ in range(refinement):
        mesh = mesh.refined()
    node_x, node_z = mesh.p
    expected_dofs = 2 * node_x.size
    if displacement.size != expected_dofs:
        raise ValueError(
            f"checkpoint displacement has {displacement.size} DOFs; "
            f"reconstructed mesh expects {expected_dofs}"
        )

    thickness_profile = []
    dof_x = 2 * np.arange(node_x.size)
    dof_z = dof_x + 1
    for x_value in np.unique(node_x):
        column = np.flatnonzero(np.isclose(node_x, x_value))
        top_node = column[np.argmax(node_z[column])]
        bottom_node = column[np.argmin(node_z[column])]
        bottom = np.array([node_x[bottom_node], node_z[bottom_node]]) + np.array([
            displacement[dof_x[bottom_node]], displacement[dof_z[bottom_node]]
        ])
        top = np.array([node_x[top_node], node_z[top_node]]) + displacement[
            [dof_x[top_node], dof_z[top_node]]
        ]
        thickness_profile.append((float(x_value), float(np.linalg.norm(top - bottom))))

    with thickness_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "stroke_mm", "x_mm", "deformed_column_thickness_mm",
            "initial_thickness_mm", "thickness_change_percent",
        ])
        for x_value, current_thickness in thickness_profile:
            writer.writerow([
                records[-1]["stroke_mm"], x_value, current_thickness,
                initial_thickness, 100. * (current_thickness / initial_thickness - 1.),
            ])

    fields = [
        "step",
        "stroke_mm",
        "punch_force_N_half_model",
        "punch_force_N_full_blank_width_scaled",
        "equilibrium_residual_inf_N_per_mm",
        "equilibrium_tolerance_inf_N_per_mm",
        "maximum_contact_penetration_mm",
        "contact_complementarity_N_mm",
        "minimum_punch_gap_mm",
        "active_punch_points",
        "punch_contact_state",
        "newton_iterations",
        "max_equivalent_plastic_strain",
        "force_resolution_unresolved",
        "holder_jump_unresolved",
        "punch_detachment_event",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {field: record.get(field) for field in fields}
            if int(record.get("active_punch_points", 0)) > 0:
                row["punch_contact_state"] = "active"
            elif float(record.get("minimum_punch_gap_mm", 0.0)) > 0.01:
                row["punch_contact_state"] = "separated_over_0.01_mm"
            else:
                row["punch_contact_state"] = "inactive_near_contact"
            row["punch_force_N_full_blank_width_scaled"] = (
                float(record["punch_force_N_half_model"]) * full_blank_scale
            )
            writer.writerow(row)

    stroke = np.asarray([record["stroke_mm"] for record in records], dtype=float)
    force = np.asarray(
        [record["punch_force_N_half_model"] for record in records], dtype=float
    )
    unresolved = np.asarray(
        [record.get("force_resolution_unresolved", False) for record in records],
        dtype=bool,
    )
    separated = np.asarray(
        [
            int(record.get("active_punch_points", 0)) == 0
            and float(record.get("minimum_punch_gap_mm", 0.0)) > 0.01
            for record in records
        ],
        dtype=bool,
    )
    active_punch = np.asarray(
        [record.get("active_punch_points", 0) for record in records],
        dtype=float,
    )
    active_die = np.asarray(
        [record.get("active_die_points", 0) for record in records],
        dtype=float,
    )
    active_holder = np.asarray(
        [record.get("active_holder_points", 0) for record in records],
        dtype=float,
    )
    fig, (ax, contact_ax) = plt.subplots(
        2, 1, figsize=(8.2, 6.2), sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.0]},
        constrained_layout=True,
    )
    ax.plot(stroke, force * full_blank_scale, color="#1769aa", linewidth=1.6)
    if np.any(unresolved):
        ax.scatter(
            stroke[unresolved],
            force[unresolved] * full_blank_scale,
            color="#d1495b",
            s=14,
            label="Unresolved force jump",
            zorder=3,
        )
        ax.legend(loc="upper right", frameon=False)
    if np.any(separated):
        ax.scatter(
            stroke[separated],
            force[separated] * full_blank_scale,
            color="#f28e2b",
            marker="x",
            s=16,
            label="No active contact, gap > 0.01 mm",
            zorder=3,
        )
        ax.legend(loc="upper right", frameon=False)
    ax.set_xlabel("Punch stroke [mm]")
    ax.set_ylabel("Punch force, full blank width scale [N]")
    ax.set_title(
        "NUMISHEET '93 draw-bending progress "
        f"(accepted to {stroke[-1]:.3f} mm)"
    )
    ax.grid(True, alpha=0.28)
    contact_ax.step(
        stroke, active_punch, where="post", color="#1769aa",
        linewidth=1.2, label="Punch",
    )
    contact_ax.step(
        stroke, active_die, where="post", color="#e07a1f",
        linewidth=1.1, label="Die",
    )
    contact_ax.step(
        stroke, active_holder, where="post", color="#4b8b3b",
        linewidth=1.1, label="Holder",
    )
    contact_ax.set_xlabel("Punch stroke [mm]")
    contact_ax.set_ylabel("Active\npoints")
    contact_ax.set_ylim(bottom=-0.1)
    contact_ax.grid(True, alpha=0.28)
    contact_ax.legend(loc="upper right", frameon=False, ncol=3, fontsize=8)
    ax.text(
        0.02,
        0.98,
        f"{metadata['nx']}x{metadata['nz']} cells; finite-strain J2\n"
        f"full blank scale x{full_blank_scale:g}; frictionless\n"
        f"unresolved jumps: {int(np.count_nonzero(unresolved))}; "
        f"separated records: {int(np.count_nonzero(separated))}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="#555555",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
    )
    fig.savefig(png_path, dpi=170)
    plt.close(fig)
    print(json.dumps({"csv": str(csv_path), "plot": str(png_path),
                      "thickness_csv": str(thickness_path),
                      "accepted_steps": len(records),
                      "last_stroke_mm": float(stroke[-1]),
                      "full_blank_force_scale": full_blank_scale,
                      "current_minimum_column_thickness_mm": min(
                          thickness for _, thickness in thickness_profile
                      )}, indent=2))


if __name__ == "__main__":
    main()
