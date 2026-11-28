"""Render saved Paris-law fatigue VTU states as an H.264 video.

Run from the repository root after ``grow_edge_crack_fatigue.py``:

    python tutorials/fracture_mode_i/fatigue/render_fatigue_video.py

The renderer only reads existing VTU/CSV results; it does not solve the model.
Requires ``meshio``, ``matplotlib``, ``numpy``, and an ``ffmpeg`` executable.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import meshio
import numpy as np
from matplotlib.animation import FFMpegWriter


def render_video(
    results_dir: Path,
    output: Path,
    *,
    seconds_per_step: int = 2,
) -> Path:
    """Render displacement and crack-history panels from saved result files."""
    if seconds_per_step < 1:
        raise ValueError("seconds_per_step must be at least 1")
    with (results_dir / "fatigue-growth.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        records = list(csv.DictReader(stream))
    files = sorted(results_dir.glob("fatigue-step-*.vtu"))
    if not files or len(files) != len(records):
        raise ValueError(
            "expected one fatigue-step-*.vtu per row in fatigue-growth.csv; "
            f"found {len(files)} VTU files and {len(records)} records in {results_dir}"
        )

    meshes = [meshio.read(path) for path in files]
    try:
        triangles = meshes[0].cells_dict["triangle"]
        xy = meshes[0].points[:, :2]
        fields = [
            np.linalg.norm(mesh.point_data["displacement"][:, :2], axis=1)
            for mesh in meshes
        ]
    except (KeyError, IndexError) as error:
        raise ValueError(
            "fatigue VTU files must contain triangle cells and point displacement"
        ) from error
    if any(mesh.cells_dict["triangle"].shape != triangles.shape for mesh in meshes):
        raise ValueError("all fatigue VTU files must use the same triangle connectivity")

    triangulation = mtri.Triangulation(xy[:, 0], xy[:, 1], triangles)
    vmax = max(float(field.max()) for field in fields)
    vmax = max(vmax, 1e-15)
    tips = np.array([float(row["crack_tip_x"]) for row in records])
    cycles = np.array([float(row["cumulative_cycles"]) for row in records])
    levels = np.linspace(0.0, vmax, 32)

    figure, (field_ax, history_ax) = plt.subplots(
        1,
        2,
        figsize=(12.8, 7.2),
        dpi=100,
        gridspec_kw={"width_ratios": [1.55, 1]},
    )
    figure.patch.set_facecolor("#101820")
    for axis in (field_ax, history_ax):
        axis.set_facecolor("#17232d")
        axis.tick_params(colors="white", labelsize=10)
        for spine in axis.spines.values():
            spine.set_color("#9aa7b2")
        axis.xaxis.label.set_color("white")
        axis.yaxis.label.set_color("white")

    contour = field_ax.tricontourf(
        triangulation, fields[0], levels=levels, cmap="magma"
    )
    colorbar = figure.colorbar(contour, ax=field_ax, fraction=0.046, pad=0.04)
    colorbar.set_label("displacement magnitude (dimensionless)", color="white")
    colorbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")
    crack_line, = field_ax.plot(
        [], [], color="#071018", linewidth=5, solid_capstyle="round", zorder=5
    )
    field_ax.set(
        xlim=(float(xy[:, 0].min()) - 0.03, float(xy[:, 0].max()) + 0.03),
        ylim=(float(xy[:, 1].min()) - 0.03, float(xy[:, 1].max()) + 0.03),
        xlabel="x",
        ylabel="y",
    )
    field_ax.set_aspect("equal")
    field_ax.set_title(
        "XFEM displacement field and growing crack", color="white", fontsize=13
    )

    history_ax.plot(cycles, tips, color="#67d5c5", linewidth=2, alpha=0.65)
    history_ax.set(
        xlabel="cumulative cycles",
        ylabel="crack-tip x",
        xlim=(0.0, max(float(cycles[-1]), 1.0)),
        ylim=(float(tips.min()) - 0.08, float(tips.max()) + 0.08),
    )
    history_ax.grid(True, alpha=0.2, color="white")
    history_ax.set_title("Paris-law growth history", color="white", fontsize=13)
    marker, = history_ax.plot(
        [], [], "o", color="#ffcc66", markersize=11, markeredgecolor="white"
    )
    cycle_label = figure.text(
        0.5, 0.965, "", ha="center", color="white", fontsize=16, weight="bold"
    )
    figure.tight_layout(rect=(0.02, 0.03, 0.98, 0.94))

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(
        fps=1,
        codec="h264",
        bitrate=2200,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    with writer.saving(figure, str(output), dpi=100):
        for index, (mesh, field, row) in enumerate(
            zip(meshes, fields, records, strict=True)
        ):
            for collection in list(field_ax.collections):
                collection.remove()
            field_ax.tricontourf(triangulation, field, levels=levels, cmap="magma")
            line_cells = mesh.cells_dict.get("line")
            if line_cells is not None and len(line_cells):
                line_xy = mesh.points[line_cells[0], :2]
                crack_line.set_data(line_xy[:, 0], line_xy[:, 1])
            else:
                crack_line.set_data([-1.0, tips[index]], [0.0, 0.0])
            marker.set_data([cycles[index]], [tips[index]])
            cycle_label.set_text(
                f"step {index:02d}   crack tip x = {tips[index]:.3f}   "
                f"N = {cycles[index]:,.0f} cycles"
            )
            for _ in range(seconds_per_step):
                writer.grab_frame()
    plt.close(figure)
    return output


def main() -> None:
    default_results = Path(__file__).resolve().parent / "results"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=default_results)
    parser.add_argument(
        "--output",
        type=Path,
        default=default_results / "fatigue-growth.mp4",
    )
    parser.add_argument("--seconds-per-step", type=int, default=2)
    args = parser.parse_args()
    output = render_video(
        args.results_dir,
        args.output,
        seconds_per_step=args.seconds_per_step,
    )
    print(f"video={output} duration_seconds={args.seconds_per_step * len(list(args.results_dir.glob('fatigue-step-*.vtu')))}")


if __name__ == "__main__":
    main()
