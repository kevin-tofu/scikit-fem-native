"""Mesh refinement of the plane-strain J2 punch/die load-unload example.

The finest requested mesh is used as a provisional numerical reference. This
does not replace comparison against an independent forming benchmark.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from plane_strain_three_point_bend_contact import run


def parse_meshes(values: list[str]) -> list[tuple[int, int]]:
    meshes = []
    for value in values:
        try:
            nx, ny = (int(part) for part in value.lower().split("x", maxsplit=1))
        except (ValueError, TypeError) as error:
            raise argparse.ArgumentTypeError(
                f"invalid mesh {value!r}; use NXxNY, for example 32x12"
            ) from error
        if nx < 16 or nx % 16 != 0 or ny < 2:
            raise argparse.ArgumentTypeError(
                "NX must be at least 16 and divisible by 16; NY must be at least 2"
            )
        meshes.append((nx, ny))
    if len(meshes) < 2:
        raise argparse.ArgumentTypeError("provide at least two meshes")
    return meshes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--meshes", nargs="+", default=["16x3", "16x6", "16x12", "32x12", "64x12"],
        help="mesh sequence from coarser to provisional reference, e.g. 16x3 32x12",
    )
    parser.add_argument(
        "--orders", nargs="+", type=int, choices=(1, 2), default=[1],
        help="polynomial orders to include (TriP1 and/or TriP2)",
    )
    parser.add_argument("--maximum-stroke", type=float, default=0.02)
    parser.add_argument("--output")
    args = parser.parse_args()
    meshes = parse_meshes(args.meshes)

    results = []
    for order in args.orders:
        for nx, ny in meshes:
            result = run(nx, ny, maximum_stroke=args.maximum_stroke, order=order)
            result.pop("steps")
            results.append(result)

    reference = results[-1]
    reference_mesh = reference["mesh"]
    common_path = np.linspace(0.0, 1.0, 501)
    ref_samples = np.asarray(reference["punch_force_by_stroke"])[:, 1]
    ref_curve = np.interp(
        common_path, np.linspace(0.0, 1.0, ref_samples.size), ref_samples
    )
    comparisons = []
    for result in results:
        peak = result["maximum_punch_force"]
        springback = result["centerline_springback_vertical_displacement"]
        thickness_change = result["centerline_thickness_change"]
        samples = np.asarray(result["punch_force_by_stroke"])[:, 1]
        curve = np.interp(common_path, np.linspace(0.0, 1.0, samples.size), samples)
        comparisons.append({
            "mesh": result["mesh"],
            "peak_force": peak,
            "peak_force_relative_error_percent": 100.0 * abs(
                peak - reference["maximum_punch_force"]
            ) / max(abs(reference["maximum_punch_force"]), np.finfo(float).tiny),
            "load_unload_curve_relative_l2_error_percent": 100.0 * np.linalg.norm(
                curve - ref_curve
            ) / max(np.linalg.norm(ref_curve), np.finfo(float).tiny),
            "thickness_change": thickness_change,
            "thickness_change_error_as_initial_thickness_percent": 100.0 * abs(
                thickness_change - reference["centerline_thickness_change"]
            ) / reference["centerline_thickness_initial"],
            "springback": springback,
            "springback_relative_error_percent": 100.0 * abs(
                springback - reference["centerline_springback_vertical_displacement"]
            ) / max(abs(reference["centerline_springback_vertical_displacement"]),
                    np.finfo(float).tiny),
            "max_equivalent_plastic_strain": result["max_equivalent_plastic_strain"],
            "maximum_equilibrium_residual_inf": result["maximum_equilibrium_residual_inf"],
        })

    payload = {
        "problem": reference["problem"],
        "units": reference["units"],
        "geometry": reference["geometry"],
        "maximum_stroke": args.maximum_stroke,
        "material": reference["material"],
        "reference_type": "last requested mesh/order; provisional internal numerical reference",
        "reference_mesh": reference_mesh,
        "independent_reference_available": False,
        "comparisons": comparisons,
    }
    rendered = json.dumps(payload, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as stream:
            stream.write(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
