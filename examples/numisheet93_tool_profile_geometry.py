"""Load and check NUMISHEET '93 analytical tool profiles from the reference data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import skfemntv


def run() -> dict:
    reference_path = Path(__file__).with_name("numisheet93_draw_bending_reference.json")
    reference = json.loads(reference_path.read_text())
    source = reference["analytical_tool_profiles_from_reference_input"]
    axis = np.asarray(source["surface_axis_line"], dtype=float)
    generator = np.asarray(source["surface_reference_direction"], dtype=float)
    profiles = {}
    for name, body in source["rigid_bodies"].items():
        # The deck uses metres; the benchmark table and continuum mesh use mm.
        local_segments = np.asarray([
            row[1:] for row in body["profile"] if row[0] == "START"
        ])
        if local_segments.shape != (1, 2):
            raise AssertionError(f"{name} has an invalid START coordinate")
        transformed_start_m = skfemntv.AnalyticalRigidProfile2D.keyword_local_to_global_xz(
            local_segments, axis, generator
        )[0]
        # Exercise the exact line/arc projection using the raw local profile;
        # its frame maps local (x,y) to global (X,Z), both in mm here.
        local_profile_m = skfemntv.AnalyticalRigidProfile2D.from_keyword_profile(
            body["profile"]
        )
        transform, offset = skfemntv.AnalyticalRigidProfile2D.keyword_local_transform_xz(
            axis, generator
        )
        profile_mm = local_profile_m.transformed(1000. * transform, 1000. * offset)
        profiles[name] = {
            "profile": profile_mm,
            "start_global_mm": transformed_start_m * 1000.,
            "reference_node_mm": np.asarray(body["reference_node"]) * 1000.,
        }

    probes = {
        "punch": np.array([[10.0, 0.39]]),
        "die": np.array([[100.0, -0.39]]),
        "blank_holder": np.array([[50.0, 0.39]]),
    }
    gaps = {name: float(profiles[name]["profile"].project(point).gaps[0])
            for name, point in probes.items()}
    expected = {"punch": 0.61, "die": 0.0, "blank_holder": 0.0}
    errors = {name: abs(gaps[name] - value) for name, value in expected.items()}
    if max(errors.values()) > 1e-8:
        raise AssertionError(f"reference-tool initial clearance check failed: {gaps}")
    return {
        "coordinate_mapping": "local profile (x,y) -> global (X,Z), sweep along global Y",
        "input_units": "m", "reported_profile_units": "mm",
        "profile_segment_counts": {
            name: profile["profile"].segment_count for name, profile in profiles.items()
        },
        "initial_clearance_mm": gaps,
        "clearance_error_mm": errors,
        "normal_directions": {
            name: profiles[name]["profile"].project(point).normals[0].tolist()
            for name, point in probes.items()
        },
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
