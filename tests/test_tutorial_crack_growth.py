import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).parents[1]
    / "tutorials"
    / "fracture_mode_i"
    / "grow_edge_crack.py"
)
SPEC = importlib.util.spec_from_file_location("tutorial_crack_growth", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_mode_i_growth_advances_tip_and_keeps_equilibrium_finite():
    records, states = MODULE.run_growth(
        resolution=16,
        initial_tip_x=-0.15,
        growth_step=0.15,
        toughness=1.0e-4,
        maximum_steps=1,
        applied_strain=0.01,
    )

    assert len(records) == 2
    assert len(states) == 2
    assert records[0].criterion_met
    assert records[0].advanced
    assert records[1].criterion_met
    assert not records[1].advanced  # maximum-step guard
    np.testing.assert_allclose(records[1].crack_tip_x, 0.0, atol=1.0e-14)
    assert all(np.isfinite(record.stress_intensity_k1) for record in records)
    assert all(record.equilibrium_residual_l2 < 1.0e-10 for record in records)
    assert all(state.enriched_node_count > 0 for state in states)


def test_growth_outputs_include_each_solved_state(tmp_path):
    records, states = MODULE.run_growth(
        resolution=16,
        initial_tip_x=-0.15,
        growth_step=0.15,
        toughness=1.0e-4,
        maximum_steps=1,
        applied_strain=0.01,
    )
    MODULE.write_results(tmp_path, records, states)

    assert (tmp_path / "crack-growth.csv").is_file()
    assert (tmp_path / "crack-growth-history.png").is_file()
    assert (tmp_path / "crack-growth-final.vtu").is_file()
    assert (tmp_path / "crack-growth-step-000.vtu").is_file()
    assert (tmp_path / "crack-growth-step-001.vtu").is_file()
