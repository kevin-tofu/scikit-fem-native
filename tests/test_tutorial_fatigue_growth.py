from tutorials.fracture_mode_i.grow_edge_crack_fatigue import run_fatigue


def test_paris_law_advances_crack_and_accumulates_cycles():
    records, states = run_fatigue(
        resolution=16,
        initial_tip_x=-0.15,
        growth_step=0.15,
        maximum_steps=1,
        applied_peak_strain=0.01,
        load_ratio=0.1,
        paris_c=1.0,
        paris_m=3.0,
        threshold_delta_k=1.0e-6,
    )

    assert len(records) == len(states) == 2
    first, second = records
    assert first.threshold_met and first.advanced
    assert first.cycles_this_increment > 0.0
    assert first.cumulative_cycles == 0.0
    expected_rate = first.paris_da_d_cycles
    assert expected_rate > 0.0
    assert abs(first.cycles_this_increment * expected_rate - 0.15) < 1.0e-12
    assert second.cumulative_cycles == first.cycles_this_increment
    assert second.crack_tip_x > first.crack_tip_x
    assert not second.advanced  # maximum-step guard
    assert all(record.equilibrium_residual_l2 < 1.0e-10 for record in records)


def test_fatigue_threshold_stops_growth():
    records, states = run_fatigue(
        resolution=16,
        initial_tip_x=-0.15,
        growth_step=0.15,
        maximum_steps=2,
        applied_peak_strain=0.01,
        load_ratio=0.1,
        paris_c=1.0,
        paris_m=3.0,
        threshold_delta_k=1.0,
    )

    assert len(records) == len(states) == 1
    assert not records[0].threshold_met
    assert not records[0].advanced
    assert records[0].cycles_this_increment == 0.0
