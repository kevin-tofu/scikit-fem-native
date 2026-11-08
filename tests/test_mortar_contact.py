import numpy as np

import skfemntv


def make_curved_contact():
    angles = np.deg2rad([-20.0, -10.0])
    coordinates = 1.2 * np.column_stack((np.cos(angles), np.sin(angles)))
    contact = skfemntv.RigidProfileMortarContact2D(
        [[0, 1]], coordinates, [0, 2], [1, 3],
        variable_count=5,
        tool_translation_dof=4,
        tool_translation_direction=(0.0, 1.0),
        quadrature_order=5,
    )
    profile = skfemntv.AnalyticalRigidProfile2D.from_keyword_profile([
        ["START", 1.0, 0.0],
        ["CIRCL", 0.0, -1.0, 0.0, 0.0],
    ])
    return contact, profile


def test_mortar_gap_jacobian_matches_directional_finite_difference():
    contact, profile = make_curved_contact()
    state = np.zeros(5)
    state[-1] = 0.03
    direction = np.array([0.07, -0.03, -0.02, 0.05, 0.11])
    step = 2e-7

    evaluation = contact.evaluate(
        state, profile, translation=[0.0, state[-1]]
    )
    plus = contact.evaluate(
        state + step * direction,
        profile,
        translation=[0.0, state[-1] + step * direction[-1]],
    ).gaps
    minus = contact.evaluate(
        state - step * direction,
        profile,
        translation=[0.0, state[-1] - step * direction[-1]],
    ).gaps
    finite_difference = (plus - minus) / (2.0 * step)
    tangent_action = evaluation.jacobian @ direction

    np.testing.assert_allclose(tangent_action, finite_difference, rtol=2e-7, atol=2e-9)
    assert evaluation.minimum_sample_gaps[0] > 0.0


def test_mortar_geometric_tangent_matches_jacobian_derivative():
    contact, profile = make_curved_contact()
    state = np.zeros(5)
    state[-1] = 0.03
    direction = np.array([0.07, -0.03, -0.02, 0.05, 0.11])
    multipliers = np.array([3.4])
    step = 2e-6

    evaluation = contact.evaluate(
        state, profile, translation=[0.0, state[-1]]
    )
    plus = contact.evaluate(
        state + step * direction,
        profile,
        translation=[0.0, state[-1] + step * direction[-1]],
    )
    minus = contact.evaluate(
        state - step * direction,
        profile,
        translation=[0.0, state[-1] - step * direction[-1]],
    )
    finite_difference = (
        (plus.jacobian.T @ multipliers)
        - (minus.jacobian.T @ multipliers)
    ) / (2.0 * step)
    tangent_action = contact.geometric_tangent(
        evaluation, multipliers
    ) @ direction

    np.testing.assert_allclose(
        tangent_action, finite_difference, rtol=3e-6, atol=3e-8
    )


def test_mortar_multiplier_is_facet_resultant_and_pressure_is_area_scaled():
    contact, profile = make_curved_contact()
    evaluation = contact.evaluate(np.zeros(5), profile)
    multiplier = np.array([12.0])

    pressure = contact.facet_pressure(evaluation, multiplier)
    force = contact.contact_force(evaluation, multiplier)

    np.testing.assert_allclose(
        pressure * evaluation.facet_lengths, multiplier, rtol=0.0, atol=1e-14
    )
    assert force.shape == (5,)
    # The holder translation receives the equal and opposite vertical force.
    np.testing.assert_allclose(force[1] + force[3] + force[4], 0.0, atol=1e-14)
