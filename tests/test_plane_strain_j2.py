import numpy as np

import skfemntv


def make_model():
    mesh = skfemntv.MeshTri.init_tensor(
        np.linspace(0.0, 1.0, 5), np.linspace(0.0, 0.25, 2)
    )
    basis = skfemntv.Basis(
        mesh,
        skfemntv.ElementVector(skfemntv.ElementTriP1(), dim=2),
        intorder=2,
    )
    model = skfemntv.PlaneStrainJ2Assembler(
        basis,
        young_modulus=1000.0,
        poisson_ratio=0.3,
        yield_stress=2.0,
        hardening_modulus=20.0,
    )
    return basis, model


def affine_x_displacement(basis, strain):
    displacement = np.zeros(basis.N)
    x_dofs = np.unique(basis.element_dofs[0::2])
    displacement[x_dofs] = strain * basis.doflocs[0, x_dofs]
    return displacement


def test_plane_strain_j2_elastic_tangent_and_transactional_trial():
    basis, model = make_model()
    committed = model.initial_state()
    original = committed.copy()

    trial = model.evaluate(affine_x_displacement(basis, 1e-5), committed)
    assert not trial.plastic_points.any()
    np.testing.assert_array_equal(committed, original)
    np.testing.assert_array_equal(trial.trial_state, committed)
    np.testing.assert_allclose(
        trial.stress[..., 0, 0],
        (model.lambda_ + 2.0 * model.mu) * 1e-5,
        rtol=2e-13,
        atol=2e-13,
    )
    np.testing.assert_allclose(
        trial.stress[..., 1, 1], model.lambda_ * 1e-5,
        rtol=2e-13,
        atol=2e-13,
    )
    np.testing.assert_allclose(
        trial.stress[..., 2, 2], model.lambda_ * 1e-5,
        rtol=2e-13,
        atol=2e-13,
    )
    np.testing.assert_allclose(
        trial.stress[..., 0, 1], 0.0, atol=1e-14
    )


def test_plane_strain_j2_plastic_consistent_tangent_and_commit():
    basis, model = make_model()
    committed = model.initial_state()
    original = committed.copy()
    displacement = affine_x_displacement(basis, 0.02)
    trial = model.evaluate(displacement, committed)

    assert trial.plastic_points.all()
    assert np.max(trial.trial_state[..., 6]) > 0.0
    np.testing.assert_array_equal(committed, original)

    direction = np.random.default_rng(42).standard_normal(basis.N)
    direction /= np.linalg.norm(direction)
    step = 1e-7
    plus = model.evaluate(displacement + step * direction, committed).residual
    minus = model.evaluate(displacement - step * direction, committed).residual
    finite_difference = (plus - minus) / (2.0 * step)
    tangent_action = trial.tangent @ direction
    relative_error = np.linalg.norm(tangent_action - finite_difference) / max(
        np.linalg.norm(finite_difference), 1e-30
    )
    assert relative_error < 2e-6
    np.testing.assert_array_equal(committed, original)

    adopted = model.commit(trial)
    np.testing.assert_array_equal(adopted, trial.trial_state)
    assert not np.shares_memory(adopted, trial.trial_state)

    unload = model.evaluate(affine_x_displacement(basis, 0.019), adopted)
    assert not unload.plastic_points.any()
    np.testing.assert_array_equal(unload.trial_state, adopted)
    np.testing.assert_array_equal(adopted, trial.trial_state)

    # A failed Newton or line-search trial is discarded by simply not committing it.
    discarded = model.evaluate(affine_x_displacement(basis, 0.03), adopted)
    assert np.max(discarded.trial_state[..., 6]) > np.max(adopted[..., 6])
    np.testing.assert_array_equal(adopted, trial.trial_state)
