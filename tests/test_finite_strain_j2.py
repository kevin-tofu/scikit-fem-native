import numpy as np

import skfemntv


def make_material():
    return skfemntv.FiniteStrainJ2Material(
        young_modulus=1000.0,
        poisson_ratio=0.3,
        yield_stress=2.0,
        hardening_modulus=20.0,
    )


def test_finite_strain_j2_material_tangent_and_transactional_history():
    material = make_material()
    committed = material.initial_state()
    original = committed.copy()
    deformation_gradient = np.array([
        [1.025, 0.018, 0.0],
        [-0.006, 0.994, 0.0],
        [0.0, 0.0, 1.0],
    ])

    trial = material.evaluate(deformation_gradient, committed)
    assert bool(trial.plastic_points)
    assert trial.plastic_multiplier > 0.0
    np.testing.assert_array_equal(committed, original)
    np.testing.assert_allclose(
        np.linalg.det(trial.trial_state[:9].reshape(3, 3)), 1.0, rtol=1e-9, atol=1e-11
    )

    direction = np.random.default_rng(42).standard_normal((3, 3))
    step = 2e-7
    plus = material.evaluate(
        deformation_gradient + step * direction, committed, tangent=False
    ).first_piola_stress
    minus = material.evaluate(
        deformation_gradient - step * direction, committed, tangent=False
    ).first_piola_stress
    finite_difference = (plus - minus) / (2.0 * step)
    tangent_action = np.einsum("ijkl,kl->ij", trial.tangent, direction)
    relative_error = np.linalg.norm(tangent_action - finite_difference) / max(
        np.linalg.norm(finite_difference), 1e-30
    )
    assert relative_error < 3e-6

    accepted = material.commit(trial)
    np.testing.assert_array_equal(accepted, trial.trial_state)
    assert not np.shares_memory(accepted, trial.trial_state)
    unload_gradient = np.eye(3) + 0.9 * (deformation_gradient - np.eye(3))
    unload = material.evaluate(unload_gradient, accepted)
    assert not bool(unload.plastic_points)
    np.testing.assert_array_equal(unload.trial_state, accepted)

    discarded = material.evaluate(
        np.diag([1.04, 0.99, 1.0]), accepted, tangent=False
    )
    assert discarded.trial_state[9] > accepted[9]
    np.testing.assert_array_equal(accepted, trial.trial_state)


def make_plane_strain_assembler():
    mesh = skfemntv.MeshTri.init_tensor(
        np.array([0.0, 1.0]), np.array([0.0, 0.5])
    )
    basis = skfemntv.Basis(
        mesh,
        skfemntv.ElementVector(skfemntv.ElementTriP1(), dim=2),
        intorder=2,
    )
    return basis, skfemntv.PlaneStrainFiniteStrainJ2Assembler(
        basis, make_material()
    )


def affine_displacement(basis, deformation_2d):
    displacement = np.zeros(basis.N)
    coordinates = basis.mesh.p
    nodal_dofs = basis.nodal_dofs
    gradient = deformation_2d - np.eye(2)
    nodal_displacement = gradient @ coordinates
    displacement[nodal_dofs[0]] = nodal_displacement[0]
    displacement[nodal_dofs[1]] = nodal_displacement[1]
    return displacement


def test_plane_strain_finite_strain_j2_assembler_tangent_and_commit():
    basis, assembler = make_plane_strain_assembler()
    committed = assembler.initial_state()
    original = committed.copy()
    displacement = affine_displacement(
        basis, np.array([[1.025, 0.018], [-0.006, 0.994]])
    )
    trial = assembler.evaluate(displacement, committed)

    assert trial.plastic_points.all()
    np.testing.assert_array_equal(committed, original)
    assert trial.tangent is not None

    direction = np.random.default_rng(73).standard_normal(basis.N)
    step = 2e-7
    plus = assembler.evaluate(
        displacement + step * direction, committed, tangent=False
    ).residual
    minus = assembler.evaluate(
        displacement - step * direction, committed, tangent=False
    ).residual
    finite_difference = (plus - minus) / (2.0 * step)
    tangent_action = trial.tangent @ direction
    relative_error = np.linalg.norm(tangent_action - finite_difference) / max(
        np.linalg.norm(finite_difference), 1e-30
    )
    assert relative_error < 5e-6

    accepted = assembler.commit(trial)
    np.testing.assert_array_equal(accepted, trial.trial_state)
    assert not np.shares_memory(accepted, trial.trial_state)
    np.testing.assert_array_equal(committed, original)
