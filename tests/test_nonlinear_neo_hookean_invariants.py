import numpy as np
import pytest

import skfemntv


def make_basis():
    points = np.array(
        [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    mesh = skfemntv.MeshTet(points, np.array([[0], [1], [2], [3]], dtype=np.int64))
    return skfemntv.Basis(
        mesh, skfemntv.ElementVector(skfemntv.ElementTetP1(), dim=3), intorder=2
    )


def make_family_basis(family):
    if family == "tet4":
        mesh, element = skfemntv.MeshTet(), skfemntv.ElementTetP1()
    elif family == "tet10":
        linear = skfemntv.MeshTet()
        mesh, element = skfemntv.MeshTet2.from_mesh(linear), skfemntv.ElementTetP2()
    elif family == "hex8":
        mesh, element = skfemntv.MeshHex(), skfemntv.ElementHex1()
    elif family == "hex20":
        linear = skfemntv.MeshHex()
        mesh, element = skfemntv.MeshHex20.from_mesh(linear), skfemntv.ElementHex20()
    else:
        linear = skfemntv.MeshHex()
        mesh, element = skfemntv.MeshHex2.from_mesh(linear), skfemntv.ElementHex2()
    return skfemntv.Basis(
        mesh, skfemntv.ElementVector(element, dim=3), intorder=5
    )


def affine_displacement(basis, deformation_gradient):
    displacement = np.zeros(basis.N)
    for component, dofs in enumerate(basis.split_indices()):
        displacement[dofs] = (deformation_gradient - np.eye(3))[component] @ basis.doflocs[:, dofs]
    return displacement


def isotropic_stretch_response(stretch, mu, lam):
    log_j = 3.0 * np.log(stretch)
    p = mu * (stretch - 1.0 / stretch) + lam * log_j / stretch
    energy = 0.5 * mu * (3.0 * stretch**2 - 3.0) - 3.0 * mu * np.log(stretch)
    energy += 0.5 * lam * log_j**2
    return p, p / stretch**2, energy


def assert_zero_stress_response(response):
    np.testing.assert_allclose(response.stress, 0.0, atol=2e-13)
    np.testing.assert_allclose(response.cauchy_stress, 0.0, atol=2e-13)
    np.testing.assert_allclose(response.equivalent_stress, 0.0, atol=2e-13)
    np.testing.assert_allclose(response.energy, 0.0, atol=2e-13)


def test_material_point_response_is_objective_under_finite_rotation():
    angle = 0.37
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0.0],
         [np.sin(angle), np.cos(angle), 0.0],
         [0.0, 0.0, 1.0]]
    )
    response = skfemntv.NeoHookeanMaterial(mu=2.4, lambda_=3.1).evaluate(rotation)
    assert_zero_stress_response(response)


def test_assembler_is_objective_under_finite_rotation():
    basis = make_basis()
    angle = 0.37
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0.0],
         [np.sin(angle), np.cos(angle), 0.0],
         [0.0, 0.0, 1.0]]
    )
    assembler = skfemntv.NeoHookeanAssembler(basis, mu=2.4, lambda_=3.1)
    result = assembler.evaluate(affine_displacement(basis, rotation))
    assert_zero_stress_response(result)
    expected_f = np.broadcast_to(rotation, result.deformation_gradient.shape)
    np.testing.assert_allclose(result.deformation_gradient, expected_f, atol=2e-14)
    np.testing.assert_allclose(result.residual, 0.0, atol=2e-13)


@pytest.mark.parametrize("stretch", [0.85, 1.2])
def test_material_point_isotropic_stretch_matches_analytic_solution(stretch):
    mu, lam = 2.4, 3.1
    p, sigma, energy = isotropic_stretch_response(stretch, mu, lam)
    response = skfemntv.NeoHookeanMaterial(mu=mu, lambda_=lam).evaluate(
        stretch * np.eye(3)
    )
    np.testing.assert_allclose(response.stress, p * np.eye(3), rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(response.cauchy_stress, sigma * np.eye(3), rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(response.equivalent_stress, 0.0, atol=2e-13)
    assert float(response.energy) == pytest.approx(energy, rel=2e-13, abs=2e-13)


@pytest.mark.parametrize("stretch", [0.85, 1.2])
def test_assembler_isotropic_stretch_matches_analytic_solution(stretch):
    mu, lam = 2.4, 3.1
    p, sigma, energy_density = isotropic_stretch_response(stretch, mu, lam)
    basis = make_basis()
    assembler = skfemntv.NeoHookeanAssembler(basis, mu=mu, lambda_=lam)
    result = assembler.evaluate(affine_displacement(basis, stretch * np.eye(3)))
    volume = float(np.sum(basis.dx))
    expected_p = np.broadcast_to(p * np.eye(3), result.stress.shape)
    expected_sigma = np.broadcast_to(sigma * np.eye(3), result.cauchy_stress.shape)
    expected_f = np.broadcast_to(stretch * np.eye(3), result.deformation_gradient.shape)
    np.testing.assert_allclose(result.deformation_gradient, expected_f, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(result.stress, expected_p, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(
        result.cauchy_stress, expected_sigma, rtol=2e-13, atol=2e-13
    )
    np.testing.assert_allclose(result.equivalent_stress, 0.0, atol=2e-13)
    assert result.energy == pytest.approx(volume * energy_density, rel=2e-13, abs=2e-13)


@pytest.mark.parametrize("family", ["tet4", "tet10", "hex8", "hex20", "hex27"])
def test_isotropic_stretch_response_is_consistent_across_supported_element_families(family):
    mu, lam, stretch = 2.4, 3.1, 1.17
    basis = make_family_basis(family)
    assembler = skfemntv.NeoHookeanAssembler(basis, mu=mu, lambda_=lam)
    deformation = stretch * np.eye(3)
    result = assembler.evaluate(affine_displacement(basis, deformation))
    point = skfemntv.NeoHookeanMaterial(mu=mu, lambda_=lam).evaluate(deformation)
    expected_f = np.broadcast_to(deformation, result.deformation_gradient.shape)
    expected_stress = np.broadcast_to(point.stress, result.stress.shape)
    expected_cauchy = np.broadcast_to(point.cauchy_stress, result.cauchy_stress.shape)
    expected_equivalent = np.broadcast_to(point.equivalent_stress, result.equivalent_stress.shape)

    np.testing.assert_allclose(result.deformation_gradient, expected_f, rtol=3e-13, atol=3e-13)
    np.testing.assert_allclose(result.stress, expected_stress, rtol=3e-13, atol=3e-13)
    np.testing.assert_allclose(result.cauchy_stress, expected_cauchy, rtol=3e-13, atol=3e-13)
    np.testing.assert_allclose(result.equivalent_stress, expected_equivalent, atol=3e-13)
    assert result.energy == pytest.approx(
        float(point.energy) * float(np.sum(basis.dx)), rel=3e-13, abs=3e-13
    )


@pytest.mark.parametrize("family", ["tet4", "tet10", "hex8", "hex20", "hex27"])
def test_rigid_rotation_response_is_objective_across_supported_element_families(family):
    angle = 0.61
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0.0],
         [np.sin(angle), np.cos(angle), 0.0],
         [0.0, 0.0, 1.0]]
    )
    basis = make_family_basis(family)
    assembler = skfemntv.NeoHookeanAssembler(basis, mu=2.4, lambda_=3.1)
    result = assembler.evaluate(affine_displacement(basis, rotation))

    assert_zero_stress_response(result)
    expected_f = np.broadcast_to(rotation, result.deformation_gradient.shape)
    np.testing.assert_allclose(result.deformation_gradient, expected_f, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(result.residual, 0.0, atol=3e-13)
