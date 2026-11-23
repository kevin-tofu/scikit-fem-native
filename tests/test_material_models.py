import numpy as np
import pytest

from skfemntv import IsotropicLinearElasticity, StVenantKirchhoffMaterial


def test_isotropic_linear_elasticity_material_stress_and_voigt_matrix_agree():
    material = IsotropicLinearElasticity(210.0, 0.3)
    strain = np.array(
        [[0.01, 0.002, 0.0], [0.002, -0.003, 0.0], [0.0, 0.0, 0.0]]
    )
    engineering_strain = np.array(
        [strain[0, 0], strain[1, 1], strain[2, 2],
         2.0 * strain[0, 1], 2.0 * strain[1, 2], 2.0 * strain[0, 2]]
    )
    stress = material.stress(strain)
    voigt_stress = material.voigt_stiffness @ engineering_strain

    np.testing.assert_allclose(
        voigt_stress,
        [stress[0, 0], stress[1, 1], stress[2, 2],
         stress[0, 1], stress[1, 2], stress[0, 2]],
    )


def test_st_venant_kirchhoff_zero_and_rigid_rotation_response():
    material = StVenantKirchhoffMaterial(210.0, 0.3)
    angle = 0.37
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0.0],
         [np.sin(angle), np.cos(angle), 0.0],
         [0.0, 0.0, 1.0]]
    )

    zero = material.evaluate(np.eye(3))
    rotated = material.evaluate(rotation)
    np.testing.assert_allclose(zero.first_piola_kirchhoff_stress, 0.0, atol=1e-12)
    np.testing.assert_allclose(zero.strain_energy_density, 0.0, atol=1e-12)
    np.testing.assert_allclose(rotated.first_piola_kirchhoff_stress, 0.0, atol=1e-12)
    np.testing.assert_allclose(rotated.strain_energy_density, 0.0, atol=1e-12)


def test_st_venant_kirchhoff_tangent_matches_finite_difference():
    material = StVenantKirchhoffMaterial(210.0, 0.3)
    deformation_gradient = np.array(
        [[1.08, 0.03, -0.01], [0.02, 0.96, 0.04], [0.0, -0.02, 1.03]]
    )
    response = material.evaluate(deformation_gradient)
    step = 1e-7

    for k in range(3):
        for l in range(3):
            perturbation = np.zeros((3, 3))
            perturbation[k, l] = step
            plus = material.evaluate(deformation_gradient + perturbation)
            minus = material.evaluate(deformation_gradient - perturbation)
            finite_difference = (
                plus.first_piola_kirchhoff_stress
                - minus.first_piola_kirchhoff_stress
            ) / (2.0 * step)
            np.testing.assert_allclose(
                response.tangent[:, :, k, l], finite_difference,
                rtol=2e-7, atol=2e-7,
            )


@pytest.mark.parametrize(
    "young,poisson",
    [(0.0, 0.3), (210.0, 0.5), (210.0, np.nan)],
)
def test_material_models_reject_invalid_parameters(young, poisson):
    with pytest.raises(ValueError):
        StVenantKirchhoffMaterial(young, poisson)
