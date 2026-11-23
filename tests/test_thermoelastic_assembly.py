import numpy as np
import pytest

import skfemntv


def _shared_tet_bases(intorder: int = 2):
    points = np.asarray(
        (
            (0.0, 1.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0, -1.0),
        )
    )
    cells = np.asarray(((0, 1), (1, 2), (2, 3), (3, 4)), dtype=np.int64)
    mesh = skfemntv.MeshTet(points, cells)
    mechanical = skfemntv.Basis(
        mesh,
        skfemntv.ElementVector(skfemntv.ElementTetP1(), dim=3),
        intorder=intorder,
    )
    thermal = skfemntv.Basis(
        mesh,
        skfemntv.ElementTetP1(),
        intorder=intorder,
    )
    return mechanical, thermal


def test_thermoelastic_stvk_global_tangent_blocks_match_directional_difference():
    mechanical_basis, thermal_basis = _shared_tet_bases()
    material = skfemntv.ThermoelasticStVenantKirchhoffMaterial(
        young_modulus=1200.0,
        poisson_ratio=0.27,
        thermal_expansion_coefficient=1.0e-4,
    )
    assembler = skfemntv.ThermoelasticStVenantKirchhoffAssembler(
        mechanical_basis,
        thermal_basis,
        material,
    )
    random = np.random.default_rng(20261123)
    displacement = 1.0e-3 * random.normal(size=mechanical_basis.N)
    temperature = 12.0 * random.normal(size=thermal_basis.N)
    previous_displacement = 3.0e-4 * random.normal(size=mechanical_basis.N)
    previous_temperature = 4.0 * random.normal(size=thermal_basis.N)
    direction_displacement = random.normal(size=mechanical_basis.N)
    direction_temperature = random.normal(size=thermal_basis.N)
    time_step = 0.2
    reference_temperature = 300.0
    response = assembler.evaluate(
        displacement,
        temperature,
        previous_displacement,
        previous_temperature,
        time_step=time_step,
        reference_temperature=reference_temperature,
    )
    tangent_action = np.concatenate(
        (
            response.mechanical_displacement_tangent @ direction_displacement
            + response.mechanical_temperature_tangent @ direction_temperature,
            response.thermal_displacement_tangent @ direction_displacement
            + response.thermal_temperature_tangent @ direction_temperature,
        )
    )

    def residual_at(step: float) -> np.ndarray:
        evaluation = assembler.evaluate(
            displacement + step * direction_displacement,
            temperature + step * direction_temperature,
            previous_displacement,
            previous_temperature,
            time_step=time_step,
            reference_temperature=reference_temperature,
        )
        return np.concatenate(
            (evaluation.mechanical_residual, evaluation.thermal_entropy_residual)
        )

    step = 2.0e-7
    finite_difference = (residual_at(step) - residual_at(-step)) / (2.0 * step)
    np.testing.assert_allclose(
        tangent_action, finite_difference, rtol=4.0e-7, atol=3.0e-8
    )


def test_thermoelastic_assembler_requires_shared_mesh_and_quadrature():
    mechanical_basis, thermal_basis = _shared_tet_bases(intorder=2)
    different_quadrature = skfemntv.Basis(
        thermal_basis.mesh,
        skfemntv.ElementTetP1(),
        intorder=4,
    )
    material = skfemntv.ThermoelasticStVenantKirchhoffMaterial(
        young_modulus=1200.0,
        poisson_ratio=0.27,
        thermal_expansion_coefficient=1.0e-4,
    )
    with pytest.raises(ValueError, match="quadrature shapes"):
        skfemntv.ThermoelasticStVenantKirchhoffAssembler(
            mechanical_basis,
            different_quadrature,
            material,
        )
