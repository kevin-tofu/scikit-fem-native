import numpy as np

import skfemntv
from skfemntv.helpers import ddot, div, sym_grad


def test_hex8_kernel_matches_native_form_assembly_on_distorted_cell():
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.1, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.2, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )
    native_hex_nodes = np.asarray([0, 3, 1, 4, 2, 7, 5, 6])
    mesh = skfemntv.MeshHex(points.T, native_hex_nodes[:, None])
    basis = skfemntv.Basis(
        mesh,
        skfemntv.ElementVector(skfemntv.ElementHex1()),
        intorder=2,
    )
    material = skfemntv.IsotropicLinearElasticity(210.0, 0.3)

    @skfemntv.BilinearForm
    def elasticity(u, v, w):
        return (
            2.0 * material.lame_mu * ddot(sym_grad(u), sym_grad(v))
            + material.lame_lambda * div(u) * div(v)
        )

    assembled = skfemntv.asm(elasticity, basis).toarray()
    local_dofs = basis.element_dofs[:, 0]
    expected_local = assembled[np.ix_(local_dofs, local_dofs)]
    integration = skfemntv.Hex8LinearElasticityKernel(material).integrate(
        points[native_hex_nodes]
    )

    np.testing.assert_allclose(
        integration.stiffness,
        expected_local,
        rtol=5.0e-14,
        atol=5.0e-14,
    )
    np.testing.assert_allclose(
        integration.stiffness,
        integration.stiffness.T,
        rtol=0.0,
        atol=5.0e-14,
    )


def test_hex8_stvk_kernel_tangent_matches_local_force_difference():
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )
    material = skfemntv.StVenantKirchhoffMaterial(1000.0, 0.25)
    integration = skfemntv.Hex8LinearElasticityKernel(
        skfemntv.IsotropicLinearElasticity(1000.0, 0.25)
    ).integrate(points)
    kernel = skfemntv.Hex8StVenantKirchhoffKernel(material)
    displacement = np.linspace(-0.01, 0.015, 24)
    direction = np.linspace(0.02, -0.03, 24)
    step = 1.0e-7
    plus = kernel.evaluate(integration, displacement + step * direction).residual
    minus = kernel.evaluate(integration, displacement - step * direction).residual
    response = kernel.evaluate(integration, displacement)

    np.testing.assert_allclose(
        response.tangent @ direction,
        (plus - minus) / (2.0 * step),
        rtol=2.0e-7,
        atol=2.0e-7,
    )
