import numpy as np
import pytest

import skfemntv
from skfemntv.helpers import ddot, div, sym_grad


def _tet10_mesh_data():
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ]
    )
    elements = np.arange(10, dtype=np.int64)[None, :]
    return points, elements


def test_tet10_kernel_stiffness_matches_native_form_assembly():
    points, elements = _tet10_mesh_data()
    material = skfemntv.IsotropicLinearElasticity(100.0, 0.25)
    kernel = skfemntv.Tet10LinearElasticityKernel(
        points, elements, material, integration_order=4
    )
    basis = kernel.basis

    @skfemntv.BilinearForm
    def elasticity(u, v, w):
        return (
            2.0 * material.lame_mu * ddot(sym_grad(u), sym_grad(v))
            + material.lame_lambda * div(u) * div(v)
        )

    assembled = skfemntv.asm(elasticity, basis).toarray()
    element_dofs = basis.element_dofs[:, 0]
    expected = assembled[np.ix_(element_dofs, element_dofs)]
    actual = kernel.integration.element_stiffness[0]

    assert kernel.integration_point_count == 11
    np.testing.assert_allclose(actual, expected, rtol=3.0e-14, atol=3.0e-14)


@pytest.mark.parametrize("integration_order", [2, 3, 4])
def test_tet10_quadrature_preserves_existing_low_order_rules(integration_order):
    points, elements = _tet10_mesh_data()
    kernel = skfemntv.Tet10LinearElasticityKernel(
        points,
        elements,
        skfemntv.IsotropicLinearElasticity(100.0, 0.25),
        integration_order=integration_order,
    )
    reference_basis = skfemntv.Basis(
        skfemntv.MeshTet2(points.T, elements.T),
        skfemntv.ElementVector(skfemntv.ElementTetP2(), dim=3),
        intorder=integration_order,
    )

    np.testing.assert_allclose(
        kernel.integration.reference_points,
        reference_basis.X.T,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        kernel.integration.quadrature_weights,
        reference_basis.W,
        atol=1.0e-15,
    )


def test_tet10_stvk_kernel_tangent_matches_local_force_difference():
    points, elements = _tet10_mesh_data()
    linear_kernel = skfemntv.Tet10LinearElasticityKernel(
        points,
        elements,
        skfemntv.IsotropicLinearElasticity(80.0, 0.22),
        integration_order=4,
    )
    kernel = skfemntv.Tet10StVenantKirchhoffKernel(
        skfemntv.StVenantKirchhoffMaterial(80.0, 0.22)
    )
    displacement = np.linspace(-1.0e-3, 1.5e-3, 30)
    direction = np.linspace(0.02, -0.03, 30)
    step = 1.0e-7
    plus = kernel.evaluate(
        linear_kernel.integration, 0, displacement + step * direction
    ).residual
    minus = kernel.evaluate(
        linear_kernel.integration, 0, displacement - step * direction
    ).residual
    response = kernel.evaluate(linear_kernel.integration, 0, displacement)

    np.testing.assert_allclose(
        response.tangent @ direction,
        (plus - minus) / (2.0 * step),
        rtol=2.0e-6,
        atol=2.0e-6,
    )
