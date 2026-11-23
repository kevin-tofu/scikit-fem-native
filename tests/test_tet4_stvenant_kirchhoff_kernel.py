from __future__ import annotations

import numpy as np

from skfemntv import StVenantKirchhoffMaterial, Tet4StVenantKirchhoffKernel


def test_tet4_kernel_local_tangent_matches_residual_difference() -> None:
    points = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    kernel = Tet4StVenantKirchhoffKernel(StVenantKirchhoffMaterial(80.0, 0.22))
    gradients, volume = kernel.integrate(points)
    displacement = np.asarray(
        [[0.001, -0.002, 0.001], [0.003, 0.001, -0.001],
         [-0.001, 0.002, 0.001], [0.001, -0.001, 0.002]]
    )
    response = kernel.evaluate(gradients, volume, displacement)
    direction = np.arange(1.0, 13.0).reshape((4, 3)) / 13.0
    epsilon = 1.0e-7
    plus = kernel.evaluate(gradients, volume, displacement + epsilon * direction)
    minus = kernel.evaluate(gradients, volume, displacement - epsilon * direction)

    np.testing.assert_allclose(
        response.tangent @ direction.ravel(),
        (plus.internal_force - minus.internal_force).ravel() / (2.0 * epsilon),
        rtol=2.0e-6,
        atol=2.0e-7,
    )
    np.testing.assert_allclose(response.tangent, response.tangent.T, atol=2.0e-12)
