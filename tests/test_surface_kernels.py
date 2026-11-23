from __future__ import annotations

import numpy as np

from skfemntv import Quad4SurfaceKernel, Tri6SurfaceKernel


def test_quad4_surface_kernel_integrates_area_and_nodal_weights() -> None:
    points = np.asarray(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 3.0, 0.0], [0.0, 3.0, 0.0]]
    )
    kernel = Quad4SurfaceKernel(points, [[0, 1, 2, 3]])

    np.testing.assert_allclose(kernel.integrate_scalar(np.ones((1, 4))), [6.0])
    np.testing.assert_allclose(kernel.element_nodal_areas(), [[1.5] * 4])
    np.testing.assert_allclose(kernel.data.normals, [[[0.0, 0.0, 1.0]] * 4])


def test_tri6_surface_kernel_integrates_area_and_total_traction() -> None:
    coordinates = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.5, 0.0],
            [0.0, 0.5, 0.0],
        ]
    )
    kernel = Tri6SurfaceKernel(integration_order=4)
    force = np.asarray([2.0, -3.0, 5.0])
    area = kernel.area(coordinates)
    nodal_force = kernel.integrate_traction(coordinates, force / area)

    np.testing.assert_allclose(area, 1.0, atol=1.0e-13)
    np.testing.assert_allclose(nodal_force.sum(axis=0), force, atol=1.0e-13)
    np.testing.assert_allclose(
        kernel.nodal_tributary_areas(coordinates).sum(), area, atol=1.0e-13
    )
