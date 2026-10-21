"""Analytic invariants for the native-only rational Pyramid5 H1 element."""

import numpy as np
import pytest

import skfemntv
from skfemntv.helpers import ddot, dot, grad


def test_pyramid5_partition_of_unity_and_volume_forms():
    basis = skfemntv.Basis(
        skfemntv.MeshPyramid1(), skfemntv.ElementPyramid1(), intorder=4,
    )

    np.testing.assert_allclose(
        basis.tabulated_shape.sum(axis=-1), 1., rtol=0., atol=2e-14,
    )
    np.testing.assert_allclose(
        basis.tabulated_gradients.sum(axis=-2), 0., rtol=0., atol=2e-14,
    )
    assert basis.dx.sum() == pytest.approx(1. / 3., abs=2e-14)

    @skfemntv.BilinearForm
    def mass(u, v, w):
        return dot(u, v)

    @skfemntv.BilinearForm
    def stiffness(u, v, w):
        return ddot(grad(u), grad(v))

    mass_matrix = skfemntv.asm(mass, basis).toarray()
    stiffness_matrix = skfemntv.asm(stiffness, basis).toarray()
    np.testing.assert_allclose(mass_matrix, mass_matrix.T, atol=2e-14)
    np.testing.assert_allclose(stiffness_matrix, stiffness_matrix.T, atol=2e-14)
    np.testing.assert_allclose(mass_matrix.sum(), 1. / 3., atol=2e-14)
    np.testing.assert_allclose(stiffness_matrix.sum(axis=0), 0., atol=2e-14)
