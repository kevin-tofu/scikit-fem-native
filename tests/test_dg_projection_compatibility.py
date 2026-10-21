"""Discontinuous simplex projection parity against scikit-fem."""

import numpy as np
import pytest
import skfem
import skfemntv


@pytest.mark.parametrize("dimension,order", [(2, 1), (2, 2), (3, 1), (3, 2)])
@pytest.mark.parametrize("vector", [False, True])
def test_discontinuous_projection_matches_skfem(dimension, order, vector):
    if dimension == 2:
        mesh = skfemntv.MeshTri.init_tensor([0., 1., 2.], [0., 1.])
        reference_mesh = skfem.MeshTri(mesh.p, mesh.t)
        native_base = skfemntv.ElementTriP1() if order == 1 else skfemntv.ElementTriP2()
        reference_base = skfem.ElementTriP1() if order == 1 else skfem.ElementTriP2()
    else:
        mesh = skfemntv.MeshTet.init_tensor([0., 1.], [0., 1.], [0., 1.])
        reference_mesh = skfem.MeshTet(mesh.p, mesh.t)
        native_base = skfemntv.ElementTetP1() if order == 1 else skfemntv.ElementTetP2()
        reference_base = skfem.ElementTetP1() if order == 1 else skfem.ElementTetP2()
    native_element = skfemntv.ElementDG(native_base)
    reference_element = skfem.ElementDG(reference_base)
    if vector:
        native_element = skfemntv.ElementVector(native_element, dim=dimension)
        reference_element = skfem.ElementVector(reference_element, dim=dimension)
    native = skfemntv.Basis(mesh, native_element, intorder=6)
    reference = skfem.Basis(reference_mesh, reference_element, intorder=6)

    def field(x):
        if vector:
            return np.stack([
                x[axis] ** 2 + (axis + 1.) * x[0] for axis in range(dimension)
            ])
        return x[0] ** 2 + .7 * x[-1]

    selected = np.arange(mesh.nelements)[::2]
    for elements in (None, selected):
        actual = native.project(field, elements=elements)
        expected = reference.project(field, elements=elements)
        np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-10)
