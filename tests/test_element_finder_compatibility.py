"""Small point-location parity checks against scikit-fem meshes."""

import numpy as np
import pytest
import skfem

import skfemntv


@pytest.mark.parametrize("dimension", [2, 3])
def test_simplex_element_finder_matches_scikit_fem(dimension):
    if dimension == 2:
        native = skfemntv.MeshTri.init_tensor(
            [0., .5, 1.], [0., .5, 1.]
        )
        reference = skfem.MeshTri(native.p, native.t)
    else:
        native = skfemntv.MeshTet.init_tensor(
            [0., 1.], [0., 1.], [0., 1.]
        )
        reference = skfem.MeshTet(native.p, native.t)

    points = native.p[:, native.t].mean(axis=1)
    expected = reference.element_finder()(*points)
    actual = native.element_finder()(*points)
    np.testing.assert_array_equal(actual, expected)


def test_curved_tet10_element_finder_matches_scikit_fem():
    linear = skfemntv.MeshTet()
    native = skfemntv.MeshTet2.from_mesh(linear)
    points = native.p.copy()
    points[2, 4] += .12
    native = skfemntv.MeshTet2(points, native.t.copy())

    reference_linear = skfem.MeshTet(linear.p, linear.t)
    reference = skfem.MeshTet2.from_mesh(reference_linear)
    reference_points = reference.doflocs.copy()
    reference_points[2, 4] += .12
    reference = skfem.MeshTet2(
        doflocs=reference_points, t=reference.t.copy()
    )
    reference_coordinates = np.array(((.2, .15), (.2, .25), (.25, .2)))
    query = reference.mapping().F(
        reference_coordinates[:, None, :], tind=np.zeros(2, dtype=int)
    )[:, 0, :]

    np.testing.assert_array_equal(
        native.element_finder()(*query), reference.element_finder()(*query)
    )


def test_element_finder_raises_for_points_outside_mesh():
    finder = skfemntv.MeshTri().element_finder()
    with pytest.raises(ValueError, match="outside of the mesh"):
        finder(np.array([2.]), np.array([2.]))


@pytest.mark.parametrize("dimension", [2, 3])
def test_tensor_element_finder_matches_scikit_fem(dimension):
    if dimension == 2:
        native = skfemntv.MeshQuad.init_tensor(
            [0., .5, 1.], [0., .5, 1.]
        )
        reference = skfem.MeshQuad(native.p, native.t)
    else:
        native = skfemntv.MeshHex.init_tensor(
            [0., 1., 2.], [0., 1.], [0., 1.]
        )
        reference = skfem.MeshHex(native.p, native.t)
    points = native.p[:, native.t].mean(axis=1)
    np.testing.assert_array_equal(
        native.element_finder()(*points), reference.element_finder()(*points)
    )


def test_element_finder_rejects_custom_mapping():
    with pytest.raises(NotImplementedError, match="custom"):
        skfemntv.MeshTri().element_finder(mapping=object())
