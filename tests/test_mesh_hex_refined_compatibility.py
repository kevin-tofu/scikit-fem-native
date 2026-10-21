"""Parity checks for uniform MeshHex.refined behavior."""

import numpy as np
import pytest
import skfem

import skfemntv


def _cell_keys(mesh, elements=None):
    cells = mesh.t if elements is None else mesh.t[:, elements]
    return sorted(
        tuple(sorted(map(tuple, np.round(mesh.p[:, cell].T, 13))))
        for cell in cells.T
    )


def _facet_keys(mesh, facets):
    return sorted(
        tuple(sorted(map(tuple, np.round(mesh.p[:, mesh.facets[:, facet]].T, 13))))
        for facet in facets
    )


@pytest.mark.parametrize("steps, elements", [(0, 2), (1, 16), (2, 128)])
def test_uniform_hexahedron_refinement_matches_scikit_fem(steps, elements):
    native = skfemntv.MeshHex.init_tensor(
        np.array([0., 1., 2.]), np.array([0., 1.]), np.array([0., 1.])
    )
    reference = skfem.MeshHex(native.p, native.t)

    actual = native.refined(steps)
    expected = reference.refined(steps)

    assert actual.nelements == expected.nelements == elements
    assert _cell_keys(actual) == _cell_keys(expected)


def test_uniform_hexahedron_refinement_preserves_named_regions():
    native = skfemntv.MeshHex.init_tensor(
        [0., 1., 2.], [0., 1., 2.], [0., 1.]
    ).with_boundaries({
        "left": lambda x: np.isclose(x[0], 0.),
        "top": lambda x: np.isclose(x[2], 1.),
    }).with_subdomains({
        "left_cell": lambda x: x[0] < 1.,
    })
    reference = skfem.MeshHex(native.p, native.t).with_boundaries({
        "left": lambda x: np.isclose(x[0], 0.),
        "top": lambda x: np.isclose(x[2], 1.),
    }).with_subdomains({
        "left_cell": lambda x: x[0] < 1.,
    })

    actual = native.refined(1)
    expected = reference.refined(1).with_boundaries({
        "left": lambda x: np.isclose(x[0], 0.),
        "top": lambda x: np.isclose(x[2], 1.),
    }).with_subdomains({
        "left_cell": lambda x: x[0] < 1.,
    })
    assert set(actual.boundaries) == set(expected.boundaries)
    for name in actual.boundaries:
        assert _facet_keys(actual, actual.boundaries[name]) == _facet_keys(
            expected, expected.boundaries[name]
        )
    assert set(actual.subdomains) == set(expected.subdomains)
    for name in actual.subdomains:
        assert _cell_keys(actual, actual.subdomains[name]) == _cell_keys(
            expected, expected.subdomains[name]
        )


def test_hexahedron_refinement_input_contract():
    mesh = skfemntv.MeshHex()
    assert mesh.refined(0) is mesh
    with pytest.raises(ValueError, match="nonnegative"):
        mesh.refined(-1)
    with pytest.raises(NotImplementedError, match="adaptive"):
        mesh.refined(np.array([0]))


def test_hexahedron_restrict_and_remove_match_scikit_fem():
    native = skfemntv.MeshHex.init_tensor(
        [0., 1., 2.], [0., 1.], [0., 1.]
    ).with_boundaries({
        "left": lambda x: np.isclose(x[0], 0.),
        "right": lambda x: np.isclose(x[0], 2.),
    }).with_subdomains({
        "first": lambda x: x[0] < 1.,
    })
    reference = skfem.MeshHex(native.p, native.t).with_boundaries({
        "left": lambda x: np.isclose(x[0], 0.),
        "right": lambda x: np.isclose(x[0], 2.),
    }).with_subdomains({
        "first": lambda x: x[0] < 1.,
    })

    actual, used_nodes = native.restrict("first", return_mapping=True)
    expected, expected_nodes = reference.restrict(
        "first", return_mapping=True
    )
    np.testing.assert_array_equal(used_nodes, expected_nodes)
    np.testing.assert_array_equal(actual.p, expected.p)
    np.testing.assert_array_equal(actual.t, expected.t)
    for name in actual.boundaries:
        assert _facet_keys(actual, actual.boundaries[name]) == _facet_keys(
            expected, expected.boundaries[name]
        )
    assert _cell_keys(actual, actual.subdomains["first"]) == _cell_keys(
        expected, expected.subdomains["first"]
    )

    actual_removed = native.remove_elements([0])
    expected_removed = reference.remove_elements([0])
    np.testing.assert_array_equal(actual_removed.p, expected_removed.p)
    np.testing.assert_array_equal(actual_removed.t, expected_removed.t)
