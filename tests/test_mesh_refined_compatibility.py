"""Parity checks for uniform MeshTri.refined behavior."""

import numpy as np
import pytest
import skfem

import skfemntv


def _point_keys(points):
    values = np.round(points.T, 13)
    return sorted(map(tuple, values))


def _cell_keys(mesh):
    return sorted(
        tuple(sorted(map(tuple, np.round(mesh.p[:, cell].T, 13))))
        for cell in mesh.t.T
    )


@pytest.mark.parametrize("steps,expected_elements", [(0, 2), (1, 8), (2, 32)])
def test_uniform_refinement_matches_scikit_fem_geometry(steps, expected_elements):
    native = skfemntv.MeshTri.init_tensor(
        np.array([0.0, 1.0]), np.array([0.0, 1.0])
    )
    reference = skfem.MeshTri.init_tensor(
        np.array([0.0, 1.0]), np.array([0.0, 1.0])
    )
    actual = native.refined(steps)
    expected = reference.refined(steps)

    assert actual.nelements == expected.nelements == expected_elements
    assert actual.p.shape[1] == expected.p.shape[1]
    assert _point_keys(actual.p) == _point_keys(expected.p)
    assert _cell_keys(actual) == _cell_keys(expected)


def test_refinement_preserves_named_boundaries_and_subdomains():
    native = skfemntv.MeshTri.init_tensor(
        np.array([0.0, 1.0]), np.array([0.0, 1.0])
    ).with_boundaries(
        {
            "left": lambda x: np.isclose(x[0], 0.0),
            "right": lambda x: np.isclose(x[0], 1.0),
        }
    ).with_subdomains({"lower": lambda x: x[1] < 0.5})
    reference = skfem.MeshTri.init_tensor(
        np.array([0.0, 1.0]), np.array([0.0, 1.0])
    ).with_boundaries(
        {
            "left": lambda x: np.isclose(x[0], 0.0),
            "right": lambda x: np.isclose(x[0], 1.0),
        }
    ).with_subdomains({"lower": lambda x: x[1] < 0.5})

    actual = native.refined(2)
    expected = reference.refined(2)
    assert set(actual.boundaries) == set(expected.boundaries)
    assert set(actual.subdomains) == set(expected.subdomains)
    for name in actual.boundaries:
        assert len(actual.boundaries[name]) == len(expected.boundaries[name]) == 4
    for name in actual.subdomains:
        actual_region = skfemntv.MeshTri(
            actual.p, actual.t[:, np.asarray(actual.subdomains[name])]
        )
        expected_region = skfem.MeshTri(
            expected.p, expected.t[:, np.asarray(expected.subdomains[name])]
        )
        assert _cell_keys(actual_region) == _cell_keys(expected_region)


def test_adaptive_refinement_is_conforming_and_preserves_domain():
    mesh = skfemntv.MeshTri.init_tensor(
        np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0])
    )
    refined = mesh.refined(np.array([0]))
    assert refined.nelements > mesh.nelements
    vectors = refined.p[:, refined.t[1:3]] - refined.p[:, refined.t[0]][:, None, :]
    area = np.abs(
        vectors[0, 0] * vectors[1, 1] - vectors[1, 0] * vectors[0, 1]
    ).sum() / 2.0
    assert np.isclose(area, 4.0)
    assert np.all(refined.f2t[1, refined.boundary_facets()] == -1)
