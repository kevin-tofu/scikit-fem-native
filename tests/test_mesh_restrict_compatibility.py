"""Parity checks for MeshTri submesh extraction APIs."""

import numpy as np
import pytest
import skfem

import skfemntv


def _mesh_pair():
    x = np.array([0.0, 1.0, 2.0])
    native = skfemntv.MeshTri.init_tensor(x, x)
    reference = skfem.MeshTri.init_tensor(x, x)
    boundary_selectors = {
        "left": lambda points: np.isclose(points[0], 0.0),
        "right": lambda points: np.isclose(points[0], 2.0),
    }
    subdomain_selectors = {"lower": lambda points: points[1] < 1.0}
    return (
        native.with_boundaries(boundary_selectors).with_subdomains(subdomain_selectors),
        reference.with_boundaries(boundary_selectors).with_subdomains(subdomain_selectors),
    )


def _cell_keys(mesh):
    return sorted(
        tuple(sorted(map(tuple, np.round(mesh.p[:, cell].T, 13))))
        for cell in mesh.t.T
    )


def _cell_key(mesh, element):
    return tuple(sorted(map(tuple, np.round(mesh.p[:, mesh.t[:, element]].T, 13))))


def _point_keys(points):
    return sorted(map(tuple, np.round(points.T, 13)))


def _corresponding_ids(native, reference, reference_ids):
    native_by_geometry = {
        _cell_key(native, element): element for element in range(native.nelements)
    }
    return np.asarray(
        [native_by_geometry[_cell_key(reference, element)] for element in reference_ids],
        dtype=np.int64,
    )


def _tagged_edge_keys(mesh, name):
    return sorted(
        tuple(sorted(map(tuple, np.round(mesh.p[:, mesh.facets[:, facet]].T, 13))))
        for facet in mesh.boundaries[name]
    )


@pytest.mark.parametrize(
    "selector",
    [
        np.array([0, 1, 4]),
        lambda centers: centers[0] > 0.5,
        "lower",
    ],
    ids=["ids", "callable", "subdomain"],
)
def test_restrict_matches_scikit_fem_for_common_selectors(selector):
    native, reference = _mesh_pair()
    if isinstance(selector, np.ndarray):
        reference_selector = selector
        native_selector = _corresponding_ids(native, reference, reference_selector)
    else:
        native_selector = reference_selector = selector
    native_result, native_mapping = native.restrict(native_selector, return_mapping=True)
    reference_result, reference_mapping = reference.restrict(reference_selector, return_mapping=True)

    assert _point_keys(native.p[:, native_mapping]) == _point_keys(
        reference.p[:, reference_mapping]
    )
    assert _cell_keys(native_result) == _cell_keys(reference_result)
    assert set(native_result.boundaries) == set(reference_result.boundaries)
    assert set(native_result.subdomains) == set(reference_result.subdomains)
    for name in native_result.boundaries:
        assert _tagged_edge_keys(native_result, name) == _tagged_edge_keys(reference_result, name)
    for name in native_result.subdomains:
        native_region = skfemntv.MeshTri(
            native_result.p,
            native_result.t[:, np.asarray(native_result.subdomains[name])],
        )
        reference_region = skfem.MeshTri(
            reference_result.p,
            reference_result.t[:, np.asarray(reference_result.subdomains[name])],
        )
        assert _cell_keys(native_region) == _cell_keys(reference_region)


def test_restrict_skip_flags_and_empty_selection():
    native, reference = _mesh_pair()
    actual = native.restrict([0, 1], skip_boundaries=True, skip_subdomains=True)
    expected = reference.restrict([0, 1], skip_boundaries=True, skip_subdomains=True)
    assert (actual.boundaries or {}) == (expected.boundaries or {}) == {}
    assert (actual.subdomains or {}) == (expected.subdomains or {}) == {}
    empty = native.restrict(np.array([], dtype=np.int64), return_mapping=True)
    assert empty[0].p.shape == (2, 0)
    assert empty[0].t.shape == (3, 0)
    assert empty[1].shape == (0,)


@pytest.mark.parametrize("selector", [0, [1, 2], "lower", lambda centers: centers[0] > 0.5])
def test_remove_elements_matches_scikit_fem(selector):
    native, reference = _mesh_pair()
    native_selector = selector
    if isinstance(selector, int):
        native_selector = _corresponding_ids(native, reference, [selector])
    elif isinstance(selector, list):
        native_selector = _corresponding_ids(native, reference, selector)
    actual = native.remove_elements(native_selector)
    expected = reference.remove_elements(selector)
    assert _cell_keys(actual) == _cell_keys(expected)
    for name in actual.boundaries:
        assert _tagged_edge_keys(actual, name) == _tagged_edge_keys(expected, name)
