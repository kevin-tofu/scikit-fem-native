"""Parity checks for MeshQuad submesh extraction APIs."""

import numpy as np
import pytest
import skfem

import skfemntv


def _mesh_pair():
    native=skfemntv.MeshQuad.init_tensor(
        [0.,1.,2.],[0.,1.,2.]
    ).with_boundaries({
        "left":lambda x:np.isclose(x[0],0.),
        "right":lambda x:np.isclose(x[0],2.),
    }).with_subdomains({
        "lower":lambda x:x[1]<1.,
    })
    reference=skfem.MeshQuad(native.p,native.t).with_boundaries({
        "left":lambda x:np.isclose(x[0],0.),
        "right":lambda x:np.isclose(x[0],2.),
    }).with_subdomains({
        "lower":lambda x:x[1]<1.,
    })
    return native,reference


def _points(points):
    return sorted(map(tuple,np.round(points.T,13)))


def _cell_keys(mesh,elements=None):
    cells=mesh.t if elements is None else mesh.t[:,elements]
    return sorted(
        tuple(sorted(map(tuple,np.round(mesh.p[:,cell].T,13))))
        for cell in cells.T
    )


def _facet_keys(mesh,facets):
    return sorted(
        tuple(sorted(map(tuple,np.round(mesh.p[:,mesh.facets[:,facet]].T,13))))
        for facet in facets
    )


@pytest.mark.parametrize("selector",[
    np.array([0,1,3]),lambda centers:centers[0]>.5,"lower",
],ids=["ids","callable","subdomain"])
def test_quadrilateral_restrict_matches_scikit_fem(selector):
    native,reference=_mesh_pair()
    actual,actual_mapping=native.restrict(selector,return_mapping=True)
    expected,expected_mapping=reference.restrict(selector,return_mapping=True)

    assert _points(native.p[:,actual_mapping])==_points(
        reference.p[:,expected_mapping]
    )
    assert _cell_keys(actual)==_cell_keys(expected)
    assert set(actual.boundaries)==set(expected.boundaries)
    for name in actual.boundaries:
        assert _facet_keys(actual,actual.boundaries[name])==_facet_keys(
            expected,expected.boundaries[name]
        )
    assert set(actual.subdomains)==set(expected.subdomains)
    for name in actual.subdomains:
        assert _cell_keys(actual,actual.subdomains[name])==_cell_keys(
            expected,expected.subdomains[name]
        )


def test_quadrilateral_restrict_skip_flags_and_empty_selection():
    native,reference=_mesh_pair()
    actual=native.restrict(
        [0,1],skip_boundaries=True,skip_subdomains=True
    )
    expected=reference.restrict(
        [0,1],skip_boundaries=True,skip_subdomains=True
    )
    assert (actual.boundaries or {})==(expected.boundaries or {})=={}
    assert (actual.subdomains or {})==(expected.subdomains or {})=={}

    actual,mapping=native.restrict([],return_mapping=True)
    assert actual.p.shape==(2,0)
    assert actual.t.shape==(4,0)
    assert mapping.shape==(0,)


@pytest.mark.parametrize("selector",[
    0,[1,2],"lower",lambda centers:centers[0]>.5,
],ids=["id","ids","subdomain","callable"])
def test_quadrilateral_remove_elements_matches_scikit_fem(selector):
    native,reference=_mesh_pair()
    actual=native.remove_elements(selector)
    expected=reference.remove_elements(selector)
    assert _cell_keys(actual)==_cell_keys(expected)
    for name in actual.boundaries:
        assert _facet_keys(actual,actual.boundaries[name])==_facet_keys(
            expected,expected.boundaries[name]
        )


def test_quadrilateral_restrict_rejects_invalid_selectors():
    mesh=skfemntv.MeshQuad()
    with pytest.raises(IndexError,match="out of range"):
        mesh.restrict([1])
    with pytest.raises(ValueError,match="one value per element"):
        mesh.restrict(np.array([True,False]))
