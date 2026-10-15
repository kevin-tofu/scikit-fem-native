"""Parity checks for uniform MeshQuad.refined behavior."""

import numpy as np
import pytest
import skfem

import skfemntv


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


@pytest.mark.parametrize("steps,elements",[(0,2),(1,8),(2,32)])
def test_uniform_quadrilateral_refinement_matches_scikit_fem(steps,elements):
    native=skfemntv.MeshQuad.init_tensor(
        np.array([0.,1.,2.]),np.array([0.,1.])
    )
    reference=skfem.MeshQuad(native.p,native.t)

    actual=native.refined(steps)
    expected=reference.refined(steps)

    assert actual.nelements==expected.nelements==elements
    np.testing.assert_array_equal(actual.p,expected.p)
    np.testing.assert_array_equal(actual.t,expected.t)


def test_uniform_quadrilateral_refinement_preserves_named_regions():
    native=skfemntv.MeshQuad.init_tensor(
        [0.,1.,2.],[0.,1.,2.]
    ).with_boundaries({
        "left":lambda x:np.isclose(x[0],0.),
        "top":lambda x:np.isclose(x[1],2.),
    }).with_subdomains({
        "lower":lambda x:x[1]<1.,
    })
    reference=skfem.MeshQuad(native.p,native.t).with_boundaries({
        "left":lambda x:np.isclose(x[0],0.),
        "top":lambda x:np.isclose(x[1],2.),
    }).with_subdomains({
        "lower":lambda x:x[1]<1.,
    })

    actual=native.refined(2)
    expected=reference.refined(2)
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


def test_quadrilateral_refinement_input_contract():
    mesh=skfemntv.MeshQuad()
    assert mesh.refined(0) is mesh
    with pytest.raises(ValueError,match="nonnegative"):
        mesh.refined(-1)
    with pytest.raises(NotImplementedError,match="adaptive"):
        mesh.refined(np.array([0]))
