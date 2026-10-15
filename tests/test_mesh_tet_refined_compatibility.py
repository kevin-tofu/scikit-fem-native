"""Parity checks for uniform MeshTet.refined behavior."""

import numpy as np
import pytest
import skfem

import skfemntv


def _cell_keys(mesh,cell_ids=None):
    cells=mesh.t if cell_ids is None else mesh.t[:,cell_ids]
    return sorted(
        tuple(sorted(map(tuple,np.round(mesh.p[:,cell].T,13))))
        for cell in cells.T
    )


def _facet_keys(mesh,facet_ids):
    return sorted(
        tuple(sorted(map(tuple,np.round(mesh.p[:,mesh.facets[:,facet]].T,13))))
        for facet in facet_ids
    )


@pytest.mark.parametrize("steps,elements",[(0,6),(1,48),(2,384)])
def test_uniform_tetrahedron_refinement_matches_scikit_fem(steps,elements):
    native=skfemntv.MeshTet.init_tensor([0.,1.],[0.,1.],[0.,1.])
    reference=skfem.MeshTet(native.p,native.t)

    actual=native.refined(steps)
    expected=reference.refined(steps)

    assert actual.nelements==expected.nelements==elements
    np.testing.assert_array_equal(actual.p,expected.p)
    np.testing.assert_array_equal(actual.t,expected.t)


def test_uniform_tetrahedron_refinement_preserves_named_regions():
    native=skfemntv.MeshTet.init_tensor(
        [0.,.5,1.],[0.,1.],[0.,1.]
    ).with_boundaries({
        "left":lambda x:np.isclose(x[0],0.),
        "top":lambda x:np.isclose(x[2],1.),
    }).with_subdomains({
        "lower":lambda x:x[2]<.5,
    })
    reference=skfem.MeshTet(native.p,native.t).with_boundaries({
        "left":lambda x:np.isclose(x[0],0.),
        "top":lambda x:np.isclose(x[2],1.),
    }).with_subdomains({
        "lower":lambda x:x[2]<.5,
    })

    actual=native.refined(2)
    expected=reference.refined(2).with_boundaries({
        "left":lambda x:np.isclose(x[0],0.),
        "top":lambda x:np.isclose(x[2],1.),
    })

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


@pytest.mark.parametrize("marked",[[0],[0,3]])
def test_adaptive_tetrahedron_refinement_matches_scikit_fem(marked):
    native=skfemntv.MeshTet.init_tensor([0.,1.,2.],[0.,1.],[0.,1.])
    reference=skfem.MeshTet(native.p,native.t)

    actual=native.refined(np.asarray(marked,dtype=np.int64))
    expected=reference.refined(np.asarray(marked,dtype=np.int64))

    np.testing.assert_array_equal(actual.p,expected.p)
    np.testing.assert_array_equal(actual.t,expected.t)
    assert np.all(actual.f2t[1,actual.boundary_facets()]==-1)


def test_adaptive_tetrahedron_refinement_preserves_named_regions():
    native=skfemntv.MeshTet.init_tensor(
        [0.,1.,2.],[0.,1.],[0.,1.]
    ).with_boundaries({
        "left":lambda x:np.isclose(x[0],0.),
        "top":lambda x:np.isclose(x[2],1.),
    }).with_subdomains({
        "left_half":lambda x:x[0]<1.,
    })
    refined=native.refined(np.array([0]))

    assert set(refined.boundaries)=={"left","top"}
    for name,axis,value in (("left",0,0.),("top",2,1.)):
        points=refined.p[:,refined.facets[:,refined.boundaries[name]]]
        assert np.allclose(points[axis],value)
    assert set(refined.subdomains)=={"left_half"}
    centers=refined.p[:,refined.t].mean(axis=1)
    assert np.all(centers[0,refined.subdomains["left_half"]]<1.)
    assert np.all(centers[0,np.setdiff1d(
        np.arange(refined.nelements),refined.subdomains["left_half"]
    )]>=1.)


def test_tetrahedron_refinement_input_contract():
    mesh=skfemntv.MeshTet()
    assert mesh.refined(0) is mesh
    with pytest.raises(ValueError,match="nonnegative"):
        mesh.refined(-1)
    assert mesh.refined([]) is mesh
    with pytest.raises(ValueError,match="1D integer"):
        mesh.refined(np.array([[0]]))
    with pytest.raises(IndexError,match="out of range"):
        mesh.refined(np.array([1]))
