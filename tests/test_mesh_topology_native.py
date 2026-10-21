import numpy as np
import pytest
import skfem

import skfemntv
import skfemntv.basis as basis_module


def _python_tri_topology(mesh):
    local = mesh._local_facets()
    indexing = np.hstack(tuple(mesh.t[np.asarray(face)] for face in local))
    canonical, first, inverse = np.unique(
        np.sort(indexing, axis=0), axis=1,
        return_index=True, return_inverse=True,
    )
    facets = canonical
    t2f = inverse.reshape(len(local), mesh.nelements)
    f2t = np.full((2, facets.shape[1]), -1, dtype=np.int64)
    flat = t2f.ravel(order="C")
    elements = np.tile(np.arange(mesh.nelements), len(local))
    for facet, element in zip(flat, elements):
        row = 0 if f2t[0, facet] == -1 else 1
        f2t[row, facet] = element
    return facets, t2f, f2t


def test_native_tri_topology_matches_numpy_facet_order_and_adjacency():
    if basis_module._build_tri_facet_topology_native is None:
        pytest.skip("native triangle topology kernel is unavailable")
    base = skfemntv.MeshTri.init_tensor(
        np.linspace(0., 1., 8), np.linspace(0., 1., 6)
    )
    mesh = skfemntv.MeshTri(base.p, base.t[:, ::-1].copy())
    expected = _python_tri_topology(mesh)

    assert np.array_equal(mesh.facets, expected[0])
    assert np.array_equal(mesh.t2f, expected[1])
    assert np.array_equal(mesh.f2t, expected[2])
    np.testing.assert_array_equal(
        mesh.boundary_facets(), np.flatnonzero(expected[2][1] == -1)
    )


def test_native_tri_topology_accepts_empty_connectivity():
    if basis_module._build_tri_facet_topology_native is None:
        pytest.skip("native triangle topology kernel is unavailable")
    mesh = skfemntv.MeshTri(
        np.empty((2, 0)), np.empty((3, 0), dtype=np.int64)
    )
    assert mesh.facets.shape == (2, 0)
    assert mesh.t2f.shape == (3, 0)
    assert mesh.f2t.shape == (2, 0)


def test_native_topology_keeps_refinement_regions_identical_to_python(monkeypatch):
    if basis_module._build_tri_facet_topology_native is None:
        pytest.skip("native triangle topology kernel is unavailable")
    native_builder = basis_module._build_tri_facet_topology_native

    def make_mesh():
        return skfemntv.MeshTri.init_tensor(
            np.linspace(0., 1., 4), np.linspace(0., 1., 3)
        ).with_boundaries({
            "bottom": lambda x: np.isclose(x[1], 0.),
        }).with_subdomains({
            "left": lambda x: x[0] < .5,
        })

    monkeypatch.setattr(basis_module, "_build_tri_facet_topology_native", None)
    python_refined = make_mesh().refined(1)
    monkeypatch.setattr(
        basis_module, "_build_tri_facet_topology_native", native_builder
    )
    native_refined = make_mesh().refined(1)

    np.testing.assert_array_equal(native_refined.p, python_refined.p)
    np.testing.assert_array_equal(native_refined.t, python_refined.t)
    np.testing.assert_array_equal(native_refined.facets, python_refined.facets)
    np.testing.assert_array_equal(native_refined.t2f, python_refined.t2f)
    np.testing.assert_array_equal(native_refined.f2t, python_refined.f2t)
    assert set(native_refined.boundaries) == set(python_refined.boundaries)
    assert set(native_refined.subdomains) == set(python_refined.subdomains)
    for name in native_refined.boundaries:
        np.testing.assert_array_equal(
            np.asarray(native_refined.boundaries[name]),
            np.asarray(python_refined.boundaries[name]),
        )
    for name in native_refined.subdomains:
        np.testing.assert_array_equal(
            np.asarray(native_refined.subdomains[name]),
            np.asarray(python_refined.subdomains[name]),
        )


@pytest.mark.parametrize("dimension",[2,3])
def test_facets_around_matches_scikit_fem_for_cell_subdomains_and_flip(dimension):
    if dimension==2:
        native=skfemntv.MeshTri.init_tensor(
            np.linspace(0.,1.,4),np.linspace(0.,1.,3)
        )
        upstream=skfem.MeshTri(native.p,native.t)
    else:
        native=skfemntv.MeshTet.init_tensor(
            np.linspace(0.,1.,3),np.linspace(0.,1.,2),np.linspace(0.,1.,2)
        )
        upstream=skfem.MeshTet(native.p,native.t)
    selected=np.arange(native.nelements//2,dtype=np.int64)
    for flip in (False,True):
        actual=native.facets_around(selected,flip=flip)
        expected=upstream.facets_around(selected,flip=flip)
        np.testing.assert_array_equal(actual,expected)
        np.testing.assert_array_equal(actual.sides,expected.ori)


def test_facets_around_empty_and_full_mesh():
    mesh=skfemntv.MeshTri.init_tensor([0.,1.],[0.,1.])
    empty=mesh.facets_around([])
    assert len(empty)==0
    full=mesh.facets_around(np.arange(mesh.nelements))
    np.testing.assert_array_equal(full,mesh.boundary_facets())
    np.testing.assert_array_equal(full.sides,np.zeros(len(full),dtype=np.int8))


def test_normalize_facets_matches_scikit_fem_selector_forms():
    native=skfemntv.MeshTri.init_tensor(
        np.linspace(0.,1.,3),np.linspace(0.,1.,3)
    ).with_boundaries({"left":lambda x:np.isclose(x[0],0.)})
    upstream=skfem.MeshTri(native.p,native.t).with_boundaries(
        {"left":lambda x:np.isclose(x[0],0.)}
    )
    predicate=lambda x:np.isclose(x[1],1.)
    array=np.array([0,2],dtype=np.int64)
    mask=np.zeros(native.facets.shape[1],dtype=bool)
    mask[[1,3]]=True
    for selector in (None,1,array,mask,predicate,"left",["left",predicate]):
        actual=native.normalize_facets(selector)
        expected=upstream.normalize_facets(selector)
        np.testing.assert_array_equal(actual,expected)


def test_normalize_nodes_matches_scikit_fem_selector_forms():
    native=skfemntv.MeshTri.init_tensor(
        np.linspace(0.,1.,3),np.linspace(0.,1.,3)
    )
    upstream=skfem.MeshTri(native.p,native.t)
    predicate=lambda x:np.isclose(x[0],0.)
    ids=np.array([0,4],dtype=np.int64)
    for selector in ((0.,0.),ids,predicate,[(0.,0.),(1.,1.)]):
        np.testing.assert_array_equal(
            native.normalize_nodes(selector),upstream.normalize_nodes(selector)
        )


def test_boundary_and_subdomain_labels_accumulate_like_scikit_fem():
    native=skfemntv.MeshTri.init_tensor([0.,1.,2.],[0.,1.,2.])
    upstream=skfem.MeshTri(native.p,native.t)
    native=native.with_boundaries({"left":lambda x:np.isclose(x[0],0.)})
    upstream=upstream.with_boundaries({"left":lambda x:np.isclose(x[0],0.)})
    native=native.with_boundaries({"right":lambda x:np.isclose(x[0],2.)})
    upstream=upstream.with_boundaries({"right":lambda x:np.isclose(x[0],2.)})
    assert set(native.boundaries)==set(upstream.boundaries)=={"left","right"}
    for name in native.boundaries:
        np.testing.assert_array_equal(native.boundaries[name],upstream.boundaries[name])

    native=native.with_subdomains({"lower":lambda x:x[1]<1.})
    upstream=upstream.with_subdomains({"lower":lambda x:x[1]<1.})
    native=native.with_subdomains({"upper":lambda x:x[1]>1.})
    upstream=upstream.with_subdomains({"upper":lambda x:x[1]>1.})
    assert set(native.subdomains)==set(upstream.subdomains)=={"lower","upper"}
    for name in native.subdomains:
        np.testing.assert_array_equal(native.subdomains[name],upstream.subdomains[name])


def test_with_boundaries_can_name_interior_facets():
    native=skfemntv.MeshTri.init_tensor(
        np.linspace(0.,1.,3),np.linspace(0.,1.,2)
    )
    upstream=skfem.MeshTri(native.p,native.t)
    predicate=lambda x:np.isclose(x[0],.5)
    actual=native.with_boundaries(
        {"interface":predicate},boundaries_only=False
    )
    expected=upstream.with_boundaries(
        {"interface":predicate},boundaries_only=False
    )
    np.testing.assert_array_equal(
        actual.boundaries["interface"],expected.boundaries["interface"]
    )
    assert np.any(actual.f2t[1,actual.boundaries["interface"]]>=0)

    default=native.with_boundaries({"side":predicate})
    reference=upstream.with_boundaries({"side":predicate})
    np.testing.assert_array_equal(
        default.boundaries["side"],reference.boundaries["side"]
    )


def test_wedge_restrict_and_remove_preserve_regions_and_mapping():
    native=skfemntv.MeshWedge1.init_tensor([0.,1.],[0.,1.],[0.,1.])
    upstream=skfem.MeshWedge1(native.p,native.t)
    native=native.with_boundaries(
        {"bottom":lambda x:np.isclose(x[2],0.)},boundaries_only=False
    ).with_subdomains({"first":[0],"both":[0,1]})
    upstream=upstream.with_boundaries(
        {"bottom":lambda x:np.isclose(x[2],0.)},boundaries_only=False
    ).with_subdomains({"first":[0],"both":[0,1]})
    actual,mapping=native.restrict([0],return_mapping=True)
    expected,expected_mapping=upstream.restrict([0],return_mapping=True)
    np.testing.assert_array_equal(mapping,expected_mapping)
    np.testing.assert_array_equal(actual.p,expected.p)
    np.testing.assert_array_equal(actual.t,expected.t)
    assert set(actual.boundaries)==set(expected.boundaries)
    for name in actual.boundaries:
        actual_facets={
            tuple(sorted(map(int,np.unique(actual.facets[:,facet]))))
            for facet in actual.boundaries[name]
        }
        expected_facets={
            tuple(sorted(map(int,np.unique(expected.facets[:,facet]))))
            for facet in expected.boundaries[name]
        }
        assert actual_facets==expected_facets
    for name in actual.subdomains:
        np.testing.assert_array_equal(actual.subdomains[name],expected.subdomains[name])
    removed=native.remove_elements([1])
    np.testing.assert_array_equal(removed.p,actual.p)
    np.testing.assert_array_equal(removed.t,actual.t)


@pytest.mark.parametrize("family",["line","tri","quad","tet","hex","wedge"])
def test_default_mesh_tags_and_cell_sizes_match_scikit_fem(family):
    if family=="line":
        native=skfemntv.MeshLine.init_tensor([0.,.4,1.])
        upstream=skfem.MeshLine(native.p,native.t)
    elif family=="tri":
        native=skfemntv.MeshTri.init_tensor([0.,.5,1.],[0.,.5,1.])
        upstream=skfem.MeshTri(native.p,native.t)
    elif family=="quad":
        native=skfemntv.MeshQuad.init_tensor([0.,.5,1.],[0.,.5,1.])
        upstream=skfem.MeshQuad(native.p,native.t)
    elif family=="tet":
        native=skfemntv.MeshTet.init_tensor([0.,1.],[0.,1.],[0.,1.])
        upstream=skfem.MeshTet(native.p,native.t)
    elif family=="hex":
        native=skfemntv.MeshHex()
        upstream=skfem.MeshHex(native.p,native.t)
    else:
        native=skfemntv.MeshWedge1.init_tensor([0.,1.],[0.,1.],[0.,1.])
        upstream=skfem.MeshWedge1(native.p,native.t)

    if family!="line":
        np.testing.assert_allclose(native.params(),upstream.params())
    assert native.param()==pytest.approx(upstream.param())
    actual=native.with_defaults()
    expected=(None if family=="line" else upstream.with_defaults().boundaries)
    if family=="line":
        assert set(actual.boundaries)=={"left","right"}
        for name,coordinate in (("left",0.),("right",1.)):
            nodes=np.unique(native.facets[:,actual.boundaries[name]])
            np.testing.assert_allclose(native.p[0,nodes],coordinate)
        return
    assert set(actual.boundaries)==set(expected)
    for name in actual.boundaries:
        if family!="wedge":
            np.testing.assert_array_equal(actual.boundaries[name],expected[name])
        else:
            def facet_keys(mesh,ids):
                return sorted(
                    tuple(sorted(map(int,np.unique(mesh.facets[:,facet]))))
                    for facet in ids
                )
            assert facet_keys(native,actual.boundaries[name])==facet_keys(
                upstream,expected[name]
            )
