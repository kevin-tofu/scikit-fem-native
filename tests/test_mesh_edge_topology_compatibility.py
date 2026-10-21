import numpy as np
import pytest

import skfem
import skfemntv


def test_tetrahedral_edges_match_scikit_fem_numbering():
    native_mesh=skfemntv.MeshTet.init_tensor([0.,1.],[0.,1.],[0.,1.])
    upstream_mesh=skfem.MeshTet(native_mesh.p,native_mesh.t)
    _=upstream_mesh.t2e
    np.testing.assert_array_equal(native_mesh.t2e, upstream_mesh.t2e)
    np.testing.assert_array_equal(native_mesh.f2e, upstream_mesh.f2e)
    np.testing.assert_array_equal(
        native_mesh.boundary_edges(), upstream_mesh.boundary_edges()
    )


def test_triangle_edges_have_lexicographic_ids_and_boundary_selection():
    mesh=skfemntv.MeshTri(
        np.array([[0.,1.,0.,1.],[0.,0.,1.,1.]]),
        np.array([[0,1],[1,3],[2,2]]),
    )
    np.testing.assert_array_equal(mesh.edges,[[0,0,1,1,2],[1,2,2,3,3]])
    np.testing.assert_array_equal(mesh.boundary_edges(),np.array([0,1,3,4]))


def test_tetrahedral_edge_selectors_match_scikit_fem():
    p=np.array([[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]])
    t=np.array([[0],[1],[2],[3]])
    native = skfemntv.MeshTet(p,t)
    upstream = skfem.MeshTet(p,t)
    _=upstream.t2e
    predicate = lambda x: x[0] < .5
    np.testing.assert_array_equal(
        native.edges_satisfying(predicate), upstream.edges_satisfying(predicate)
    )
    np.testing.assert_array_equal(native.interior_edges(), upstream.interior_edges())


def test_hexahedral_edge_topology_matches_scikit_fem():
    native=skfemntv.MeshHex()
    upstream=skfem.MeshHex(native.p,native.t)
    np.testing.assert_array_equal(native.edges,upstream.edges)
    np.testing.assert_array_equal(native.t2e,upstream.t2e)
    np.testing.assert_array_equal(native.f2e,upstream.f2e)
    np.testing.assert_array_equal(
        native.boundary_edges(),upstream.boundary_edges()
    )
    np.testing.assert_array_equal(native.interior_edges(),upstream.interior_edges())


def test_hex20_edge_topology_uses_corner_connectivity():
    native=skfemntv.MeshHex20.from_mesh(skfemntv.MeshHex())
    upstream=skfem.MeshHex(native.p,native.t[:8])
    np.testing.assert_array_equal(native.edges,upstream.edges)
    np.testing.assert_array_equal(native.t2e,upstream.t2e)
    np.testing.assert_array_equal(native.f2e,upstream.f2e)
    np.testing.assert_array_equal(
        native.boundary_edges(),upstream.boundary_edges()
    )


def test_wedge_edge_topology_matches_scikit_fem():
    native=skfemntv.MeshWedge1()
    upstream=skfem.MeshWedge1(native.p,native.t)
    np.testing.assert_array_equal(native.edges,upstream.edges)
    np.testing.assert_array_equal(native.t2e,upstream.t2e)
    np.testing.assert_array_equal(
        native.boundary_edges(),np.arange(native.edges.shape[1])
    )
    assert native.interior_edges().size==0


def test_two_wedges_exclude_shared_face_edges_from_boundary():
    mesh=skfemntv.MeshWedge1.init_tensor([0.,1.],[0.,1.],[0.,1.])
    assert mesh.nelements==2
    assert np.count_nonzero(mesh.f2t[1]>=0)==1
    assert mesh.boundary_edges().size==mesh.edges.shape[1]
    assert mesh.interior_edges().size==0
