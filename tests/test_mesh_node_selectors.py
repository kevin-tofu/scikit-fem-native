"""Node boundary and predicate selectors across linear and quadratic meshes."""

import numpy as np
import pytest
import skfem

import skfemntv


def _mesh_pairs():
    line = skfemntv.MeshLine.init_tensor([0., .5, 1.])
    tri = skfemntv.MeshTri.init_tensor([0., .5, 1.], [0., .5, 1.])
    quad = skfemntv.MeshQuad.init_tensor([0., .5, 1.], [0., .5, 1.])
    tet = skfemntv.MeshTet.init_tensor([0., .5, 1.], [0., .5, 1.], [0., .5, 1.])
    hex_mesh = skfemntv.MeshHex.init_tensor([0., .5, 1.], [0., .5, 1.], [0., .5, 1.])
    return (
        (line, skfem.MeshLine(line.p, line.t)),
        (tri, skfem.MeshTri(tri.p, tri.t)),
        (quad, skfem.MeshQuad(quad.p, quad.t)),
        (tet, skfem.MeshTet(tet.p, tet.t)),
        (hex_mesh, skfem.MeshHex(hex_mesh.p, hex_mesh.t)),
        (skfemntv.MeshTri2.from_mesh(tri), skfem.MeshTri2.from_mesh(skfem.MeshTri(tri.p, tri.t))),
        (skfemntv.MeshQuad2.from_mesh(quad), skfem.MeshQuad2.from_mesh(skfem.MeshQuad(quad.p, quad.t))),
        (skfemntv.MeshTet2.from_mesh(tet), skfem.MeshTet2.from_mesh(skfem.MeshTet(tet.p, tet.t))),
        (skfemntv.MeshHex2.from_mesh(hex_mesh), skfem.MeshHex2.from_mesh(skfem.MeshHex(hex_mesh.p, hex_mesh.t))),
    )


def _sorted_points(points):
    return points[:, np.lexsort(tuple(points[::-1]))]


@pytest.mark.parametrize("native,reference", _mesh_pairs())
def test_boundary_and_interior_nodes_match_scikit_fem(native, reference):
    native_boundary = native.boundary_nodes()
    reference_boundary = reference.boundary_nodes()
    np.testing.assert_allclose(
        _sorted_points(native.p[:, native_boundary]),
        _sorted_points(reference.p[:, reference_boundary]),
        rtol=0., atol=2e-14,
    )
    native_interior = native.interior_nodes()
    reference_interior = reference.interior_nodes()
    np.testing.assert_allclose(
        _sorted_points(native.p[:, native_interior]),
        _sorted_points(reference.p[:, reference_interior]),
        rtol=0., atol=2e-14,
    )


@pytest.mark.parametrize("native,reference", _mesh_pairs())
def test_nodes_satisfying_matches_scikit_fem(native, reference):
    predicate = lambda points: points[0] >= .5 - 1e-12
    for boundaries_only in (False, True):
        native_nodes = native.nodes_satisfying(
            predicate,boundaries_only=boundaries_only,
        )
        reference_nodes = reference.nodes_satisfying(
            predicate,boundaries_only=boundaries_only,
        )
        np.testing.assert_allclose(
            _sorted_points(native.p[:, native_nodes]),
            _sorted_points(reference.p[:, reference_nodes]),
            rtol=0., atol=2e-14,
        )


def test_nodes_satisfying_validates_predicate_shape():
    with pytest.raises(ValueError,match="one boolean per node"):
        skfemntv.MeshTri().nodes_satisfying(lambda points: np.array([True]))


def test_hex_mesh_local_order_and_facets_match_scikit_fem():
    x=np.asarray((0.,.7,2.1))
    y=np.asarray((-1.,.4,1.8))
    z=np.asarray((.2,1.3,3.7))
    native=skfemntv.MeshHex.init_tensor(x,y,z)
    reference=skfem.MeshHex.init_tensor(x,y,z)
    np.testing.assert_allclose(native.p,reference.p,rtol=0.,atol=0.)
    np.testing.assert_array_equal(native.t,reference.t)

    native_element=skfemntv.ElementHex1()
    reference_element=skfem.ElementHex1()
    np.testing.assert_array_equal(native_element.doflocs,reference_element.doflocs)
    native_basis=skfemntv.Basis(native,native_element)
    reference_basis=skfem.Basis(reference,reference_element)
    for element in range(native.nelements):
        native_local=native_basis.doflocs[:,native_basis.element_dofs[:,element]]
        reference_local=reference_basis.doflocs[:,reference_basis.element_dofs[:,element]]
        np.testing.assert_allclose(native_local,reference_local,rtol=0.,atol=0.)

    native_faces=native._local_facets()
    reference_faces=tuple(tuple(face) for face in reference.elem.refdom.facets)
    assert native_faces==reference_faces
    native_face_coordinates=tuple(
        tuple(map(tuple,native.p[:,native.t[np.asarray(face),0]].T))
        for face in native_faces
    )
    reference_face_coordinates=tuple(
        tuple(map(tuple,reference.p[:,reference.t[np.asarray(face),0]].T))
        for face in reference_faces
    )
    assert native_face_coordinates==reference_face_coordinates


def test_hex2_local_dofs_and_promoted_connectivity_match_scikit_fem():
    native_linear=skfemntv.MeshHex.init_tensor(
        (0.,.7,2.1),(-1.,.4,1.8),(.2,1.3,3.7)
    )
    reference_linear=skfem.MeshHex.init_tensor(
        (0.,.7,2.1),(-1.,.4,1.8),(.2,1.3,3.7)
    )
    native=skfemntv.MeshHex2.from_mesh(native_linear)
    reference=skfem.MeshHex2.from_mesh(reference_linear)
    native_element=skfemntv.ElementHex2()
    reference_element=skfem.ElementHex2()
    np.testing.assert_array_equal(native_element.doflocs,reference_element.doflocs)
    native_basis=skfemntv.Basis(native,native_element)
    reference_basis=skfem.Basis(reference,reference_element)
    for element in range(native.nelements):
        native_local=native_basis.doflocs[:,native_basis.element_dofs[:,element]]
        reference_local=reference_basis.doflocs[:,reference_basis.element_dofs[:,element]]
        np.testing.assert_allclose(native_local,reference_local,rtol=0.,atol=1e-15)
