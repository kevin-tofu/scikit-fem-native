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
