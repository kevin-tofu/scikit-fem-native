"""Line-mesh boundary and refinement lifecycle parity."""

import numpy as np
import skfem

import skfemntv


def _tagged_mesh():
    return skfemntv.MeshLine.init_tensor([0.,.3,1.]).with_boundaries({
        "left":lambda x:np.isclose(x[0],0.),
        "right":lambda x:np.isclose(x[0],1.),
    }).with_subdomains({"first":lambda x:x[0]<.5})


def test_meshline_boundary_refinement_and_restriction_match_skfem():
    native=_tagged_mesh()
    reference=skfem.MeshLine.init_tensor(np.array([0.,.3,1.])).with_boundaries({
        "left":lambda x:np.isclose(x[0],0.),
        "right":lambda x:np.isclose(x[0],1.),
    }).with_subdomains({"first":lambda x:x[0]<.5})

    np.testing.assert_array_equal(native.facets,reference.facets)
    np.testing.assert_array_equal(native.boundary_facets(),reference.boundary_facets())
    np.testing.assert_array_equal(native.boundaries["left"],reference.boundaries["left"])
    np.testing.assert_array_equal(native.subdomains["first"],reference.subdomains["first"])

    refined=native.refined(2)
    expected=reference.refined(2)
    np.testing.assert_allclose(refined.p,expected.p)
    np.testing.assert_array_equal(refined.t,expected.t)
    np.testing.assert_array_equal(refined.subdomains["first"],expected.subdomains["first"])
    for name in ("left","right"):
        np.testing.assert_array_equal(refined.boundaries[name],expected.boundaries[name])

    restricted=native.restrict([1])
    expected_restricted=reference.restrict([1])
    np.testing.assert_allclose(restricted.p,expected_restricted.p)
    np.testing.assert_array_equal(restricted.t,expected_restricted.t)
    for name in ("left","right"):
        np.testing.assert_array_equal(
            restricted.boundaries[name],expected_restricted.boundaries[name],
        )
    removed=native.remove_elements([0])
    expected_removed=reference.remove_elements([0])
    np.testing.assert_allclose(removed.p,expected_removed.p)
    np.testing.assert_array_equal(removed.t,expected_removed.t)


def test_meshline_adaptive_refinement_preserves_named_boundary_and_subdomain():
    native=_tagged_mesh()
    reference=skfem.MeshLine.init_tensor(np.array([0.,.3,1.])).with_boundaries({
        "left":lambda x:np.isclose(x[0],0.),
        "right":lambda x:np.isclose(x[0],1.),
    }).with_subdomains({"first":lambda x:x[0]<.5})
    refined=native.refined(np.array([0],dtype=np.int64))
    expected=reference.refined(np.array([0],dtype=np.int64))
    np.testing.assert_allclose(refined.p,expected.p)
    np.testing.assert_array_equal(refined.t,expected.t)
    np.testing.assert_array_equal(refined.subdomains["first"],expected.subdomains["first"])
    for name in ("left","right"):
        np.testing.assert_array_equal(refined.boundaries[name],expected.boundaries[name])
