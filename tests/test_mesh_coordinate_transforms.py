"""Mesh coordinate transforms and copy semantics against scikit-fem."""

import numpy as np
import pytest
import skfem

import skfemntv


def _mesh_pairs():
    line = skfemntv.MeshLine.init_tensor([0., .4, 1.])
    tri = skfemntv.MeshTri.init_tensor([0., 1.], [0., 1.])
    quad = skfemntv.MeshQuad.init_tensor([0., 1.], [0., 1.])
    tet = skfemntv.MeshTet.init_tensor([0., 1.], [0., 1.], [0., 1.])
    hex_mesh = skfemntv.MeshHex.init_tensor([0., 1.], [0., 1.], [0., 1.])
    wedge = skfemntv.MeshWedge1()
    tri2 = skfemntv.MeshTri2.from_mesh(tri)
    quad2 = skfemntv.MeshQuad2.from_mesh(quad)
    tet2 = skfemntv.MeshTet2.from_mesh(tet)
    hex2 = skfemntv.MeshHex2.from_mesh(hex_mesh)
    pairs = (
        (line, skfem.MeshLine), (tri, skfem.MeshTri),
        (quad, skfem.MeshQuad), (tet, skfem.MeshTet),
        (hex_mesh, skfem.MeshHex), (wedge, skfem.MeshWedge1),
        (tri2, skfem.MeshTri2), (quad2, skfem.MeshQuad2),
        (tet2, skfem.MeshTet2), (hex2, skfem.MeshHex2),
    )
    return [(native, reference_type(native.p, native.t))
            for native, reference_type in pairs]


def _sorted_points(points):
    order = np.lexsort(tuple(points[::-1]))
    return points[:, order]


@pytest.mark.parametrize("operation", ["copy", "translated", "scaled", "morphed"])
def test_mesh_coordinate_transforms_preserve_connectivity_and_regions(operation):
    for native, reference in _mesh_pairs():
        native = native.with_boundaries({"marked": [0]}).with_subdomains({"marked": [0]})
        reference = reference.with_boundaries({"marked": [0]}).with_subdomains({"marked": [0]})
        original_points = native.p.copy()
        dimension = native.dim()
        if operation == "copy":
            actual, expected = native.copy(), reference.copy()
        elif operation == "translated":
            offsets = np.arange(1., dimension + 1.) / 7.
            actual, expected = native.translated(offsets), reference.translated(offsets)
        elif operation == "scaled":
            factors = np.arange(1., dimension + 1.) + .5
            actual, expected = native.scaled(factors), reference.scaled(factors)
        else:
            functions = tuple(
                (lambda points, axis=axis: points[axis] + .03 * points[-1] ** 2)
                for axis in range(dimension)
            )
            actual, expected = native.morphed(*functions), reference.morphed(*functions)

        np.testing.assert_allclose(
            _sorted_points(actual.p), _sorted_points(expected.p),
            rtol=0., atol=2e-14,
        )
        np.testing.assert_array_equal(actual.t, native.t)
        np.testing.assert_array_equal(expected.t, reference.t)
        np.testing.assert_array_equal(actual.boundaries["marked"], expected.boundaries["marked"])
        np.testing.assert_array_equal(actual.subdomains["marked"], expected.subdomains["marked"])
        np.testing.assert_array_equal(native.p, original_points)
        assert actual is not native


def test_mesh_transform_arguments_are_checked():
    mesh = skfemntv.MeshTri()
    with pytest.raises(ValueError, match="diffs must have shape"):
        mesh.translated([1.])
    with pytest.raises(ValueError, match="factors must be a scalar"):
        mesh.scaled([1.])
    with pytest.raises(ValueError, match="at most one morph function"):
        mesh.morphed(None, None, None)
    with pytest.raises(ValueError, match="one coordinate per node"):
        mesh.morphed(lambda points: np.zeros(points.shape[1] + 1))
