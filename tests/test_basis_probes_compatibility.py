"""scikit-fem parity tests for physical-point probes and interpolators."""

import numpy as np
import pytest
import skfem
from scipy.sparse import issparse

import skfemntv


def _mesh_pair(dim=2, quadratic=False):
    if dim == 2:
        native = skfemntv.MeshTri.init_tensor(
            np.array([0.0, 1.0]), np.array([0.0, 1.0])
        )
        reference = skfem.MeshTri.init_tensor(
            np.array([0.0, 1.0]), np.array([0.0, 1.0])
        )
        if quadratic:
            native = skfemntv.MeshTri2.from_mesh(native)
        return native, reference
    native = skfemntv.MeshTet.init_tensor(
        np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([0.0, 1.0])
    )
    reference = skfem.MeshTet.init_tensor(
        np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([0.0, 1.0])
    )
    if quadratic:
        native = skfemntv.MeshTet2.from_mesh(native)
    return native, reference


@pytest.mark.parametrize(
    "dim,quadratic,element_name",
    [
        (2, False, "ElementTriP1"),
        (2, True, "ElementTriP2"),
        (3, False, "ElementTetP1"),
        (3, True, "ElementTetP2"),
    ],
)
def test_scalar_probe_matrix_matches_scikit_fem(dim, quadratic, element_name):
    native_mesh, reference_mesh = _mesh_pair(dim, quadratic)
    native_basis = skfemntv.Basis(
        native_mesh, getattr(skfemntv, element_name)()
    )
    reference_basis = skfem.Basis(
        reference_mesh, getattr(skfem, element_name)()
    )
    points = np.array(
        [[0.21, 0.68, 0.22], [0.19, 0.17, 0.65]]
        if dim == 2
        else [[0.12, 0.55], [0.18, 0.17], [0.21, 0.12]]
    )
    actual = native_basis.probes(points)
    expected = reference_basis.probes(points)
    assert issparse(actual) and issparse(expected)
    assert actual.shape == expected.shape == (points.shape[1], native_basis.N)

    def polynomial(x):
        return 1.0 + 2.0 * x[0] - 3.0 * x[1] + x[0] * x[1]

    native_coefficients = polynomial(native_basis.doflocs)
    reference_coefficients = polynomial(reference_basis.doflocs)
    np.testing.assert_allclose(
        actual @ native_coefficients,
        expected @ reference_coefficients,
        rtol=2e-12,
        atol=2e-12,
    )
    if dim == 2:
        grid = np.array(
            [[[0.21, 0.68], [0.22, 0.31]],
             [[0.19, 0.17], [0.65, 0.42]]]
        )
        actual_interpolator = native_basis.interpolator(native_coefficients)
        expected_interpolator = reference_basis.interpolator(reference_coefficients)
        np.testing.assert_allclose(
            actual_interpolator(grid), expected_interpolator(grid),
            rtol=2e-12, atol=2e-12,
        )


@pytest.mark.parametrize(
    "dim,element_name",
    [(2, "ElementTriP2"), (3, "ElementTetP2")],
)
def test_quadratic_field_on_linear_mesh_matches_scikit_fem(dim, element_name):
    native_mesh, reference_mesh = _mesh_pair(dim)
    element = getattr(skfemntv, element_name)()
    reference_element = getattr(skfem, element_name)()
    native_basis = skfemntv.Basis(native_mesh, element, intorder=4)
    reference_basis = skfem.Basis(reference_mesh, reference_element, intorder=4)
    points = np.array(
        [[0.21, 0.68, 0.22], [0.19, 0.17, 0.65]]
        if dim == 2
        else [[0.12, 0.55], [0.18, 0.17], [0.21, 0.12]]
    )

    def polynomial(x):
        value = 1.0 + 2.0 * x[0] - 3.0 * x[1] + x[0] * x[1]
        if dim == 3:
            value += 0.7 * x[2] + 0.4 * x[0] * x[2] - 0.2 * x[1] * x[2]
        return value

    native_coefficients = polynomial(native_basis.doflocs)
    reference_coefficients = polynomial(reference_basis.doflocs)
    np.testing.assert_allclose(
        native_basis.probes(points) @ native_coefficients,
        reference_basis.probes(points) @ reference_coefficients,
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        native_basis.dx.sum(axis=1),
        reference_basis.dx.sum(axis=1),
        rtol=2e-12,
        atol=2e-12,
    )


@pytest.mark.parametrize(
    "dim,element_name",
    [(2, "ElementTriP2"), (3, "ElementTetP2")],
)
def test_p2_promotion_preserves_named_regions(dim, element_name):
    native_mesh, reference_linear = _mesh_pair(dim)
    native_mesh = native_mesh.with_subdomains({"first-cell": [0]}).with_boundaries(
        {"left": lambda x: np.isclose(x[0], 0.0)}
    )
    quadratic_type = skfem.MeshTri2 if dim == 2 else skfem.MeshTet2
    reference_mesh = quadratic_type.from_mesh(reference_linear)
    reference_mesh = reference_mesh.with_subdomains({"first-cell": [0]}).with_boundaries(
        {"left": lambda x: np.isclose(x[0], 0.0)}
    )
    native_basis = skfemntv.Basis(
        native_mesh, getattr(skfemntv, element_name)(), elements="first-cell"
    )
    reference_basis = skfem.Basis(
        reference_mesh, getattr(skfem, element_name)()
    )
    assert native_basis.mesh.boundaries is not None
    assert "left" in native_basis.mesh.boundaries
    np.testing.assert_array_equal(native_basis.tind, [0])
    native_left = native_basis.get_dofs("left").all()
    reference_left = reference_basis.get_dofs("left").all()
    native_locations = native_basis.doflocs[:, native_left]
    reference_locations = reference_basis.doflocs[:, reference_left]
    np.testing.assert_allclose(
        np.sort(native_locations, axis=1),
        np.sort(reference_locations, axis=1),
        atol=1e-14,
    )
def test_vector_probe_matrix_and_interpolator_match_scikit_fem():
    native_mesh, reference_mesh = _mesh_pair()
    native_element = skfemntv.ElementVector(skfemntv.ElementTriP1(), dim=2)
    reference_element = skfem.ElementVector(skfem.ElementTriP1(), dim=2)
    native_basis = skfemntv.Basis(native_mesh, native_element)
    reference_basis = skfem.Basis(reference_mesh, reference_element)
    points = np.array([[0.2, 0.7], [0.3, 0.2]])
    actual_probe = native_basis.probes(points)
    expected_probe = reference_basis.probes(points)
    native_dofs = np.arange(native_basis.N) % 2
    reference_dofs = np.arange(reference_basis.N) % 2
    native_coefficients = np.where(
        native_dofs == 0,
        native_basis.doflocs[0] + 2.0 * native_basis.doflocs[1],
        2.0 * native_basis.doflocs[0] - native_basis.doflocs[1],
    )
    reference_coefficients = np.where(
        reference_dofs == 0,
        reference_basis.doflocs[0] + 2.0 * reference_basis.doflocs[1],
        2.0 * reference_basis.doflocs[0] - reference_basis.doflocs[1],
    )
    np.testing.assert_allclose(
        actual_probe @ native_coefficients,
        expected_probe @ reference_coefficients,
        rtol=2e-13,
        atol=2e-13,
    )
    actual_interpolator = native_basis.interpolator(native_coefficients)
    expected_interpolator = reference_basis.interpolator(reference_coefficients)
    np.testing.assert_allclose(
        actual_interpolator(points), expected_interpolator(points),
        rtol=2e-13, atol=2e-13,
    )


def test_probes_outside_mesh_raise_like_scikit_fem():
    native_mesh, reference_mesh = _mesh_pair()
    native_basis = skfemntv.Basis(native_mesh, skfemntv.ElementTriP1())
    reference_basis = skfem.Basis(reference_mesh, skfem.ElementTriP1())
    outside = np.array([[2.0], [2.0]])
    with pytest.raises((ValueError, RuntimeError)):
        native_basis.probes(outside)
    with pytest.raises((ValueError, RuntimeError)):
        reference_basis.probes(outside)


def test_interpolator_rejects_wrong_coefficient_shape():
    mesh, _ = _mesh_pair()
    basis = skfemntv.Basis(mesh, skfemntv.ElementTriP1())
    with pytest.raises(ValueError, match="y must have shape"):
        basis.interpolator(np.zeros(basis.N + 1))
