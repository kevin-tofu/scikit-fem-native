"""Parity checks for scikit-fem's sparse solver utility API."""

import numpy as np
import pytest
import skfem
from scipy.sparse import csr_matrix, diags

import skfemntv
from skfemntv import utils as native_utils


def _system():
    matrix = csr_matrix(
        np.array(
            [
                [4.0, -1.0, 0.0, 0.0],
                [-1.0, 4.0, -1.0, 0.0],
                [0.0, -1.0, 4.0, -1.0],
                [0.0, 0.0, -1.0, 3.0],
            ]
        )
    )
    rhs = np.array([1.0, 2.0, 3.0, 4.0])
    values = np.array([0.5, 0.0, 0.0, -0.25])
    return matrix, rhs, values


@pytest.mark.parametrize("expand", [True, False])
def test_condense_matches_scikit_fem_for_nonzero_boundary_values(expand):
    matrix, rhs, values = _system()
    actual = native_utils.condense(matrix, rhs, x=values, D=np.array([0, 3]), expand=expand)
    expected = skfem.condense(matrix, rhs, x=values, D=np.array([0, 3]), expand=expand)

    actual_parts = actual if isinstance(actual, tuple) else (actual,)
    expected_parts = expected if isinstance(expected, tuple) else (expected,)
    np.testing.assert_allclose(actual_parts[0].toarray(), expected_parts[0].toarray())
    if len(expected_parts) > 1:
        np.testing.assert_allclose(actual_parts[1], expected_parts[1])
    if expand:
        np.testing.assert_array_equal(actual_parts[2], expected_parts[2])
        np.testing.assert_array_equal(actual_parts[3], expected_parts[3])


def test_condense_supports_dofs_views_and_dictionary_selections():
    native_mesh = skfemntv.MeshTri.init_tensor(
        np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0])
    )
    reference_mesh = skfem.MeshTri.init_tensor(
        np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0])
    )
    native_basis = skfemntv.Basis(
        native_mesh,
        skfemntv.ElementTriP1(),
    )
    reference_basis = skfem.Basis(
        reference_mesh,
        skfem.ElementTriP1(),
    )
    native_view = native_basis.get_dofs()
    reference_view = reference_basis.get_dofs()
    np.testing.assert_array_equal(native_view.flatten(), reference_view.flatten())

    matrix = diags(np.arange(1.0, native_basis.N + 1.0), format="csr")
    rhs = np.arange(1.0, native_basis.N + 1.0)
    native_views = {"left": native_view, "right": native_view}
    reference_views = {"left": reference_view, "right": reference_view}
    for native_dofs, reference_dofs in (
        (native_view, reference_view),
        (native_views, reference_views),
    ):
        actual = native_utils.condense(matrix, rhs, D=native_dofs, expand=False)
        expected = skfem.condense(matrix, rhs, D=reference_dofs, expand=False)
        np.testing.assert_allclose(actual[0].toarray(), expected[0].toarray())
        np.testing.assert_allclose(actual[1], expected[1])


@pytest.mark.parametrize("function", ["enforce", "penalize"])
def test_boundary_modification_matches_scikit_fem(function):
    matrix, rhs, values = _system()
    native_function = getattr(native_utils, function)
    reference_function = getattr(skfem, function)
    actual = native_function(matrix, rhs, x=values, D=np.array([0, 3]))
    expected = reference_function(matrix, rhs, x=values, D=np.array([0, 3]))
    np.testing.assert_allclose(actual[0].toarray(), expected[0].toarray())
    np.testing.assert_allclose(actual[1], expected[1])


def test_enforce_overwrite_and_sparse_rhs_match_scikit_fem():
    matrix, rhs, values = _system()
    native_matrix, reference_matrix = matrix.copy(), matrix.copy()
    native_rhs = csr_matrix(np.diag(rhs))
    reference_rhs = native_rhs.copy()
    actual = native_utils.enforce(
        native_matrix, native_rhs, x=values, D=np.array([0]), overwrite=True
    )
    expected = skfem.enforce(
        reference_matrix, reference_rhs, x=values, D=np.array([0]), overwrite=True
    )
    assert actual[0] is native_matrix
    assert expected[0] is reference_matrix
    np.testing.assert_allclose(actual[0].toarray(), expected[0].toarray())
    np.testing.assert_allclose(actual[1].toarray(), expected[1].toarray())


def test_solve_linear_and_custom_solver_match_scikit_fem():
    matrix, rhs, values = _system()
    actual_condensed = native_utils.condense(matrix, rhs, x=values, D=np.array([0, 3]))
    expected_condensed = skfem.condense(matrix, rhs, x=values, D=np.array([0, 3]))
    actual = native_utils.solve(*actual_condensed)
    expected = skfem.solve(*expected_condensed)
    np.testing.assert_allclose(actual, expected)

    def custom(A, b, shift=0.0):
        return np.linalg.solve(A.toarray() + shift * np.eye(A.shape[0]), b)

    actual = native_utils.solve(matrix, rhs, solver=custom, shift=0.25)
    expected = skfem.solve(matrix, rhs, solver=custom, shift=0.25)
    np.testing.assert_allclose(actual, expected)


def test_generalized_eigenvalue_solve_and_expansion_match_scikit_fem():
    stiffness = diags(np.array([1.0, 2.0, 4.0, 7.0, 11.0]), format="csr")
    mass = diags(np.array([1.0, 1.0, 2.0, 2.0, 4.0]), format="csr")
    actual_values, _ = native_utils.solve(
        stiffness, mass, sigma=0.5, k=2
    )
    expected_values, _ = skfem.solve(
        stiffness, mass, sigma=0.5, k=2
    )
    np.testing.assert_allclose(actual_values, expected_values, rtol=1e-10, atol=1e-12)
    fixed = np.array([5.0, 0.0, 0.0, 0.0, 0.0], dtype=complex)
    reduced_stiffness = stiffness[1:, 1:]
    reduced_mass = mass[1:, 1:]
    values, vectors = native_utils.solve(
        reduced_stiffness, reduced_mass, x=fixed, I=np.arange(1, 5), sigma=0.5, k=2
    )
    reference_values, reference_vectors = skfem.solve(
        reduced_stiffness, reduced_mass, x=fixed, I=np.arange(1, 5), sigma=0.5, k=2
    )
    np.testing.assert_allclose(values, reference_values, rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(vectors[0], fixed[0])
    np.testing.assert_array_equal(reference_vectors[0], fixed[0])
    for index, eigenvalue in enumerate(values):
        residual = reduced_stiffness @ vectors[1:, index] - eigenvalue * (
            reduced_mass @ vectors[1:, index]
        )
        assert np.linalg.norm(residual) < 1e-9
