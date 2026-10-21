"""General Basis/asm integration for the affine triangular N1 space."""

import numpy as np
import pytest
import skfem
from skfem.helpers import curl as reference_curl
from skfem.helpers import dot as reference_dot

import skfemntv
from skfemntv import utils as native_utils
from skfemntv.helpers import curl, dot


def _mesh():
    return skfemntv.MeshTri.init_tensor(
        np.linspace(0., 1., 3), np.linspace(0., 1., 3),
    ).with_boundaries({
        "left": lambda x: np.isclose(x[0], 0.),
        "bottom": lambda x: np.isclose(x[1], 0.),
    })


def _reference_basis(mesh, native_basis):
    reference_mesh = skfem.MeshTri(mesh.p, mesh.t[:3]).with_boundaries({
        "left": lambda x: np.isclose(x[0], 0.),
        "bottom": lambda x: np.isclose(x[1], 0.),
    })
    return skfem.Basis(
        reference_mesh,
        skfem.ElementTriN1(),
        quadrature=(native_basis.X, native_basis.W),
    )


def _tet_mesh():
    return skfemntv.MeshTet.init_tensor(
        np.linspace(0., 1., 3), np.linspace(0., 1., 3),
        np.linspace(0., 1., 3),
    ).with_boundaries({
        "left": lambda x: np.isclose(x[0], 0.),
        "bottom": lambda x: np.isclose(x[1], 0.),
    })


def _reference_tet_basis(mesh, native_basis):
    reference_mesh = skfem.MeshTet(mesh.p, mesh.t[:4]).with_boundaries({
        "left": lambda x: np.isclose(x[0], 0.),
        "bottom": lambda x: np.isclose(x[1], 0.),
    })
    return skfem.Basis(
        reference_mesh, skfem.ElementTetN1(),
        quadrature=(native_basis.X, native_basis.W),
    )


def test_elementtrin1_general_basis_matches_scikit_fem_dofs_and_interpolation():
    mesh = _mesh()
    native = skfemntv.Basis(mesh, skfemntv.ElementTriN1(), intorder=3)
    reference = _reference_basis(mesh, native)

    assert native.N == reference.N
    np.testing.assert_array_equal(native.element_dofs, reference.element_dofs)
    np.testing.assert_allclose(native.doflocs, reference.doflocs)
    np.testing.assert_allclose(native.dx, reference.dx)
    np.testing.assert_array_equal(
        native.get_dofs().all(), reference.get_dofs().all(),
    )
    np.testing.assert_array_equal(
        native.get_dofs().all("u^t"), reference.get_dofs().all("u^t")
    )
    np.testing.assert_array_equal(
        native.get_dofs("left").all(), reference.get_dofs("left").all(),
    )
    np.testing.assert_array_equal(
        native.get_dofs("bottom").all(), reference.get_dofs("bottom").all(),
    )

    coefficients = np.linspace(-.4, .7, native.N)
    actual = native.interpolate(coefficients)
    expected = reference.interpolate(coefficients)
    np.testing.assert_allclose(actual.value, np.asarray(expected), atol=4e-14)
    np.testing.assert_allclose(actual.curl, expected.curl, atol=4e-14)
    np.testing.assert_allclose(curl(actual), expected.curl, atol=4e-14)


def test_elementtrin1_public_asm_matches_scikit_fem_with_fields():
    mesh = _mesh()
    native = skfemntv.Basis(mesh, skfemntv.ElementTriN1(), intorder=3)
    reference = _reference_basis(mesh, native)

    @skfemntv.BilinearForm
    def native_maxwell(u, v, w):
        return (1. + .2 * w.x[0]) * dot(u, v) + (
            .5 + .3 * w.x[1]
        ) * curl(u) * curl(v)

    @skfem.BilinearForm
    def reference_maxwell(u, v, w):
        return (1. + .2 * w.x[0]) * reference_dot(u, v) + (
            .5 + .3 * w.x[1]
        ) * reference_curl(u) * reference_curl(v)

    np.testing.assert_allclose(
        skfemntv.asm(native_maxwell, native).toarray(),
        skfem.asm(reference_maxwell, reference).toarray(),
        rtol=3e-13, atol=3e-13,
    )

    load = np.stack((1. + native.global_coordinates[..., 0],
                     -.25 + native.global_coordinates[..., 1]))

    @skfemntv.LinearForm
    def native_load(v, w):
        return dot(w.force, v)

    @skfem.LinearForm
    def reference_load(v, w):
        return reference_dot(w.force, v)

    actual_rhs = skfemntv.asm(native_load, native, force=load)
    expected_rhs = skfem.asm(reference_load, reference, force=load)
    np.testing.assert_allclose(actual_rhs, expected_rhs, atol=3e-13)


def test_elementtrin1_dirichlet_maxwell_solve_matches_scikit_fem():
    mesh = _mesh()
    native = skfemntv.Basis(mesh, skfemntv.ElementTriN1(), intorder=3)
    reference = _reference_basis(mesh, native)

    @skfemntv.BilinearForm
    def native_maxwell(u, v, w):
        return dot(u, v) + .2 * curl(u) * curl(v)

    @skfem.BilinearForm
    def reference_maxwell(u, v, w):
        return reference_dot(u, v) + .2 * reference_curl(u) * reference_curl(v)

    @skfemntv.LinearForm
    def native_load(v, w):
        return dot(w.force, v)

    @skfem.LinearForm
    def reference_load(v, w):
        return reference_dot(w.force, v)

    force = np.stack((np.ones(native.dx.shape), .5 * np.ones(native.dx.shape)))
    matrix = skfemntv.asm(native_maxwell, native)
    rhs = skfemntv.asm(native_load, native, force=force)
    reference_matrix = skfem.asm(reference_maxwell, reference)
    reference_rhs = skfem.asm(reference_load, reference, force=force)
    native_dofs = native.get_dofs().all()
    reference_dofs = reference.get_dofs().all()
    actual = native_utils.solve(*native_utils.condense(
        matrix, rhs, D=native_dofs,
    ))
    expected = skfem.solve(*skfem.condense(
        reference_matrix, reference_rhs, D=reference_dofs,
    ))
    np.testing.assert_allclose(actual, expected, atol=2e-13)
    np.testing.assert_array_equal(actual[native_dofs], 0.)


def test_elementtrin1_projection_matches_scikit_fem():
    mesh = _mesh()
    native = skfemntv.Basis(mesh, skfemntv.ElementTriN1(), intorder=3)
    reference = _reference_basis(mesh, native)

    def vector_field(x):
        return np.array((1. + x[0], -.3 + x[1]))

    np.testing.assert_allclose(
        native.project(vector_field), reference.project(vector_field),
        rtol=2e-12, atol=2e-12,
    )
    subset = np.array([0, 1, 4])
    np.testing.assert_allclose(
        native.project(vector_field, elements=subset),
        reference.project(vector_field, elements=subset),
        rtol=2e-12, atol=2e-12,
    )


def test_elementtrin1_point_evaluation_and_probes_match_scikit_fem():
    mesh = _mesh()
    native = skfemntv.Basis(mesh, skfemntv.ElementTriN1(), intorder=3)
    reference = _reference_basis(mesh, native)
    coefficients = np.linspace(-.7, .9, native.N)
    points = np.array(((.17, .73, .42), (.21, .64, .83)))

    actual = native.evaluate_at(points, coefficients)
    expected = reference.interpolator(coefficients)(points)
    np.testing.assert_allclose(actual.value, expected, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(
        (native.probes(points) @ coefficients).reshape(2, -1), expected,
        rtol=2e-12, atol=2e-12,
    )
    np.testing.assert_allclose(
        native.interpolator(coefficients)(points), expected,
        rtol=2e-12, atol=2e-12,
    )
    reference_field = reference.interpolate(coefficients)
    expected_curl = np.array([
        reference_field.curl[cell, 0] for cell in actual.cell_ids
    ])
    np.testing.assert_allclose(actual.curl, expected_curl, atol=2e-12)
    step = 1e-6
    expected_gradient = np.stack([
        (
            reference.interpolator(coefficients)(
                points + np.eye(2)[:, [axis]] * step
            )
            -reference.interpolator(coefficients)(
                points - np.eye(2)[:, [axis]] * step
            )
        )/(2.*step)
        for axis in range(2)
    ], axis=1)
    np.testing.assert_allclose(
        actual.grad, expected_gradient, rtol=2e-8, atol=2e-8
    )
    actual_mesh, actual_values = native.refinterp(coefficients, nrefs=2)
    expected_mesh, expected_values = reference.refinterp(coefficients, nrefs=2)
    np.testing.assert_allclose(actual_mesh.p, expected_mesh.p, atol=2e-14)
    np.testing.assert_array_equal(actual_mesh.t, expected_mesh.t)
    np.testing.assert_allclose(actual_values, expected_values, atol=2e-13)

    with pytest.raises(ValueError, match="outside the basis mesh"):
        native.evaluate_at(np.array(((1.5,), (.5,))), coefficients)
    outside = native.evaluate_at(
        np.array(((1.5,), (.5,))), coefficients, outside="nan"
    )
    assert outside.outside[0]
    assert np.isnan(outside.value).all()


def test_elementtetn1_general_basis_workflow_matches_scikit_fem():
    mesh = _tet_mesh()
    np.testing.assert_allclose(
        skfemntv.ElementTetN1().doflocs, skfem.ElementTetN1().doflocs
    )
    native = skfemntv.Basis(mesh, skfemntv.ElementTetN1(), intorder=3)
    reference = _reference_tet_basis(mesh, native)
    assert native.N == reference.N
    np.testing.assert_array_equal(native.element_dofs, reference.element_dofs)
    np.testing.assert_allclose(native.doflocs, reference.doflocs)
    np.testing.assert_array_equal(
        native.get_dofs().all(), reference.get_dofs().all()
    )
    np.testing.assert_array_equal(
        native.get_dofs().all("u^t"), reference.get_dofs().all("u^t")
    )
    np.testing.assert_array_equal(
        native.get_dofs("left").all(), reference.get_dofs("left").all()
    )

    coefficients = np.linspace(-.4, .8, native.N)
    actual_field = native.interpolate(coefficients)
    expected_field = reference.interpolate(coefficients)
    np.testing.assert_allclose(actual_field.value, expected_field.value)
    np.testing.assert_allclose(actual_field.curl, expected_field.curl)
    points = np.array((
        (.13, .71, .2), (.17, .12, .31), (.19, .1, .42),
    ))
    actual_points = native.evaluate_at(points, coefficients)
    expected_points = reference.interpolator(coefficients)(points)
    np.testing.assert_allclose(
        actual_points.value, expected_points, rtol=2e-12, atol=2e-12
    )
    np.testing.assert_allclose(
        (native.probes(points) @ coefficients).reshape(3, -1),
        expected_points, rtol=2e-12, atol=2e-12,
    )
    np.testing.assert_allclose(
        native.interpolator(coefficients)(points), expected_points,
        rtol=2e-12, atol=2e-12,
    )
    expected_curl = np.stack([
        expected_field.curl[:, cell, 0]
        for cell in actual_points.cell_ids
    ], axis=1)
    np.testing.assert_allclose(
        actual_points.curl, expected_curl, rtol=2e-12, atol=2e-12
    )
    step = 1e-6
    expected_gradient = np.stack([
        (
            reference.interpolator(coefficients)(
                points + np.eye(3)[:, [axis]] * step
            )
            -reference.interpolator(coefficients)(
                points - np.eye(3)[:, [axis]] * step
            )
        )/(2.*step)
        for axis in range(3)
    ], axis=1)
    np.testing.assert_allclose(
        actual_points.grad, expected_gradient, rtol=3e-8, atol=3e-8
    )
    plot_mesh, plot_values = native.refinterp(coefficients, nrefs=2)
    reference_plot_mesh, reference_plot_values = reference.refinterp(
        coefficients, nrefs=2
    )
    np.testing.assert_allclose(plot_mesh.p, reference_plot_mesh.p, atol=3e-14)
    np.testing.assert_array_equal(plot_mesh.t, reference_plot_mesh.t)
    np.testing.assert_allclose(plot_values, reference_plot_values, atol=3e-13)

    @skfemntv.BilinearForm
    def native_maxwell(u, v, w):
        return dot(u, v) + .2 * dot(curl(u), curl(v))

    @skfem.BilinearForm
    def reference_maxwell(u, v, w):
        return reference_dot(u, v) + .2 * reference_dot(
            reference_curl(u), reference_curl(v)
        )

    np.testing.assert_allclose(
        skfemntv.asm(native_maxwell, native).toarray(),
        skfem.asm(reference_maxwell, reference).toarray(),
        rtol=4e-13, atol=4e-13,
    )

    force = np.stack((1. + native.global_coordinates[..., 0],
                      .5 + native.global_coordinates[..., 1],
                      -.2 + native.global_coordinates[..., 2]))

    @skfemntv.LinearForm
    def native_load(v, w):
        return dot(w.force, v)

    @skfem.LinearForm
    def reference_load(v, w):
        return reference_dot(w.force, v)

    np.testing.assert_allclose(
        skfemntv.asm(native_load, native, force=force),
        skfem.asm(reference_load, reference, force=force),
        atol=5e-13,
    )

    field = lambda x: np.array((1. + x[0], .5 - x[1], x[2]))
    np.testing.assert_allclose(
        native.project(field), reference.project(field),
        rtol=2e-12, atol=2e-12,
    )
    selected_elements = np.array([0, 1, 4])
    np.testing.assert_array_equal(
        native.get_dofs(elements=selected_elements).all(),
        reference.get_dofs(elements=selected_elements).all(),
    )
    np.testing.assert_allclose(
        native.project(field, elements=selected_elements),
        reference.project(field, elements=selected_elements),
        rtol=2e-12, atol=2e-12,
    )

    native_dofs = native.get_dofs().all()
    reference_dofs = reference.get_dofs().all()
    actual_solution = native_utils.solve(*native_utils.condense(
        skfemntv.asm(native_maxwell, native),
        skfemntv.asm(native_load, native, force=force), D=native_dofs,
    ))
    expected_solution = skfem.solve(*skfem.condense(
        skfem.asm(reference_maxwell, reference),
        skfem.asm(reference_load, reference, force=force), D=reference_dofs,
    ))
    np.testing.assert_allclose(actual_solution, expected_solution, atol=2e-12)


def test_elementtetn1_general_basis_handles_outside_points():
    basis = skfemntv.Basis(_tet_mesh(), skfemntv.ElementTetN1())
    coefficients = np.zeros(basis.N)
    outside_point = np.array(((1.5,), (.5,), (.5,)))
    with pytest.raises(ValueError, match="outside the basis mesh"):
        basis.evaluate_at(outside_point, coefficients)
    result = basis.evaluate_at(
        outside_point, coefficients, outside="nan"
    )
    assert result.outside[0]
    assert np.isnan(result.value).all()


def test_elementtrin1_general_basis_rejects_unsupported_mesh_and_subsets():
    with pytest.raises(NotImplementedError, match="affine MeshTri"):
        skfemntv.Basis(skfemntv.MeshTet(), skfemntv.ElementTriN1())
    with pytest.raises(NotImplementedError, match="element subsets"):
        skfemntv.Basis(
            _mesh(), skfemntv.ElementTriN1(), elements=[0],
        )
    basis = skfemntv.Basis(_mesh(), skfemntv.ElementTriN1())
    np.testing.assert_allclose(
        basis.project(lambda x: np.zeros_like(x)), 0.
    )
