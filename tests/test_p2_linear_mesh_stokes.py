"""End-to-end Taylor--Hood solves on linear simplex meshes."""

import numpy as np
import pytest
import skfem
from skfem.helpers import div as reference_div
from skfem.helpers import ddot as reference_ddot
from skfem.helpers import dot as reference_dot
from skfem.helpers import grad as reference_grad

import skfemntv
from skfemntv import utils as native_utils
from skfemntv.helpers import ddot, div, dot, grad


def _mesh_pair(dim):
    if dim == 2:
        native = skfemntv.MeshTri.init_tensor(
            np.linspace(0.0, 1.0, 4), np.linspace(0.0, 1.0, 4)
        )
        reference = skfem.MeshTri(native.p, native.t)
        native_velocity = skfemntv.ElementVector(skfemntv.ElementTriP2())
        reference_velocity = skfem.ElementVector(skfem.ElementTriP2())
        native_pressure = skfemntv.ElementTriP1()
        reference_pressure = skfem.ElementTriP1()
    else:
        axis = np.linspace(0.0, 1.0, 3)
        native = skfemntv.MeshTet.init_tensor(axis, axis, axis)
        reference = skfem.MeshTet(native.p, native.t)
        native_velocity = skfemntv.ElementVector(skfemntv.ElementTetP2())
        reference_velocity = skfem.ElementVector(skfem.ElementTetP2())
        native_pressure = skfemntv.ElementTetP1()
        reference_pressure = skfem.ElementTetP1()

    def wall(points):
        return np.any(
            np.isclose(points, 0.0) | np.isclose(points, 1.0), axis=0
        )

    native = native.with_boundaries({"wall": wall})
    reference = reference.with_boundaries({"wall": wall})
    return (
        native,
        reference,
        native_velocity * native_pressure,
        reference_velocity * reference_pressure,
    )


def _origin_dof(basis):
    selected = basis.get_dofs(
        nodes=lambda points: np.all(np.isclose(points, 0.0), axis=0)
    ).all()
    assert selected.size == 1
    return int(selected[0])


@pytest.mark.parametrize("dim", [2, 3])
def test_p2_p1_stokes_solve_on_linear_mesh_matches_scikit_fem(dim):
    native_mesh, reference_mesh, native_element, reference_element = _mesh_pair(dim)
    native = skfemntv.Basis(native_mesh, native_element, intorder=4)
    reference = skfem.Basis(reference_mesh, reference_element, intorder=4)

    @skfemntv.BilinearForm
    def native_stokes(u, p, v, q, w):
        return ddot(grad(u), grad(v)) - p * div(v) - q * div(u)

    @skfem.BilinearForm
    def reference_stokes(u, p, v, q, w):
        return (
            reference_ddot(reference_grad(u), reference_grad(v))
            - p * reference_div(v)
            - q * reference_div(u)
        )

    @skfemntv.LinearForm
    def native_force(v, q, w):
        return dot(w.force, v)

    @skfem.LinearForm
    def reference_force(v, q, w):
        return reference_dot(w.force, v)

    native_matrix = skfemntv.asm(native_stokes, native)
    native_points = np.moveaxis(native.global_coordinates, -1, 0)
    force = np.zeros((dim,) + native_points.shape[1:])
    force[0] = native_points[1]
    force[1] = -native_points[0]
    native_rhs = skfemntv.asm(native_force, native, force=force)

    reference_matrix = skfem.asm(reference_stokes, reference)
    reference_points = np.asarray(reference.global_coordinates())
    reference_force_values = np.zeros_like(reference_points)
    reference_force_values[0] = reference_points[1]
    reference_force_values[1] = -reference_points[0]
    reference_rhs = skfem.asm(
        reference_force, reference, force=reference_force_values
    )

    np.testing.assert_array_equal(native.doflocs, reference.doflocs)
    np.testing.assert_array_equal(native.element_dofs, reference.element_dofs)
    np.testing.assert_allclose(
        native_matrix.toarray(),
        reference_matrix.toarray(),
        rtol=5e-11,
        atol=5e-12,
    )
    np.testing.assert_allclose(
        native_rhs, reference_rhs, rtol=5e-11, atol=5e-12
    )

    native_velocity, native_pressure = native.split_bases()
    reference_velocity, reference_pressure = reference.split_bases()
    native_velocity_ids, native_pressure_ids = native.split_indices()
    reference_velocity_ids, reference_pressure_ids = reference.split_indices()
    native_wall = native_velocity_ids[native_velocity.get_dofs("wall").all()]
    reference_wall = reference_velocity_ids[
        reference_velocity.get_dofs("wall").all()
    ]
    native_pin = native_pressure_ids[_origin_dof(native_pressure)]
    reference_pin = reference_pressure_ids[_origin_dof(reference_pressure)]
    native_fixed = np.concatenate((native_wall, [native_pin]))
    reference_fixed = np.concatenate((reference_wall, [reference_pin]))
    np.testing.assert_array_equal(native_fixed, reference_fixed)

    native_solution = native_utils.solve(
        *native_utils.condense(native_matrix, native_rhs, D=native_fixed)
    )
    reference_solution = skfem.solve(
        *skfem.condense(reference_matrix, reference_rhs, D=reference_fixed)
    )
    np.testing.assert_allclose(
        native_solution,
        reference_solution,
        rtol=3e-9,
        atol=3e-11,
    )
    np.testing.assert_allclose(native_solution[native_wall], 0.0, atol=1e-13)
    native_free = np.setdiff1d(np.arange(native.N), native_fixed)
    residual = (
        native_matrix[native_free] @ native_solution
        - native_rhs[native_free]
    )
    assert np.linalg.norm(residual) < 1e-9
