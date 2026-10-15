"""Manufactured Poisson convergence for P2 fields on linear meshes."""

import numpy as np
import pytest
import skfem
from skfem.helpers import dot as reference_dot
from skfem.helpers import grad as reference_grad

import skfemntv
from skfemntv import utils as native_utils
from skfemntv.helpers import ddot, dot, grad


def _mesh_pair(dim, divisions):
    axes = np.linspace(0.0, 1.0, divisions + 1)
    if dim == 2:
        native = skfemntv.MeshTri.init_tensor(axes, axes)
        reference = skfem.MeshTri(native.p, native.t)
        native_element = skfemntv.ElementTriP2()
        reference_element = skfem.ElementTriP2()
    else:
        native = skfemntv.MeshTet.init_tensor(axes, axes, axes)
        reference = skfem.MeshTet(native.p, native.t)
        native_element = skfemntv.ElementTetP2()
        reference_element = skfem.ElementTetP2()

    def boundary(points):
        return np.any(
            np.isclose(points, 0.0) | np.isclose(points, 1.0), axis=0
        )

    return (
        native.with_boundaries({"dirichlet": boundary}),
        reference.with_boundaries({"dirichlet": boundary}),
        native_element,
        reference_element,
    )


def _manufactured_solution(points):
    sine = np.sin(np.pi * points)
    product = np.prod(sine, axis=0)
    dimension = points.shape[0]
    value = 1.0 + product
    gradient = np.empty_like(points)
    for axis in range(dimension):
        factors = np.cos(np.pi * points[axis])
        if dimension > 1:
            factors = factors * np.prod(
                np.delete(sine, axis, axis=0), axis=0
            )
        gradient[axis] = np.pi * factors
    source = dimension * np.pi**2 * product
    return value, gradient, source


def _native_error(basis, solution):
    coordinates = np.moveaxis(basis.global_coordinates, -1, 0)
    exact_value, exact_gradient, _ = _manufactured_solution(coordinates)
    numerical = basis.interpolate(solution)
    value_error = numerical.value - exact_value
    gradient_error = numerical.grad - exact_gradient
    l2 = np.sqrt(np.sum(value_error**2 * basis.dx))
    h1_seminorm = np.sqrt(
        np.sum(np.sum(gradient_error**2, axis=0) * basis.dx)
    )
    return l2, h1_seminorm


@pytest.mark.parametrize("dim", [2, 3])
def test_p2_manufactured_poisson_errors_converge_on_linear_meshes(dim):
    errors = []
    for divisions in (2, 4):
        native_mesh, reference_mesh, native_element, reference_element = (
            _mesh_pair(dim, divisions)
        )
        native_basis = skfemntv.Basis(
            native_mesh, native_element, intorder=6
        )
        reference_basis = skfem.Basis(
            reference_mesh, reference_element, intorder=6
        )

        @skfemntv.BilinearForm
        def native_stiffness(u, v, w):
            return ddot(grad(u), grad(v))

        @skfem.BilinearForm
        def reference_stiffness(u, v, w):
            return reference_dot(reference_grad(u), reference_grad(v))

        @skfemntv.LinearForm
        def native_load(v, w):
            return dot(w.force, v)

        @skfem.LinearForm
        def reference_load(v, w):
            return w.force * v

        native_points = np.moveaxis(native_basis.global_coordinates, -1, 0)
        _, _, native_force = _manufactured_solution(native_points)
        reference_points = np.asarray(reference_basis.global_coordinates())
        _, _, reference_force = _manufactured_solution(reference_points)
        native_matrix = skfemntv.asm(native_stiffness, native_basis)
        native_rhs = skfemntv.asm(
            native_load, native_basis, force=native_force[None]
        )
        reference_matrix = skfem.asm(reference_stiffness, reference_basis)
        reference_rhs = skfem.asm(
            reference_load, reference_basis, force=reference_force
        )

        native_fixed = native_basis.get_dofs("dirichlet").all()
        reference_fixed = reference_basis.get_dofs("dirichlet").all()
        native_values = np.zeros(native_basis.N)
        reference_values = np.zeros(reference_basis.N)
        native_values[native_fixed] = 1.0
        reference_values[reference_fixed] = 1.0
        native_solution = native_utils.solve(
            *native_utils.condense(
                native_matrix, native_rhs, x=native_values, D=native_fixed
            )
        )
        reference_solution = skfem.solve(
            *skfem.condense(
                reference_matrix, reference_rhs,
                x=reference_values, D=reference_fixed,
            )
        )
        np.testing.assert_allclose(
            native_solution[native_fixed], 1.0, rtol=0.0, atol=1e-13
        )
        errors.append(_native_error(native_basis, native_solution))

    coarse_l2, coarse_h1 = errors[0]
    fine_l2, fine_h1 = errors[1]
    assert fine_l2 < coarse_l2
    assert fine_h1 < coarse_h1
    # Halving h should approach the P2 rates h^3 in L2 and h^2 in H1.
    assert coarse_l2 / fine_l2 > 4.0
    assert coarse_h1 / fine_h1 > 2.5
