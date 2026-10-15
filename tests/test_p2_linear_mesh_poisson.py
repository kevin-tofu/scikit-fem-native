"""End-to-end Poisson solves using P2 fields on linear meshes."""

import numpy as np
import pytest
import skfem
from skfem.helpers import dot as reference_dot
from skfem.helpers import grad as reference_grad

import skfemntv
from skfemntv import utils as native_utils
from skfemntv.helpers import ddot, dot, grad


def _meshes_and_elements(dim):
    if dim == 2:
        native = skfemntv.MeshTri.init_tensor(
            np.linspace(0.0, 1.0, 4), np.linspace(0.0, 1.0, 4)
        )
        reference = skfem.MeshTri(native.p, native.t)
        native_element = skfemntv.ElementTriP2()
        reference_element = skfem.ElementTriP2()
    else:
        native = skfemntv.MeshTet.init_tensor(
            [0.0, 0.5, 1.0], [0.0, 1.0], [0.0, 1.0]
        )
        reference = skfem.MeshTet(native.p, native.t)
        native_element = skfemntv.ElementTetP2()
        reference_element = skfem.ElementTetP2()
    named_boundary = {"left": lambda x: np.isclose(x[0], 0.0)}
    return (
        native.with_boundaries(named_boundary),
        reference.with_boundaries(named_boundary),
        native_element,
        reference_element,
    )


def _coordinate_permutation(native, reference):
    reference_by_point = {
        tuple(np.round(point, 13)): dof
        for dof, point in enumerate(reference.doflocs.T)
    }
    return np.asarray(
        [
            reference_by_point[tuple(np.round(point, 13))]
            for point in native.doflocs.T
        ],
        dtype=np.int64,
    )


@pytest.mark.parametrize("dim", [2, 3])
def test_p2_poisson_solve_on_linear_mesh_matches_scikit_fem(dim):
    native_mesh, reference_mesh, native_element, reference_element = (
        _meshes_and_elements(dim)
    )
    native_basis = skfemntv.Basis(native_mesh, native_element, intorder=4)
    reference_basis = skfem.Basis(
        reference_mesh, reference_element, intorder=4
    )

    @skfemntv.BilinearForm
    def native_stiffness(u, v, w):
        return ddot(grad(u), grad(v))

    @skfem.BilinearForm
    def reference_stiffness(u, v, w):
        return reference_dot(reference_grad(u), reference_grad(v))

    @skfemntv.LinearForm
    def native_load(v, w):
        return dot((1.0 + w.x[0])[None], v)

    @skfem.LinearForm
    def reference_load(v, w):
        return (1.0 + w.x[0]) * v

    native_matrix = skfemntv.asm(native_stiffness, native_basis)
    native_rhs = skfemntv.asm(native_load, native_basis)
    reference_matrix = skfem.asm(reference_stiffness, reference_basis)
    reference_rhs = skfem.asm(reference_load, reference_basis)
    permutation = _coordinate_permutation(native_basis, reference_basis)
    np.testing.assert_allclose(
        native_rhs,
        reference_rhs[permutation],
        rtol=3e-11,
        atol=3e-12,
    )

    native_fixed = native_basis.get_dofs("left").all()
    reference_fixed = reference_basis.get_dofs("left").all()
    native_fixed_points = sorted(
        map(tuple, np.round(native_basis.doflocs[:, native_fixed].T, 13))
    )
    reference_fixed_points = sorted(
        map(tuple, np.round(reference_basis.doflocs[:, reference_fixed].T, 13))
    )
    assert native_fixed_points == reference_fixed_points

    native_system = native_utils.condense(
        native_matrix, native_rhs, D=native_fixed
    )
    reference_system = skfem.condense(
        reference_matrix, reference_rhs, D=reference_fixed
    )
    native_solution = native_utils.solve(*native_system)
    reference_solution = skfem.solve(*reference_system)
    np.testing.assert_allclose(
        native_solution,
        reference_solution[permutation],
        rtol=2e-10,
        atol=2e-12,
    )
    np.testing.assert_allclose(native_solution[native_fixed], 0.0, atol=1e-14)
    native_free = np.setdiff1d(np.arange(native_basis.N), native_fixed)
    assert np.linalg.norm(
        native_matrix[native_free][:, native_free]
        @ native_solution[native_free]
        - native_rhs[native_free]
    ) < 1e-10
