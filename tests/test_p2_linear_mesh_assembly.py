"""End-to-end assembly parity for P2 fields on linear meshes."""

import numpy as np
import pytest
import skfem
from skfem.helpers import dot as reference_dot
from skfem.helpers import grad as reference_grad

import skfemntv
from skfemntv.helpers import ddot, dot, grad


def _spaces(dim):
    if dim == 2:
        native_linear = skfemntv.MeshTri.init_tensor(
            np.linspace(0.0, 1.0, 3), np.linspace(0.0, 1.0, 3)
        )
        reference_linear = skfem.MeshTri(native_linear.p, native_linear.t)
        native_quadratic = skfemntv.ElementTriP2()
        reference_quadratic = skfem.ElementTriP2()
    else:
        native_linear = skfemntv.MeshTet.init_tensor(
            [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]
        )
        reference_linear = skfem.MeshTet(native_linear.p, native_linear.t)
        native_quadratic = skfemntv.ElementTetP2()
        reference_quadratic = skfem.ElementTetP2()
    boundary = {"left": lambda x: np.isclose(x[0], 0.0)}
    return (
        native_linear.with_boundaries(boundary),
        reference_linear.with_boundaries(boundary),
        native_quadratic,
        reference_quadratic,
    )


@pytest.mark.parametrize("dim", [2, 3])
def test_p2_mass_and_stiffness_on_linear_mesh_match_scikit_fem(dim):
    native_mesh, reference_mesh, native_element, reference_element = _spaces(dim)
    native_basis = skfemntv.Basis(
        native_mesh, native_element, intorder=4
    )
    reference_basis = skfem.Basis(
        reference_mesh, reference_element, intorder=4
    )
    assert native_basis.mesh is native_mesh
    assert native_basis.mesh.t.shape[0] == (3 if dim == 2 else 4)
    assert native_basis._mesh.t.shape[0] == (6 if dim == 2 else 10)

    @skfemntv.BilinearForm
    def native_mass(u, v, w):
        return dot(u, v)

    @skfemntv.BilinearForm
    def native_stiffness(u, v, w):
        return ddot(grad(u), grad(v))

    @skfem.BilinearForm
    def reference_mass(u, v, w):
        return u * v

    @skfem.BilinearForm
    def reference_stiffness(u, v, w):
        return reference_dot(reference_grad(u), reference_grad(v))

    np.testing.assert_array_equal(native_basis.doflocs, reference_basis.doflocs)
    np.testing.assert_array_equal(
        native_basis.element_dofs, reference_basis.element_dofs
    )
    for native_form, reference_form in (
        (native_mass, reference_mass),
        (native_stiffness, reference_stiffness),
    ):
        actual = skfemntv.asm(native_form, native_basis)
        expected = skfem.asm(reference_form, reference_basis)
        np.testing.assert_allclose(
            actual.toarray(),
            expected.toarray(),
            rtol=3e-11,
            atol=3e-12,
        )

    native_boundary = native_basis.get_dofs("left").all()
    reference_boundary = reference_basis.get_dofs("left").all()
    np.testing.assert_array_equal(native_boundary, reference_boundary)
    native_points = sorted(map(tuple, np.round(native_basis.doflocs[:, native_boundary].T, 13)))
    reference_points = sorted(map(tuple, np.round(reference_basis.doflocs[:, reference_boundary].T, 13)))
    assert native_points == reference_points
