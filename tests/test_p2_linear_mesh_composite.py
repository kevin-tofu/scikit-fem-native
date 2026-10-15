"""P2/P1 composite basis parity on linear simplex meshes."""

import numpy as np
import pytest
import skfem
from skfem.helpers import div as reference_div
from skfem.helpers import ddot as reference_ddot
from skfem.helpers import grad as reference_grad

import skfemntv
from skfemntv.helpers import ddot, div, grad


def _spaces(dim):
    if dim == 2:
        linear = skfemntv.MeshTri.init_tensor(
            np.linspace(0.0, 1.0, 3), np.linspace(0.0, 1.0, 3)
        )
        reference_linear = skfem.MeshTri(linear.p, linear.t)
        native_velocity = skfemntv.ElementVector(skfemntv.ElementTriP2())
        reference_velocity = skfem.ElementVector(skfem.ElementTriP2())
        native_pressure = skfemntv.ElementTriP1()
        reference_pressure = skfem.ElementTriP1()
    else:
        linear = skfemntv.MeshTet.init_tensor(
            [0.0, 0.5, 1.0], [0.0, 1.0], [0.0, 1.0]
        )
        reference_linear = skfem.MeshTet(linear.p, linear.t)
        native_velocity = skfemntv.ElementVector(skfemntv.ElementTetP2())
        reference_velocity = skfem.ElementVector(skfem.ElementTetP2())
        native_pressure = skfemntv.ElementTetP1()
        reference_pressure = skfem.ElementTetP1()

    wall = {"wall": lambda x: np.isclose(x[0], 0.0)}
    native_mesh = linear.with_boundaries(wall)
    reference_mesh = reference_linear.with_boundaries(wall)
    return (
        native_mesh,
        reference_mesh,
        native_velocity * native_pressure,
        reference_velocity * reference_pressure,
    )


@pytest.mark.parametrize("dim", [2, 3])
def test_p2_p1_composite_on_linear_mesh_matches_scikit_fem(dim):
    native_mesh, reference_mesh, native_element, reference_element = _spaces(dim)
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

    actual = skfemntv.asm(native_stokes, native)
    expected = skfem.asm(reference_stokes, reference)
    np.testing.assert_array_equal(native.elem.doflocs, reference.elem.doflocs)
    np.testing.assert_array_equal(native.doflocs, reference.doflocs)
    np.testing.assert_array_equal(native.element_dofs, reference.element_dofs)
    np.testing.assert_array_equal(native.nodal_dofs, reference.nodal_dofs)
    np.testing.assert_allclose(
        actual.toarray(),
        expected.toarray(),
        rtol=3e-11,
        atol=3e-12,
    )

    native_velocity, _ = native.split_bases()
    reference_velocity, _ = reference.split_bases()
    native_boundary = native_velocity.get_dofs("wall").all()
    reference_boundary = reference_velocity.get_dofs("wall").all()
    native_points = sorted(
        map(tuple, np.round(native_velocity.doflocs[:, native_boundary].T, 13))
    )
    reference_points = sorted(
        map(tuple, np.round(reference_velocity.doflocs[:, reference_boundary].T, 13))
    )
    assert native_points == reference_points
    np.testing.assert_array_equal(
        native_velocity.get_dofs("wall").all(),
        reference_velocity.get_dofs("wall").all(),
    )
    assert native.mesh is native_mesh
    assert native.mesh.t.shape[0] == (3 if dim == 2 else 4)
    assert native._mesh.t.shape[0] == (6 if dim == 2 else 10)
