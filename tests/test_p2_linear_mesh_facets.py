"""Boundary integration parity for P2 fields on linear simplex meshes."""

import numpy as np
import pytest
import skfem

import skfemntv
from skfemntv.helpers import dot


def _spaces(dim):
    if dim == 2:
        linear = skfemntv.MeshTri.init_tensor(
            np.linspace(0.0, 1.0, 3), np.linspace(0.0, 1.0, 3)
        )
        reference_linear = skfem.MeshTri(linear.p, linear.t)
        native_scalar = skfemntv.ElementTriP2()
        reference_scalar = skfem.ElementTriP2()
    else:
        linear = skfemntv.MeshTet.init_tensor(
            [0.0, 0.5, 1.0], [0.0, 1.0], [0.0, 1.0]
        )
        reference_linear = skfem.MeshTet(linear.p, linear.t)
        native_scalar = skfemntv.ElementTetP2()
        reference_scalar = skfem.ElementTetP2()
    named = {"right": lambda x: np.isclose(x[0], 1.0)}
    native_mesh = linear.with_boundaries(named)
    reference_mesh = reference_linear.with_boundaries(named)
    native_element = skfemntv.ElementVector(native_scalar, dim=1)
    return native_mesh, reference_mesh, native_element, reference_scalar


@pytest.mark.parametrize("dim", [2, 3])
def test_p2_boundary_mass_and_load_on_linear_mesh_match_scikit_fem(dim):
    native_mesh, reference_mesh, native_element, reference_element = _spaces(dim)
    native_basis = skfemntv.FacetBasis(
        native_mesh, native_element, facets="right", intorder=4
    )
    reference_basis = skfem.FacetBasis(
        reference_mesh, reference_element, facets="right", intorder=4
    )

    @skfemntv.BilinearForm
    def native_mass(u, v, w):
        return dot(u, v)

    @skfem.BilinearForm
    def reference_mass(u, v, w):
        return u * v

    @skfemntv.LinearForm
    def native_load(v, w):
        return dot((1.0 + w.x[1])[None], v)

    @skfem.LinearForm
    def reference_load(v, w):
        return (1.0 + w.x[1]) * v

    actual_matrix = skfemntv.asm(native_mass, native_basis)
    expected_matrix = skfem.asm(reference_mass, reference_basis)
    actual_load = skfemntv.asm(native_load, native_basis)
    expected_load = skfem.asm(reference_load, reference_basis)
    np.testing.assert_array_equal(native_basis.doflocs, reference_basis.doflocs)
    np.testing.assert_array_equal(
        native_basis.element_dofs, reference_basis.element_dofs
    )
    np.testing.assert_allclose(
        actual_matrix.toarray(),
        expected_matrix.toarray(),
        rtol=3e-11,
        atol=3e-12,
    )
    np.testing.assert_allclose(
        actual_load,
        expected_load,
        rtol=3e-11,
        atol=3e-12,
    )
    np.testing.assert_allclose(
        native_basis.dx.sum(), reference_basis.dx.sum(),
        rtol=2e-12, atol=2e-12,
    )
