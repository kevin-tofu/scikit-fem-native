"""Wedge H1 basis parity on a small user-supplied prism mesh."""

import numpy as np
import skfem

import skfemntv
from skfem.helpers import ddot as skfem_ddot
from skfem.helpers import dot as skfem_dot
from skfem.helpers import grad as skfem_grad
from skfemntv.helpers import ddot, dot, grad


def test_wedge_basis_geometry_and_assembly_match_skfem():
    native_mesh = skfemntv.MeshWedge1()
    reference_mesh = skfem.MeshWedge1(native_mesh.p, native_mesh.t)
    reference = skfem.Basis(
        reference_mesh, skfem.ElementWedge1(), intorder=3,
    )
    # scikit-fem 12.0.2's default wedge rule is not the product of its
    # triangle and line rules; share its points and weights for this parity
    # check so geometry and form assembly are compared independently.
    native = skfemntv.Basis(
        native_mesh, skfemntv.ElementWedge1(), quadrature=reference.quadrature,
    )

    np.testing.assert_array_equal(native.element_dofs, reference.element_dofs)
    np.testing.assert_allclose(native.doflocs, reference.doflocs)
    np.testing.assert_allclose(
        native.X, reference.X, rtol=0., atol=0.,
    )
    np.testing.assert_allclose(
        native.W, reference.W, rtol=2e-14, atol=2e-14,
    )
    np.testing.assert_allclose(
        native.dx, reference.dx, rtol=2e-14, atol=2e-14,
    )
    np.testing.assert_allclose(
        native.global_coordinates,
        reference.global_coordinates().transpose(1, 2, 0),
        rtol=2e-14, atol=2e-14,
    )
    reference_shape = np.stack(
        [np.asarray(field[0]) for field in reference.basis], axis=-1,
    )
    reference_gradient = np.stack(
        [field[0].grad.transpose(1, 2, 0) for field in reference.basis], axis=2,
    )
    np.testing.assert_allclose(
        native.tabulated_shape, reference_shape,
    )
    np.testing.assert_allclose(
        native.tabulated_gradients, reference_gradient,
    )

    @skfemntv.BilinearForm
    def native_mass(u, v, w):
        return dot(u, v)

    @skfem.BilinearForm
    def reference_mass(u, v, w):
        return u * v

    @skfemntv.BilinearForm
    def native_stiffness(u, v, w):
        return ddot(grad(u), grad(v))

    @skfem.BilinearForm
    def reference_stiffness(u, v, w):
        return skfem_ddot(skfem_grad(u), skfem_grad(v))

    np.testing.assert_allclose(
        skfemntv.asm(native_mass, native).toarray(),
        skfem.asm(reference_mass, reference).toarray(),
        rtol=2e-13, atol=2e-13,
    )
    np.testing.assert_allclose(
        skfemntv.asm(native_stiffness, native).toarray(),
        skfem.asm(reference_stiffness, reference).toarray(),
        rtol=2e-13, atol=2e-13,
    )


def test_wedge_default_product_rule_integrates_unit_prism():
    basis = skfemntv.Basis(
        skfemntv.MeshWedge1(), skfemntv.ElementWedge1(), intorder=3,
    )
    np.testing.assert_allclose(
        basis.W.sum(), .5, rtol=0., atol=2e-14,
    )
    np.testing.assert_allclose(
        basis.dx.sum(), .5, rtol=0., atol=2e-14,
    )
    np.testing.assert_allclose(
        basis.tabulated_shape.sum(axis=-1), 1., rtol=0., atol=2e-14,
    )

    @skfemntv.BilinearForm
    def mass(u, v, w):
        return dot(u, v)

    mass_matrix = skfemntv.asm(mass, basis).toarray()
    np.testing.assert_allclose(mass_matrix, mass_matrix.T, atol=2e-14)
    np.testing.assert_allclose(mass_matrix.sum(), .5, atol=2e-14)


def test_pinned_scikit_fem_wedge_default_rule_difference_is_recorded():
    native_mesh = skfemntv.MeshWedge1()
    reference_mesh = skfem.MeshWedge1(native_mesh.p, native_mesh.t)
    native = skfemntv.Basis(
        native_mesh, skfemntv.ElementWedge1(), intorder=3,
    )
    reference = skfem.Basis(
        reference_mesh, skfem.ElementWedge1(), intorder=3,
    )

    # scikit-fem 12.0.2 builds a 32-point product over independent triangle
    # coordinates and its weights integrate this one-cell prism to 1/4.
    # Keep the mismatch visible while the native default remains physically
    # integrated to its geometric volume of 1/2.
    assert native.X.shape[1] == 18
    assert reference.X.shape[1] == 32
    np.testing.assert_allclose(native.dx.sum(), .5, rtol=0., atol=2e-14)
    np.testing.assert_allclose(reference.dx.sum(), .25, rtol=0., atol=2e-14)
