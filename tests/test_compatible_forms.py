import numpy as np
import pytest
import skfem
from skfem.helpers import ddot as skfem_ddot
from skfem.helpers import dot as skfem_dot
from skfem.helpers import grad as skfem_grad

import skfn
from skfn.helpers import ddot, dot, grad


def test_native_linear_form_preserves_constant_coefficient_shape():
    value = np.array([2.], dtype=np.float64)
    gradient = np.array([[3., 4.]], dtype=np.float64)

    assert skfn.NativeLinearForm._coefficient(
        "value", value, (100, 3, 1)
    ).shape == (1,)
    assert skfn.NativeLinearForm._coefficient(
        "gradient", gradient, (100, 3, 1, 2)
    ).shape == (1, 2)


def vector_basis():
    mesh = skfn.MeshTet.init_tensor(
        np.linspace(0., 1., 3),
        np.linspace(0., 1., 3),
        np.linspace(0., 1., 3),
    )
    return skfn.Basis(
        mesh, skfn.ElementVector(skfn.ElementTetP1()), intorder=2
    )


def reference_basis(basis, *, facets=False):
    mesh = skfem.MeshTet(basis.mesh.p, basis.mesh.t)
    element = skfem.ElementVector(skfem.ElementTetP1())
    if facets:
        return skfem.FacetBasis(
            mesh, element, facets=mesh.boundary_facets(), intorder=2
        )
    return skfem.Basis(mesh, element, intorder=2)


def test_import_skfn_as_skfem_uses_independent_core_api():
    assert skfn.MeshTet is not skfem.MeshTet
    assert skfn.Basis is not skfem.Basis
    assert skfn.FacetBasis is not skfem.FacetBasis


def test_native_volume_linear_form_uses_skfem_syntax():
    basis = vector_basis()

    @skfn.LinearForm
    def native_load(v, w):
        return dot(w.force, v)

    @skfem.LinearForm
    def reference_load(v, w):
        return skfem_dot(w.force, v)

    force = np.array([1.2, -0.7, 2.1])[:, None, None]
    actual = skfn.asm(native_load, basis, force=force)
    expected = skfem.asm(
        reference_load, reference_basis(basis), force=force
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)


def test_native_facet_traction_uses_same_form():
    basis = vector_basis()
    facets = basis.mesh.boundary_facets()
    facet_basis = skfn.FacetBasis(
        basis.mesh, basis.elem, facets=facets, intorder=2
    )

    @skfn.LinearForm
    def native_traction(v, w):
        return dot(w.traction, v)

    @skfem.LinearForm
    def reference_traction(v, w):
        return skfem_dot(w.traction, v)

    traction = np.array([0.3, 1.1, -0.2])[:, None, None]
    actual = skfn.asm(native_traction, facet_basis, traction=traction)
    expected = skfem.asm(
        reference_traction,
        reference_basis(basis, facets=True),
        traction=traction,
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)


def test_gradient_linear_form_uses_native_path():
    basis = vector_basis()

    @skfn.LinearForm
    def native_form(v, w):
        return ddot(w.tensor, grad(v))

    @skfem.LinearForm
    def reference_form(v, w):
        return skfem_ddot(w.tensor, skfem_grad(v))

    tensor = np.arange(9, dtype=float).reshape(3, 3, 1, 1) / 10
    actual = native_form.assemble(basis, tensor=tensor)
    expected = reference_form.assemble(
        reference_basis(basis), tensor=tensor
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)


def test_coordinate_linear_form_uses_quadrature_context():
    basis = vector_basis()

    def load(x):
        return np.stack((1.+x[0],x[1]**2,-.5*x[2]),axis=0)

    @skfn.LinearForm
    def coordinate_load(v, w):
        return dot(load(w.x), v)

    @skfem.LinearForm
    def reference_load(v,w):
        return skfem_dot(load(w.x),v)

    actual=skfn.asm(coordinate_load,basis)
    expected=skfem.asm(reference_load,reference_basis(basis))
    np.testing.assert_allclose(actual,expected,rtol=3e-14,atol=3e-14)


def test_facet_normal_linear_form_uses_outward_normal():
    basis=vector_basis()
    native=skfn.FacetBasis(
        basis.mesh,basis.elem,facets=basis.mesh.boundary_facets(),intorder=2
    )

    @skfn.LinearForm
    def normal_load(v,w):
        return dot(w.n,v)

    @skfem.LinearForm
    def reference_load(v,w):
        return skfem_dot(w.n,v)

    actual=skfn.asm(normal_load,native)
    expected=skfem.asm(
        reference_load,reference_basis(basis,facets=True)
    )
    np.testing.assert_allclose(actual,expected,rtol=3e-14,atol=3e-14)


def test_native_bilinear_mass_form():
    basis = vector_basis()

    @skfn.BilinearForm
    def mass(u, v, w):
        return dot(u, v)

    actual = skfn.asm(mass, basis)

    @skfem.BilinearForm
    def reference(u, v, w):
        return skfem_dot(u, v)

    expected = skfem.asm(reference, reference_basis(basis))
    np.testing.assert_allclose(
        actual.toarray(), expected.toarray(), rtol=3e-13, atol=3e-13
    )


def test_coordinate_dependent_bilinear_coefficient():
    basis=vector_basis()

    @skfn.BilinearForm
    def weighted_mass(u,v,w):
        return (1.+w.x[0]**2)*dot(u,v)

    @skfem.BilinearForm
    def reference(u,v,w):
        return (1.+w.x[0]**2)*skfem_dot(u,v)

    actual=skfn.asm(weighted_mass,basis)
    expected=skfem.asm(reference,reference_basis(basis))
    np.testing.assert_allclose(
        actual.toarray(),expected.toarray(),rtol=3e-13,atol=3e-13
    )


def test_user_parameter_composes_with_volume_geometry():
    basis=vector_basis()

    @skfn.BilinearForm
    def weighted_mass(u,v,w):
        return w.scale*(1.+w.x[0])*dot(u,v)

    @skfem.BilinearForm
    def reference(u,v,w):
        return w.scale*(1.+w.x[0])*skfem_dot(u,v)

    actual=skfn.asm(weighted_mass,basis,scale=1.8)
    expected=skfem.asm(
        reference,reference_basis(basis),scale=1.8
    )
    np.testing.assert_allclose(
        actual.toarray(),expected.toarray(),rtol=3e-13,atol=3e-13
    )


def test_multiterm_volume_bilinear_form_matches_reference():
    basis=vector_basis()

    @skfn.BilinearForm
    def reaction_diffusion(u,v,w):
        return (
            w.reaction*(1.+w.x[0]**2)*dot(u,v)
            +w.diffusivity*np.exp(-w.x[1])
                *ddot(grad(u),grad(v))
        )

    @skfem.BilinearForm
    def reference(u,v,w):
        return (
            w.reaction*(1.+w.x[0]**2)*skfem_dot(u,v)
            +w.diffusivity*np.exp(-w.x[1])
                *skfem_ddot(skfem_grad(u),skfem_grad(v))
        )

    actual=skfn.asm(
        reaction_diffusion,basis,reaction=1.3,diffusivity=.4
    )
    expected=skfem.asm(
        reference,reference_basis(basis),
        reaction=1.3,diffusivity=.4,
    )
    np.testing.assert_allclose(
        actual.toarray(),expected.toarray(),rtol=4e-13,atol=4e-13
    )


def test_multiterm_bilinear_subtraction_is_supported():
    basis=vector_basis()

    @skfn.BilinearForm
    def form(u,v,w):
        return (
            2.*dot(u,v)+3.*ddot(grad(u),grad(v))
            -.5*dot(u,v)-ddot(grad(u),grad(v))
        )

    actual=skfn.asm(form,basis)

    @skfn.BilinearForm
    def reduced(u,v,w):
        return 1.5*dot(u,v)+2.*ddot(grad(u),grad(v))

    expected=skfn.asm(reduced,basis)
    np.testing.assert_allclose(actual.toarray(),expected.toarray())


def test_upstream_form_is_rejected_by_native_asm():
    basis = vector_basis()

    @skfem.LinearForm
    def upstream(v, w):
        return v[0]

    with pytest.raises(TypeError, match="use skfem.asm explicitly"):
        skfn.asm(upstream, basis)
