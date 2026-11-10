import numpy as np
import skfem
import skfemntv
from skfem.helpers import dot as reference_dot
from skfem.helpers import grad as reference_grad
from skfemntv.helpers import dot, grad


def test_scalar_h1_bilinear_linear_and_elemental_functional_match_skfem():
    native_mesh = skfemntv.MeshTri().refined(1)
    reference_mesh = skfem.MeshTri(native_mesh.p, native_mesh.t)
    native_basis = skfemntv.Basis(
        native_mesh, skfemntv.ElementTriP1()
    )
    reference_basis = skfem.Basis(
        reference_mesh, skfem.ElementTriP1()
    )

    @skfemntv.BilinearForm
    def native_bilinear(u, v, w):
        return 2.0 * u * v + dot(grad(u), grad(v))

    @skfem.BilinearForm
    def reference_bilinear(u, v, w):
        return 2.0 * u * v + reference_dot(
            reference_grad(u), reference_grad(v)
        )

    np.testing.assert_allclose(
        skfemntv.asm(native_bilinear, native_basis).toarray(),
        skfem.asm(reference_bilinear, reference_basis).toarray(),
        rtol=2e-13,
        atol=2e-13,
    )

    force = np.ones(native_basis.dx.shape)

    @skfemntv.LinearForm
    def native_linear(v, w):
        return w.force * v

    @skfem.LinearForm
    def reference_linear(v, w):
        return w.force * v

    np.testing.assert_allclose(
        skfemntv.asm(native_linear, native_basis, force=force),
        skfem.asm(reference_linear, reference_basis, force=force),
        rtol=2e-13,
        atol=2e-13,
    )

    @skfemntv.Functional
    def native_energy(w):
        return w.x[0] ** 2

    @skfem.Functional
    def reference_energy(w):
        return w.x[0] ** 2

    elemental = native_energy.elemental(native_basis)
    np.testing.assert_allclose(
        elemental,
        reference_energy.elemental(reference_basis),
        rtol=2e-13,
        atol=2e-13,
    )
    np.testing.assert_allclose(
        native_energy.assemble(native_basis),
        elemental.sum(),
        rtol=2e-13,
        atol=2e-13,
    )


def test_vector_linear_form_component_index_matches_skfem():
    native_mesh = skfemntv.MeshTri().refined(1)
    reference_mesh = skfem.MeshTri(native_mesh.p, native_mesh.t)
    native_element = skfemntv.ElementVector(
        skfemntv.ElementTriP1(), dim=2
    )
    reference_element = skfem.ElementVector(skfem.ElementTriP1(), dim=2)
    facets = native_mesh.boundary_facets()
    native_basis = skfemntv.FacetBasis(
        native_mesh, native_element, facets=facets
    )
    reference_basis = skfem.FacetBasis(
        reference_mesh, reference_element, facets=facets
    )

    @skfemntv.LinearForm
    def native_load(v, w):
        return 2.5 * v[1]

    @skfem.LinearForm
    def reference_load(v, w):
        return 2.5 * v[1]

    np.testing.assert_allclose(
        skfemntv.asm(native_load, native_basis),
        skfem.asm(reference_load, reference_basis),
        rtol=2e-13,
        atol=2e-13,
    )
