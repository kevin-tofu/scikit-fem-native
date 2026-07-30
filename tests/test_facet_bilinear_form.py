import numpy as np
import skfem
from skfem.helpers import ddot as reference_ddot
from skfem.helpers import dot as reference_dot
from skfem.helpers import grad as reference_grad

import skfn
from skfn.helpers import ddot, dot, grad


def bases():
    mesh = skfn.MeshHex()
    native = skfn.FacetBasis(
        mesh, skfn.ElementVector(skfn.ElementHex1()), intorder=4
    )
    reference_mesh = skfem.MeshHex(mesh.p, mesh.t)
    reference = skfem.FacetBasis(
        reference_mesh,
        skfem.ElementVector(skfem.ElementHex1()),
        facets=reference_mesh.boundary_facets(),
        intorder=4,
    )
    return native, reference


def test_native_facet_penalty_bilinear_form():
    native, reference = bases()

    @skfn.BilinearForm
    def penalty(u, v, w):
        return w.alpha * dot(u, v)

    @skfem.BilinearForm
    def expected_form(u, v, w):
        return w.alpha * reference_dot(u, v)

    actual = skfn.asm(penalty, native, alpha=7.5)
    expected = skfem.asm(expected_form, reference, alpha=7.5)
    np.testing.assert_allclose(
        actual.toarray(), expected.toarray(), rtol=5e-13, atol=5e-13
    )


def test_native_facet_gradient_bilinear_form():
    native, reference = bases()

    @skfn.BilinearForm
    def gradient_form(u, v, w):
        return ddot(grad(u), grad(v))

    @skfem.BilinearForm
    def expected_form(u, v, w):
        return reference_ddot(reference_grad(u), reference_grad(v))

    actual = gradient_form.assemble(native)
    expected = expected_form.assemble(reference)
    np.testing.assert_allclose(
        actual.toarray(), expected.toarray(), rtol=8e-13, atol=8e-13
    )


def test_repeated_facet_assembly_reuses_csr_structure():
    native, _ = bases()

    @skfn.BilinearForm
    def penalty(u, v, w):
        return w.alpha * dot(u, v)

    first = penalty.assemble(native, alpha=1.)
    first_values = first.toarray().copy()
    data_id = id(first.data)
    second = penalty.assemble(native, alpha=2.)
    assert id(second.data) == data_id
    np.testing.assert_allclose(second.toarray(), 2 * first_values)
