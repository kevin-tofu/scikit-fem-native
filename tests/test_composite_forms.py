import numpy as np
import skfem
from skfem.helpers import dot as reference_dot
from skfem.helpers import grad as reference_grad

import skfn
from skfn.helpers import dot,grad


def test_element_multiplication_creates_composite_basis():
    mesh=skfn.MeshTet()
    element=skfn.ElementTetP1()*skfn.ElementTetP1()
    basis=skfn.Basis(mesh,element,intorder=2)
    assert isinstance(element,skfn.ElementComposite)
    assert len(element.elems)==2
    assert len(basis.subbases)==2
    assert basis.N==2*mesh.p.shape[1]
    np.testing.assert_array_equal(
        basis.subbases[0].nodal_dofs[0],np.arange(0,basis.N,2)
    )
    np.testing.assert_array_equal(
        basis.subbases[1].nodal_dofs[0],np.arange(1,basis.N,2)
    )


def test_composite_form_signature_and_all_scalar_blocks_match_skfem():
    mesh=skfn.MeshTet.init_tensor(
        np.linspace(0.,1.,3),
        np.linspace(0.,1.,3),
        np.linspace(0.,1.,3),
    )
    basis=skfn.Basis(
        mesh,skfn.ElementTetP1()*skfn.ElementTetP1(),intorder=2
    )

    @skfn.BilinearForm
    def form(u1,u2,v1,v2,w):
        return (
            (1.+w.x[0])*u1*v1
            +2.*u2*v2+.3*u2*v1-.4*u1*v2
            +w.diffusion*dot(grad(u1),grad(v1))
        )

    actual=skfn.asm(form,basis,diffusion=.6)

    reference_mesh=skfem.MeshTet(mesh.p,mesh.t)
    reference_basis=skfem.Basis(
        reference_mesh,
        skfem.ElementTetP1()*skfem.ElementTetP1(),
        intorder=2,
    )

    @skfem.BilinearForm
    def reference(u1,u2,v1,v2,w):
        return (
            (1.+w.x[0])*u1*v1
            +2.*u2*v2+.3*u2*v1-.4*u1*v2
            +w.diffusion*reference_dot(
                reference_grad(u1),reference_grad(v1)
            )
        )

    expected=skfem.asm(reference,reference_basis,diffusion=.6)
    np.testing.assert_allclose(
        actual.toarray(),expected.toarray(),rtol=4e-13,atol=4e-13
    )


def test_repeated_composite_assembly_reuses_native_blocks():
    basis=skfn.Basis(
        skfn.MeshTet(),
        skfn.ElementTetP1()*skfn.ElementTetP1(),
    )

    @skfn.BilinearForm
    def form(u1,u2,v1,v2,w):
        return w.a*u1*v1+w.b*u2*v2

    first=skfn.asm(form,basis,a=1.,b=2.)
    native=form._native_cache[basis]
    ids={key:id(value) for key,value in native._assemblers.items()}
    second=skfn.asm(form,basis,a=3.,b=4.)
    assert ids=={
        key:id(value) for key,value in native._assemblers.items()
    }
    assert first.shape==second.shape==(basis.N,basis.N)
