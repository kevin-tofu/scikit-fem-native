import numpy as np
import pytest
import skfem

import skfn
from skfn.helpers import ddot,dot,grad
from skfem.helpers import dot as reference_dot
from skfem.helpers import grad as reference_grad


def _meshes():
    mesh=skfn.MeshTri.init_tensor(
        np.linspace(0.,1.,5),np.linspace(0.,1.,4)
    )
    return mesh,skfem.MeshTri(mesh.p,mesh.t)


def test_constructor_and_with_elements_match_skfem():
    mesh,reference_mesh=_meshes()
    elements=np.array([1,4,7,10,15],dtype=np.int64)
    element=skfn.ElementVector(skfn.ElementTriP1(),dim=1)
    direct=skfn.Basis(mesh,element,elements=elements)
    restricted=skfn.Basis(mesh,element).with_elements(elements)
    reference=skfem.Basis(
        reference_mesh,skfem.ElementTriP1(),elements=elements
    )

    @skfn.BilinearForm
    def form(u,v,w):
        return (1.+w.x[0])*dot(u,v)+.3*ddot(grad(u),grad(v))

    @skfem.BilinearForm
    def reference_form(u,v,w):
        return (
            (1.+w.x[0])*u*v
            +.3*reference_dot(reference_grad(u),reference_grad(v))
        )

    expected=skfem.asm(reference_form,reference)
    for basis in (direct,restricted):
        np.testing.assert_array_equal(basis.tind,elements)
        assert basis.nelems==len(elements)
        np.testing.assert_allclose(
            skfn.asm(form,basis).toarray(),expected.toarray(),
            rtol=3e-13,atol=3e-13,
        )


def test_restricted_linear_functional_and_interpolation_match_skfem():
    mesh,reference_mesh=_meshes()
    mask=mesh.p[0,mesh.t[:3]].mean(axis=0)>.55
    basis=skfn.Basis(
        mesh,skfn.ElementVector(skfn.ElementTriP1(),dim=1),
        elements=mask,
    )
    reference=skfem.Basis(
        reference_mesh,skfem.ElementTriP1(),elements=np.flatnonzero(mask)
    )
    coefficients=1.+basis.doflocs[0]-.2*basis.doflocs[1]

    @skfn.LinearForm
    def load(v,w):
        return dot((1.+w.x[1])[None],v)

    @skfem.LinearForm
    def reference_load(v,w):
        return (1.+w.x[1])*v

    @skfn.Functional
    def energy(w):
        return w.u*w.u

    @skfem.Functional
    def reference_energy(w):
        return w.u*w.u

    np.testing.assert_allclose(
        skfn.asm(load,basis),skfem.asm(reference_load,reference),
        rtol=3e-13,atol=3e-13,
    )
    np.testing.assert_allclose(
        skfn.asm(energy,basis,u=basis.interpolate(coefficients)),
        skfem.asm(
            reference_energy,reference,
            u=reference.interpolate(coefficients),
        ),
        rtol=3e-13,atol=3e-13,
    )


def test_cross_basis_and_composite_restriction():
    mesh,reference_mesh=_meshes()
    elements=np.array([0,3,6,9],dtype=np.int64)
    trial=skfn.Basis(
        mesh,skfn.ElementVector(skfn.ElementTriP0(),dim=1),
        elements=elements,
    )
    test=skfn.Basis(
        mesh,skfn.ElementVector(skfn.ElementTriP1(),dim=1),
        elements=elements,
    )
    reference_trial=skfem.Basis(
        reference_mesh,skfem.ElementTriP0(),elements=elements
    )
    reference_test=skfem.Basis(
        reference_mesh,skfem.ElementTriP1(),elements=elements
    )

    @skfn.BilinearForm
    def coupling(u,v,w):
        return (1.+w.x[0])*dot(u,v)

    @skfem.BilinearForm
    def reference_coupling(u,v,w):
        return (1.+w.x[0])*u*v

    np.testing.assert_allclose(
        skfn.asm(coupling,trial,test).toarray(),
        skfem.asm(
            reference_coupling,reference_trial,reference_test
        ).toarray(),
        rtol=3e-13,atol=3e-13,
    )

    composite=skfn.Basis(
        mesh,skfn.ElementTriP1()*skfn.ElementTriP1(),intorder=4
    ).with_elements(elements)
    reference_composite=skfem.Basis(
        reference_mesh,
        skfem.ElementTriP1()*skfem.ElementTriP1(),
        elements=elements,intorder=4,
    )

    @skfn.BilinearForm
    def mixed(u1,u2,v1,v2,w):
        return (1.+w.x[0])*u1*v1+2.*u2*v2+.2*u1*v2

    @skfem.BilinearForm
    def reference_mixed(u1,u2,v1,v2,w):
        return (1.+w.x[0])*u1*v1+2.*u2*v2+.2*u1*v2

    assert composite.nelems==len(elements)
    assert all(
        subbasis.nelems==len(elements)
        for subbasis in composite.subbases
    )
    np.testing.assert_array_equal(composite.tind,elements)
    np.testing.assert_allclose(
        skfn.asm(mixed,composite).toarray(),
        skfem.asm(reference_mixed,reference_composite).toarray(),
        rtol=3e-13,atol=3e-13,
    )


def test_element_selection_validation():
    basis=skfn.Basis(
        skfn.MeshTri(),
        skfn.ElementVector(skfn.ElementTriP1(),dim=1),
    )
    with pytest.raises(ValueError,match="mask"):
        basis.with_elements(np.array([True,False]))
    with pytest.raises(ValueError,match="duplicates"):
        basis.with_elements([0,0])
    with pytest.raises(IndexError):
        basis.with_elements([basis.mesh.nelements])

    empty=basis.with_elements([])

    @skfn.BilinearForm
    def mass(u,v,w):
        return dot(u,v)

    matrix=skfn.asm(mass,empty)
    assert empty.nelems==0
    assert matrix.shape==(basis.N,basis.N)
    assert matrix.nnz==0
