import numpy as np
import pytest
import skfem
from skfem.helpers import dot as reference_dot
from skfem.helpers import jump as reference_jump

import skfn
from skfn.helpers import dot,jump


@pytest.mark.parametrize(
    "mesh,element,reference_mesh,reference_element",
    [
        (
            skfn.MeshTri.init_tensor([0.,.4,1.],[0.,1.]),
            skfn.ElementTriP0(),skfem.MeshTri,skfem.ElementTriP0,
        ),
        (
            skfn.MeshQuad.init_tensor([0.,.4,1.],[0.,1.]),
            skfn.ElementQuad0(),skfem.MeshQuad,skfem.ElementQuad0,
        ),
        (
            skfn.MeshTet(),skfn.ElementTetP0(),
            skfem.MeshTet,skfem.ElementTetP0,
        ),
        (
            skfn.MeshHex(),skfn.ElementHex0(),
            skfem.MeshHex,skfem.ElementHex0,
        ),
    ],
)
def test_p0_volume_forms_and_interpolation_match_skfem(
    mesh,element,reference_mesh,reference_element
):
    basis=skfn.Basis(
        mesh,skfn.ElementVector(element,dim=1),intorder=2
    )
    reference_basis=skfem.Basis(
        reference_mesh(mesh.p,mesh.t),reference_element(),intorder=2
    )

    @skfn.BilinearForm
    def mass(u,v,w):
        return (1.+w.x[0])*dot(u,v)

    @skfem.BilinearForm
    def reference_mass(u,v,w):
        return (1.+w.x[0])*u*v

    np.testing.assert_allclose(
        skfn.asm(mass,basis).toarray(),
        skfem.asm(reference_mass,reference_basis).toarray(),
        rtol=3e-12,atol=3e-12,
    )
    coefficients=np.arange(1.,basis.N+1.)
    field=basis.interpolate(coefficients)
    np.testing.assert_allclose(
        field.value,reference_basis.interpolate(coefficients),
        rtol=2e-14,atol=2e-14,
    )
    np.testing.assert_allclose(field.grad,0.,atol=0.)


@pytest.mark.parametrize("kind",["tri","quad"])
@pytest.mark.parametrize("order",[0,1,2])
def test_element_dg_jump_penalty_matches_skfem(kind,order):
    axis=np.linspace(0.,1.,4)
    if kind=="tri":
        mesh=skfn.MeshTri.init_tensor(axis,axis)
        reference_mesh=skfem.MeshTri(mesh.p,mesh.t)
        element=(
            skfn.ElementTriP0() if order==0 else
            skfn.ElementTriP1() if order==1 else skfn.ElementTriP2()
        )
        reference_element=(
            skfem.ElementTriP0() if order==0 else
            skfem.ElementTriP1() if order==1 else skfem.ElementTriP2()
        )
        if order==2:
            mesh=skfn.MeshTri2.from_mesh(mesh)
            reference_mesh=skfem.MeshTri2.from_mesh(reference_mesh)
    else:
        mesh=skfn.MeshQuad.init_tensor(axis,axis)
        reference_mesh=skfem.MeshQuad(mesh.p,mesh.t)
        element=(
            skfn.ElementQuad0() if order==0 else
            skfn.ElementQuad1() if order==1 else skfn.ElementQuad2()
        )
        reference_element=(
            skfem.ElementQuad0() if order==0 else
            skfem.ElementQuad1() if order==1 else skfem.ElementQuad2()
        )
        if order==2:
            mesh=skfn.MeshQuad2.from_mesh(mesh)
            reference_mesh=skfem.MeshQuad2.from_mesh(reference_mesh)
    intorder=4 if order==2 else 2
    basis=[
        skfn.InteriorFacetBasis(
            mesh,
            skfn.ElementVector(
                element if order==0 else skfn.ElementDG(element),dim=1
            ),
            side=side,intorder=intorder,
        )
        for side in (0,1)
    ]
    reference_basis=[
        skfem.InteriorFacetBasis(
            reference_mesh,
            skfem.ElementVector(
                (
                    reference_element if order==0 else
                    skfem.ElementDG(reference_element)
                ),dim=1
            ),
            side=side,intorder=intorder,
        )
        for side in (0,1)
    ]

    @skfn.BilinearForm
    def penalty(u,v,w):
        return (1.+w.x[1])*dot(jump(w,u),jump(w,v))

    @skfem.BilinearForm
    def reference_penalty(u,v,w):
        return (
            (1.+w.x[1])
            *reference_dot(
                reference_jump(w,u),reference_jump(w,v)
            )
        )

    actual=skfn.asm(penalty,basis,basis)
    expected=skfem.asm(
        reference_penalty,reference_basis,reference_basis
    )
    assert actual.nnz>0
    np.testing.assert_allclose(
        actual.toarray(),expected.toarray(),
        rtol=5e-12,atol=5e-12,
    )


def test_dg_traces_are_independent_between_cells():
    mesh=skfn.MeshQuad.init_tensor([0.,.5,1.],[0.,1.])
    bases=[
        skfn.InteriorFacetBasis(
            mesh,
            skfn.ElementVector(
                skfn.ElementDG(skfn.ElementQuad1()),dim=1
            ),
            side=side,
        )
        for side in (0,1)
    ]
    coefficients=np.zeros(bases[0].N)
    coefficients[bases[0].element_dofs[:,0]]=2.
    traces=[basis.interpolate(coefficients).value for basis in bases]
    np.testing.assert_allclose(np.abs(traces[0]-traces[1]),2.)


def test_dg_get_dofs_matches_scikit_fem_policy():
    mesh=skfn.MeshTri.init_tensor([0.,.5,1.],[0.,1.])
    basis=skfn.Basis(
        mesh,
        skfn.ElementVector(
            skfn.ElementDG(skfn.ElementTriP1()),dim=1
        ),
    )
    assert basis.get_dofs().all().size==0
    np.testing.assert_array_equal(
        basis.get_dofs(elements=[1]).all(),
        basis.element_dofs[:,1],
    )
