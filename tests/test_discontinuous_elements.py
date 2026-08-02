import numpy as np
import pytest
import skfem
from skfem.helpers import dot as reference_dot
from skfem.helpers import jump as reference_jump

import skfemntv
from skfemntv.helpers import dot,jump


@pytest.mark.parametrize(
    "mesh,element,reference_mesh,reference_element",
    [
        (
            skfemntv.MeshTri.init_tensor([0.,.4,1.],[0.,1.]),
            skfemntv.ElementTriP0(),skfem.MeshTri,skfem.ElementTriP0,
        ),
        (
            skfemntv.MeshQuad.init_tensor([0.,.4,1.],[0.,1.]),
            skfemntv.ElementQuad0(),skfem.MeshQuad,skfem.ElementQuad0,
        ),
        (
            skfemntv.MeshTet(),skfemntv.ElementTetP0(),
            skfem.MeshTet,skfem.ElementTetP0,
        ),
        (
            skfemntv.MeshHex(),skfemntv.ElementHex0(),
            skfem.MeshHex,skfem.ElementHex0,
        ),
    ],
)
def test_p0_volume_forms_and_interpolation_match_skfem(
    mesh,element,reference_mesh,reference_element
):
    basis=skfemntv.Basis(
        mesh,skfemntv.ElementVector(element,dim=1),intorder=2
    )
    reference_basis=skfem.Basis(
        reference_mesh(mesh.p,mesh.t),reference_element(),intorder=2
    )

    @skfemntv.BilinearForm
    def mass(u,v,w):
        return (1.+w.x[0])*dot(u,v)

    @skfem.BilinearForm
    def reference_mass(u,v,w):
        return (1.+w.x[0])*u*v

    np.testing.assert_allclose(
        skfemntv.asm(mass,basis).toarray(),
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
        mesh=skfemntv.MeshTri.init_tensor(axis,axis)
        reference_mesh=skfem.MeshTri(mesh.p,mesh.t)
        element=(
            skfemntv.ElementTriP0() if order==0 else
            skfemntv.ElementTriP1() if order==1 else skfemntv.ElementTriP2()
        )
        reference_element=(
            skfem.ElementTriP0() if order==0 else
            skfem.ElementTriP1() if order==1 else skfem.ElementTriP2()
        )
        if order==2:
            mesh=skfemntv.MeshTri2.from_mesh(mesh)
            reference_mesh=skfem.MeshTri2.from_mesh(reference_mesh)
    else:
        mesh=skfemntv.MeshQuad.init_tensor(axis,axis)
        reference_mesh=skfem.MeshQuad(mesh.p,mesh.t)
        element=(
            skfemntv.ElementQuad0() if order==0 else
            skfemntv.ElementQuad1() if order==1 else skfemntv.ElementQuad2()
        )
        reference_element=(
            skfem.ElementQuad0() if order==0 else
            skfem.ElementQuad1() if order==1 else skfem.ElementQuad2()
        )
        if order==2:
            mesh=skfemntv.MeshQuad2.from_mesh(mesh)
            reference_mesh=skfem.MeshQuad2.from_mesh(reference_mesh)
    intorder=4 if order==2 else 2
    basis=[
        skfemntv.InteriorFacetBasis(
            mesh,
            skfemntv.ElementVector(
                element if order==0 else skfemntv.ElementDG(element),dim=1
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

    @skfemntv.BilinearForm
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

    actual=skfemntv.asm(penalty,basis,basis)
    expected=skfem.asm(
        reference_penalty,reference_basis,reference_basis
    )
    assert actual.nnz>0
    np.testing.assert_allclose(
        actual.toarray(),expected.toarray(),
        rtol=5e-12,atol=5e-12,
    )


def test_dg_traces_are_independent_between_cells():
    mesh=skfemntv.MeshQuad.init_tensor([0.,.5,1.],[0.,1.])
    bases=[
        skfemntv.InteriorFacetBasis(
            mesh,
            skfemntv.ElementVector(
                skfemntv.ElementDG(skfemntv.ElementQuad1()),dim=1
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
    mesh=skfemntv.MeshTri.init_tensor([0.,.5,1.],[0.,1.])
    basis=skfemntv.Basis(
        mesh,
        skfemntv.ElementVector(
            skfemntv.ElementDG(skfemntv.ElementTriP1()),dim=1
        ),
    )
    assert basis.get_dofs().all().size==0
    np.testing.assert_array_equal(
        basis.get_dofs(elements=[1]).all(),
        basis.element_dofs[:,1],
    )
