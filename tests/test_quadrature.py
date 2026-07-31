import math

import numpy as np
import pytest
import skfem

import skfn


def _volume_case(kind):
    if kind=="tri":
        mesh=skfn.MeshTri()
        return (
            mesh,skfn.ElementTriP1(),
            skfem.MeshTri(mesh.p,mesh.t),skfem.ElementTriP1(),(4,3),
        )
    if kind=="quad":
        mesh=skfn.MeshQuad()
        return (
            mesh,skfn.ElementQuad1(),
            skfem.MeshQuad(mesh.p,mesh.t),skfem.ElementQuad1(),(4,3),
        )
    if kind=="tet":
        mesh=skfn.MeshTet()
        return (
            mesh,skfn.ElementTetP1(),
            skfem.MeshTet(mesh.p,mesh.t),skfem.ElementTetP1(),(3,2,2),
        )
    mesh=skfn.MeshHex()
    return (
        mesh,skfn.ElementHex1(),
        skfem.MeshHex(mesh.p,mesh.t),skfem.ElementHex1(),(3,2,2),
    )


def _exact_monomial(kind,powers):
    if kind in {"tri","tet"}:
        numerator=np.prod([math.factorial(power) for power in powers])
        return numerator/math.factorial(sum(powers)+len(powers))
    return np.prod([1./(power+1) for power in powers])


@pytest.mark.parametrize("kind",["tri","quad","tet","hex"])
def test_high_order_volume_quadrature_integrates_monomials(kind):
    mesh,element,_,_,powers=_volume_case(kind)
    basis=skfn.Basis(
        mesh,skfn.ElementVector(element,dim=1),intorder=8
    )

    @skfn.Functional
    def monomial(w):
        value=1.
        for axis,power in enumerate(powers):
            value=value*w.x[axis]**power
        return value

    np.testing.assert_allclose(
        skfn.asm(monomial,basis),_exact_monomial(kind,powers),
        rtol=3e-13,atol=3e-13,
    )
    assert basis.X.shape[1]>(
        6 if kind=="tri" else 11 if kind=="tet" else
        9 if kind=="quad" else 27
    )


@pytest.mark.parametrize("kind",["tri","quad","tet","hex"])
def test_high_order_facet_quadrature_matches_skfem(kind):
    mesh,element,reference_mesh,reference_element,_=_volume_case(kind)
    basis=skfn.FacetBasis(
        mesh,skfn.ElementVector(element,dim=1),intorder=8
    )
    reference=skfem.FacetBasis(
        reference_mesh,reference_element,intorder=8
    )

    @skfn.Functional
    def functional(w):
        return (
            1.+w.x[0]**6+.3*w.x[-1]**7
            +.2*w.n[0]**2
        )

    @skfem.Functional
    def reference_functional(w):
        return (
            1.+w.x[0]**6+.3*w.x[-1]**7
            +.2*w.n[0]**2
        )

    np.testing.assert_allclose(
        skfn.asm(functional,basis),
        skfem.asm(reference_functional,reference),
        rtol=2e-12,atol=2e-12,
    )


@pytest.mark.parametrize("kind",["tri","quad","tet","hex"])
def test_custom_volume_quadrature_matches_skfem(kind):
    mesh,element,reference_mesh,reference_element,_=_volume_case(kind)
    dimension=mesh.dim()
    point=np.full((dimension,1),1./(dimension+1))
    if kind=="quad":
        point[:]=.37
    elif kind=="hex":
        # The two packages use opposite reference-node conventions for Hex8.
        point[:]=.5
    weight=np.array([.123])
    quadrature=(point,weight)
    basis=skfn.Basis(
        mesh,skfn.ElementVector(element,dim=1),
        quadrature=quadrature,
    )
    reference=skfem.Basis(
        reference_mesh,reference_element,quadrature=quadrature
    )

    @skfn.Functional
    def functional(w):
        return 1.+w.x[0]+.2*w.x[-1]**2

    @skfem.Functional
    def reference_functional(w):
        return 1.+w.x[0]+.2*w.x[-1]**2

    np.testing.assert_allclose(
        skfn.asm(functional,basis),
        skfem.asm(reference_functional,reference),
        rtol=3e-14,atol=3e-14,
    )
    np.testing.assert_array_equal(basis.X,point)
    np.testing.assert_array_equal(basis.W,weight)


@pytest.mark.parametrize("kind",["tri","quad","tet","hex"])
def test_custom_facet_quadrature_matches_skfem(kind):
    mesh,element,reference_mesh,reference_element,_=_volume_case(kind)
    facet_dimension=mesh.dim()-1
    point=np.full((facet_dimension,1),.37)
    if kind=="tet":
        point[:]=.25
    weight=np.array([.41])
    quadrature=(point,weight)
    basis=skfn.FacetBasis(
        mesh,skfn.ElementVector(element,dim=1),
        quadrature=quadrature,
    )
    reference=skfem.FacetBasis(
        reference_mesh,reference_element,quadrature=quadrature
    )

    @skfn.Functional
    def functional(w):
        return 1.+w.x[0]+.2*w.n[-1]**2

    @skfem.Functional
    def reference_functional(w):
        return 1.+w.x[0]+.2*w.n[-1]**2

    np.testing.assert_allclose(
        skfn.asm(functional,basis),
        skfem.asm(reference_functional,reference),
        rtol=3e-13,atol=3e-13,
    )


def test_custom_interior_quadrature_aligns_both_sides():
    mesh=skfn.MeshHex.init_tensor([0.,.5,1.],[0.,1.],[0.,1.])
    quadrature=(np.array([[.2,.7],[.3,.6]]),np.array([.4,.6]))
    bases=[
        skfn.InteriorFacetBasis(
            mesh,skfn.ElementVector(skfn.ElementHex1(),dim=1),
            side=side,quadrature=quadrature,
        )
        for side in (0,1)
    ]
    np.testing.assert_allclose(
        bases[0].global_coordinates,bases[1].global_coordinates,
        rtol=2e-14,atol=2e-14,
    )
    np.testing.assert_array_equal(bases[0].X,quadrature[0])
    np.testing.assert_array_equal(bases[0].W,quadrature[1])


@pytest.mark.parametrize(
    "quadrature",
    [
        (np.zeros((1,2)),np.ones(2)),
        (np.zeros((2,2)),np.ones(3)),
        (np.array([[np.nan],[0.]]),np.ones(1)),
    ],
)
def test_invalid_custom_quadrature_raises(quadrature):
    with pytest.raises(ValueError):
        skfn.Basis(
            skfn.MeshTri(),
            skfn.ElementVector(skfn.ElementTriP1(),dim=1),
            quadrature=quadrature,
        )
