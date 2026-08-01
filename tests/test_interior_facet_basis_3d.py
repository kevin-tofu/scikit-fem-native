import numpy as np
import pytest
import skfem
from skfem.helpers import dot as reference_dot
from skfem.helpers import jump as reference_jump

import skfemntv
from skfemntv.helpers import dot,jump


def _meshes(kind,order):
    if kind=="tet":
        linear=skfemntv.MeshTet.init_tensor([0.,1.],[0.,1.],[0.,1.])
        reference_linear=skfem.MeshTet(linear.p,linear.t)
        if order==2:
            return (
                skfemntv.MeshTet2.from_mesh(linear),
                skfem.MeshTet2.from_mesh(reference_linear),
                skfemntv.ElementTetP2(),skfem.ElementTetP2(),
            )
        return (
            linear,reference_linear,
            skfemntv.ElementTetP1(),skfem.ElementTetP1(),
        )
    linear=skfemntv.MeshHex.init_tensor([0.,.5,1.],[0.,1.],[0.,1.])
    reference_linear=skfem.MeshHex(linear.p,linear.t)
    if order==2:
        return (
            skfemntv.MeshHex2.from_mesh(linear),
            skfem.MeshHex2.from_mesh(reference_linear),
            skfemntv.ElementHex2(),skfem.ElementHex2(),
        )
    return (
        linear,reference_linear,
        skfemntv.ElementHex1(),skfem.ElementHex1(),
    )


def _coordinate_permutation(native,reference,components):
    lookup={}
    for dof in range(reference.N):
        coordinate=tuple(np.round(reference.doflocs[:,dof],14))
        lookup.setdefault(coordinate,[]).append(dof)
    permutation=np.empty(native.N,dtype=np.int64)
    for dof in range(native.N):
        coordinate=tuple(np.round(native.doflocs[:,dof],14))
        permutation[dof]=lookup[coordinate][dof%components]
    return permutation


@pytest.mark.parametrize("kind",["tet","hex"])
@pytest.mark.parametrize("order",[1,2])
@pytest.mark.parametrize("side",[0,1])
def test_3d_interior_facet_forms_match_skfem(kind,order,side):
    mesh,reference_mesh,element,reference_element=_meshes(kind,order)
    intorder=4 if order==2 else 2
    basis=skfemntv.InteriorFacetBasis(
        mesh,skfemntv.ElementVector(element),side=side,intorder=intorder
    )
    reference_basis=skfem.InteriorFacetBasis(
        reference_mesh,skfem.ElementVector(reference_element),
        side=side,intorder=intorder,
    )

    @skfemntv.Functional
    def measure(w):
        return (
            1.+w.x[0]+.2*w.x[2]
            +.1*w.n[0]-.07*w.n[1]+.03*w.n[2]
        )

    @skfem.Functional
    def reference_measure(w):
        return (
            1.+w.x[0]+.2*w.x[2]
            +.1*w.n[0]-.07*w.n[1]+.03*w.n[2]
        )

    np.testing.assert_allclose(
        skfemntv.asm(measure,basis),
        skfem.asm(reference_measure,reference_basis),
        rtol=8e-12,atol=8e-12,
    )

    @skfemntv.LinearForm
    def flux(v,w):
        return dot((1.+w.x[1])*w.n,v)

    @skfem.LinearForm
    def reference_flux(v,w):
        return reference_dot((1.+w.x[1])*w.n,v)

    actual=skfemntv.asm(flux,basis)
    expected=skfem.asm(reference_flux,reference_basis)
    permutation=_coordinate_permutation(basis,reference_basis,3)
    np.testing.assert_allclose(
        actual,expected[permutation],rtol=8e-12,atol=8e-12
    )


@pytest.mark.parametrize("kind",["tet","hex"])
def test_3d_interior_sides_share_points_normals_and_continuous_trace(kind):
    mesh,_,element,_=_meshes(kind,1)
    bases=[
        skfemntv.InteriorFacetBasis(
            mesh,skfemntv.ElementVector(element,dim=1),side=side
        )
        for side in (0,1)
    ]
    np.testing.assert_allclose(
        bases[0].global_coordinates,bases[1].global_coordinates,
        rtol=3e-14,atol=3e-14,
    )
    np.testing.assert_allclose(
        bases[0].normals,bases[1].normals,
        rtol=3e-14,atol=3e-14,
    )
    coefficients=(
        bases[0].doflocs[0]
        +2.*bases[0].doflocs[1]-bases[0].doflocs[2]
    )
    traces=[basis.interpolate(coefficients).value for basis in bases]
    np.testing.assert_allclose(
        traces[0],traces[1],rtol=3e-14,atol=3e-14
    )


@pytest.mark.parametrize("kind",["tet","hex"])
@pytest.mark.parametrize("space",["p0","dg1"])
def test_3d_discontinuous_jump_penalty_matches_skfem(kind,space):
    mesh,reference_mesh,h1,reference_h1=_meshes(kind,1)
    if space=="p0":
        element=skfemntv.ElementTetP0() if kind=="tet" else skfemntv.ElementHex0()
        reference_element=(
            skfem.ElementTetP0() if kind=="tet" else skfem.ElementHex0()
        )
    else:
        element=skfemntv.ElementDG(h1)
        reference_element=skfem.ElementDG(reference_h1)
    bases=[
        skfemntv.InteriorFacetBasis(
            mesh,skfemntv.ElementVector(element,dim=1),side=side
        )
        for side in (0,1)
    ]
    reference_bases=[
        skfem.InteriorFacetBasis(
            reference_mesh,
            skfem.ElementVector(reference_element,dim=1),side=side,
        )
        for side in (0,1)
    ]

    @skfemntv.BilinearForm
    def penalty(u,v,w):
        return (1.+w.x[2])*dot(jump(w,u),jump(w,v))

    @skfem.BilinearForm
    def reference_penalty(u,v,w):
        return (
            (1.+w.x[2])
            *reference_dot(
                reference_jump(w,u),reference_jump(w,v)
            )
        )

    actual=skfemntv.asm(penalty,bases,bases)
    expected=skfem.asm(
        reference_penalty,reference_bases,reference_bases
    )
    np.testing.assert_allclose(
        actual.toarray(),expected.toarray(),
        rtol=8e-12,atol=8e-12,
    )
