import numpy as np
import pytest
import skfem
from skfem.helpers import dot as reference_dot
from skfem.helpers import jump as reference_jump

import skfemntv
from skfemntv.helpers import dot,jump


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


@pytest.mark.parametrize("kind",["tri","quad"])
@pytest.mark.parametrize("order",[1,2])
@pytest.mark.parametrize("side",[0,1])
def test_interior_facet_functional_and_linear_form_match_skfem(
    kind,order,side
):
    axis=np.linspace(0.,1.,4)
    if kind=="tri":
        linear=skfemntv.MeshTri.init_tensor(axis,axis)
        reference_linear=skfem.MeshTri(linear.p,linear.t)
        if order==2:
            mesh=skfemntv.MeshTri2.from_mesh(linear)
            reference_mesh=skfem.MeshTri2.from_mesh(reference_linear)
            element=skfemntv.ElementTriP2()
            reference_element=skfem.ElementTriP2()
        else:
            mesh=linear;reference_mesh=reference_linear
            element=skfemntv.ElementTriP1()
            reference_element=skfem.ElementTriP1()
    else:
        linear=skfemntv.MeshQuad.init_tensor(axis,axis)
        reference_linear=skfem.MeshQuad(linear.p,linear.t)
        if order==2:
            mesh=skfemntv.MeshQuad2.from_mesh(linear)
            reference_mesh=skfem.MeshQuad2.from_mesh(reference_linear)
            element=skfemntv.ElementQuad2()
            reference_element=skfem.ElementQuad2()
        else:
            mesh=linear;reference_mesh=reference_linear
            element=skfemntv.ElementQuad1()
            reference_element=skfem.ElementQuad1()
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
        return 1.+w.x[0]+.2*w.n[0]-.1*w.n[1]

    @skfem.Functional
    def reference_measure(w):
        return 1.+w.x[0]+.2*w.n[0]-.1*w.n[1]

    np.testing.assert_allclose(
        skfemntv.asm(measure,basis),
        skfem.asm(reference_measure,reference_basis),
        rtol=3e-12,atol=3e-12,
    )

    @skfemntv.LinearForm
    def flux(v,w):
        return dot((1.+w.x[1])*w.n,v)

    @skfem.LinearForm
    def reference_flux(v,w):
        return reference_dot((1.+w.x[1])*w.n,v)

    actual=skfemntv.asm(flux,basis)
    expected=skfem.asm(reference_flux,reference_basis)
    permutation=_coordinate_permutation(basis,reference_basis,2)
    np.testing.assert_allclose(
        actual,expected[permutation],rtol=4e-12,atol=4e-12
    )


@pytest.mark.parametrize("kind",["tri","quad"])
def test_interior_facet_sides_share_points_and_normals(kind):
    axis=np.linspace(0.,1.,5)
    if kind=="tri":
        mesh=skfemntv.MeshTri.init_tensor(axis,axis)
        element=skfemntv.ElementTriP1()
    else:
        mesh=skfemntv.MeshQuad.init_tensor(axis,axis)
        element=skfemntv.ElementQuad1()
    bases=[
        skfemntv.InteriorFacetBasis(
            mesh,skfemntv.ElementVector(element,dim=1),side=side
        )
        for side in (0,1)
    ]
    np.testing.assert_allclose(
        bases[0].global_coordinates,bases[1].global_coordinates,
        rtol=2e-14,atol=2e-14,
    )
    np.testing.assert_allclose(
        bases[0].normals,bases[1].normals,
        rtol=2e-14,atol=2e-14,
    )

    coefficients=bases[0].doflocs[0]+2.*bases[0].doflocs[1]
    traces=[basis.interpolate(coefficients) for basis in bases]
    np.testing.assert_allclose(
        traces[0].value,traces[1].value,rtol=2e-14,atol=2e-14
    )


@pytest.mark.parametrize("kind",["tri","quad"])
@pytest.mark.parametrize("order",[1,2])
def test_jump_penalty_assembly_matches_skfem(kind,order):
    axis=np.linspace(0.,1.,4)
    if kind=="tri":
        linear=skfemntv.MeshTri.init_tensor(axis,axis)
        reference_linear=skfem.MeshTri(linear.p,linear.t)
        if order==2:
            mesh=skfemntv.MeshTri2.from_mesh(linear)
            reference_mesh=skfem.MeshTri2.from_mesh(reference_linear)
            element=skfemntv.ElementTriP2()
            reference_element=skfem.ElementTriP2()
        else:
            mesh=linear;reference_mesh=reference_linear
            element=skfemntv.ElementTriP1()
            reference_element=skfem.ElementTriP1()
    else:
        linear=skfemntv.MeshQuad.init_tensor(axis,axis)
        reference_linear=skfem.MeshQuad(linear.p,linear.t)
        if order==2:
            mesh=skfemntv.MeshQuad2.from_mesh(linear)
            reference_mesh=skfem.MeshQuad2.from_mesh(reference_linear)
            element=skfemntv.ElementQuad2()
            reference_element=skfem.ElementQuad2()
        else:
            mesh=linear;reference_mesh=reference_linear
            element=skfemntv.ElementQuad1()
            reference_element=skfem.ElementQuad1()
    intorder=4 if order==2 else 2
    bases=[
        skfemntv.InteriorFacetBasis(
            mesh,skfemntv.ElementVector(element,dim=1),
            side=side,intorder=intorder,
        )
        for side in (0,1)
    ]
    reference_bases=[
        skfem.InteriorFacetBasis(
            reference_mesh,skfem.ElementVector(reference_element),
            side=side,intorder=intorder,
        )
        for side in (0,1)
    ]

    @skfemntv.BilinearForm
    def penalty(u,v,w):
        return (1.+w.x[0])*dot(jump(w,u),jump(w,v))

    @skfem.BilinearForm
    def reference_penalty(u,v,w):
        return (
            (1.+w.x[0])
            *reference_dot(
                reference_jump(w,u),reference_jump(w,v)
            )
        )

    actual=skfemntv.asm(penalty,bases,bases)
    expected=skfem.asm(
        reference_penalty,reference_bases,reference_bases
    )
    permutation=_coordinate_permutation(
        bases[0],reference_bases[0],1
    )
    np.testing.assert_allclose(
        actual.toarray(),
        expected[permutation][:,permutation].toarray(),
        rtol=5e-12,atol=5e-12,
    )
