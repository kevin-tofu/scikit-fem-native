import numpy as np
import pytest
import skfem
from skfem.helpers import ddot as reference_ddot
from skfem.helpers import div as reference_div
from skfem.helpers import dot as reference_dot
from skfem.helpers import grad as reference_grad

import skfemntv
from skfemntv.helpers import ddot,div,dot,grad


def _reference_mesh(mesh,quadratic):
    if not quadratic:
        return skfem.MeshQuad(mesh.p,mesh.t)
    linear_nodes=np.unique(mesh.t[:4])
    linear=skfem.MeshQuad(mesh.p[:,linear_nodes],mesh.t[:4])
    reference=skfem.MeshQuad2.from_mesh(linear)
    lookup={
        tuple(np.round(mesh.p[:,node],14)):mesh.p[:,node]
        for node in range(mesh.p.shape[1])
    }
    for node in range(reference.p.shape[1]):
        key=tuple(np.round(reference.p[:,node],14))
        if key in lookup:
            reference.p[:,node]=lookup[key]
    return reference


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


def _composite_permutation(native,reference):
    permutation=np.empty(native.N,dtype=np.int64)
    for native_dofs,reference_dofs,components in zip(
        native.split_indices(),reference.split_indices(),(2,1)
    ):
        lookup={}
        for dof in reference_dofs:
            coordinate=tuple(np.round(reference.doflocs[:,dof],14))
            lookup.setdefault(coordinate,[]).append(int(dof))
        for offset,dof in enumerate(native_dofs):
            coordinate=tuple(np.round(native.doflocs[:,dof],14))
            permutation[dof]=lookup[coordinate][offset%components]
    return permutation


@pytest.mark.parametrize(
    "mesh,element,reference_element,intorder",
    [
        (
            skfemntv.MeshQuad(),skfemntv.ElementQuad1(),
            skfem.ElementQuad1,2,
        ),
        (
            skfemntv.MeshQuad2(),skfemntv.ElementQuad2(),
            skfem.ElementQuad2,4,
        ),
    ],
)
def test_quad_volume_forms_match_skfem(
    mesh,element,reference_element,intorder
):
    basis=skfemntv.Basis(
        mesh,skfemntv.ElementVector(element,dim=1),intorder=intorder
    )

    @skfemntv.BilinearForm
    def form(u,v,w):
        return (1.+w.x[0])*dot(u,v)+.7*ddot(grad(u),grad(v))

    actual=skfemntv.asm(form,basis)
    reference_basis=skfem.Basis(
        _reference_mesh(mesh,intorder>=4),
        reference_element(),
        intorder=intorder,
    )

    @skfem.BilinearForm
    def reference(u,v,w):
        return (
            (1.+w.x[0])*u*v
            +.7*reference_dot(reference_grad(u),reference_grad(v))
        )

    expected=skfem.asm(reference,reference_basis)
    permutation=_coordinate_permutation(basis,reference_basis,1)
    np.testing.assert_allclose(
        actual.toarray(),expected[permutation][:,permutation].toarray(),
        rtol=3e-12,atol=3e-12,
    )


@pytest.mark.parametrize(
    "mesh,element,reference_element,intorder",
    [
        (
            skfemntv.MeshQuad(),skfemntv.ElementQuad1(),
            skfem.ElementQuad1,2,
        ),
        (
            skfemntv.MeshQuad2(),skfemntv.ElementQuad2(),
            skfem.ElementQuad2,4,
        ),
    ],
)
def test_quad_facet_forms_match_skfem(
    mesh,element,reference_element,intorder
):
    basis=skfemntv.FacetBasis(
        mesh,skfemntv.ElementVector(element),intorder=intorder
    )

    @skfemntv.LinearForm
    def pressure(v,w):
        return dot((1.+w.x[0])*w.n,v)

    @skfemntv.Functional
    def measure(w):
        return 1.+w.x[1]+w.n[0]**2

    actual=skfemntv.asm(pressure,basis)
    actual_measure=skfemntv.asm(measure,basis)
    reference_basis=skfem.FacetBasis(
        _reference_mesh(mesh,intorder>=4),
        skfem.ElementVector(reference_element()),
        intorder=intorder,
    )

    @skfem.LinearForm
    def reference_pressure(v,w):
        return reference_dot((1.+w.x[0])*w.n,v)

    @skfem.Functional
    def reference_measure(w):
        return 1.+w.x[1]+w.n[0]**2

    expected=skfem.asm(reference_pressure,reference_basis)
    permutation=_coordinate_permutation(basis,reference_basis,2)
    np.testing.assert_allclose(
        actual,expected[permutation],rtol=3e-12,atol=3e-12
    )
    np.testing.assert_allclose(
        actual_measure,skfem.asm(reference_measure,reference_basis),
        rtol=3e-12,atol=3e-12,
    )


def test_quad_q2_q1_composite_divergence_matches_skfem():
    linear=skfemntv.MeshQuad.init_tensor(
        np.linspace(0.,1.,4),np.linspace(0.,1.,3)
    )
    mesh=skfemntv.MeshQuad2.from_mesh(linear)
    basis=skfemntv.Basis(
        mesh,
        skfemntv.ElementVector(skfemntv.ElementQuad2())*skfemntv.ElementQuad1(),
        intorder=4,
    )

    @skfemntv.BilinearForm
    def stokes(u,p,v,q,w):
        return ddot(grad(u),grad(v))-p*div(v)-q*div(u)

    actual=skfemntv.asm(stokes,basis)
    reference_mesh=skfem.MeshQuad2.from_mesh(
        skfem.MeshQuad(linear.p,linear.t)
    )
    reference_basis=skfem.Basis(
        reference_mesh,
        skfem.ElementVector(skfem.ElementQuad2())
        *skfem.ElementQuad1(),
        intorder=4,
    )

    @skfem.BilinearForm
    def reference(u,p,v,q,w):
        return (
            reference_ddot(reference_grad(u),reference_grad(v))
            -p*reference_div(v)-q*reference_div(u)
        )

    expected=skfem.asm(reference,reference_basis)
    permutation=_composite_permutation(basis,reference_basis)
    np.testing.assert_allclose(
        actual.toarray(),expected[permutation][:,permutation].toarray(),
        rtol=5e-12,atol=5e-12,
    )


def test_quad2_reproduces_quadratic_field_and_curved_edge_geometry():
    mesh=skfemntv.MeshQuad2()
    basis=skfemntv.Basis(
        mesh,skfemntv.ElementVector(skfemntv.ElementQuad2(),dim=1),intorder=4
    )
    coefficients=(
        basis.doflocs[0]**2
        +basis.doflocs[0]*basis.doflocs[1]
        -2.*basis.doflocs[1]**2
    )
    field=basis.interpolate(coefficients)
    x=np.moveaxis(basis.global_coordinates,-1,0)
    expected=x[0]**2+x[0]*x[1]-2.*x[1]**2
    np.testing.assert_allclose(field.value,expected,rtol=2e-13,atol=2e-13)

    mesh.p[:,5]=[1.15,.5]
    facet=skfemntv.FacetBasis(
        mesh,skfemntv.ElementVector(skfemntv.ElementQuad2()),intorder=4
    )
    lengths=facet.dx.sum(axis=1)
    assert lengths.max()>1.
    np.testing.assert_allclose(
        np.linalg.norm(facet.normals,axis=2),1.,rtol=2e-14,atol=2e-14
    )

    @skfemntv.Functional
    def geometry_measure(w):
        return 1.+w.x[0]+.3*w.n[0]**2

    reference_mesh=skfem.MeshQuad2()
    right_midpoint=np.flatnonzero(np.all(
        np.isclose(reference_mesh.p,np.array([[1.],[.5]])),axis=0
    ))[0]
    reference_mesh.p[:,right_midpoint]=[1.15,.5]
    reference_facet=skfem.FacetBasis(
        reference_mesh,skfem.ElementVector(skfem.ElementQuad2()),
        intorder=4,
    )

    @skfem.Functional
    def reference_measure(w):
        return 1.+w.x[0]+.3*w.n[0]**2

    np.testing.assert_allclose(
        skfemntv.asm(geometry_measure,facet),
        skfem.asm(reference_measure,reference_facet),
        rtol=3e-12,atol=3e-12,
    )
