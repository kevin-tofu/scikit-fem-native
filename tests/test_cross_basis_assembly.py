import numpy as np
import pytest
import skfem
from skfem.helpers import div as reference_div
from skfem.helpers import dot as reference_dot
from skfem.helpers import grad as reference_grad

import skfn
from skfn.helpers import ddot,div,dot,grad


def _permutation(native,reference,components):
    lookup={}
    for dof in range(reference.N):
        key=tuple(np.round(reference.doflocs[:,dof],14))
        lookup.setdefault(key,[]).append(dof)
    result=np.empty(native.N,dtype=np.int64)
    for dof in range(native.N):
        key=tuple(np.round(native.doflocs[:,dof],14))
        result[dof]=lookup[key][dof%components]
    return result


def _spaces(kind):
    if kind=="tri":
        linear=skfn.MeshTri.init_tensor(
            np.linspace(0.,1.,4),np.linspace(0.,1.,3)
        )
        mesh=skfn.MeshTri2.from_mesh(linear)
        reference_mesh=skfem.MeshTri2.from_mesh(
            skfem.MeshTri(linear.p,linear.t)
        )
        return (
            mesh,reference_mesh,
            skfn.ElementTriP1(),skfn.ElementTriP2(),
            skfem.ElementTriP1(),skfem.ElementTriP2(),
        )
    if kind=="quad":
        linear=skfn.MeshQuad.init_tensor(
            np.linspace(0.,1.,4),np.linspace(0.,1.,3)
        )
        mesh=skfn.MeshQuad2.from_mesh(linear)
        reference_mesh=skfem.MeshQuad2.from_mesh(
            skfem.MeshQuad(linear.p,linear.t)
        )
        return (
            mesh,reference_mesh,
            skfn.ElementQuad1(),skfn.ElementQuad2(),
            skfem.ElementQuad1(),skfem.ElementQuad2(),
        )
    if kind=="tet":
        linear=skfn.MeshTet.init_tensor(
            [0.,1.],[0.,1.],[0.,1.]
        )
        mesh=skfn.MeshTet2.from_mesh(linear)
        reference_mesh=skfem.MeshTet2.from_mesh(
            skfem.MeshTet(linear.p,linear.t)
        )
        return (
            mesh,reference_mesh,
            skfn.ElementTetP1(),skfn.ElementTetP2(),
            skfem.ElementTetP1(),skfem.ElementTetP2(),
        )
    linear=skfn.MeshHex.init_tensor(
        [0.,.5,1.],[0.,1.],[0.,1.]
    )
    mesh=skfn.MeshHex2.from_mesh(linear)
    reference_mesh=skfem.MeshHex2.from_mesh(
        skfem.MeshHex(linear.p,linear.t)
    )
    return (
        mesh,reference_mesh,
        skfn.ElementHex1(),skfn.ElementHex2(),
        skfem.ElementHex1(),skfem.ElementHex2(),
    )


@pytest.mark.parametrize("kind",["tri","quad","tet","hex"])
def test_rectangular_value_gradient_assembly_matches_skfem(kind):
    (
        mesh,reference_mesh,low,high,reference_low,reference_high
    )=_spaces(kind)
    trial=skfn.Basis(
        mesh,skfn.ElementVector(low,dim=1),intorder=4
    )
    test=skfn.Basis(
        mesh,skfn.ElementVector(high,dim=1),intorder=4
    )
    reference_trial=skfem.Basis(
        reference_mesh,reference_low,intorder=4,
    )
    reference_test=skfem.Basis(
        reference_mesh,reference_high,intorder=4,
    )

    @skfn.BilinearForm
    def form(u,v,w):
        return (
            (1.+w.x[0])*dot(u,v)
            +.4*ddot(grad(u),grad(v))
        )

    @skfem.BilinearForm
    def reference(u,v,w):
        return (
            (1.+w.x[0])*u*v
            +.4*reference_dot(reference_grad(u),reference_grad(v))
        )

    actual=skfn.asm(form,trial,test)
    expected=skfem.asm(reference,reference_trial,reference_test)
    rows=_permutation(test,reference_test,1)
    columns=_permutation(trial,reference_trial,1)
    assert actual.shape==(test.N,trial.N)
    np.testing.assert_allclose(
        actual.toarray(),expected[rows][:,columns].toarray(),
        rtol=8e-12,atol=8e-12,
    )

@pytest.mark.parametrize("kind",["tri","quad","tet","hex"])
def test_rectangular_divergence_block_matches_skfem(kind):
    (
        mesh,reference_mesh,low,high,reference_low,reference_high
    )=_spaces(kind)
    dimension=mesh.dim()
    velocity=skfn.Basis(
        mesh,skfn.ElementVector(high),intorder=4
    )
    pressure=skfn.Basis(
        mesh,skfn.ElementVector(low,dim=1),intorder=4
    )
    reference_velocity=skfem.Basis(
        reference_mesh,skfem.ElementVector(reference_high),intorder=4
    )
    reference_pressure=skfem.Basis(
        reference_mesh,reference_low,intorder=4,
    )

    @skfn.BilinearForm
    def divergence(u,q,w):
        return (1.+.2*w.x[-1])*div(u)*q

    @skfem.BilinearForm
    def reference(u,q,w):
        return (1.+.2*w.x[-1])*reference_div(u)*q

    actual=skfn.asm(divergence,velocity,pressure)
    expected=skfem.asm(
        reference,reference_velocity,reference_pressure
    )
    rows=_permutation(pressure,reference_pressure,1)
    columns=_permutation(
        velocity,reference_velocity,dimension
    )
    np.testing.assert_allclose(
        actual.toarray(),expected[rows][:,columns].toarray(),
        rtol=8e-12,atol=8e-12,
    )

    @skfn.BilinearForm
    def transpose_divergence(p,v,w):
        return div(v)*p*(1.+.2*w.x[-1])

    @skfem.BilinearForm
    def reference_transpose(p,v,w):
        return reference_div(v)*p*(1.+.2*w.x[-1])

    actual_transpose=skfn.asm(
        transpose_divergence,pressure,velocity
    )
    expected_transpose=skfem.asm(
        reference_transpose,reference_pressure,reference_velocity
    )
    transpose_rows=_permutation(
        velocity,reference_velocity,dimension
    )
    transpose_columns=_permutation(
        pressure,reference_pressure,1
    )
    np.testing.assert_allclose(
        actual_transpose.toarray(),
        expected_transpose[transpose_rows][:,transpose_columns].toarray(),
        rtol=8e-12,atol=8e-12,
    )


@pytest.mark.parametrize("kind",["tri","quad","tet","hex"])
def test_rectangular_facet_assembly_matches_skfem(kind):
    (
        mesh,reference_mesh,low,high,reference_low,reference_high
    )=_spaces(kind)
    dimension=mesh.dim()
    trial=skfn.FacetBasis(
        mesh,skfn.ElementVector(low),intorder=4
    )
    test=skfn.FacetBasis(
        mesh,skfn.ElementVector(high),intorder=4
    )
    reference_trial=skfem.FacetBasis(
        reference_mesh,skfem.ElementVector(reference_low),intorder=4
    )
    reference_test=skfem.FacetBasis(
        reference_mesh,skfem.ElementVector(reference_high),intorder=4
    )

    @skfn.BilinearForm
    def boundary_mass(u,v,w):
        return (1.+w.x[0]+.1*w.n[-1]**2)*dot(u,v)

    @skfem.BilinearForm
    def reference(u,v,w):
        return (1.+w.x[0]+.1*w.n[-1]**2)*reference_dot(u,v)

    actual=skfn.asm(boundary_mass,trial,test)
    expected=skfem.asm(reference,reference_trial,reference_test)
    rows=_permutation(test,reference_test,dimension)
    columns=_permutation(trial,reference_trial,dimension)
    np.testing.assert_allclose(
        actual.toarray(),expected[rows][:,columns].toarray(),
        rtol=8e-12,atol=8e-12,
    )
