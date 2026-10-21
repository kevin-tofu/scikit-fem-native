import numpy as np
import pytest
import skfem
from skfem.helpers import ddot,det,eye,grad,inv,transpose

import skfemntv


def reference_forms(mu,lame_lambda):
    @skfem.LinearForm
    def residual(v,w):
        field=w["displacement"]
        deformation=eye(1.+0.*field.grad[0,0],3)+field.grad
        inverse_transpose=transpose(inv(deformation))
        stress=(
            mu*(deformation-inverse_transpose)
            +lame_lambda*np.log(det(deformation))*inverse_transpose
        )
        return ddot(stress,grad(v))

    @skfem.BilinearForm
    def tangent(increment,v,w):
        field=w["displacement"]
        deformation=eye(1.+0.*field.grad[0,0],3)+field.grad
        inverse_transpose=transpose(inv(deformation))
        log_j=np.log(det(deformation))
        increment_gradient=grad(increment)
        test_gradient=grad(v)
        return (
            mu*ddot(increment_gradient,test_gradient)
            +(mu-lame_lambda*log_j)*np.einsum(
                "il...,kj...,kl...,ij...",
                inverse_transpose,inverse_transpose,
                increment_gradient,test_gradient,
            )
            +lame_lambda*np.einsum(
                "ij...,kl...,kl...,ij...",
                inverse_transpose,inverse_transpose,
                increment_gradient,test_gradient,
            )
        )

    return residual,tangent


def native_to_reference_dofs(native_basis,reference_basis):
    permutation=np.empty(native_basis.N,dtype=np.int64)
    native_components=native_basis.split_indices()
    reference_components=reference_basis.split_indices()
    for native_ids,reference_ids in zip(
        native_components,reference_components
    ):
        lookup={
            tuple(np.round(reference_basis.doflocs[:,dof],13)):int(dof)
            for dof in reference_ids
        }
        for dof in native_ids:
            key=tuple(np.round(native_basis.doflocs[:,dof],13))
            permutation[dof]=lookup[key]
    return permutation


@pytest.mark.parametrize("family",["tet10","hex8","hex20","hex27"])
def test_nonlinear_residual_and_tangent_match_scikit_fem(family):
    if family=="tet10":
        linear=skfemntv.MeshTet()
        mesh=skfemntv.MeshTet2.from_mesh(linear)
        reference_mesh=skfem.MeshTet(linear.p,linear.t)
        reference_mesh=skfem.MeshTet2.from_mesh(reference_mesh)
        native_element=skfemntv.ElementTetP2()
        reference_element=skfem.ElementTetP2()
    elif family=="hex8":
        mesh=skfemntv.MeshHex()
        reference_mesh=skfem.MeshHex(mesh.p,mesh.t)
        native_element=skfemntv.ElementHex1()
        reference_element=skfem.ElementHex1()
    elif family=="hex20":
        linear=skfemntv.MeshHex()
        mesh=skfemntv.MeshHex20.from_mesh(linear)
        reference_linear=skfem.MeshHex(linear.p,linear.t)
        reference_mesh=skfem.MeshHex2.from_mesh(reference_linear)
        native_element=skfemntv.ElementHex20()
        reference_element=skfem.ElementHexS2()
    else:
        linear=skfemntv.MeshHex()
        mesh=skfemntv.MeshHex2.from_mesh(linear)
        reference_linear=skfem.MeshHex(linear.p,linear.t)
        reference_mesh=skfem.MeshHex2.from_mesh(reference_linear)
        native_element=skfemntv.ElementHex2()
        reference_element=skfem.ElementHex2()
    native_basis=skfemntv.Basis(
        mesh,skfemntv.ElementVector(native_element,dim=3),intorder=5
    )
    reference_basis=skfem.Basis(
        reference_mesh,
        skfem.ElementVector(reference_element,dim=3),
        quadrature=native_basis.quadrature,
    )
    rng=np.random.default_rng(401)
    displacement=.015*rng.normal(size=native_basis.N)
    mu=2.4
    lame_lambda=3.1
    native=skfemntv.NeoHookeanAssembler(native_basis,mu,lame_lambda)
    actual=native.evaluate(displacement)
    reference_residual,reference_tangent=reference_forms(mu,lame_lambda)
    permutation=native_to_reference_dofs(native_basis,reference_basis)
    reference_displacement=np.empty_like(displacement)
    reference_displacement[permutation]=displacement
    reference_field=reference_basis.interpolate(reference_displacement)
    expected_residual=reference_residual.assemble(
        reference_basis,displacement=reference_field
    )
    expected_tangent=reference_tangent.assemble(
        reference_basis,displacement=reference_field
    )
    np.testing.assert_allclose(
        actual.residual,expected_residual[permutation],
        rtol=4e-11,atol=4e-11,
    )
    np.testing.assert_allclose(
        actual.tangent.toarray(),
        expected_tangent.toarray()[permutation][:,permutation],
        rtol=4e-11,atol=4e-11,
    )
    direction=rng.normal(size=native_basis.N)
    step=1e-7
    residual_difference=(
        native.evaluate(displacement+step*direction).residual
        -native.evaluate(displacement-step*direction).residual
    )/(2.*step)
    np.testing.assert_allclose(
        actual.tangent@direction,residual_difference,
        rtol=3e-7,atol=3e-8,
    )
