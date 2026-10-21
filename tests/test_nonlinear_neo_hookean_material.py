import numpy as np
import pytest

import skfemntv


def make_basis(family="tet4"):
    if family=="tet4":
        mesh=skfemntv.MeshTet()
        element=skfemntv.ElementTetP1()
    elif family=="tet10":
        mesh=skfemntv.MeshTet2.from_mesh(skfemntv.MeshTet())
        element=skfemntv.ElementTetP2()
    else:
        mesh=skfemntv.MeshHex2.from_mesh(skfemntv.MeshHex())
        element=skfemntv.ElementHex2()
    return skfemntv.Basis(
        mesh,skfemntv.ElementVector(element,dim=3),intorder=5)


def deformation_gradients(basis,model,displacement):
    result=np.broadcast_to(np.eye(3),(basis.nelems,basis.dx.shape[1],3,3)).copy()
    for e in range(basis.nelems):
        for q in range(basis.dx.shape[1]):
            for a in range(model._dofs.shape[1]):
                for i in range(3):
                    for j in range(3):
                        result[e,q,i,j]+=displacement[model._dofs[e,a,i]]*model._gradients[e,q,a,j]
    return result


def test_reference_configuration_has_zero_energy_and_linear_elastic_tangent():
    material=skfemntv.NeoHookeanMaterial(mu=2.4,lambda_=3.1)
    result=material.evaluate(np.eye(3))
    np.testing.assert_allclose(result.stress,0.,atol=1e-14)
    np.testing.assert_allclose(result.cauchy_stress,0.,atol=1e-14)
    np.testing.assert_allclose(result.equivalent_stress,0.,atol=1e-14)
    assert float(result.energy)==pytest.approx(0.,abs=1e-14)
    lamé=material.lambda_
    mu=material.mu
    expected=np.empty((3,3,3,3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    expected[i,j,k,l]=(
                        mu*(i==k and j==l)+mu*(i==l and j==k)
                        +lamé*(i==j and k==l))
    np.testing.assert_allclose(result.tangent,expected,atol=1e-14)


def test_material_point_tangent_and_energy_match_finite_differences():
    material=skfemntv.NeoHookeanMaterial(mu=2.4,lambda_=3.1)
    rng=np.random.default_rng(52)
    f=np.eye(3)+.08*rng.normal(size=(2,3,3))
    direction=rng.normal(size=f.shape)
    result=material.evaluate(f)
    j=np.linalg.det(f)
    cauchy=np.einsum("...ij,...kj->...ik",result.stress,f)/j[...,None,None]
    np.testing.assert_allclose(result.cauchy_stress,cauchy,rtol=2e-13,atol=2e-13)
    np.testing.assert_allclose(
        result.cauchy_stress,result.cauchy_stress.swapaxes(-1,-2),
        rtol=2e-12,atol=2e-12)
    dev=result.cauchy_stress-np.trace(
        result.cauchy_stress,axis1=-2,axis2=-1)[...,None,None]*np.eye(3)/3.
    equivalent=np.sqrt(1.5*np.einsum("...ij,...ij->...",dev,dev))
    np.testing.assert_allclose(result.equivalent_stress,equivalent,rtol=2e-13,atol=2e-13)
    step=1e-7
    fd=(material.evaluate(f+step*direction).stress-
        material.evaluate(f-step*direction).stress)/(2.*step)
    action=np.einsum("...ijkl,...kl->...ij",result.tangent,direction)
    np.testing.assert_allclose(action,fd,rtol=2e-7,atol=2e-8)
    energy_fd=(material.evaluate(f+step*direction).energy-
               material.evaluate(f-step*direction).energy)/(2.*step)
    energy_action=np.einsum("...ij,...ij->...",result.stress,direction)
    np.testing.assert_allclose(energy_action,energy_fd,rtol=2e-7,atol=2e-9)


@pytest.mark.parametrize("family",["tet4","tet10","hex27"])
def test_material_point_response_integrates_to_assembler_residual_tangent_and_energy(family):
    basis=make_basis(family)
    material=skfemntv.NeoHookeanMaterial(mu=2.4,lambda_=3.1)
    assembler=skfemntv.NeoHookeanAssembler(basis,mu=material.mu,lambda_=material.lambda_)
    displacement=.03*np.random.default_rng(72).normal(size=basis.N)
    f=deformation_gradients(basis,assembler,displacement)
    point=material.evaluate(f)
    assembled=assembler.evaluate(displacement)
    np.testing.assert_allclose(assembled.deformation_gradient,f,rtol=2e-14,atol=2e-14)
    np.testing.assert_allclose(assembled.stress,point.stress,rtol=2e-13,atol=2e-13)
    np.testing.assert_allclose(assembled.cauchy_stress,point.cauchy_stress,rtol=2e-13,atol=2e-13)
    np.testing.assert_allclose(assembled.equivalent_stress,point.equivalent_stress,rtol=2e-13,atol=2e-13)
    local_residual=np.einsum(
        "eqij,eqaj,eq->eai",point.stress,assembler._gradients,assembler._weights)
    local_tangent=np.einsum(
        "eqaj,eqijkl,eqbl,eq->eaibk",
        assembler._gradients,point.tangent,assembler._gradients,assembler._weights)
    dofs=assembler._dofs.reshape(basis.nelems,-1)
    residual=np.zeros(basis.N)
    np.add.at(residual,dofs.reshape(-1),local_residual.reshape(-1))
    local_dofs=dofs.shape[1]
    rows=np.broadcast_to(
        dofs[:,:,None],(basis.nelems,local_dofs,local_dofs)).reshape(-1)
    cols=np.broadcast_to(
        dofs[:,None,:],(basis.nelems,local_dofs,local_dofs)).reshape(-1)
    from scipy.sparse import coo_matrix
    tangent=coo_matrix(
        (local_tangent.reshape(-1),(rows,cols)),shape=(basis.N,basis.N)).tocsr()
    np.testing.assert_allclose(assembled.residual,residual,rtol=3e-13,atol=3e-13)
    np.testing.assert_allclose(assembled.tangent.toarray(),tangent.toarray(),rtol=3e-13,atol=3e-13)
    assert assembled.energy==pytest.approx(
        np.sum(point.energy*assembler._weights),rel=3e-13,abs=3e-13)
    direction=np.random.default_rng(73).normal(size=basis.N)
    step=1e-7
    residual_difference=(
        assembler.evaluate(displacement+step*direction).residual
        -assembler.evaluate(displacement-step*direction).residual
    )/(2.*step)
    np.testing.assert_allclose(
        assembled.tangent@direction,residual_difference,rtol=3e-7,atol=3e-8)


def test_nonpositive_jacobian_and_invalid_parameters_are_rejected():
    material=skfemntv.NeoHookeanMaterial(mu=2.4,lambda_=3.1)
    with pytest.raises(ValueError,match=r"nonpositive det\(F\)"):
        material.evaluate(np.diag([1.,1.,-1.]))
    with pytest.raises(ValueError,match="lambda_"):
        skfemntv.NeoHookeanMaterial(mu=1.,lambda_=-1.)
