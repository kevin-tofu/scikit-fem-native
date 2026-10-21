import numpy as np
import pytest
from scipy.sparse import coo_matrix

import skfemntv
from skfemntv._skfn import evaluate_j2_plasticity,evaluate_neo_hookean


def make_shared_tet_basis(intorder=2):
    points=np.array([
        [0.,1.,0.,0.,0.],
        [0.,0.,1.,0.,0.],
        [0.,0.,0.,1.,-1.],
    ])
    cells=np.array([[0,1],[1,2],[2,3],[3,4]],dtype=np.int64)
    mesh=skfemntv.MeshTet(points,cells)
    return skfemntv.Basis(
        mesh,skfemntv.ElementVector(skfemntv.ElementTetP1(),dim=3),
        intorder=intorder)


def affine_x(basis,amount):
    u=np.zeros(basis.N)
    x_dofs=basis.split_indices()[0]
    u[x_dofs]=amount*basis.doflocs[0,x_dofs]
    return u


def scatter_local(basis,local_residual,local_tangent):
    dofs=basis.element_dofs.T
    residual=np.zeros(basis.N)
    np.add.at(residual,dofs.reshape(-1),local_residual.reshape(-1))
    rows=np.broadcast_to(dofs[:,:,None],local_tangent.shape).reshape(-1)
    cols=np.broadcast_to(dofs[:,None,:],local_tangent.shape).reshape(-1)
    tangent=coo_matrix(
        (local_tangent.reshape(-1),(rows,cols)),shape=(basis.N,basis.N)
    ).tocsr()
    return residual,tangent


def test_shared_dof_j2_global_assembly_sums_element_contributions():
    basis=make_shared_tet_basis()
    model=skfemntv.J2PlasticityAssembler(
        basis,young_modulus=1000.,poisson_ratio=.25,
        yield_stress=2.,hardening_modulus=20.)
    displacement=affine_x(basis,.02)
    committed=model.initial_state()
    result=model.evaluate(displacement,committed)
    local=evaluate_j2_plasticity(
        model._dofs,model._gradients,model._weights,displacement,committed,
        model.lambda_,model.mu,model.yield_stress,model.hardening_modulus)
    expected_residual,expected_tangent=scatter_local(basis,local[0],local[1])
    np.testing.assert_allclose(result.residual,expected_residual,rtol=2e-13,atol=2e-13)
    np.testing.assert_allclose(
        result.tangent.toarray(),expected_tangent.toarray(),rtol=2e-13,atol=2e-13)
    assert result.trial_state.shape[:2]==(basis.nelems,basis.dx.shape[1])
    assert result.plastic_points.all()
    direction=np.random.default_rng(12).normal(size=basis.N)
    step=1e-8
    finite_difference=(
        model.evaluate(displacement+step*direction,committed).residual
        -model.evaluate(displacement-step*direction,committed).residual
    )/(2.*step)
    np.testing.assert_allclose(
        result.tangent@direction,finite_difference,rtol=3e-7,atol=3e-8)
    # Point history is indexed by element: changing one cell's history affects
    # only that cell's returned trial history.
    changed=committed.copy()
    changed[0,:,6]=.01
    changed_result=model.evaluate(displacement,changed)
    assert not np.allclose(changed_result.trial_state[0],changed_result.trial_state[1])
    np.testing.assert_array_equal(committed,0.)


def test_shared_dof_neo_hookean_global_assembly_sums_element_contributions():
    basis=make_shared_tet_basis()
    model=skfemntv.NeoHookeanAssembler(basis,mu=2.4,lambda_=3.1)
    displacement=affine_x(basis,.012)
    result=model.evaluate(displacement)
    local=evaluate_neo_hookean(
        model._dofs,model._gradients,model._weights,
        displacement,model.mu,model.lambda_)
    expected_residual,expected_tangent=scatter_local(basis,local[0],local[1])
    np.testing.assert_allclose(result.residual,expected_residual,rtol=2e-13,atol=2e-13)
    np.testing.assert_allclose(
        result.tangent.toarray(),expected_tangent.toarray(),rtol=2e-13,atol=2e-13)
    direction=np.random.default_rng(33).normal(size=basis.N)
    step=1e-7
    fd=(model.evaluate(displacement+step*direction).residual-
        model.evaluate(displacement-step*direction).residual)/(2.*step)
    np.testing.assert_allclose(result.tangent@direction,fd,rtol=3e-7,atol=3e-8)


@pytest.mark.parametrize("material",["j2","neo_hookean"])
def test_nonlinear_assembler_reuses_csr_graph_across_evaluations(material):
    basis=make_shared_tet_basis()
    if material=="j2":
        model=skfemntv.J2PlasticityAssembler(
            basis,young_modulus=1000.,poisson_ratio=.25,
            yield_stress=2.,hardening_modulus=20.)
        state=model.initial_state()
        evaluate=lambda u:model.evaluate(u,state)
    else:
        model=skfemntv.NeoHookeanAssembler(basis,mu=2.4,lambda_=3.1)
        evaluate=model.evaluate

    graph=model._tangent_graph
    first=evaluate(affine_x(basis,.008)).tangent
    first_data=first.data.copy()
    second=evaluate(affine_x(basis,.013)).tangent
    assert model._tangent_graph is graph
    np.testing.assert_array_equal(first.indptr,second.indptr)
    np.testing.assert_array_equal(first.indices,second.indices)
    np.testing.assert_array_equal(first.data,first_data)


@pytest.mark.parametrize("material",["j2","neo_hookean"])
def test_native_multielement_kernel_is_stable_across_thread_limits(material):
    one=make_shared_tet_basis(intorder=1)
    # Repeated cells force the native element loop past its parallel threshold
    # while accumulating many local blocks onto the same global DOFs.
    cells=np.repeat(one.mesh.t,512,axis=1)
    mesh=skfemntv.MeshTet(one.mesh.p,cells)
    assert cells.shape[1]>=1024
    basis=skfemntv.Basis(
        mesh,skfemntv.ElementVector(skfemntv.ElementTetP1(),dim=3),intorder=1)
    displacement=affine_x(basis,.006)
    if material=="j2":
        model=skfemntv.J2PlasticityAssembler(
            basis,young_modulus=1000.,poisson_ratio=.25,
            yield_stress=2.,hardening_modulus=20.)
        evaluate=lambda:model.evaluate(displacement,model.initial_state())
    else:
        model=skfemntv.NeoHookeanAssembler(basis,mu=2.4,lambda_=3.1)
        evaluate=lambda:model.evaluate(displacement)
    with skfemntv.thread_limit(1):
        serial=evaluate()
    with skfemntv.thread_limit(4):
        parallel=evaluate()
    np.testing.assert_allclose(parallel.residual,serial.residual,rtol=1e-13,atol=1e-13)
    np.testing.assert_allclose(
        parallel.tangent.toarray(),serial.tangent.toarray(),rtol=1e-13,atol=1e-13)
