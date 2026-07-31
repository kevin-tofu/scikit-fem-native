from pathlib import Path
import sys

import numpy as np
import skfem

import skfn


sys.path.insert(0,str(Path(__file__).parents[1]/"benchmarks"))
from skfem_j2 import forms as reference_forms


def material():
    return skfn.StandardLinearSolid(
        equilibrium_modulus=100.,branch_modulus=60.,poisson_ratio=.25,
        relaxation_time=2.,time_step=.2,
    )


def elastic_tangent(young,poisson):
    mu=young/(2.*(1.+poisson))
    lmbda=young*poisson/((1.+poisson)*(1.-2.*poisson))
    tangent=np.zeros((6,6))
    tangent[:3,:3]=lmbda
    tangent[0,0]+=2.*mu;tangent[1,1]+=2.*mu;tangent[2,2]+=2.*mu
    tangent[3,3]=tangent[4,4]=tangent[5,5]=2.*mu
    return tangent


def test_standard_linear_solid_backward_euler_update():
    model=material();state=model.initial_state(2)
    strain=np.array([
        [.004,-.001,-.002,.0005,0.,0.],
        [.002,.001,-.001,0.,.0003,0.],
    ])
    result=model.evaluate(strain,state,num_threads=2)
    factor=1./(1.+model.time_step/model.relaxation_time)
    np.testing.assert_allclose(
        result.trial_state.viscous_strain,(1.-factor)*strain,
        rtol=2e-15,atol=2e-15,
    )
    expected_tangent=elastic_tangent(
        model.equilibrium_modulus+factor*model.branch_modulus,
        model.poisson_ratio,
    )
    expected_stress=np.einsum("ij,nj->ni",expected_tangent,strain)
    np.testing.assert_allclose(result.stress,expected_stress,rtol=2e-15,atol=2e-15)
    np.testing.assert_allclose(
        result.tangent,np.broadcast_to(expected_tangent,result.tangent.shape),
        rtol=2e-15,atol=2e-15,
    )
    np.testing.assert_array_equal(state.storage,0.)


def test_standard_linear_solid_relaxes_under_held_strain():
    model=material();state=model.initial_state(1)
    strain=np.array([[.003,-.0015,-.0015,0.,0.,0.]])
    equilibrium=elastic_tangent(
        model.equilibrium_modulus,model.poisson_ratio
    )@strain[0]
    excess=[]
    for _ in range(8):
        result=model.evaluate(strain,state)
        excess.append(result.stress[0,0]-equilibrium[0])
        state=result.trial_state
    factor=1./(1.+model.time_step/model.relaxation_time)
    np.testing.assert_allclose(
        np.array(excess[1:])/np.array(excess[:-1]),factor,
        rtol=3e-14,atol=3e-14,
    )


def test_standard_linear_solid_consistent_tangent_matches_difference():
    model=material()
    state=skfn.StandardLinearSolidState(
        np.array([[.001,-.0004,-.0006,.0002,0.,0.]])
    )
    strain=np.array([[.003,-.001,-.002,.0005,.0002,0.]])
    direction=np.array([[.2,-.1,-.1,.3,-.2,.15]])
    result=model.evaluate(strain,state)
    step=1e-7
    plus=model.evaluate(strain+step*direction,state).stress
    minus=model.evaluate(strain-step*direction,state).stress
    np.testing.assert_allclose(
        np.einsum("nij,nj->ni",result.tangent,direction),
        (plus-minus)/(2.*step),rtol=2e-11,atol=2e-11,
    )


def test_standard_linear_solid_tet_assembly_matches_scikit_fem():
    axis=np.linspace(0.,1.,3)
    mesh=skfn.MeshTet.init_tensor(axis,axis,axis)
    basis=skfn.Basis(
        mesh,skfn.ElementVector(skfn.ElementTetP1(),dim=3),intorder=2
    )
    reference_basis=skfem.Basis(
        skfem.MeshTet(mesh.p,mesh.t),skfem.ElementVector(skfem.ElementTetP1()),
        quadrature=(basis.X,basis.W),
    )
    model=material();assembler=skfn.MaterialAssembler(basis,model)
    state=assembler.initial_state()
    coordinates=basis.doflocs
    u=np.zeros(basis.N)
    u[0::3]=.003*coordinates[0,0::3]
    u[1::3]=-.001*coordinates[1,1::3]
    u[2::3]=-.002*coordinates[2,2::3]
    actual=assembler.assemble(u,state,num_threads=4)
    residual=actual.residual.copy();tangent=actual.tangent.copy()

    field=reference_basis.interpolate(u)
    strain=np.stack((
        field.grad[0,0],field.grad[1,1],field.grad[2,2],
        .5*(field.grad[0,1]+field.grad[1,0]),
        .5*(field.grad[1,2]+field.grad[2,1]),
        .5*(field.grad[0,2]+field.grad[2,0]),
    ))
    factor=1./(1.+model.time_step/model.relaxation_time)
    constitutive=elastic_tangent(
        model.equilibrium_modulus+factor*model.branch_modulus,
        model.poisson_ratio,
    )
    stress=np.einsum("ij,j...->i...",constitutive,strain)
    constitutive=np.broadcast_to(
        constitutive[:,:,None,None],(6,6)+strain.shape[1:]
    )
    residual_form,tangent_form=reference_forms()
    expected_residual=residual_form.assemble(reference_basis,stress=stress)
    expected_tangent=tangent_form.assemble(
        reference_basis,constitutive=constitutive
    )
    np.testing.assert_allclose(residual,expected_residual,rtol=3e-12,atol=3e-12)
    np.testing.assert_allclose(
        tangent.toarray(),expected_tangent.toarray(),rtol=3e-12,atol=3e-12
    )
    np.testing.assert_allclose(
        actual.trial_state.viscous_strain,
        np.moveaxis((1.-factor)*strain,0,-1).reshape(-1,6),
        rtol=3e-12,atol=3e-12,
    )


def test_standard_linear_solid_hex_parallel_matches_serial():
    axis=np.linspace(0.,1.,4)
    mesh=skfn.MeshHex.init_tensor(axis,axis,axis)
    basis=skfn.Basis(
        mesh,skfn.ElementVector(skfn.ElementHex1(),dim=3),intorder=2
    )
    assembler=skfn.MaterialAssembler(basis,material())
    state=assembler.initial_state()
    assert state.storage.shape==(assembler.state_count,6)
    u=np.ascontiguousarray(
        np.random.default_rng(7).normal(scale=1e-3,size=basis.N)
    )
    serial=assembler.assemble(u,state,num_threads=1)
    residual=serial.residual.copy();tangent=serial.tangent.copy()
    trial=serial.trial_state.storage.copy()
    parallel=assembler.assemble(u,state,num_threads=4)
    np.testing.assert_allclose(parallel.residual,residual,rtol=2e-14,atol=2e-14)
    np.testing.assert_allclose(
        parallel.tangent.data,tangent.data,rtol=2e-14,atol=2e-14
    )
    np.testing.assert_allclose(
        parallel.trial_state.storage,trial,rtol=2e-14,atol=2e-14
    )


def test_standard_linear_solid_time_step_override_without_rebuild():
    model=material();state=model.initial_state(1)
    strain=np.array([[.003,-.001,-.002,.0004,0.,0.]])
    dt=.05
    result=model.evaluate(strain,state,time_step=dt)
    factor=1./(1.+dt/model.relaxation_time)
    expected=elastic_tangent(
        model.equilibrium_modulus+factor*model.branch_modulus,
        model.poisson_ratio,
    )
    np.testing.assert_allclose(result.tangent[0],expected,rtol=2e-15,atol=2e-15)

    mesh=skfn.MeshTet();basis=skfn.Basis(
        mesh,skfn.ElementVector(skfn.ElementTetP1(),dim=3),intorder=2
    )
    assembler=skfn.MaterialAssembler(basis,model)
    native_identity=id(assembler._native)
    committed=assembler.initial_state();u=np.linspace(0.,.003,basis.N)
    small=assembler.assemble(u,committed,time_step=.05)
    small_tangent=small.tangent.copy()
    large=assembler.assemble(u,committed,time_step=.8)
    assert id(assembler._native)==native_identity
    assert not np.allclose(small_tangent.data,large.tangent.data)
    np.testing.assert_array_equal(committed.storage,0.)


def test_standard_linear_solid_rejected_step_then_cutback():
    model=material();state=model.initial_state(3)
    strain=np.tile(np.array([.003,-.001,-.002,0.,0.,0.]),(3,1))
    rejected=model.evaluate(strain,state,time_step=1.)
    assert np.linalg.norm(rejected.trial_state.storage)>0.
    cutback=model.evaluate(strain,state,time_step=.05)
    clean=model.evaluate(strain,model.initial_state(3),time_step=.05)
    np.testing.assert_array_equal(state.storage,0.)
    np.testing.assert_allclose(
        cutback.trial_state.storage,clean.trial_state.storage,rtol=0.,atol=0.
    )
    np.testing.assert_allclose(cutback.stress,clean.stress,rtol=0.,atol=0.)


def test_standard_linear_solid_rejects_invalid_time_step_override():
    model=material();state=model.initial_state(1);strain=np.zeros((1,6))
    for invalid in (0.,-1.,True):
        with np.testing.assert_raises_regex(ValueError,"time_step"):
            model.evaluate(strain,state,time_step=invalid)
