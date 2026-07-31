import numpy as np

import skfn


def material():
    return skfn.J2Plasticity(young_modulus=210.,poisson_ratio=.3,
                            yield_stress=.25,hardening_modulus=1.5)


def test_j2_elastic_response_matches_isotropic_elasticity():
    model=material();state=model.initial_state(1)
    strain=np.array([[1e-5,-2e-5,3e-5,4e-6,-5e-6,6e-6]])
    result=model.evaluate(strain,state)
    mu=210./(2.*1.3);lmbda=210.*.3/(1.3*.4)
    trace=strain[0,:3].sum()
    expected=np.r_[lmbda*trace+2.*mu*strain[0,:3],
                   2.*mu*strain[0,3:]]
    np.testing.assert_allclose(result.stress[0],expected,rtol=2e-13,atol=2e-13)
    np.testing.assert_array_equal(result.trial_state.plastic_strain,0.)


def test_j2_plastic_update_and_trial_state_do_not_mutate_committed_state():
    model=material();state=model.initial_state(2)
    strain=np.array([[.004,-.002,-.002,0.,0.,0.],
                     [.003,-.0015,-.0015,.001,0.,0.]])
    result=model.evaluate(strain,state,num_threads=4)
    assert np.all(result.trial_state.equivalent_plastic_strain>0.)
    np.testing.assert_array_equal(state.plastic_strain,0.)
    np.testing.assert_array_equal(state.equivalent_plastic_strain,0.)


def test_j2_consistent_tangent_matches_stress_difference():
    model=material();state=model.initial_state(1)
    strain=np.array([[.004,-.002,-.002,.0007,0.,0.]])
    direction=np.array([[.3,-.2,-.1,.4,-.15,.2]])
    result=model.evaluate(strain,state)
    step=2e-7
    plus=model.evaluate(strain+step*direction,state).stress
    minus=model.evaluate(strain-step*direction,state).stress
    action=np.einsum("nij,nj->ni",result.tangent,direction)
    np.testing.assert_allclose(
        action,(plus-minus)/(2.*step),rtol=2e-5,atol=2e-7
    )


def test_j2_global_assembler_tangent_and_trial_state():
    axis=np.linspace(0.,1.,3)
    mesh=skfn.MeshTet.init_tensor(axis,axis,axis)
    basis=skfn.Basis(
        mesh,skfn.ElementVector(skfn.ElementTetP1(),dim=3),intorder=2
    )
    assembler=skfn.J2Assembler(basis,material())
    state=assembler.initial_state()
    coordinates=basis.doflocs
    u=np.zeros(basis.N)
    u[0::3]=.004*coordinates[0,0::3]
    u[1::3]=-.002*coordinates[1,1::3]
    u[2::3]=-.002*coordinates[2,2::3]
    direction=np.random.default_rng(4).normal(size=basis.N)
    result=assembler.assemble(u,state,num_threads=4)
    tangent=result.tangent.copy()
    residual=result.residual.copy()
    step=2e-7
    plus=assembler.assemble(
        np.ascontiguousarray(u+step*direction),state,
        mode="residual",num_threads=4,
    ).residual.copy()
    minus=assembler.assemble(
        np.ascontiguousarray(u-step*direction),state,
        mode="residual",num_threads=4,
    ).residual.copy()
    np.testing.assert_allclose(
        tangent@direction,(plus-minus)/(2.*step),rtol=3e-5,atol=3e-7
    )
    assert np.linalg.norm(residual)>0.
    assert np.all(result.trial_state.equivalent_plastic_strain>0.)
    np.testing.assert_array_equal(state.equivalent_plastic_strain,0.)


def test_j2_hex8_global_assembler_parallel_matches_serial():
    axis=np.linspace(0.,1.,3)
    mesh=skfn.MeshHex.init_tensor(axis,axis,axis)
    basis=skfn.Basis(
        mesh,skfn.ElementVector(skfn.ElementHex1(),dim=3),intorder=2
    )
    assembler=skfn.J2Assembler(basis,material())
    state=assembler.initial_state()
    u=np.zeros(basis.N)
    u[0::3]=.003*basis.doflocs[0,0::3]
    u[1::3]=-.0015*basis.doflocs[1,1::3]
    u[2::3]=-.0015*basis.doflocs[2,2::3]
    serial=assembler.assemble(u,state,num_threads=1)
    residual=serial.residual.copy();tangent=serial.tangent.copy()
    parallel=assembler.assemble(u,state,num_threads=4)
    np.testing.assert_allclose(parallel.residual,residual,rtol=2e-14,atol=2e-14)
    np.testing.assert_allclose(
        parallel.tangent.toarray(),tangent.toarray(),rtol=2e-14,atol=2e-14
    )
    assert parallel.trial_state.plastic_strain.shape==(
        mesh.nelements*basis.dx.shape[1],6
    )


def test_common_material_assembler_entry_point_preserves_j2_api():
    mesh=skfn.MeshTet()
    basis=skfn.Basis(
        mesh,skfn.ElementVector(skfn.ElementTetP1(),dim=3),intorder=2
    )
    common=skfn.MaterialAssembler(basis,material())
    legacy=skfn.J2Assembler(basis,material())
    assert type(common) is type(legacy)
    assert common.material.state_size==7
    assert common.material.state_fields==(
        "plastic_strain","equivalent_plastic_strain"
    )
    state=common.initial_state()
    result=common.assemble(np.zeros(basis.N),state)
    np.testing.assert_array_equal(result.residual,0.)


def test_material_assembler_rejects_non_native_material_before_setup():
    mesh=skfn.MeshTet()
    basis=skfn.Basis(
        mesh,skfn.ElementVector(skfn.ElementTetP1(),dim=3),intorder=2
    )
    with np.testing.assert_raises_regex(TypeError,"unsupported material kernel"):
        skfn.MaterialAssembler(basis,object())


def test_j2_named_fields_share_contiguous_generic_state_storage():
    state=material().initial_state(5)
    assert state.storage.shape==(5,7)
    assert state.storage.flags.c_contiguous
    assert np.shares_memory(state.plastic_strain,state.storage)
    assert np.shares_memory(
        state.equivalent_plastic_strain,state.storage
    )
    state.plastic_strain[2,3]=.4
    state.equivalent_plastic_strain[2]=.7
    assert state.storage[2,3]==.4
    assert state.storage[2,6]==.7


def test_zero_state_material_uses_same_generic_assembler():
    axis=np.linspace(0.,1.,3)
    mesh=skfn.MeshHex.init_tensor(axis,axis,axis)
    basis=skfn.Basis(
        mesh,skfn.ElementVector(skfn.ElementHex1(),dim=3),intorder=2
    )
    kernel=skfn.LinearElasticity(120.,.27)
    common=skfn.MaterialAssembler(basis,kernel)
    established=skfn.NativeAssembler.from_basis(basis,kernel)
    state=common.initial_state()
    assert state.storage.shape==(common.state_count,0)
    assert state.storage.flags.c_contiguous
    displacement=np.ascontiguousarray(
        np.random.default_rng(12).normal(scale=1e-3,size=basis.N)
    )
    actual=common.assemble(displacement,state,num_threads=4)
    residual=actual.residual.copy();tangent=actual.tangent.copy()
    expected=established.assemble(displacement,num_threads=1)
    np.testing.assert_allclose(
        residual,expected.residual,rtol=3e-13,atol=3e-13
    )
    np.testing.assert_allclose(
        tangent.toarray(),expected.tangent.toarray(),
        rtol=3e-13,atol=3e-13,
    )
    assert actual.trial_state.storage.shape==(common.state_count,0)
