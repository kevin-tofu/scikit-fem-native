import numpy as np
import pytest
from scipy.sparse import coo_matrix

import skfemntv
from skfemntv.helpers import ddot,div,sym_grad


def make_model():
    points=np.array([[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]])
    mesh=skfemntv.MeshTet(points,np.array([[0],[1],[2],[3]],dtype=np.int64))
    basis=skfemntv.Basis(
        mesh,skfemntv.ElementVector(skfemntv.ElementTetP1(),dim=3),intorder=2)
    return basis,skfemntv.J2PlasticityTet4Assembler(
        basis,young_modulus=1000.,poisson_ratio=.25,yield_stress=2.,hardening_modulus=20.)


def affine_displacement(basis,exx):
    u=np.zeros(basis.N)
    for node in range(basis.mesh.p.shape[1]):
        for component in range(3):
            dof=basis.nodal_dofs[component,node]
            u[dof]=exx*basis.mesh.p[0,node] if component==0 else 0.
    return u


def test_elastic_trial_matches_linear_elastic_tangent_and_does_not_change_state():
    basis,model=make_model()
    state=model.initial_state()
    original=state.copy()
    result=model.evaluate(affine_displacement(basis,1e-5),state)
    assert not result.plastic_points.any()
    np.testing.assert_array_equal(state,original)
    np.testing.assert_allclose(result.trial_state,state)
    np.testing.assert_allclose(result.tangent.toarray(),result.tangent.toarray().T,atol=1e-12)

    @skfemntv.BilinearForm
    def elasticity(u,v,w):
        return 2.*model.mu*ddot(sym_grad(u),sym_grad(v)) \
            +model.lambda_*div(u)*div(v)

    np.testing.assert_allclose(
        result.tangent.toarray(),elasticity.assemble(basis).toarray(),
        rtol=2e-13,atol=2e-13)


def test_plastic_consistent_tangent_matches_directional_difference():
    basis,model=make_model()
    state=model.initial_state()
    u=affine_displacement(basis,.02)
    result=model.evaluate(u,state)
    assert result.plastic_points.all()
    direction=np.random.default_rng(81).normal(size=basis.N)
    h=1e-8
    fd=(model.evaluate(u+h*direction,state).residual-
        model.evaluate(u-h*direction,state).residual)/(2*h)
    np.testing.assert_allclose(result.tangent@direction,fd,rtol=2e-7,atol=2e-8)


def test_commit_unload_and_discarded_trial_state():
    basis,model=make_model()
    initial=model.initial_state()
    loaded=model.evaluate(affine_displacement(basis,.02),initial)
    committed=model.commit(loaded)
    assert np.all(committed[...,6]>0.)
    np.testing.assert_array_equal(initial,0.)
    unload=model.evaluate(affine_displacement(basis,.019),committed)
    assert not unload.plastic_points.any()
    np.testing.assert_array_equal(unload.trial_state,committed)
    # A discarded trial is not adopted and repeated evaluation is deterministic.
    trial=model.evaluate(affine_displacement(basis,.03),committed)
    again=model.evaluate(affine_displacement(basis,.03),committed)
    np.testing.assert_array_equal(trial.trial_state,again.trial_state)
    np.testing.assert_array_equal(committed,loaded.trial_state)


def test_invalid_material_and_state_shape_are_rejected():
    basis,_=make_model()
    with pytest.raises(ValueError,match="poisson_ratio"):
        skfemntv.J2PlasticityTet4Assembler(basis,1.,.5,1.)
    _,model=make_model()
    with pytest.raises(ValueError,match="committed_state"):
        model.evaluate(np.zeros(basis.N),np.zeros((1,1,6)))


def test_j2_material_point_api_has_consistent_tangent_and_transactional_history():
    material=skfemntv.J2PlasticityMaterial(
        young_modulus=1000.,poisson_ratio=.25,yield_stress=2.,hardening_modulus=20.)
    rng=np.random.default_rng(9)
    raw=rng.normal(size=(2,3,3))
    strain=.02*.5*(raw+raw.swapaxes(-1,-2))
    committed=material.initial_state(2)
    original=committed.copy()
    result=material.evaluate(strain,committed)
    assert result.plastic_points.all()
    np.testing.assert_array_equal(committed,original)
    np.testing.assert_allclose(
        result.equivalent_stress,
        material.yield_stress+material.hardening_modulus*result.trial_state[...,6],
        rtol=2e-12,atol=2e-12)
    direction=rng.normal(size=strain.shape)
    direction=.5*(direction+direction.swapaxes(-1,-2))
    h=1e-8
    fd=(material.evaluate(strain+h*direction,committed).stress-
        material.evaluate(strain-h*direction,committed).stress)/(2*h)
    tangent_action=np.einsum("...ijkl,...kl->...ij",result.tangent,direction)
    np.testing.assert_allclose(tangent_action,fd,rtol=3e-7,atol=3e-8)
    adopted=material.commit(result)
    np.testing.assert_array_equal(adopted,result.trial_state)
    assert not np.shares_memory(adopted,result.trial_state)


def test_j2_material_point_api_matches_assembler_quadrature_response():
    basis,assembler=make_model()
    displacement=affine_displacement(basis,.02)
    committed=assembler.initial_state()
    assembled=assembler.evaluate(displacement,committed)
    strains=np.zeros((basis.nelems,basis.dx.shape[1],3,3))
    for e in range(basis.nelems):
        for q in range(basis.dx.shape[1]):
            for i in range(3):
                for j in range(3):
                    for a in range(4):
                        di=assembler._dofs[e,a,i]
                        dj=assembler._dofs[e,a,j]
                        gij=assembler._gradients[e,q,a,j]
                        gji=assembler._gradients[e,q,a,i]
                        strains[e,q,i,j]+=.5*(displacement[di]*gij+displacement[dj]*gji)
    material=skfemntv.J2PlasticityMaterial(
        assembler.young_modulus,assembler.poisson_ratio,
        assembler.yield_stress,assembler.hardening_modulus)
    response=material.evaluate(strains,committed)
    np.testing.assert_allclose(assembled.stress,response.stress,rtol=1e-13,atol=1e-13)
    np.testing.assert_allclose(
        assembled.trial_state,response.trial_state,rtol=1e-13,atol=1e-13)


def test_j2_assembler_incremental_energy_matches_material_points_and_residual_gradient():
    basis, model = make_model()
    committed = model.initial_state()
    displacement = affine_displacement(basis, .02)
    assembled = model.evaluate(displacement, committed)
    local_u = displacement[model._dofs]
    grad_u = np.einsum("eai,eqaj->eqij", local_u, model._gradients)
    strain = .5 * (grad_u + grad_u.swapaxes(-1, -2))
    material = skfemntv.J2PlasticityMaterial(
        model.young_modulus, model.poisson_ratio,
        model.yield_stress, model.hardening_modulus)
    point = material.evaluate(strain, committed)
    assert assembled.energy == pytest.approx(
        float(np.sum(point.energy * model._weights)), rel=2e-13, abs=2e-13)

    direction = np.random.default_rng(187).normal(size=basis.N)
    step = 1e-8
    energy_difference = (
        model.evaluate(displacement + step * direction, committed).energy
        - model.evaluate(displacement - step * direction, committed).energy
    ) / (2. * step)
    assert energy_difference == pytest.approx(
        float(assembled.residual @ direction), rel=3e-7, abs=3e-8)


def test_nonuniform_plastic_material_response_integrates_to_assembler():
    basis, assembler = make_model()
    material = skfemntv.J2PlasticityMaterial(
        assembler.young_modulus,
        assembler.poisson_ratio,
        assembler.yield_stress,
        assembler.hardening_modulus,
    )
    nq = basis.dx.shape[1]
    committed = material.initial_state((basis.nelems, nq))
    prior_strain = np.zeros((basis.nelems, nq, 3, 3))
    for q, exx in enumerate(np.linspace(0.008, 0.02, nq)):
        prior_strain[:, q, 0, 0] = exx
    committed = material.commit(material.evaluate(prior_strain, committed))
    assert np.ptp(committed[0, :, 6]) > 0.0
    original_state = committed.copy()

    displacement = 0.008 * np.random.default_rng(184).normal(size=basis.N)
    local_u = displacement[assembler._dofs]
    grad_u = np.einsum("eai,eqaj->eqij", local_u, assembler._gradients)
    strain = 0.5 * (grad_u + grad_u.swapaxes(-1, -2))
    point = material.evaluate(strain, committed)
    assembled = assembler.evaluate(displacement, committed)

    local_residual = np.einsum(
        "eqij,eqaj,eq->eai", point.stress, assembler._gradients, assembler._weights
    )
    local_tangent = np.einsum(
        "eqaj,eqijkl,eqbl,eq->eaibk",
        assembler._gradients,
        point.tangent,
        assembler._gradients,
        assembler._weights,
    )
    dofs = assembler._dofs.reshape(basis.nelems, -1)
    residual = np.zeros(basis.N)
    np.add.at(residual, dofs.reshape(-1), local_residual.reshape(-1))
    local_matrix_shape = (basis.nelems, 12, 12)
    rows = np.broadcast_to(dofs[:, :, None], local_matrix_shape)
    columns = np.broadcast_to(dofs[:, None, :], local_matrix_shape)
    tangent = coo_matrix(
        (local_tangent.reshape(-1), (rows.reshape(-1), columns.reshape(-1))),
        shape=(basis.N, basis.N),
    ).tocsr()

    np.testing.assert_allclose(assembled.stress, point.stress, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(assembled.trial_state, point.trial_state, rtol=2e-13, atol=2e-13)
    np.testing.assert_array_equal(assembled.plastic_points, point.plastic_points)
    np.testing.assert_allclose(assembled.residual, residual, rtol=3e-13, atol=3e-13)
    np.testing.assert_allclose(
        assembled.tangent.toarray(), tangent.toarray(), rtol=3e-13, atol=3e-13
    )
    np.testing.assert_array_equal(committed, original_state)


def make_family(family):
    if family in ("tet4","tet10"):
        linear=skfemntv.MeshTet()
        if family=="tet4":
            mesh=linear
            element=skfemntv.ElementTetP1()
        else:
            mesh=skfemntv.MeshTet2.from_mesh(linear)
            element=skfemntv.ElementTetP2()
    elif family=="hex8":
        mesh=skfemntv.MeshHex()
        element=skfemntv.ElementHex1()
    elif family=="hex20":
        mesh=skfemntv.MeshHex20.from_mesh(skfemntv.MeshHex())
        element=skfemntv.ElementHex20()
    else:
        mesh=skfemntv.MeshHex2.from_mesh(skfemntv.MeshHex())
        element=skfemntv.ElementHex2()
    return skfemntv.Basis(
        mesh,skfemntv.ElementVector(element,dim=3),intorder=5)


def affine_x_displacement(basis,strain):
    displacement=np.zeros(basis.N)
    component_dofs=basis.split_indices()
    displacement[component_dofs[0]]=strain*basis.doflocs[0,component_dofs[0]]
    return displacement


@pytest.mark.parametrize("family",["tet4","tet10","hex8","hex20","hex27"])
def test_j2_plastic_tangent_history_and_elastic_stiffness_for_supported_families(family):
    basis=make_family(family)
    model=skfemntv.J2PlasticityAssembler(
        basis,young_modulus=1000.,poisson_ratio=.25,
        yield_stress=2.,hardening_modulus=20.)
    state=model.initial_state()

    @skfemntv.BilinearForm
    def elasticity(u,v,w):
        return 2.*model.mu*ddot(sym_grad(u),sym_grad(v)) \
            +model.lambda_*div(u)*div(v)

    elastic=model.evaluate(affine_x_displacement(basis,1e-5),state)
    assert not elastic.plastic_points.any()
    np.testing.assert_allclose(
        elastic.stress[...,0,0],(model.lambda_+2.*model.mu)*1e-5,
        rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(
        elastic.stress[...,1,1],model.lambda_*1e-5,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(elastic.stress,elastic.stress.swapaxes(-1,-2),atol=1e-13)
    np.testing.assert_allclose(
        elastic.equivalent_stress,
        np.sqrt(0.5*((elastic.stress[...,0,0]-elastic.stress[...,1,1])**2
                     +(elastic.stress[...,1,1]-elastic.stress[...,2,2])**2
                     +(elastic.stress[...,2,2]-elastic.stress[...,0,0])**2
                     +6.*(elastic.stress[...,0,1]**2+elastic.stress[...,1,2]**2
                          +elastic.stress[...,0,2]**2))),
        rtol=2e-13,atol=2e-13)
    np.testing.assert_allclose(
        elastic.tangent.toarray(),elasticity.assemble(basis).toarray(),
        rtol=3e-12,atol=3e-12)
    loaded_u=affine_x_displacement(basis,.02)
    loaded=model.evaluate(loaded_u,state)
    assert loaded.plastic_points.all()
    local_u=loaded_u[model._dofs]
    grad_u=np.einsum("eai,eqaj->eqij",local_u,model._gradients)
    strain=.5*(grad_u+grad_u.swapaxes(-1,-2))
    material=skfemntv.J2PlasticityMaterial(
        young_modulus=model.young_modulus,poisson_ratio=model.poisson_ratio,
        yield_stress=model.yield_stress,hardening_modulus=model.hardening_modulus)
    point=material.evaluate(strain,state)
    local_tangent=np.einsum(
        "eqaj,eqijkl,eqbl,eq->eaibk",model._gradients,point.tangent,
        model._gradients,model._weights)
    dofs=model._dofs.reshape(basis.nelems,-1)
    local_size=dofs.shape[1]
    matrix_shape=(basis.nelems,local_size,local_size)
    rows=np.broadcast_to(dofs[:,:,None],matrix_shape).reshape(-1)
    columns=np.broadcast_to(dofs[:,None,:],matrix_shape).reshape(-1)
    independent_tangent=coo_matrix(
        (local_tangent.reshape(-1),(rows,columns)),
        shape=(basis.N,basis.N)).tocsr()
    np.testing.assert_allclose(
        loaded.tangent.toarray(),independent_tangent.toarray(),
        rtol=3e-12,atol=3e-12)
    np.testing.assert_allclose(
        loaded.equivalent_stress,
        model.yield_stress+model.hardening_modulus*loaded.trial_state[...,6],
        rtol=2e-12,atol=2e-12)
    rng=np.random.default_rng(2026)
    direction=rng.normal(size=basis.N)
    step=1e-8
    fd=(model.evaluate(loaded_u+step*direction,state).residual-
        model.evaluate(loaded_u-step*direction,state).residual)/(2.*step)
    np.testing.assert_allclose(loaded.tangent@direction,fd,rtol=3e-7,atol=3e-8)
    energy_fd=(model.evaluate(loaded_u+step*direction,state).energy-
               model.evaluate(loaded_u-step*direction,state).energy)/(2.*step)
    assert energy_fd==pytest.approx(
        float(loaded.residual@direction),rel=3e-7,abs=3e-8)
    committed=model.commit(loaded)
    np.testing.assert_array_equal(state,0.)
    unload=model.evaluate(affine_x_displacement(basis,.019),committed)
    assert not unload.plastic_points.any()
    np.testing.assert_array_equal(unload.trial_state,committed)
