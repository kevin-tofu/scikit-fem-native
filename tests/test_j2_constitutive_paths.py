import numpy as np
import pytest

import skfemntv


def material(hardening=20.):
    return skfemntv.J2PlasticityMaterial(
        young_modulus=1000.,poisson_ratio=.25,
        yield_stress=2.,hardening_modulus=hardening)


def test_hydrostatic_strain_does_not_activate_j2_plasticity():
    model=material()
    strain=.01*np.eye(3)
    state=model.initial_state()
    response=model.evaluate(strain,state)
    assert not bool(response.plastic_points)
    assert float(response.equivalent_stress)==pytest.approx(0.,abs=1e-13)
    np.testing.assert_allclose(response.stress,3.*(model.lambda_+2.*model.mu/3.)*.01*np.eye(3))
    np.testing.assert_array_equal(response.trial_state,state)


def test_pure_shear_uses_von_mises_shear_yield_condition():
    model=material()
    strain=np.zeros((3,3))
    strain[0,1]=strain[1,0]=1.5*model.yield_stress/(2.*np.sqrt(3.)*model.mu)
    state=model.initial_state()
    response=model.evaluate(strain,state)
    assert bool(response.plastic_points)
    assert float(response.equivalent_stress)==pytest.approx(
        model.yield_stress+model.hardening_modulus*response.trial_state[6],
        rel=2e-12)
    assert response.stress[0,1]==pytest.approx(float(response.equivalent_stress)/np.sqrt(3.))


def test_perfect_plasticity_holds_equivalent_stress_at_yield():
    model=material(hardening=0.)
    strain=np.zeros((3,3))
    strain[0,1]=strain[1,0]=5.*model.yield_stress/(2.*np.sqrt(3.)*model.mu)
    response=model.evaluate(strain,model.initial_state())
    assert bool(response.plastic_points)
    assert response.trial_state[6]>0.
    assert float(response.equivalent_stress)==pytest.approx(model.yield_stress,rel=2e-12)


def test_multistep_loading_unloading_and_reverse_yield_use_only_committed_history():
    model=material()
    committed=model.initial_state()
    path=[.01,.009,.012,.010,.0095,.004]
    yielded=[]
    alphas=[]
    for epsilon in path:
        strain=np.diag([epsilon,0.,0.])
        before=committed.copy()
        trial=model.evaluate(strain,committed)
        np.testing.assert_array_equal(committed,before)
        yielded.append(bool(trial.plastic_points))
        alphas.append(float(trial.trial_state[6]))
        if epsilon==.01:
            # A failed line-search trial is deliberately discarded.
            discarded=model.evaluate(np.diag([.1,0.,0.]),committed)
            assert discarded.trial_state[6]>trial.trial_state[6]
            np.testing.assert_array_equal(committed,before)
        committed=model.commit(trial)
    assert yielded==[True,False,True,False,False,True]
    assert alphas[0]<alphas[2]<alphas[5]
    assert committed[6]>0.
    # A subsequent reverse load eventually yields in the opposite direction.
    reverse=model.evaluate(np.diag([-.006,0.,0.]),committed)
    assert bool(reverse.plastic_points)
    assert reverse.trial_state[0]<committed[0]
    assert reverse.trial_state[6]>committed[6]


@pytest.mark.parametrize("case", ["elastic", "plastic", "unloading"])
def test_incremental_potential_matches_energy_formula_and_stress_gradient(case):
    model=material()
    committed=model.initial_state()
    if case=="elastic":
        strain=np.diag([1e-4,0.,0.])
    elif case=="plastic":
        strain=np.diag([.01,0.,0.])
    else:
        committed=model.commit(model.evaluate(np.diag([.01,0.,0.]),committed))
        strain=np.diag([.006,0.,0.])
    result=model.evaluate(strain,committed)
    plastic_strain=np.array([
        [result.trial_state[0],result.trial_state[3],result.trial_state[5]],
        [result.trial_state[3],result.trial_state[1],result.trial_state[4]],
        [result.trial_state[5],result.trial_state[4],result.trial_state[2]],
    ])
    elastic_strain=strain-plastic_strain
    alpha_increment=result.trial_state[6]-committed[6]
    expected=(model.mu*np.einsum("ij,ij->",elastic_strain,elastic_strain)
              +.5*model.lambda_*np.trace(elastic_strain)**2
              +.5*model.hardening_modulus*result.trial_state[6]**2
              +model.yield_stress*alpha_increment)
    assert float(result.energy)==pytest.approx(expected,rel=2e-13,abs=2e-13)
    assert bool(result.plastic_points)==(case=="plastic")

    direction=np.random.default_rng(93).normal(size=(3,3))
    direction=.5*(direction+direction.T)
    h=1e-8
    energy_fd=(model.evaluate(strain+h*direction,committed).energy-
               model.evaluate(strain-h*direction,committed).energy)/(2.*h)
    stress_action=np.einsum("ij,ij->",result.stress,direction)
    assert float(energy_fd)==pytest.approx(float(stress_action),rel=3e-7,abs=3e-8)
