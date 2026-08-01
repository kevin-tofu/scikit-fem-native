import numpy as np
from scipy.sparse.linalg import spsolve

import skfemntv


def test_load_controlled_j2_newton_commits_only_after_convergence():
    axis=np.linspace(0.,1.,3)
    mesh=skfemntv.MeshHex.init_tensor(axis,axis,axis)
    basis=skfemntv.Basis(
        mesh,skfemntv.ElementVector(skfemntv.ElementHex1(),dim=3),intorder=2
    )
    material=skfemntv.J2Plasticity(
        young_modulus=210.,poisson_ratio=.3,
        yield_stress=.25,hardening_modulus=2.,
    )
    assembler=skfemntv.J2Assembler(basis,material)
    state=assembler.initial_state()
    coordinates=basis.doflocs
    fixed=np.flatnonzero(np.isclose(coordinates[0],0.))
    free=np.setdiff1d(np.arange(basis.N),fixed)
    loaded_nodes=np.flatnonzero(
        np.isclose(mesh.p[0],1.)
    )
    loaded_dofs=3*loaded_nodes
    unit_load=np.zeros(basis.N)
    unit_load[loaded_dofs]=1./len(loaded_dofs)
    displacement=np.zeros(basis.N)
    iteration_counts=[]

    for load_factor in np.linspace(.05,.55,11):
        external=load_factor*unit_load
        committed_alpha=state.equivalent_plastic_strain.copy()
        for iteration in range(15):
            evaluation=assembler.assemble(
                np.ascontiguousarray(displacement),state,num_threads=4
            )
            residual=evaluation.residual-external
            norm=np.linalg.norm(residual[free])
            if norm<2e-10:
                break
            increment=spsolve(
                evaluation.tangent[free][:,free],-residual[free]
            )
            displacement[free]+=increment
            np.testing.assert_array_equal(
                state.equivalent_plastic_strain,committed_alpha
            )
        else:
            raise AssertionError("J2 Newton iteration did not converge")
        state=evaluation.trial_state
        iteration_counts.append(iteration)

    assert np.max(state.equivalent_plastic_strain)>0.
    assert max(iteration_counts)<10
    final=assembler.assemble(displacement,state,num_threads=4)
    np.testing.assert_allclose(
        final.residual[loaded_dofs].sum(),.55,rtol=2e-8,atol=2e-8
    )


def test_j2_trial_state_can_be_rolled_back():
    mesh=skfemntv.MeshTet()
    basis=skfemntv.Basis(
        mesh,skfemntv.ElementVector(skfemntv.ElementTetP1(),dim=3),intorder=2
    )
    assembler=skfemntv.J2Assembler(
        basis,skfemntv.J2Plasticity(100.,.3,.1,1.)
    )
    committed=assembler.initial_state()
    trial=assembler.assemble(
        np.ascontiguousarray(np.linspace(0.,.02,basis.N)),committed
    ).trial_state
    assert np.max(trial.equivalent_plastic_strain)>0.
    np.testing.assert_array_equal(committed.plastic_strain,0.)
    np.testing.assert_array_equal(committed.equivalent_plastic_strain,0.)
