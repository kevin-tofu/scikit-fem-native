import numpy as np
import pytest

import skfemntv


def _assembler():
    mesh=skfemntv.MeshTet()
    basis=skfemntv.Basis(
        mesh,skfemntv.ElementVector(skfemntv.ElementTetP1(),dim=3),intorder=2
    )
    material=skfemntv.J2Plasticity(210.,.3,.25,1.5)
    return skfemntv.MaterialAssembler(basis,material),basis


def _plastic_displacement(basis,scale=1.):
    u=np.zeros(basis.N)
    u[0::3]=scale*.004*basis.doflocs[0,0::3]
    u[1::3]=-scale*.002*basis.doflocs[1,1::3]
    u[2::3]=-scale*.002*basis.doflocs[2,2::3]
    return np.ascontiguousarray(u)


def test_trial_assembly_does_not_mutate_committed_state_and_can_rollback():
    assembler,basis=_assembler();history=assembler.initial_history()
    committed_before=history.committed.storage.copy()
    result=assembler.assemble_trial(_plastic_displacement(basis),history)

    assert history.trial is not None
    assert np.linalg.norm(result.trial_state.storage)>0.
    np.testing.assert_array_equal(history.committed.storage,committed_before)
    assert not history.committed.storage.flags.writeable
    assert not history.trial.storage.flags.writeable

    committed=history.rollback()
    assert history.trial is None
    assert history.commit_count==0
    np.testing.assert_array_equal(committed.storage,committed_before)


def test_commit_advances_history_only_after_explicit_acceptance():
    assembler,basis=_assembler();history=assembler.initial_history()
    first=assembler.assemble_trial(_plastic_displacement(basis),history)
    expected=first.trial_state.storage.copy()
    committed=history.commit()

    assert history.trial is None
    assert history.commit_count==1
    np.testing.assert_allclose(committed.storage,expected,rtol=0.,atol=0.)
    second=assembler.assemble_trial(
        _plastic_displacement(basis,1.2),history,mode="residual"
    )
    assert second.tangent is None
    np.testing.assert_array_equal(history.committed.storage,expected)
    assert np.linalg.norm(history.trial.storage-expected)>0.


def test_rejected_trial_does_not_pollute_cutback_evaluation():
    assembler,basis=_assembler();history=assembler.initial_history()
    assembler.assemble_trial(_plastic_displacement(basis,2.),history)
    history.rollback()
    cutback=assembler.assemble_trial(_plastic_displacement(basis,.5),history)

    clean=assembler.initial_history()
    reference=assembler.assemble_trial(_plastic_displacement(basis,.5),clean)
    np.testing.assert_allclose(
        cutback.residual,reference.residual,rtol=0.,atol=0.
    )
    np.testing.assert_allclose(
        history.trial.storage,clean.trial.storage,rtol=0.,atol=0.
    )


def test_history_rejects_wrong_state_type_shape_and_empty_commit():
    assembler,_basis=_assembler();history=assembler.initial_history()
    with pytest.raises(RuntimeError,match="no trial"):
        history.commit()
    with pytest.raises(TypeError,match="J2State"):
        history.stage(skfemntv.MaterialState(np.zeros((1,7))))
    with pytest.raises(ValueError,match="shape"):
        history.stage(skfemntv.J2Plasticity(1.,.2,.1).initial_state(1))
