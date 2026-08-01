from pathlib import Path
import sys

import numpy as np
import pytest
import skfem

import skfemntv


sys.path.insert(0,str(Path(__file__).parents[1]/"benchmarks"))
from skfem_j2 import forms as reference_forms,update as reference_update


def _strain(field):
    return np.stack((
        field.grad[0,0],field.grad[1,1],field.grad[2,2],
        .5*(field.grad[0,1]+field.grad[1,0]),
        .5*(field.grad[1,2]+field.grad[2,1]),
        .5*(field.grad[0,2]+field.grad[2,0]),
    ))


@pytest.mark.parametrize("topology",["tet","hex"])
def test_j2_load_unload_reload_history_matches_scikit_fem(topology):
    axis=np.linspace(0.,1.,3)
    if topology=="tet":
        mesh=skfemntv.MeshTet.init_tensor(axis,axis,axis)
        element=skfemntv.ElementTetP1()
        reference_mesh=skfem.MeshTet(mesh.p,mesh.t)
        reference_element=skfem.ElementTetP1()
    else:
        mesh=skfemntv.MeshHex.init_tensor(axis,axis,axis)
        element=skfemntv.ElementHex1()
        reference_mesh=skfem.MeshHex(mesh.p,mesh.t)
        reference_element=skfem.ElementHex1()
    basis=skfemntv.Basis(
        mesh,skfemntv.ElementVector(element,dim=3),intorder=2
    )
    reference_basis=skfem.Basis(
        reference_mesh,skfem.ElementVector(reference_element),
        quadrature=(basis.X,basis.W),
    )
    material=skfemntv.J2Plasticity(210.,.3,.25,2.)
    assembler=skfemntv.J2Assembler(basis,material)
    residual_form,tangent_form=reference_forms()
    state=assembler.initial_state()
    entities,quadrature=basis.dx.shape
    reference_plastic=np.zeros((6,entities,quadrature))
    reference_alpha=np.zeros((entities,quadrature))
    coordinates=basis.doflocs
    committed_alpha=[]

    # Elastic loading, plastic loading, unloading, reverse loading, reloading.
    for amplitude in (.0005,.004,.003,-.003,.0025):
        displacement=np.zeros(basis.N)
        displacement[0::3]=amplitude*coordinates[0,0::3]
        displacement[1::3]=-.5*amplitude*coordinates[1,1::3]
        displacement[2::3]=-.5*amplitude*coordinates[2,2::3]
        native=assembler.assemble(displacement,state,num_threads=4)
        native_residual=native.residual.copy()
        native_tangent=native.tangent.copy()

        field=reference_basis.interpolate(displacement)
        stress,constitutive,trial_plastic,trial_alpha=reference_update(
            _strain(field),material.young_modulus,material.poisson_ratio,
            material.yield_stress,material.hardening_modulus,
            reference_plastic,reference_alpha,
        )
        reference_residual=residual_form.assemble(
            reference_basis,stress=stress
        )
        reference_tangent=tangent_form.assemble(
            reference_basis,constitutive=constitutive
        )
        np.testing.assert_allclose(
            native_residual,reference_residual,rtol=4e-11,atol=4e-11
        )
        difference=native_tangent-reference_tangent
        difference.eliminate_zeros()
        assert (difference.nnz==0
                or np.max(np.abs(difference.data))<4e-10)
        np.testing.assert_allclose(
            native.trial_state.plastic_strain,
            np.moveaxis(trial_plastic,0,-1).reshape(-1,6),
            rtol=4e-11,atol=4e-11,
        )
        np.testing.assert_allclose(
            native.trial_state.equivalent_plastic_strain,
            trial_alpha.reshape(-1),rtol=4e-11,atol=4e-11,
        )

        # Commit only the accepted trial state.
        state=native.trial_state
        reference_plastic=trial_plastic
        reference_alpha=trial_alpha
        committed_alpha.append(np.max(reference_alpha))

    assert committed_alpha[0]==0.
    assert committed_alpha[1]>0.
    assert committed_alpha[2]==pytest.approx(committed_alpha[1])
    assert committed_alpha[-1]>=committed_alpha[1]


def test_j2_history_rejected_trial_does_not_affect_reload():
    mesh=skfemntv.MeshTet()
    basis=skfemntv.Basis(
        mesh,skfemntv.ElementVector(skfemntv.ElementTetP1(),dim=3),intorder=2
    )
    assembler=skfemntv.J2Assembler(
        basis,skfemntv.J2Plasticity(210.,.3,.25,2.)
    )
    committed=assembler.initial_state()
    rejected=np.ascontiguousarray(np.linspace(-.02,.02,basis.N))
    assembler.assemble(rejected,committed,num_threads=4)
    accepted=np.ascontiguousarray(np.linspace(-.003,.003,basis.N))
    after_rejection=assembler.assemble(accepted,committed,num_threads=4)
    residual=after_rejection.residual.copy()
    tangent=after_rejection.tangent.copy()
    clean=assembler.assemble(accepted,assembler.initial_state(),num_threads=1)
    np.testing.assert_allclose(clean.residual,residual,rtol=0.,atol=0.)
    np.testing.assert_allclose(clean.tangent.data,tangent.data,rtol=0.,atol=0.)
