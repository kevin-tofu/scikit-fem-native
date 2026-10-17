"""Show J2 material updates and assembly for prescribed Tet4 strain steps.

This example intentionally leaves load application and equilibrium solving to
the caller.  It demonstrates how to evaluate and commit material history.
"""

import numpy as np

import skfemntv


def main():
    points=np.array([
        [0.,1.,0.,0.],
        [0.,0.,1.,0.],
        [0.,0.,0.,1.],
    ])
    mesh=skfemntv.MeshTet(
        points,np.array([[0],[1],[2],[3]],dtype=np.int64))
    basis=skfemntv.Basis(
        mesh,skfemntv.ElementVector(skfemntv.ElementTetP1(),dim=3),
        intorder=2)
    material=skfemntv.J2PlasticityAssembler(
        basis,young_modulus=1000.,poisson_ratio=.25,
        yield_stress=2.,hardening_modulus=20.)
    committed=material.initial_state()
    x_dofs=basis.split_indices()[0]

    print("strain_xx  ||internal force||  increment potential  mean von Mises  mean ep_eq  plastic q-points")
    for strain_xx in (0.001,0.01,0.004,0.012,0.002):
        displacement=np.zeros(basis.N)
        displacement[x_dofs]=strain_xx*basis.doflocs[0,x_dofs]
        trial=material.evaluate(displacement,committed)
        print(
            f"{strain_xx:9.4f}  {np.linalg.norm(trial.residual):17.8e}  "
            f"{trial.energy:19.8e}  "
            f"{np.mean(trial.equivalent_stress):13.6f}  "
            f"{np.mean(trial.trial_state[...,6]):10.6f}  "
            f"{np.count_nonzero(trial.plastic_points):16d}"
        )
        # In a nonlinear solve this assignment belongs after step convergence.
        committed=material.commit(trial)


if __name__=="__main__":
    main()
