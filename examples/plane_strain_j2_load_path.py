"""Incremental displacement control with transactional J2 history commits."""

import numpy as np
from scipy.sparse.linalg import spsolve

import skfemntv


def main():
    mesh = skfemntv.MeshTri.init_tensor(
        np.linspace(0.0, 1.0, 9), np.linspace(0.0, 0.25, 3)
    )
    basis = skfemntv.Basis(
        mesh,
        skfemntv.ElementVector(skfemntv.ElementTriP1(), dim=2),
        intorder=2,
    )
    assembler = skfemntv.PlaneStrainJ2Assembler(
        basis,
        young_modulus=1000.0,
        poisson_ratio=0.3,
        yield_stress=2.0,
        hardening_modulus=20.0,
    )

    dof_ids = np.arange(basis.N)
    x = basis.doflocs[0]
    y = basis.doflocs[1]
    component = dof_ids % 2
    left_x = np.flatnonzero((component == 0) & np.isclose(x, 0.0))
    right_x = np.flatnonzero((component == 0) & np.isclose(x, 1.0))
    bottom_left_y = np.flatnonzero(
        (component == 1) & np.isclose(x, 0.0) & np.isclose(y, 0.0)
    )
    fixed = np.unique(np.r_[left_x, right_x, bottom_left_y])
    free = np.setdiff1d(dof_ids, fixed)
    committed = assembler.initial_state()
    displacement = np.zeros(basis.N)
    target_strain = 0.01
    load_fractions = np.linspace(0.2, 1.0, 5)
    history = []

    for fraction in load_fractions:
        prescribed = np.zeros(fixed.size)
        prescribed[np.isin(fixed, right_x)] = fraction * target_strain
        displacement[fixed] = prescribed
        accepted = False
        for iteration in range(1, 31):
            trial = assembler.evaluate(displacement, committed)
            residual = trial.residual[free]
            norm = float(np.linalg.norm(residual, ord=np.inf))
            if norm <= 1.0e-9:
                accepted = True
                break
            increment = spsolve(trial.tangent[free][:, free], -residual)
            line_search = False
            for backtrack in range(16):
                alpha = 0.5**backtrack
                candidate = displacement.copy()
                candidate[free] += alpha * increment
                candidate[fixed] = prescribed
                candidate_trial = assembler.evaluate(candidate, committed)
                candidate_norm = np.linalg.norm(candidate_trial.residual[free], ord=np.inf)
                if candidate_norm < norm:
                    displacement = candidate
                    trial = candidate_trial
                    line_search = True
                    break
            if not line_search:
                raise RuntimeError(
                    f"Newton line search failed at load fraction {fraction:.3f}"
                )
        if not accepted:
            raise RuntimeError(f"Newton failed at load fraction {fraction:.3f}")

        # The trial was evaluated against the previous converged state. Commit
        # exactly once, and only after the equilibrium residual has converged.
        committed = assembler.commit(trial)
        reactions = trial.residual[right_x]
        history.append((fraction, iteration, np.linalg.norm(reactions),
                        np.max(committed[..., 6])))

    for fraction, iterations, punch_force, ep_max in history:
        print(
            f"load={fraction:.2f} Newton={iterations:2d} "
            f"reaction={punch_force:.8e} max_ep_eq={ep_max:.8e}"
        )
    print(f"final free residual inf: {np.linalg.norm(trial.residual[free], ord=np.inf):.3e}")


if __name__ == "__main__":
    main()
