"""Plane-strain J2 element patch and consistent-tangent directional check."""

import numpy as np

import skfemntv


def main():
    mesh = skfemntv.MeshTri.init_tensor(
        np.linspace(0.0, 1.0, 5), np.linspace(0.0, 0.25, 2)
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
    committed = assembler.initial_state()
    committed_before = committed.copy()
    displacement = np.zeros(basis.N)
    x_dofs = np.unique(basis.element_dofs[0::2])
    displacement[x_dofs] = 0.01 * basis.doflocs[0, x_dofs]

    trial = assembler.evaluate(displacement, committed)
    rng = np.random.default_rng(42)
    direction = rng.standard_normal(basis.N)
    direction /= np.linalg.norm(direction)
    step = 1.0e-7
    plus = assembler.evaluate(displacement + step * direction, committed).residual
    minus = assembler.evaluate(displacement - step * direction, committed).residual
    finite_difference = (plus - minus) / (2.0 * step)
    tangent_action = trial.tangent @ direction
    tangent_error = np.linalg.norm(tangent_action - finite_difference) / max(
        np.linalg.norm(finite_difference), 1.0e-30
    )

    committed_after = assembler.commit(trial)
    print(f"elements: {basis.nelems}")
    print(f"quadrature points: {trial.trial_state.shape[0] * trial.trial_state.shape[1]}")
    print(f"plastic points: {np.count_nonzero(trial.plastic_points)}")
    print(f"mean sigma_zz: {np.mean(trial.stress[..., 2, 2]):.8e}")
    print(f"max equivalent plastic strain: {np.max(trial.trial_state[..., 6]):.8e}")
    print(f"committed input unchanged: {np.array_equal(committed, committed_before)}")
    print(f"commit advances history: {np.max(committed_after[..., 6]) > 0.0}")
    print(f"consistent tangent directional relative error: {tangent_error:.3e}")


if __name__ == "__main__":
    main()
