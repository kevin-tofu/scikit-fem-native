"""Finite-strain plane-strain J2 tension path with transactional history.

This material/assembler integration example uses a uniform tabular hardening
curve and displacement control. It is a constitutive and Newton-path check,
not a sheet-forming or contact benchmark.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import spsolve

import skfemntv


def run(target_strain: float = 0.05, increments: int = 5) -> dict:
    if not np.isfinite(target_strain) or target_strain <= 0.:
        raise ValueError("target_strain must be finite and positive")
    if increments < 1:
        raise ValueError("increments must be at least one")

    mesh = skfemntv.MeshTri.init_tensor(
        np.linspace(0., 1., 9), np.linspace(0., 0.25, 3)
    )
    basis = skfemntv.Basis(
        mesh, skfemntv.ElementVector(skfemntv.ElementTriP1(), dim=2), intorder=2
    )
    reference_path = Path(__file__).with_name("numisheet93_draw_bending_reference.json")
    reference = json.loads(reference_path.read_text())
    material_data = reference["material"]
    material = skfemntv.FiniteStrainJ2Material(
        material_data["young_modulus"],
        material_data["poisson_ratio"],
        material_data["initial_yield_stress"],
        hardening_curve=material_data[
            "hardening_curve_equivalent_plastic_strain_yield_stress_MPa"
        ],
    )
    assembler = skfemntv.PlaneStrainFiniteStrainJ2Assembler(basis, material)

    dof_ids = np.arange(basis.N)
    x = basis.doflocs[0]
    y = basis.doflocs[1]
    component = dof_ids % 2
    left_x = np.flatnonzero((component == 0) & np.isclose(x, 0.))
    right_x = np.flatnonzero((component == 0) & np.isclose(x, 1.))
    anchor_y = np.flatnonzero(
        (component == 1) & np.isclose(x, 0.) & np.isclose(y, 0.)
    )
    fixed = np.unique(np.r_[left_x, right_x, anchor_y])
    free = np.setdiff1d(dof_ids, fixed)
    committed = assembler.initial_state()
    displacement = np.zeros(basis.N)
    history = []

    for fraction in np.linspace(1. / increments, 1., increments):
        prescribed = np.zeros(fixed.size)
        prescribed[np.isin(fixed, right_x)] = fraction * target_strain
        displacement[fixed] = prescribed
        accepted = False
        for iteration in range(1, 31):
            trial = assembler.evaluate(displacement, committed)
            residual = trial.residual[free]
            residual_norm = float(np.linalg.norm(residual, ord=np.inf))
            if residual_norm <= 2e-6:
                accepted = True
                break
            increment = spsolve(trial.tangent[free][:, free], -residual)
            line_search = False
            for backtrack in range(16):
                alpha = 0.5 ** backtrack
                candidate = displacement.copy()
                candidate[free] += alpha * increment
                candidate[fixed] = prescribed
                candidate_trial = assembler.evaluate(
                    candidate, committed, tangent=False
                )
                candidate_norm = float(np.linalg.norm(
                    candidate_trial.residual[free], ord=np.inf
                ))
                if candidate_norm < residual_norm:
                    displacement = candidate
                    trial = candidate_trial
                    line_search = True
                    break
            if not line_search:
                raise RuntimeError(
                    f"Newton line search failed at load fraction {fraction:.3f}; "
                    f"residual={residual_norm:.3e}"
                )
        if not accepted:
            raise RuntimeError(f"Newton failed at load fraction {fraction:.3f}")

        committed = assembler.commit(trial)
        reaction = float(np.sum(trial.residual[right_x]))
        history.append({
            "load_fraction": float(fraction),
            "prescribed_extension": float(fraction * target_strain),
            "newton_iterations": iteration,
            "right_edge_reaction": reaction,
            "maximum_equivalent_plastic_strain": float(np.max(committed[..., 9])),
            "equilibrium_residual_inf": residual_norm,
            "plastic_points": int(np.count_nonzero(trial.plastic_points)),
        })

    return {
        "model": "finite-strain plane-strain Hencky J2 with multiplicative plasticity",
        "mesh": {"elements": int(basis.nelems), "nodes": int(mesh.p.shape[1]),
                 "displacement_dofs": int(basis.N)},
        "target_extension": target_strain,
        "hardening": "NUMISHEET '93 reference tabular curve",
        "tangent": (
            "analytic consistent first-Piola tangent dP/dF; centered finite "
            "differences are available as a material-point reference mode"
        ),
        "steps": history,
        "maximum_equivalent_plastic_strain": history[-1][
            "maximum_equivalent_plastic_strain"
        ],
        "scope": "material/assembler Newton-path check; no rigid-tool contact or forming validation",
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-strain", type=float, default=0.05)
    parser.add_argument("--increments", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(run(args.target_strain, args.increments), indent=2))


if __name__ == "__main__":
    main()
