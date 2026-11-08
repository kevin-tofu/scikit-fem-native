"""3D Tet4 Hill48 assembly and directional tangent check."""

from __future__ import annotations

import numpy as np

import skfemntv


def run() -> dict:
    mesh = skfemntv.MeshTet.init_tensor(
        np.array([0., 1.]), np.array([0., 1.]), np.array([0., 1.])
    )
    basis = skfemntv.Basis(
        mesh, skfemntv.ElementVector(skfemntv.ElementTetP1(), dim=3), intorder=2
    )
    material = {
        "young_modulus": 1000., "poisson_ratio": 0.3,
        "yield_stress": 2., "hardening_modulus": 20.,
        "stress_ratios": (1.2, 0.8, 1.0, 1.1, 1.0, 0.9),
    }
    assembler = skfemntv.Hill48PlasticityAssembler(basis, **material)
    dofs = basis.element_dofs.T.reshape(basis.nelems, -1)
    coordinates = mesh.p.T
    # Affine strain drives all material points into the anisotropic plastic range.
    gradient = np.array([[0.012, 0.004, 0.], [0.004, -0.002, 0.001],
                         [0., 0.001, -0.001]])
    displacement = (coordinates @ gradient.T).reshape(-1)
    committed = assembler.initial_state()
    evaluation = assembler.evaluate(displacement, committed)
    direction = np.sin(np.arange(basis.N) + 0.37)
    epsilon = 1e-7
    plus = assembler.evaluate(displacement + epsilon * direction, committed).residual
    minus = assembler.evaluate(displacement - epsilon * direction, committed).residual
    finite_difference = (plus - minus) / (2. * epsilon)
    tangent_action = evaluation.tangent @ direction
    relative_error = np.linalg.norm(finite_difference - tangent_action) / max(
        np.linalg.norm(finite_difference), 1e-14
    )
    return {
        "mesh": {"tetrahedra": int(mesh.nelements), "nodes": int(mesh.p.shape[1])},
        "stress_ratios_R11_R22_R33_R12_R13_R23": material["stress_ratios"],
        "plastic_points": int(np.count_nonzero(evaluation.plastic_points)),
        "max_equivalent_plastic_strain": float(np.max(evaluation.trial_state[..., 6])),
        "directional_tangent_relative_error": float(relative_error),
        "transactional_state_unchanged": bool(np.all(committed == 0.)),
        "committed_history_shape": list(assembler.commit(evaluation).shape),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run(), indent=2))
