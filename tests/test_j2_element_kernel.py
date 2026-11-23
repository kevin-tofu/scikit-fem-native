from __future__ import annotations

import numpy as np

from skfemntv import (
    Basis,
    ElementTetP1,
    ElementVector,
    J2ElementKernel,
    J2PlasticityAssembler,
    MeshTet,
)


def test_j2_element_residual_and_tangent_match_global_assembler() -> None:
    points = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    mesh = MeshTet(points.T, np.arange(4, dtype=np.int64)[:, None])
    basis = Basis(mesh, ElementVector(ElementTetP1(), dim=3), intorder=1)
    model = J2PlasticityAssembler(
        basis,
        young_modulus=200.0,
        poisson_ratio=0.28,
        yield_stress=0.05,
        hardening_modulus=4.0,
    )
    displacement = np.asarray(
        [0.0, 0.0, 0.0, 0.01, 0.001, 0.0, 0.0, -0.001, 0.0, 0.0, 0.0, 0.0]
    )
    state = model.initial_state()
    reference = model.evaluate(displacement, state)

    dofs = basis.element_dofs.T.reshape(1, 4, 3)
    kernel = J2ElementKernel(model.material)
    local = kernel.evaluate(
        displacement[dofs[0]],
        basis.tabulated_gradients[0],
        basis.dx[0],
        state[0],
    )

    np.testing.assert_allclose(local.residual, reference.residual, atol=1e-12)
    np.testing.assert_allclose(
        local.element_tangent, reference.tangent.toarray(), rtol=2e-12, atol=2e-12
    )
    np.testing.assert_allclose(local.trial_state, reference.trial_state[0])
