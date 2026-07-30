"""Minimal load-controlled Newton solve using the native Neo-Hookean kernel."""

import numpy as np
from scipy.sparse.linalg import spsolve

from skfn import NativeAssembler, NeoHookeanTet4


coordinates = np.array(
    [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
)
connectivity = np.array([[0, 1, 2, 3]], dtype=np.int64)
element_dofs = np.arange(12, dtype=np.int64).reshape(1, 4, 3)
assembler = NativeAssembler(
    coordinates, connectivity, element_dofs,
    NeoHookeanTet4.from_young_poisson(young_modulus=100., poisson_ratio=.3),
)

# Remove the six rigid modes and pull node 3 in the z direction.
fixed = np.array([0, 1, 2, 4, 5, 8])
free = np.setdiff1d(np.arange(assembler.ndofs), fixed)
loads = np.zeros(assembler.ndofs)
loads[11] = 1.
u = np.zeros(assembler.ndofs)

for iteration in range(12):
    out = assembler.evaluate(u, loads=loads)
    norm = np.linalg.norm(out.residual[free])
    print(f"{iteration:2d}: |R_free| = {norm:.3e}")
    if norm < 1e-11:
        break
    u[free] += spsolve(out.tangent[free][:, free], -out.residual[free])
else:
    raise RuntimeError("Newton iteration did not converge")

print(f"node 3 z-displacement: {u[11]:.8f}")
