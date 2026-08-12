"""Experimental affine-triangle lowest-order H(curl) solve."""

import numpy as np
from scipy.sparse.linalg import spsolve

import skfemntv


mesh=skfemntv.MeshTri.init_tensor(
    np.linspace(0.,1.,9),np.linspace(0.,1.,7)
).with_boundaries({
    "left":lambda x:np.isclose(x[0],0.),
    "right":lambda x:np.isclose(x[0],1.),
})
basis=skfemntv.AffineTriN1Basis(mesh,intorder=3)
assembler=skfemntv.TriN1Assembler(basis)
linear_assembler=skfemntv.TriN1LinearAssembler(basis)

# A positive mass term removes the gradient nullspace of pure curl-curl.
matrix=assembler.assemble_maxwell(
    mass_coefficient=1.,curl_coefficient=.05
).copy()
boundary=basis.boundary_dofs()
free=np.setdiff1d(np.arange(basis.N),boundary)

# Solver policy remains external; the load is integrated against H(curl) basis
# functions by the dedicated reusable linear assembler.
load=linear_assembler.assemble_vector_load(
    lambda x:np.array((0.*x[0],np.sin(np.pi*x[0])))
).copy()
solution=np.zeros(basis.N)
solution[free]=spsolve(matrix[free][:,free],load[free])
free_residual=np.linalg.norm((matrix@solution-load)[free])

assert np.all(solution[boundary]==0.)
assert free_residual<1.e-10
print(
    f"TriN1 DOFs={basis.N}, constrained={len(boundary)}, "
    f"free residual={free_residual:.3e}"
)
