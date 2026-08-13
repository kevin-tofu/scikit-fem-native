"""Experimental affine-tetrahedron lowest-order H(curl) solve."""

import numpy as np
from scipy.sparse.linalg import spsolve

import skfemntv


axis=np.linspace(0.,1.,3)
mesh=skfemntv.MeshTet.init_tensor(axis,axis,axis)
basis=skfemntv.AffineTetN1Basis(mesh,intorder=3)
matrix=skfemntv.TetN1Assembler(basis).assemble_maxwell(
    mass_coefficient=1.,curl_coefficient=.05
).copy()
load=skfemntv.TetN1LinearAssembler(basis).assemble_vector_load(
    lambda x:np.array((1.+0.*x[0],x[0],-x[1]))
).copy()
boundary=basis.boundary_dofs()
free=np.setdiff1d(np.arange(basis.N),boundary)
solution=np.zeros(basis.N)
solution[free]=spsolve(matrix[free][:,free],load[free])
free_residual=np.linalg.norm((matrix@solution-load)[free])

assert np.all(solution[boundary]==0.)
assert free_residual<1.e-10
print(
    f"TetN1 DOFs={basis.N}, constrained={len(boundary)}, "
    f"free residual={free_residual:.3e}"
)
