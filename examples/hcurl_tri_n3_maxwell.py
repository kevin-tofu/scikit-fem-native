"""Solve a regularized Maxwell problem with affine-triangle Nedelec N3."""

import numpy as np
from scipy.sparse.linalg import spsolve

import skfemntv
from skfemntv.helpers import curl,dot


mesh=skfemntv.MeshTri.init_tensor(
    np.linspace(0.,1.,9),np.linspace(0.,1.,7)
).with_boundaries({
    "left":lambda x:np.isclose(x[0],0.),
    "right":lambda x:np.isclose(x[0],1.),
})
basis=skfemntv.Basis(mesh,skfemntv.ElementTriN3(),intorder=8)

# The mass term removes the gradient nullspace of pure curl-curl.
@skfemntv.BilinearForm
def maxwell(u,v,w):
    return dot(u,v)+.05*curl(u)*curl(v)


@skfemntv.LinearForm
def forcing(v,w):
    return dot(w.force,v)


matrix=skfemntv.asm(maxwell,basis).tocsr()
force=np.stack((
    np.zeros_like(basis.global_coordinates[...,0]),
    np.sin(np.pi*basis.global_coordinates[...,0]),
))
load=skfemntv.asm(forcing,basis,force=force)
boundary=basis.get_dofs().all()
free=np.setdiff1d(np.arange(basis.N),boundary)
solution=np.zeros(basis.N)
solution[free]=spsolve(matrix[free][:,free],load[free])
free_residual=np.linalg.norm((matrix@solution-load)[free])

assert np.all(solution[boundary]==0.)
assert np.isfinite(solution).all()
assert free_residual<1.e-10
print(
    f"TriN3 DOFs={basis.N}, constrained={len(boundary)}, "
    f"free residual={free_residual:.3e}"
)
