"""Gallery Example 19: heat equation using the theta method."""

from __future__ import annotations

from math import ceil

import numpy as np
from scipy.sparse.linalg import splu
import skfem
from skfem.helpers import dot as reference_dot
from skfem.helpers import grad as reference_grad

import skfemntv
from skfemntv.helpers import ddot,dot,grad

from comparison import GalleryComparison,sparse_max_abs


@skfemntv.BilinearForm
def native_laplace(u,v,w):
    return ddot(grad(u),grad(v))


@skfemntv.BilinearForm
def native_mass(u,v,w):
    return dot(u,v)


@skfem.BilinearForm
def reference_laplace(u,v,w):
    return reference_dot(reference_grad(u),reference_grad(v))


@skfem.BilinearForm
def reference_mass(u,v,w):
    return u*v


def evolve(matrix,mass,initial,boundary,steps: int=8) -> np.ndarray:
    diffusivity=5.
    timestep=.01
    theta=.5
    free=np.setdiff1d(
        np.arange(initial.size,dtype=np.int64),boundary,
        assume_unique=False,
    )
    stiffness=diffusivity*matrix[free][:,free]
    reduced_mass=mass[free][:,free]
    left=(reduced_mass+theta*timestep*stiffness).tocsc()
    right=reduced_mass-(1.-theta)*timestep*stiffness
    solve=splu(left).solve
    state=initial[free].copy()
    for _ in range(steps):
        state=solve(right@state)
    result=np.zeros(initial.size,dtype=float)
    result[free]=state
    return result


def compare() -> GalleryComparison:
    halfwidth=np.array([2.,3.])
    cells=2**3
    x=np.linspace(-1.,1.,2*cells)*halfwidth[0]
    y=np.linspace(
        -1.,1.,2*cells*ceil(halfwidth[1]//halfwidth[0])
    )*halfwidth[1]
    linear=skfemntv.MeshQuad.init_tensor(x,y)
    mesh=skfemntv.MeshQuad2.from_mesh(linear)
    native_basis=skfemntv.Basis(
        mesh,skfemntv.ElementVector(skfemntv.ElementQuad2(),dim=1),
        intorder=4,
    )
    reference_linear=skfem.MeshQuad(linear.p,linear.t)
    reference_mesh=skfem.MeshQuad2.from_mesh(reference_linear)
    reference_basis=skfem.Basis(
        reference_mesh,skfem.ElementQuad2(),intorder=4
    )
    if not np.array_equal(native_basis.doflocs,reference_basis.doflocs):
        raise AssertionError(
            "Example 19 requires identical public DOF ordering"
        )

    native_stiffness=skfemntv.asm(native_laplace,native_basis)
    native_mass_matrix=skfemntv.asm(native_mass,native_basis)
    reference_stiffness=skfem.asm(reference_laplace,reference_basis)
    reference_mass_matrix=skfem.asm(reference_mass,reference_basis)
    native_initial=np.cos(
        np.pi*native_basis.doflocs/2./halfwidth[:,None]
    ).prod(0)
    reference_initial=np.cos(
        np.pi*reference_basis.doflocs/2./halfwidth[:,None]
    ).prod(0)
    native_solution=evolve(
        native_stiffness,native_mass_matrix,native_initial,
        native_basis.get_dofs().all(),
    )
    reference_solution=evolve(
        reference_stiffness,reference_mass_matrix,reference_initial,
        reference_basis.get_dofs().all(),
    )
    return GalleryComparison(
        "Example 19: heat equation",
        max(
            sparse_max_abs(native_stiffness,reference_stiffness),
            sparse_max_abs(native_mass_matrix,reference_mass_matrix),
        ),
        float(np.max(np.abs(native_initial-reference_initial))),
        float(np.max(np.abs(native_solution-reference_solution))),
    )


if __name__=="__main__":
    result=compare()
    result.assert_matches()
    print(result.summary())
