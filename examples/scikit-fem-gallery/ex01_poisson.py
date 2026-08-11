"""Gallery Example 1: Poisson equation with a unit load."""

from __future__ import annotations

import numpy as np
import skfem
from skfem.helpers import dot as reference_dot
from skfem.helpers import grad as reference_grad

import skfemntv
from skfemntv.helpers import ddot,dot,grad

from comparison import GalleryComparison,solve_dirichlet,sparse_max_abs


@skfemntv.BilinearForm
def native_laplace(u,v,w):
    return ddot(grad(u),grad(v))


@skfemntv.LinearForm
def native_load(v,w):
    return dot(w.source,v)


@skfem.BilinearForm
def reference_laplace(u,v,w):
    return reference_dot(reference_grad(u),reference_grad(v))


@skfem.LinearForm
def reference_load(v,w):
    return v


def compare() -> GalleryComparison:
    axis=np.linspace(0.,1.,17)
    mesh=skfemntv.MeshTri.init_tensor(axis,axis)
    native_basis=skfemntv.Basis(
        mesh,skfemntv.ElementVector(skfemntv.ElementTriP1(),dim=1)
    )
    reference_mesh=skfem.MeshTri(mesh.p,mesh.t)
    reference_basis=skfem.Basis(reference_mesh,skfem.ElementTriP1())

    native_matrix=skfemntv.asm(native_laplace,native_basis)
    native_rhs=skfemntv.asm(
        native_load,native_basis,source=np.array([1.])
    )
    reference_matrix=skfem.asm(reference_laplace,reference_basis)
    reference_rhs=skfem.asm(reference_load,reference_basis)
    native_solution=solve_dirichlet(
        native_matrix,native_rhs,native_basis.get_dofs().all()
    )
    reference_solution=solve_dirichlet(
        reference_matrix,reference_rhs,reference_basis.get_dofs().all()
    )
    return GalleryComparison(
        "Example 1: 2-D Poisson",
        sparse_max_abs(native_matrix,reference_matrix),
        float(np.max(np.abs(native_rhs-reference_rhs))),
        float(np.max(np.abs(native_solution-reference_solution))),
    )


if __name__=="__main__":
    result=compare()
    result.assert_matches()
    print(result.summary())
