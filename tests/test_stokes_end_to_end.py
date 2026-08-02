import numpy as np
import skfem
from scipy.sparse.linalg import spsolve
from skfem.helpers import ddot as reference_ddot
from skfem.helpers import div as reference_div
from skfem.helpers import dot as reference_dot
from skfem.helpers import grad as reference_grad

import skfemntv
from skfemntv.helpers import ddot,div,dot,grad


def _permutation(native,reference):
    permutation=np.empty(native.N,dtype=np.int64)
    for native_dofs,reference_dofs,components in zip(
        native.split_indices(),reference.split_indices(),(3,1)
    ):
        lookup={}
        for dof in reference_dofs:
            coordinate=tuple(np.round(reference.doflocs[:,dof],14))
            lookup.setdefault(coordinate,[]).append(int(dof))
        for offset,native_dof in enumerate(native_dofs):
            coordinate=tuple(np.round(
                native.doflocs[:,native_dof],14
            ))
            permutation[native_dof]=lookup[coordinate][offset%components]
    return permutation


def _solve(matrix,rhs,constrained):
    free=np.setdiff1d(
        np.arange(matrix.shape[0]),constrained,assume_unique=False
    )
    solution=np.zeros(matrix.shape[0])
    solution[free]=spsolve(matrix[free][:,free],rhs[free])
    return solution


def test_taylor_hood_stokes_end_to_end_matches_skfem():
    axis=np.linspace(0.,1.,3)
    linear=skfemntv.MeshTet.init_tensor(axis,axis,axis)
    mesh=skfemntv.MeshTet2.from_mesh(linear).with_boundaries({
        "wall":lambda x:np.ones(x.shape[1],dtype=bool),
    })
    element=(
        skfemntv.ElementVector(skfemntv.ElementTetP2())
        *skfemntv.ElementTetP1()
    )
    basis=skfemntv.Basis(mesh,element,intorder=4)

    @skfemntv.BilinearForm
    def stokes(u,p,v,q,w):
        return (
            w.viscosity*ddot(grad(u),grad(v))
            -p*div(v)-q*div(u)
        )

    @skfemntv.LinearForm
    def forcing(v,q,w):
        return dot(w.force,v)

    matrix=skfemntv.asm(stokes,basis,viscosity=.8)
    x=np.moveaxis(basis.global_coordinates,-1,0)
    force=np.stack((x[1],-x[0],x[0]*0.))
    rhs=skfemntv.asm(forcing,basis,force=force)

    velocity_basis,_=basis.split_bases()
    velocity_indices,pressure_indices=basis.split_indices()
    constrained=np.concatenate((
        velocity_indices[velocity_basis.get_dofs("wall").all()],
        pressure_indices[:1],
    ))
    solution=_solve(matrix,rhs,constrained)
    velocity,pressure=basis.interpolate(solution)

    @skfemntv.Functional
    def dissipation(w):
        return w.viscosity*ddot(
            grad(w.velocity),grad(w.velocity)
        )

    native_dissipation=skfemntv.asm(
        dissipation,basis,viscosity=.8,velocity=velocity
    )

    reference_linear=skfem.MeshTet(linear.p,linear.t).with_boundaries({
        "wall":lambda x:np.ones(x.shape[1],dtype=bool),
    })
    reference_mesh=skfem.MeshTet2.from_mesh(reference_linear)
    reference_basis=skfem.Basis(
        reference_mesh,
        skfem.ElementVector(skfem.ElementTetP2())
        *skfem.ElementTetP1(),
        intorder=4,
    )

    @skfem.BilinearForm
    def reference_stokes(u,p,v,q,w):
        return (
            w.viscosity*reference_ddot(
                reference_grad(u),reference_grad(v)
            )
            -p*reference_div(v)-q*reference_div(u)
        )

    @skfem.LinearForm
    def reference_forcing(v,q,w):
        return reference_dot(w.force,v)

    reference_matrix=skfem.asm(
        reference_stokes,reference_basis,viscosity=.8
    )
    reference_x=np.asarray(reference_basis.global_coordinates())
    reference_force=np.stack((
        reference_x[1],-reference_x[0],reference_x[0]*0.
    ))
    reference_rhs=skfem.asm(
        reference_forcing,reference_basis,force=reference_force
    )
    reference_velocity_basis,_=reference_basis.split_bases()
    reference_velocity_indices,reference_pressure_indices=(
        reference_basis.split_indices()
    )
    reference_constrained=np.concatenate((
        reference_velocity_indices[
            reference_velocity_basis.get_dofs().all()
        ],
        reference_pressure_indices[:1],
    ))
    reference_solution=_solve(
        reference_matrix,reference_rhs,reference_constrained
    )
    permutation=_permutation(basis,reference_basis)
    np.testing.assert_allclose(
        solution,reference_solution[permutation],
        rtol=2e-10,atol=2e-12,
    )

    reference_velocity,_=reference_basis.interpolate(
        reference_solution
    )

    @skfem.Functional
    def reference_dissipation(w):
        return w.viscosity*reference_ddot(
            reference_grad(w.velocity),
            reference_grad(w.velocity),
        )

    expected_dissipation=skfem.asm(
        reference_dissipation,reference_basis,
        viscosity=.8,velocity=reference_velocity,
    )
    np.testing.assert_allclose(
        native_dissipation,expected_dissipation,
        rtol=2e-10,atol=2e-12,
    )
    assert np.linalg.norm(velocity.value)>0.
    assert abs(solution[pressure_indices[0]])==0.
