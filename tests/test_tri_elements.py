import numpy as np
import pytest
import skfem
from scipy.sparse.linalg import spsolve
from skfem.helpers import ddot as reference_ddot
from skfem.helpers import div as reference_div
from skfem.helpers import dot as reference_dot
from skfem.helpers import grad as reference_grad

import skfn
from skfn.helpers import ddot,div,dot,grad


def _reference_tri_mesh(mesh,quadratic):
    if not quadratic:
        return skfem.MeshTri(mesh.p,mesh.t)
    vertex_count=len(np.unique(mesh.t[:3]))
    linear=skfem.MeshTri(
        mesh.p[:,:vertex_count],mesh.t[:3]
    )
    return skfem.MeshTri2.from_mesh(linear)


def _coordinate_permutation(native,reference,components):
    lookup={}
    for dof in range(reference.N):
        coordinate=tuple(np.round(reference.doflocs[:,dof],14))
        lookup.setdefault(coordinate,[]).append(dof)
    permutation=np.empty(native.N,dtype=np.int64)
    for dof in range(native.N):
        coordinate=tuple(np.round(native.doflocs[:,dof],14))
        permutation[dof]=lookup[coordinate][dof%components]
    return permutation


def _composite_permutation(native,reference):
    permutation=np.empty(native.N,dtype=np.int64)
    for native_dofs,reference_dofs,components in zip(
        native.split_indices(),reference.split_indices(),(2,1)
    ):
        lookup={}
        for dof in reference_dofs:
            coordinate=tuple(np.round(reference.doflocs[:,dof],14))
            lookup.setdefault(coordinate,[]).append(int(dof))
        for offset,dof in enumerate(native_dofs):
            coordinate=tuple(np.round(native.doflocs[:,dof],14))
            permutation[dof]=lookup[coordinate][offset%components]
    return permutation


@pytest.mark.parametrize(
    "mesh,element,reference_mesh,reference_element,intorder",
    [
        (
            skfn.MeshTri(),skfn.ElementTriP1(),
            skfem.MeshTri,skfem.ElementTriP1,2,
        ),
        (
            skfn.MeshTri2(),skfn.ElementTriP2(),
            skfem.MeshTri2,skfem.ElementTriP2,4,
        ),
    ],
)
def test_tri_volume_forms_match_skfem(
    mesh,element,reference_mesh,reference_element,intorder
):
    basis=skfn.Basis(
        mesh,skfn.ElementVector(element,dim=1),intorder=intorder
    )

    @skfn.BilinearForm
    def form(u,v,w):
        return (
            (1.+w.x[0])*dot(u,v)
            +.7*ddot(grad(u),grad(v))
        )

    actual=skfn.asm(form,basis)
    reference_basis=skfem.Basis(
        _reference_tri_mesh(mesh,intorder>=4),reference_element(),
        intorder=intorder,
    )

    @skfem.BilinearForm
    def reference(u,v,w):
        return (
            (1.+w.x[0])*u*v
            +.7*reference_dot(reference_grad(u),reference_grad(v))
        )

    expected=skfem.asm(reference,reference_basis)
    permutation=_coordinate_permutation(basis,reference_basis,1)
    expected=expected[permutation][:,permutation]
    np.testing.assert_allclose(
        actual.toarray(),expected.toarray(),rtol=2e-12,atol=2e-12
    )


@pytest.mark.parametrize(
    "mesh,element,reference_mesh,reference_element,intorder",
    [
        (
            skfn.MeshTri(),skfn.ElementTriP1(),
            skfem.MeshTri,skfem.ElementTriP1,2,
        ),
        (
            skfn.MeshTri2(),skfn.ElementTriP2(),
            skfem.MeshTri2,skfem.ElementTriP2,4,
        ),
    ],
)
def test_tri_facet_linear_and_functional_match_skfem(
    mesh,element,reference_mesh,reference_element,intorder
):
    basis=skfn.FacetBasis(
        mesh,skfn.ElementVector(element),intorder=intorder
    )

    @skfn.LinearForm
    def pressure(v,w):
        return dot((1.+w.x[0])*w.n,v)

    actual=skfn.asm(pressure,basis)

    @skfn.Functional
    def boundary_measure(w):
        return 1.+w.x[1]+w.n[0]**2

    actual_measure=skfn.asm(boundary_measure,basis)
    reference_basis=skfem.FacetBasis(
        _reference_tri_mesh(mesh,intorder>=4),
        skfem.ElementVector(reference_element()),
        intorder=intorder,
    )

    @skfem.LinearForm
    def reference_pressure(v,w):
        return reference_dot((1.+w.x[0])*w.n,v)

    @skfem.Functional
    def reference_measure(w):
        return 1.+w.x[1]+w.n[0]**2

    expected=skfem.asm(reference_pressure,reference_basis)
    expected_measure=skfem.asm(reference_measure,reference_basis)
    permutation=_coordinate_permutation(basis,reference_basis,2)
    expected=expected[permutation]
    np.testing.assert_allclose(actual,expected,rtol=2e-12,atol=2e-12)
    np.testing.assert_allclose(
        actual_measure,expected_measure,rtol=2e-12,atol=2e-12
    )


def test_tri_taylor_hood_blocks_match_skfem():
    linear=skfn.MeshTri.init_tensor(
        np.linspace(0.,1.,3),np.linspace(0.,1.,3)
    )
    mesh=skfn.MeshTri2.from_mesh(linear)
    basis=skfn.Basis(
        mesh,
        skfn.ElementVector(skfn.ElementTriP2())
        *skfn.ElementTriP1(),
        intorder=4,
    )

    @skfn.BilinearForm
    def stokes(u,p,v,q,w):
        return (
            ddot(grad(u),grad(v))
            -p*div(v)-q*div(u)
        )

    actual=skfn.asm(stokes,basis)
    reference_mesh=skfem.MeshTri2.from_mesh(
        skfem.MeshTri(linear.p,linear.t)
    )
    reference_basis=skfem.Basis(
        reference_mesh,
        skfem.ElementVector(skfem.ElementTriP2())
        *skfem.ElementTriP1(),
        intorder=4,
    )

    @skfem.BilinearForm
    def reference(u,p,v,q,w):
        return (
            reference_ddot(reference_grad(u),reference_grad(v))
            -p*reference_div(v)-q*reference_div(u)
        )

    expected=skfem.asm(reference,reference_basis)
    permutation=_composite_permutation(basis,reference_basis)
    expected=expected[permutation][:,permutation]
    np.testing.assert_allclose(
        actual.toarray(),expected.toarray(),rtol=2e-12,atol=2e-12
    )


def test_tri_p1_facet_on_quadratic_geometry_matches_skfem():
    linear=skfn.MeshTri.init_tensor(
        np.linspace(0.,1.,3),np.linspace(0.,1.,3)
    )
    mesh=skfn.MeshTri2.from_mesh(linear)
    basis=skfn.FacetBasis(
        mesh,skfn.ElementVector(skfn.ElementTriP1()),intorder=4
    )

    @skfn.LinearForm
    def form(v,w):
        return dot((1.+w.x[0])*w.n,v)

    actual=skfn.asm(form,basis)
    reference_mesh=skfem.MeshTri2.from_mesh(
        skfem.MeshTri(linear.p,linear.t)
    )
    reference_basis=skfem.FacetBasis(
        reference_mesh,
        skfem.ElementVector(skfem.ElementTriP1()),
        intorder=4,
    )

    @skfem.LinearForm
    def reference(v,w):
        return reference_dot((1.+w.x[0])*w.n,v)

    expected=skfem.asm(reference,reference_basis)
    permutation=_coordinate_permutation(basis,reference_basis,2)
    np.testing.assert_allclose(
        actual,expected[permutation],rtol=2e-12,atol=2e-12
    )


def test_tri_taylor_hood_solve_and_interpolation_match_skfem():
    axis=np.linspace(0.,1.,4)
    linear=skfn.MeshTri.init_tensor(axis,axis)
    mesh=skfn.MeshTri2.from_mesh(linear)
    basis=skfn.Basis(
        mesh,
        skfn.ElementVector(skfn.ElementTriP2())
        *skfn.ElementTriP1(),
        intorder=4,
    )

    @skfn.BilinearForm
    def stokes(u,p,v,q,w):
        return ddot(grad(u),grad(v))-p*div(v)-q*div(u)

    @skfn.LinearForm
    def load(v,q,w):
        return dot(w.force,v)

    matrix=skfn.asm(stokes,basis)
    x=np.moveaxis(basis.global_coordinates,-1,0)
    rhs=skfn.asm(load,basis,force=np.stack((x[1],-x[0])))
    velocity_basis,_=basis.split_bases()
    velocity_indices,pressure_indices=basis.split_indices()
    constrained=np.concatenate((
        velocity_indices[velocity_basis.get_dofs().all()],
        pressure_indices[:1],
    ))
    free=np.setdiff1d(np.arange(basis.N),constrained)
    solution=np.zeros(basis.N)
    solution[free]=spsolve(matrix[free][:,free],rhs[free])
    velocity,pressure=basis.interpolate(solution)
    assert velocity.grad.shape[:2]==(2,2)
    assert pressure.grad.shape[0]==2

    reference_mesh=skfem.MeshTri2.from_mesh(
        skfem.MeshTri(linear.p,linear.t)
    )
    reference_basis=skfem.Basis(
        reference_mesh,
        skfem.ElementVector(skfem.ElementTriP2())
        *skfem.ElementTriP1(),
        intorder=4,
    )

    @skfem.BilinearForm
    def reference_stokes(u,p,v,q,w):
        return (
            reference_ddot(reference_grad(u),reference_grad(v))
            -p*reference_div(v)-q*reference_div(u)
        )

    @skfem.LinearForm
    def reference_load(v,q,w):
        return reference_dot(w.force,v)

    reference_matrix=skfem.asm(reference_stokes,reference_basis)
    reference_x=np.asarray(reference_basis.global_coordinates())
    reference_rhs=skfem.asm(
        reference_load,reference_basis,
        force=np.stack((reference_x[1],-reference_x[0])),
    )
    reference_velocity,_=reference_basis.split_bases()
    reference_velocity_indices,reference_pressure_indices=(
        reference_basis.split_indices()
    )
    reference_constrained=np.concatenate((
        reference_velocity_indices[reference_velocity.get_dofs().all()],
        reference_pressure_indices[:1],
    ))
    reference_free=np.setdiff1d(
        np.arange(reference_basis.N),reference_constrained
    )
    reference_solution=np.zeros(reference_basis.N)
    reference_solution[reference_free]=spsolve(
        reference_matrix[reference_free][:,reference_free],
        reference_rhs[reference_free],
    )
    permutation=_composite_permutation(basis,reference_basis)
    np.testing.assert_allclose(
        solution,reference_solution[permutation],
        rtol=2e-10,atol=2e-12,
    )
