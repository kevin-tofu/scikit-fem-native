import numpy as np
import skfem
from scipy.sparse import bmat
from skfem.helpers import dot as reference_dot
from skfem.helpers import sym_grad as reference_sym_grad
from skfem.models.elasticity import linear_stress

import skfemntv
from skfemntv.helpers import (
    avg,dot,grad,isotropic_traction_tensor,jump,normal_grad,
)


def _native_scalar_interface():
    mesh=skfemntv.MeshTet()
    basis=skfemntv.FacetBasis(
        mesh,skfemntv.ElementVector(skfemntv.ElementTetP1(),dim=1),
        intorder=4,
    )
    return mesh,basis,skfemntv.TriangleSupermesh.from_facets(basis,basis)


def test_poisson_symmetric_nitsche_matches_skfem_facet_integrals():
    mesh,basis,integration=_native_scalar_interface()
    penalty=7.0

    @skfemntv.BilinearForm
    def nitsche(u,v,w):
        return (
            penalty*dot(jump(u),jump(v))
            -dot(jump(v),avg(normal_grad(u)))
            -dot(jump(u),avg(normal_grad(v)))
        )

    actual=nitsche.assemble(basis,basis,integration=integration)
    reference_mesh=skfem.MeshTet(mesh.p,mesh.t)
    reference_basis=skfem.FacetBasis(
        reference_mesh,skfem.ElementTetP1(),intorder=4
    )

    @skfem.BilinearForm
    def mass(u,v,w):
        return u*v

    @skfem.BilinearForm
    def normal_flux(u,v,w):
        return v*reference_dot(u.grad,w.n)

    mass_matrix=skfem.asm(mass,reference_basis)
    flux_matrix=skfem.asm(normal_flux,reference_basis)
    diagonal=penalty*mass_matrix-.5*(flux_matrix+flux_matrix.T)
    expected=bmat(
        [[diagonal,-diagonal],[-diagonal,diagonal]],format="csr"
    )
    np.testing.assert_allclose(
        actual.toarray(),expected.toarray(),rtol=8e-13,atol=8e-13
    )


def test_elastic_nitsche_flux_matches_skfem_and_is_action_reaction():
    mesh=skfemntv.MeshTet()
    interface_facets=mesh.facets_satisfying(
        lambda x:np.isclose(x[2],0.),boundaries_only=True
    )
    basis=skfemntv.FacetBasis(
        mesh,skfemntv.ElementVector(skfemntv.ElementTetP1()),
        facets=interface_facets,intorder=4,
    )
    integration=skfemntv.TriangleSupermesh.from_facets(basis,basis)
    lame_lambda=2.0;lame_mu=3.0;penalty=11.0

    @skfemntv.BilinearForm
    def consistency(u,v,w):
        traction=isotropic_traction_tensor(
            w.n_master,lame_lambda,lame_mu
        )
        return dot(jump(v),dot(traction,avg(grad(u))))

    @skfemntv.BilinearForm
    def penalty_form(u,v,w):
        return penalty*dot(jump(u),jump(v))

    flux=consistency.assemble(basis,basis,integration=integration)
    actual=penalty_form.assemble(
        basis,basis,integration=integration
    )-flux-flux.T

    reference_mesh=skfem.MeshTet(mesh.p,mesh.t)
    reference_basis=skfem.FacetBasis(
        reference_mesh,skfem.ElementVector(skfem.ElementTetP1()),
        facets=interface_facets,intorder=4,
    )
    stress=linear_stress(Lambda=lame_lambda,Mu=lame_mu)

    @skfem.BilinearForm
    def reference_mass(u,v,w):
        return reference_dot(u,v)

    @skfem.BilinearForm
    def reference_flux(u,v,w):
        return reference_dot(
            v,reference_dot(stress(reference_sym_grad(u)),w.n)
        )

    mass_matrix=skfem.asm(reference_mass,reference_basis)
    flux_matrix=skfem.asm(reference_flux,reference_basis)
    consistency_reference=bmat([
        [.5*flux_matrix,.5*flux_matrix],
        [-.5*flux_matrix,-.5*flux_matrix],
    ],format="csr")
    penalty_reference=bmat([
        [penalty*mass_matrix,-penalty*mass_matrix],
        [-penalty*mass_matrix,penalty*mass_matrix],
    ],format="csr")
    expected=penalty_reference-consistency_reference-consistency_reference.T
    np.testing.assert_allclose(
        actual.toarray(),expected.toarray(),rtol=9e-13,atol=9e-13
    )
    linear_displacement=mesh.p.T.reshape(-1)
    resultant=(
        flux@np.concatenate((linear_displacement,linear_displacement))
    ).reshape(2,-1,3).sum(axis=1)
    np.testing.assert_allclose(resultant[0],-resultant[1],atol=2e-13)
    assert np.linalg.norm(resultant[0])>.1


def _refined_triangle(level):
    nodes=[];lookup={}
    for j in range(level+1):
        for i in range(level+1-j):
            lookup[i,j]=len(nodes);nodes.append((i/level,j/level,0.))
    triangles=[]
    for j in range(level):
        for i in range(level-j):
            triangles.append((lookup[i,j],lookup[i+1,j],lookup[i,j+1]))
            if i+j<level-1:
                triangles.append((
                    lookup[i+1,j],lookup[i+1,j+1],lookup[i,j+1]
                ))
    return np.asarray(nodes).T,np.asarray(triangles,dtype=np.int64).T


def test_nonmatching_p1_trace_refinement_converges_for_quadratic_field():
    master_points=np.array([[0.,1.,0.],[0.,0.,1.],[0.,0.,0.]])
    master_triangles=np.array([[0],[1],[2]])
    exact=np.array([1./60.,1./20.,1./60.])
    errors=[]
    for level in (1,2,4,8):
        slave_points,slave_triangles=_refined_triangle(level)
        integration=skfemntv.TriangleSupermesh(
            master_points,master_triangles,slave_points,slave_triangles
        )
        interpolated=integration.assemble()@(slave_points[0]**2)
        errors.append(np.linalg.norm(interpolated-exact))
        np.testing.assert_allclose(
            integration.assemble()@np.ones(slave_points.shape[1]),
            np.full(3,1./6.),rtol=3e-13,atol=3e-13,
        )
    assert all(new<.35*old for old,new in zip(errors,errors[1:]))
