"""Parity checks for the third-order triangular Nedelec element."""

import numpy as np
import skfem

import skfemntv
from skfem.helpers import curl as sk_curl
from skfem.helpers import dot as sk_dot
from skfemntv.helpers import curl,dot


def _mesh():
    return skfemntv.MeshTri.init_tensor(
        np.linspace(0.,1.,3),np.linspace(0.,1.,3),
    ).with_boundaries({
        "left":lambda x:np.isclose(x[0],0.),
        "bottom":lambda x:np.isclose(x[1],0.),
    })


def _reference(mesh,basis):
    reference_mesh=skfem.MeshTri(mesh.p,mesh.t[:3]).with_boundaries({
        "left":lambda x:np.isclose(x[0],0.),
        "bottom":lambda x:np.isclose(x[1],0.),
    })
    return skfem.Basis(
        reference_mesh,skfem.ElementTriN3(),
        quadrature=(basis.X,basis.W),
    )


def _evaluate_reference_at(basis,coefficients,points):
    values=np.empty((2,points.shape[1]))
    curls=np.empty(points.shape[1])
    vertices=basis.mesh.p[:,basis.mesh.t[:3]]
    for point_id,point in enumerate(points.T):
        found=False
        for cell in range(basis.mesh.nelements):
            jacobian=vertices[:,1:,cell]-vertices[:,:1,cell]
            reference=np.linalg.solve(jacobian,point-vertices[:,0,cell])
            barycentric=np.r_[1.-reference.sum(),reference]
            if np.min(barycentric)>=-1.e-12:
                found=True
                break
        assert found
        local=np.zeros(2)
        local_curl=0.
        for dof in range(basis.element_dofs.shape[0]):
            field=basis.elem.gbasis(
                basis.mapping,reference[:,None],dof,
                tind=np.asarray([cell]),
            )[0]
            coefficient=coefficients[basis.element_dofs[dof,cell]]
            local+=coefficient*np.asarray(field.value)[:,0,0]
            local_curl+=coefficient*np.asarray(field.curl).reshape(-1)[0]
        values[:,point_id]=local
        curls[point_id]=local_curl
    return values,curls


def test_trin3_dofs_orientation_boundary_and_interpolation_match_skfem():
    mesh=_mesh()
    basis=skfemntv.Basis(mesh,skfemntv.ElementTriN3(),intorder=8)
    reference=_reference(mesh,basis)
    assert basis.N==reference.N
    assert np.any(basis._hcurl_basis.basis_signs[:9]<0.)
    assert np.any(
        basis._hcurl_basis.basis_permutation[:9]
        !=np.arange(9,dtype=np.int64)[:,None]
    )
    np.testing.assert_array_equal(basis.element_dofs,reference.element_dofs)
    np.testing.assert_allclose(basis.doflocs,reference.doflocs,atol=2e-15)
    np.testing.assert_allclose(basis.dx,reference.dx,atol=2e-15)
    for name in (None,"left","bottom"):
        np.testing.assert_array_equal(
            basis.get_dofs(name).all(),reference.get_dofs(name).all()
        )
    np.testing.assert_array_equal(
        basis.get_dofs().all("u^t"),reference.get_dofs().all("u^t")
    )
    coefficients=np.linspace(-.4,.8,basis.N)
    actual=basis.interpolate(coefficients)
    expected=reference.interpolate(coefficients)
    np.testing.assert_allclose(actual.value,np.asarray(expected),atol=5e-14)
    np.testing.assert_allclose(actual.curl,expected.curl,atol=5e-14)


def test_trin3_public_and_reusable_assembly_match_skfem():
    mesh=_mesh()
    basis=skfemntv.Basis(mesh,skfemntv.ElementTriN3(),intorder=8)
    reference=_reference(mesh,basis)

    @skfemntv.BilinearForm
    def native_form(u,v,w):
        return (1.+.2*w.x[0])*dot(u,v)+(.5+.3*w.x[1])*curl(u)*curl(v)

    @skfem.BilinearForm
    def reference_form(u,v,w):
        return (1.+.2*w.x[0])*sk_dot(u,v)+(.5+.3*w.x[1])*sk_curl(u)*sk_curl(v)

    expected=skfem.asm(reference_form,reference).toarray()
    np.testing.assert_allclose(
        skfemntv.asm(native_form,basis).toarray(),expected,
        rtol=5e-13,atol=5e-13,
    )
    direct=skfemntv.TriN3Assembler(basis._hcurl_basis)
    np.testing.assert_allclose(
        direct.assemble_maxwell().toarray(),
        skfem.asm(_unit_maxwell_reference(),reference).toarray(),
        rtol=5e-13,atol=5e-13,
    )
    force=np.stack((1.+basis.global_coordinates[...,0],
                    -.25+basis.global_coordinates[...,1]))

    @skfemntv.LinearForm
    def native_load(v,w):
        return dot(w.force,v)

    @skfem.LinearForm
    def reference_load(v,w):
        return sk_dot(w.force,v)

    expected_load=skfem.asm(reference_load,reference,force=force)
    np.testing.assert_allclose(
        skfemntv.asm(native_load,basis,force=force),expected_load,
        rtol=5e-13,atol=5e-13,
    )
    np.testing.assert_allclose(
        skfemntv.TriN3LinearAssembler(basis._hcurl_basis)
        .assemble_vector_load(force),expected_load,rtol=5e-13,atol=5e-13,
    )


def _unit_maxwell_reference():
    @skfem.BilinearForm
    def form(u,v,w):
        return sk_dot(u,v)+sk_curl(u)*sk_curl(v)
    return form


def test_trin3_point_evaluation_gradient_projection_and_refinterp():
    mesh=_mesh()
    basis=skfemntv.Basis(mesh,skfemntv.ElementTriN3(),intorder=8)
    reference=_reference(mesh,basis)
    field=lambda x:np.array((1.+x[0],-.3+x[1]))
    np.testing.assert_allclose(
        basis.project(field),reference.project(field),rtol=5e-12,atol=5e-12,
    )
    coefficients=np.linspace(-.7,.9,basis.N)
    points=np.array(((.17,.73,.42),(.21,.64,.83)))
    actual=basis.evaluate_at(points,coefficients)
    expected,expected_curl=_evaluate_reference_at(reference,coefficients,points)
    np.testing.assert_allclose(actual.value,expected,rtol=5e-12,atol=5e-12)
    np.testing.assert_allclose(actual.curl,expected_curl,rtol=5e-12,atol=5e-12)
    np.testing.assert_allclose(
        (basis.probes(points)@coefficients).reshape(2,-1),expected,
        rtol=5e-12,atol=5e-12,
    )
    step=1.e-6
    expected_gradient=np.stack([
        (_evaluate_reference_at(
            reference,coefficients,points+np.eye(2)[:,[axis]]*step
         )[0]-_evaluate_reference_at(
            reference,coefficients,points-np.eye(2)[:,[axis]]*step
         )[0])/(2.*step)
        for axis in range(2)
    ],axis=1)
    np.testing.assert_allclose(
        actual.grad,expected_gradient,rtol=3e-8,atol=3e-8,
    )
    actual_mesh,actual_values=basis.refinterp(coefficients,nrefs=2)
    expected_mesh,expected_values=reference.refinterp(coefficients,nrefs=2)
    np.testing.assert_allclose(actual_mesh.p,expected_mesh.p,atol=3e-14)
    np.testing.assert_array_equal(actual_mesh.t,expected_mesh.t)
    np.testing.assert_allclose(actual_values,expected_values,atol=5e-13)
