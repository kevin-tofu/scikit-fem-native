"""Parity checks for the second-order triangular Nedelec element."""

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
        reference_mesh,skfem.ElementTriN2(),
        quadrature=(basis.X,basis.W),
    )


def test_trin2_basis_dofs_orientation_and_interpolation_match_skfem():
    mesh=_mesh()
    basis=skfemntv.Basis(mesh,skfemntv.ElementTriN2(),intorder=4)
    reference=_reference(mesh,basis)
    assert basis.N==reference.N
    np.testing.assert_array_equal(basis.element_dofs,reference.element_dofs)
    np.testing.assert_allclose(basis.doflocs,reference.doflocs,atol=2e-15)
    np.testing.assert_allclose(basis.dx,reference.dx,atol=2e-15)
    for name in (None,"left","bottom"):
        actual=basis.get_dofs(name).all()
        expected=reference.get_dofs(name).all()
        np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(
        basis.get_dofs().all("u^t"),reference.get_dofs().all("u^t")
    )

    coefficients=np.linspace(-.4,.8,basis.N)
    actual=basis.interpolate(coefficients)
    expected=reference.interpolate(coefficients)
    np.testing.assert_allclose(actual.value,np.asarray(expected),atol=3e-14)
    np.testing.assert_allclose(actual.curl,expected.curl,atol=3e-14)


def test_trin2_public_asm_and_reusable_assemblers_match_skfem():
    mesh=_mesh()
    basis=skfemntv.Basis(mesh,skfemntv.ElementTriN2(),intorder=4)
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
        rtol=4e-13,atol=4e-13,
    )
    direct=skfemntv.TriN2Assembler(basis._hcurl_basis)
    np.testing.assert_allclose(
        direct.assemble_maxwell().toarray(),
        skfem.asm(
            _unit_maxwell_reference(),reference
        ).toarray(),rtol=4e-13,atol=4e-13,
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
        rtol=3e-13,atol=3e-13,
    )
    direct_load=skfemntv.TriN2LinearAssembler(basis._hcurl_basis)
    np.testing.assert_allclose(
        direct_load.assemble_vector_load(force),expected_load,
        rtol=3e-13,atol=3e-13,
    )


def _unit_maxwell_reference():
    @skfem.BilinearForm
    def form(u,v,w):
        return sk_dot(u,v)+sk_curl(u)*sk_curl(v)
    return form


def test_trin2_projection_point_evaluation_and_refinterp_match_skfem():
    mesh=_mesh()
    basis=skfemntv.Basis(mesh,skfemntv.ElementTriN2(),intorder=4)
    reference=_reference(mesh,basis)
    field=lambda x:np.array((1.+x[0],-.3+x[1]))
    np.testing.assert_allclose(
        basis.project(field),reference.project(field),rtol=3e-12,atol=3e-12,
    )
    coefficients=np.linspace(-.7,.9,basis.N)
    points=np.array(((.17,.73,.42),(.21,.64,.83)))
    actual=basis.evaluate_at(points,coefficients)
    expected=reference.interpolator(coefficients)(points)
    np.testing.assert_allclose(actual.value,expected,rtol=3e-12,atol=3e-12)
    np.testing.assert_allclose(
        (basis.probes(points)@coefficients).reshape(2,-1),expected,
        rtol=3e-12,atol=3e-12,
    )
    step=1.e-6
    expected_gradient=np.stack([
        (reference.interpolator(coefficients)(
            points+np.eye(2)[:,[axis]]*step
         )-reference.interpolator(coefficients)(
            points-np.eye(2)[:,[axis]]*step
         ))/(2.*step)
        for axis in range(2)
    ],axis=1)
    np.testing.assert_allclose(
        actual.grad,expected_gradient,rtol=2e-8,atol=2e-8,
    )
    actual_mesh,actual_values=basis.refinterp(coefficients,nrefs=2)
    expected_mesh,expected_values=reference.refinterp(coefficients,nrefs=2)
    np.testing.assert_allclose(actual_mesh.p,expected_mesh.p,atol=3e-14)
    np.testing.assert_array_equal(actual_mesh.t,expected_mesh.t)
    np.testing.assert_allclose(actual_values,expected_values,atol=3e-13)
