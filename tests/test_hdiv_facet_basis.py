import numpy as np
import pytest
import skfem
from skfem.helpers import dot as reference_dot

import skfemntv
from skfemntv.helpers import dot


@pytest.mark.parametrize("family", ["tri", "tet"])
def test_rt0_facet_trace_and_boundary_linear_form_match_scikit_fem(family):
    if family == "tri":
        mesh=skfemntv.MeshTri.init_tensor(
            [0.,.5,1.],[0.,.5,1.]
        )
        reference_mesh=skfem.MeshTri(mesh.p,mesh.t)
        element=skfemntv.ElementTriRT0()
        reference_element=skfem.ElementTriRT0()
    else:
        mesh=skfemntv.MeshTet.init_tensor([0.,1.],[0.,1.],[0.,1.])
        reference_mesh=skfem.MeshTet(mesh.p,mesh.t)
        element=skfemntv.ElementTetRT0()
        reference_element=skfem.ElementTetRT0()

    basis=skfemntv.FacetBasis(mesh,element,intorder=4)
    reference=skfem.FacetBasis(
        reference_mesh,reference_element,facets=reference_mesh.boundary_facets(),
        quadrature=basis.quadrature,
    )
    assert np.array_equal(basis.facet_ids,reference.find)
    np.testing.assert_allclose(basis.dx,reference.dx,rtol=2e-14,atol=2e-14)
    np.testing.assert_allclose(
        basis.global_coordinates,
        np.moveaxis(reference.global_coordinates(),0,-1),
        rtol=2e-14,atol=2e-14,
    )
    np.testing.assert_allclose(
        basis.normals,np.moveaxis(reference.normals,0,-1),
        rtol=2e-14,atol=2e-14
    )

    coefficients=np.random.default_rng(934).normal(size=basis.N)
    actual_field=basis.interpolate(coefficients)
    reference_field=reference.interpolate(coefficients)
    np.testing.assert_allclose(
        actual_field.value,reference_field.value,rtol=3e-14,atol=3e-14
    )
    np.testing.assert_allclose(
        actual_field.div,reference_field.div,rtol=3e-14,atol=3e-14
    )
    expected_trace=np.einsum(
        "dfq,dfq->fq",reference_field.value,reference.normals
    )
    np.testing.assert_allclose(
        basis.normal_trace(coefficients),expected_trace,
        rtol=3e-14,atol=3e-14,
    )

    @skfemntv.LinearForm
    def boundary_load(v,w):
        return dot(v,(1.+.2*w.x[0])*w.n)

    @skfem.LinearForm
    def reference_boundary_load(v,w):
        return reference_dot(v,(1.+.2*w.x[0])*w.n)

    @skfemntv.BilinearForm
    def boundary_normal_mass(u,v,w):
        return (1.+.2*w.x[0])*dot(u,w.n)*dot(v,w.n)

    @skfem.BilinearForm
    def reference_boundary_normal_mass(u,v,w):
        return (1.+.2*w.x[0])*reference_dot(u,w.n)*reference_dot(v,w.n)

    actual=skfemntv.asm(boundary_load,basis)
    expected=skfem.asm(reference_boundary_load,reference)
    np.testing.assert_allclose(actual,expected,rtol=4e-13,atol=4e-13)
    actual_matrix=skfemntv.asm(boundary_normal_mass,basis)
    expected_matrix=skfem.asm(reference_boundary_normal_mass,reference)
    np.testing.assert_allclose(
        actual_matrix.toarray(),expected_matrix.toarray(),
        rtol=4e-13,atol=4e-13,
    )


@pytest.mark.parametrize("family", ["tri", "tet"])
@pytest.mark.parametrize("side", [0, 1])
def test_rt0_interior_facet_side_matches_scikit_fem(family, side):
    if family == "tri":
        mesh=skfemntv.MeshTri.init_tensor([0.,.5,1.],[0.,.5,1.])
        reference_mesh=skfem.MeshTri(mesh.p,mesh.t)
        element=skfemntv.ElementTriRT0()
        reference_element=skfem.ElementTriRT0()
    else:
        mesh=skfemntv.MeshTet.init_tensor([0.,1.],[0.,1.],[0.,1.])
        reference_mesh=skfem.MeshTet(mesh.p,mesh.t)
        element=skfemntv.ElementTetRT0()
        reference_element=skfem.ElementTetRT0()

    basis=skfemntv.InteriorFacetBasis(
        mesh,element,side=side,intorder=4
    )
    reference=skfem.InteriorFacetBasis(
        reference_mesh,reference_element,side=side,
        quadrature=basis.quadrature,
    )
    assert np.array_equal(basis.facet_ids,reference.find)
    np.testing.assert_allclose(basis.dx,reference.dx,rtol=2e-14,atol=2e-14)
    np.testing.assert_allclose(
        basis.global_coordinates,
        np.moveaxis(reference.global_coordinates(),0,-1),
        rtol=2e-14,atol=2e-14,
    )
    np.testing.assert_allclose(
        basis.normals,np.moveaxis(reference.normals,0,-1),
        rtol=2e-14,atol=2e-14,
    )

    coefficients=np.random.default_rng(213+side).normal(size=basis.N)
    actual_field=basis.interpolate(coefficients)
    reference_field=reference.interpolate(coefficients)
    np.testing.assert_allclose(
        actual_field.value,reference_field.value,rtol=3e-14,atol=3e-14
    )
    np.testing.assert_allclose(
        actual_field.div,reference_field.div,rtol=3e-14,atol=3e-14
    )
    expected_trace=np.einsum(
        "dfq,dfq->fq",reference_field.value,reference.normals
    )
    np.testing.assert_allclose(
        basis.normal_trace(coefficients),expected_trace,
        rtol=3e-14,atol=3e-14,
    )

    @skfemntv.LinearForm
    def interior_flux(v,w):
        return dot(v,(1.+.2*w.x[0])*w.n)

    @skfem.LinearForm
    def reference_interior_flux(v,w):
        return reference_dot(v,(1.+.2*w.x[0])*w.n)

    np.testing.assert_allclose(
        skfemntv.asm(interior_flux,basis),
        skfem.asm(reference_interior_flux,reference),
        rtol=4e-13,atol=4e-13,
    )
