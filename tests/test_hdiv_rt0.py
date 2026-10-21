import numpy as np
import pytest
import skfem

import skfemntv
from skfem.helpers import div as sk_div
from skfem.helpers import dot as sk_dot
from skfemntv.helpers import div as native_div
from skfemntv.helpers import dot as native_dot


def test_hdiv_rt0_capabilities_are_declared_experimental():
    assert not skfemntv.supports("space.hdiv")
    assert skfemntv.supports("space.hdiv",include_experimental=True)
    assert skfemntv.get_compatibility("skfem.ElementTriRT0").native_name==(
        "skfemntv.ElementTriRT0"
    )
    assert skfemntv.get_compatibility("skfem.ElementTetRT0").status is (
        skfemntv.CapabilityStatus.EXPERIMENTAL
    )


def make_mesh(family):
    if family=="tri":
        return skfemntv.MeshTri.init_tensor([0.,1.],[0.,1.])
    return skfemntv.MeshTet.init_tensor([0.,1.],[0.,1.],[0.,1.])


def make_reference(mesh,family,quadrature):
    if family=="tri":
        return skfem.Basis(
            skfem.MeshTri(mesh.p,mesh.t),skfem.ElementTriRT0(),
            quadrature=quadrature,
        )
    return skfem.Basis(
        skfem.MeshTet(mesh.p,mesh.t),skfem.ElementTetRT0(),
        quadrature=quadrature,
    )


@pytest.mark.parametrize("family",["tri","tet"])
def test_rt0_basis_dofs_values_and_divergence_match_skfem(family):
    mesh=make_mesh(family)
    element=(skfemntv.ElementTriRT0() if family=="tri"
             else skfemntv.ElementTetRT0())
    basis=skfemntv.Basis(mesh,element,intorder=5)
    reference=make_reference(mesh,family,basis.quadrature)
    assert np.array_equal(basis.element_dofs,reference.element_dofs)
    np.testing.assert_allclose(basis.doflocs,reference.doflocs,atol=2e-14)
    assert np.array_equal(
        basis.dof_map.basis_signs,
        np.where(
            mesh.f2t[0,basis.element_dofs]==np.arange(mesh.nelements)[None,:],
            1.,-1.,
        ),
    )
    coefficients=np.random.default_rng(510).normal(size=basis.N)
    actual=basis.interpolate(coefficients)
    expected=reference.interpolate(coefficients)
    np.testing.assert_allclose(actual.value,expected.value,rtol=2e-14,atol=2e-14)
    np.testing.assert_allclose(actual.div,expected.div,rtol=2e-14,atol=2e-14)
    assert np.array_equal(basis._hdiv_basis.boundary_dofs(),mesh.boundary_facets())
    assert np.array_equal(
        basis.get_dofs().all(),reference.get_dofs().all()
    )


@pytest.mark.parametrize("family",["tri","tet"])
def test_rt0_mass_divergence_and_vector_load_match_skfem(family):
    mesh=make_mesh(family)
    element=(skfemntv.ElementTriRT0() if family=="tri"
             else skfemntv.ElementTetRT0())
    basis=skfemntv.Basis(mesh,element,intorder=5)
    reference=make_reference(mesh,family,basis.quadrature)

    @skfem.BilinearForm
    def sk_mass(u,v,w):
        return sk_dot(u,v)

    @skfemntv.BilinearForm
    def native_mass(u,v,w):
        return native_dot(u,v)

    @skfem.BilinearForm
    def sk_divergence(u,v,w):
        return sk_div(u)*sk_div(v)

    @skfemntv.BilinearForm
    def native_divergence(u,v,w):
        return native_div(u)*native_div(v)

    np.testing.assert_allclose(
        skfem.asm(sk_mass,reference).toarray(),
        skfemntv.asm(native_mass,basis).toarray(),
        rtol=2e-13,atol=2e-13,
    )
    np.testing.assert_allclose(
        skfem.asm(sk_divergence,reference).toarray(),
        skfemntv.asm(native_divergence,basis).toarray(),
        rtol=2e-13,atol=2e-13,
    )

    direction=np.arange(1,mesh.dim()+1,dtype=float)

    @skfem.LinearForm
    def sk_load(v,w):
        return sk_dot(direction,v)

    @skfemntv.LinearForm
    def native_load(v,w):
        return native_dot(direction,v)

    np.testing.assert_allclose(
        skfem.asm(sk_load,reference),skfemntv.asm(native_load,basis),
        rtol=2e-13,atol=2e-13,
    )
