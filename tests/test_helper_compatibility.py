"""Parity for helper operations on interpolated fields."""

import numpy as np
import pytest
import skfem
from skfem.helpers import curl as reference_curl
from skfem.helpers import dot as reference_dot

import skfemntv
from skfemntv.helpers import curl
from skfemntv.helpers import dot


@pytest.mark.parametrize("dim,components",[(2,1),(2,2),(3,3)])
def test_curl_of_interpolated_fields_matches_scikit_fem(dim,components):
    if dim==2:
        native_mesh=skfemntv.MeshTri.init_tensor(
            np.linspace(0.,1.,3),np.linspace(0.,1.,3)
        )
        reference_mesh=skfem.MeshTri(native_mesh.p,native_mesh.t)
        scalar=skfemntv.ElementTriP1()
        reference_scalar=skfem.ElementTriP1()
    else:
        native_mesh=skfemntv.MeshTet.init_tensor(
            [0.,1.],[0.,1.],[0.,1.]
        )
        reference_mesh=skfem.MeshTet(native_mesh.p,native_mesh.t)
        scalar=skfemntv.ElementTetP1()
        reference_scalar=skfem.ElementTetP1()
    native_element=(
        scalar if components==1 else
        skfemntv.ElementVector(scalar,dim=components)
    )
    reference_element=(
        reference_scalar if components==1 else
        skfem.ElementVector(reference_scalar,dim=components)
    )
    native_basis=skfemntv.Basis(native_mesh,native_element,intorder=3)
    reference_basis=skfem.Basis(
        reference_mesh,reference_element,
        quadrature=(native_basis.X,native_basis.W),
    )
    rng=np.random.default_rng(927)
    coefficients=rng.normal(size=native_basis.N)
    native_field=native_basis.interpolate(coefficients)
    reference_field=reference_basis.interpolate(coefficients)

    np.testing.assert_allclose(
        curl(native_field),reference_curl(reference_field),rtol=0.,atol=0.
    )


def test_curl_rejects_unsupported_field_shapes():
    class Field:
        grad=np.zeros((3,4,2))

    with pytest.raises(NotImplementedError,match="supports scalar 2D"):
        curl(Field())


@pytest.mark.parametrize("dim",[2,3])
def test_curl_curl_bilinear_form_matches_scikit_fem(dim):
    if dim==2:
        native_mesh=skfemntv.MeshTri.init_tensor([0.,.5,1.],[0.,.5,1.])
        reference_mesh=skfem.MeshTri(native_mesh.p,native_mesh.t)
        native_element=skfemntv.ElementVector(skfemntv.ElementTriP1())
        reference_element=skfem.ElementVector(skfem.ElementTriP1())
    else:
        native_mesh=skfemntv.MeshTet.init_tensor([0.,1.],[0.,1.],[0.,1.])
        reference_mesh=skfem.MeshTet(native_mesh.p,native_mesh.t)
        native_element=skfemntv.ElementVector(skfemntv.ElementTetP1())
        reference_element=skfem.ElementVector(skfem.ElementTetP1())
    native_basis=skfemntv.Basis(native_mesh,native_element,intorder=3)
    reference_basis=skfem.Basis(
        reference_mesh,reference_element,
        quadrature=(native_basis.X,native_basis.W),
    )

    @skfemntv.BilinearForm
    def native_form(u,v,w):
        if dim==2:
            return curl(u)*curl(v)
        return dot(curl(u),curl(v))

    @skfem.BilinearForm
    def reference_form(u,v,w):
        if dim==2:
            return reference_curl(u)*reference_curl(v)
        return reference_dot(reference_curl(u),reference_curl(v))

    actual=skfemntv.asm(native_form,native_basis)
    expected=skfem.asm(reference_form,reference_basis)
    np.testing.assert_allclose(
        actual.toarray(),expected.toarray(),rtol=2e-12,atol=2e-13
    )
