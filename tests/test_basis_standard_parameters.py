import numpy as np
import pytest
import skfem

import skfemntv


@pytest.mark.parametrize("dimension",[2,3])
def test_default_parameters_match_scikit_fem(dimension):
    if dimension==2:
        native_mesh=skfemntv.MeshTri()
        reference_mesh=skfem.MeshTri(native_mesh.p,native_mesh.t)
        native_element=skfemntv.ElementTriP1()
        reference_element=skfem.ElementTriP1()
    else:
        native_mesh=skfemntv.MeshTet()
        reference_mesh=skfem.MeshTet(native_mesh.p,native_mesh.t)
        native_element=skfemntv.ElementTetP1()
        reference_element=skfem.ElementTetP1()
    native=skfemntv.Basis(native_mesh,native_element,intorder=2)
    reference=skfem.Basis(reference_mesh,reference_element,intorder=2)

    actual=native.default_parameters()
    expected=reference.default_parameters()
    assert set(actual)==set(expected)=={"x","h"}
    np.testing.assert_allclose(actual["x"],native.global_coordinates())
    actual_x=np.asarray(actual["x"])
    expected_x=np.asarray(expected["x"])
    for element in range(actual_x.shape[1]):
        actual_points=actual_x[:,element,:]
        expected_points=expected_x[:,element,:]
        actual_order=np.lexsort(actual_points[::-1])
        expected_order=np.lexsort(expected_points[::-1])
        np.testing.assert_allclose(
            actual_points[:,actual_order],expected_points[:,expected_order]
        )
    np.testing.assert_allclose(
        np.sort(np.asarray(actual["h"]),axis=1),
        np.sort(np.asarray(expected["h"]),axis=1),
    )
    np.testing.assert_allclose(actual["h"],native.mesh_parameters())


def test_basis_boundary_matches_direct_facet_basis():
    mesh=skfemntv.MeshTri()
    basis=skfemntv.Basis(mesh,skfemntv.ElementTriP1(),intorder=4)
    actual=basis.boundary(intorder=4)
    expected=skfemntv.FacetBasis(
        mesh,basis.elem,facets=mesh.boundary_facets(),intorder=4
    )
    np.testing.assert_allclose(actual.dx,expected.dx)
    np.testing.assert_allclose(
        actual.global_coordinates,expected.global_coordinates
    )
    assert actual.global_coordinates().shape==(2,actual.dx.shape[0],actual.dx.shape[1])


def test_boundary_rejects_a_basis_restricted_to_a_subdomain():
    mesh=skfemntv.MeshTri().refined()
    basis=skfemntv.Basis(
        mesh,skfemntv.ElementTriP1(),elements=[0]
    )
    with pytest.raises(NotImplementedError,match="Boundary of subdomain"):
        basis.boundary()


def test_hcurl_basis_default_parameters_are_available():
    native_mesh=skfemntv.MeshTri()
    native=skfemntv.Basis(
        native_mesh,skfemntv.ElementTriN1(),intorder=2
    )
    actual=native.default_parameters()
    assert set(actual)=={"x","h"}
    assert actual["x"].shape==(2,native.nelems,native.dx.shape[1])
    np.testing.assert_allclose(actual["x"],native.global_coordinates())
    np.testing.assert_allclose(
        actual["h"],np.sqrt(np.abs(native._hcurl_tri_n1_basis.detJ))[:,None]
        *np.ones_like(native.dx)
    )
