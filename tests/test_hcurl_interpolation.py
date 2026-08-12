import numpy as np
import pytest
import skfem

import skfemntv


def _constant(points):
    return np.broadcast_to(np.array((1.25,-.4))[:,None],points.shape)


def test_edge_moment_interpolation_reproduces_constant_field_and_zero_curl():
    mesh=skfemntv.MeshTri.init_tensor(
        np.linspace(0.,1.,5),np.linspace(0.,1.,4)
    )
    basis=skfemntv.AffineTriN1Basis(mesh,intorder=3)
    coefficients=basis.interpolate_edge_moments(_constant)
    expected=_constant(basis.global_coordinates.reshape(2,-1)).reshape(
        basis.global_coordinates.shape
    )
    np.testing.assert_allclose(basis.evaluate(coefficients),expected,atol=3e-15)
    np.testing.assert_allclose(basis.evaluate_curl(coefficients),0.,atol=3e-14)


def test_edge_coefficients_follow_ascending_global_orientation():
    basis=skfemntv.AffineTriN1Basis(skfemntv.MeshTri())
    vector=np.array((.7,-1.1))
    coefficients=basis.interpolate_edge_moments(
        lambda points:np.broadcast_to(vector[:,None],points.shape)
    )
    expected=np.array([
        vector@(basis.mesh.p[:,end]-basis.mesh.p[:,start])
        for start,end in basis.dof_map.topology.edges.T
    ])
    np.testing.assert_allclose(coefficients,expected,atol=2e-15)


def test_interpolation_and_evaluation_validate_inputs():
    basis=skfemntv.AffineTriN1Basis(skfemntv.MeshTri())
    with pytest.raises(TypeError,match="callable"):
        basis.interpolate_edge_moments(np.ones(2))
    with pytest.raises(ValueError,match="positive integer"):
        basis.interpolate_edge_moments(_constant,quadrature_order=0)
    with pytest.raises(ValueError,match="broadcastable"):
        basis.interpolate_edge_moments(lambda points:np.ones(3))
    with pytest.raises(ValueError,match="coefficients"):
        basis.evaluate(np.zeros((basis.N,1)))
    with pytest.raises(ValueError,match="coefficients"):
        basis.evaluate_curl(np.zeros(basis.N+1))


def test_interpolated_values_and_curls_match_scikit_fem():
    mesh=skfemntv.MeshTri.init_tensor(
        np.linspace(0.,1.,4),np.linspace(0.,1.,3)
    )
    basis=skfemntv.AffineTriN1Basis(mesh,intorder=3)
    coefficients=basis.interpolate_edge_moments(_manufactured,quadrature_order=6)
    reference=skfem.Basis(
        skfem.MeshTri(mesh.p,mesh.t[:3]),
        skfem.ElementTriN1(),
        quadrature=(basis.X,basis.W),
    )
    native_edges={
        tuple(edge):index for index,edge in enumerate(basis.dof_map.topology.edges.T)
    }
    reference_coefficients=np.array([
        -coefficients[native_edges[tuple(edge)]]
        for edge in reference.mesh.facets.T
    ])
    # scikit-fem's global TriN1 basis is the negative of this package's
    # ascending-edge moment convention; the reference-element sign relation
    # is documented in 20260813-nedelec-reference-triangle.md.
    field=reference.interpolate(reference_coefficients)
    np.testing.assert_allclose(
        basis.evaluate(coefficients),np.asarray(field),atol=3e-14
    )
    np.testing.assert_allclose(
        basis.evaluate_curl(coefficients),field.curl,atol=5e-14
    )


def _manufactured(points):
    x,y=points
    return np.array((0.*x,x*y))


def _manufactured_curl(points):
    _,y=points
    return y


def _errors(divisions):
    mesh=skfemntv.MeshTri.init_tensor(
        np.linspace(0.,1.,divisions+1),np.linspace(0.,1.,divisions+1)
    )
    basis=skfemntv.AffineTriN1Basis(mesh,intorder=5)
    coefficients=basis.interpolate_edge_moments(
        _manufactured,quadrature_order=6
    )
    points=basis.global_coordinates
    value_error=basis.evaluate(coefficients)-_manufactured(points)
    curl_error=basis.evaluate_curl(coefficients)-_manufactured_curl(points)
    l2=np.sqrt(np.einsum("ieq,ieq,eq->",value_error,value_error,basis.dx))
    curl_l2=np.sqrt(np.einsum("eq,eq,eq->",curl_error,curl_error,basis.dx))
    return l2,curl_l2,np.hypot(l2,curl_l2)


def test_smooth_edge_interpolant_converges_in_l2_and_hcurl_norm():
    history=np.array([_errors(n) for n in (4,8,16)])
    assert np.all(history[1:]<history[:-1])
    rates=np.log2(history[:-1]/history[1:])
    assert np.all(rates[:,0]>.9)
    assert np.all(rates[:,2]>.9)
