import numpy as np
import pytest

import skfemntv


def _mesh(resolution):
    axis=np.linspace(0.,1.,resolution+1)
    return skfemntv.MeshTet.init_tensor(axis,axis,axis)


def test_tet_edge_moments_reproduce_constant_vector_field():
    basis=skfemntv.AffineTetN1Basis(_mesh(2),intorder=3)
    constant=np.array((1.25,-.5,.75))
    coefficients=basis.interpolate_edge_moments(
        lambda x:constant[:,None]+0.*x
    )
    expected=np.einsum(
        "ie,i->e",
        basis.mesh.p[:,basis.dof_map.topology.edges[1]]
        -basis.mesh.p[:,basis.dof_map.topology.edges[0]],
        constant,
    )
    np.testing.assert_allclose(coefficients,expected,atol=2e-15)
    values=basis.evaluate(coefficients)
    np.testing.assert_allclose(
        values,np.broadcast_to(constant[:,None,None],values.shape),atol=2e-14
    )
    np.testing.assert_allclose(basis.evaluate_curl(coefficients),0.,atol=2e-14)


def test_tet_interpolation_and_evaluation_validation():
    basis=skfemntv.AffineTetN1Basis(_mesh(1))
    with pytest.raises(TypeError,match="field must be callable"):
        basis.interpolate_edge_moments(np.ones(3))
    with pytest.raises(ValueError,match="positive integer"):
        basis.interpolate_edge_moments(lambda x:x,quadrature_order=0)
    with pytest.raises(ValueError,match="broadcastable"):
        basis.interpolate_edge_moments(lambda x:np.zeros((2,x.shape[1])))
    with pytest.raises(ValueError,match="coefficients must have shape"):
        basis.evaluate(np.zeros(basis.N+1))


def test_tet_n1_interpolation_converges_in_hcurl_norm():
    def field(x):
        return np.array((0.*x[0],x[0]*x[2],x[0]*x[1]))

    def exact_curl(x):
        return np.array((0.*x[0],-x[1],x[2]))

    errors=[]
    for resolution in (1,2,4):
        basis=skfemntv.AffineTetN1Basis(_mesh(resolution),intorder=4)
        coefficients=basis.interpolate_edge_moments(field,quadrature_order=4)
        value_error=basis.evaluate(coefficients)-field(basis.global_coordinates)
        curl_error=(
            basis.evaluate_curl(coefficients)
            -exact_curl(basis.global_coordinates)
        )
        l2=np.sqrt(np.einsum("ieq,ieq,eq->",value_error,value_error,basis.dx))
        curl_l2=np.sqrt(np.einsum("ieq,ieq,eq->",curl_error,curl_error,basis.dx))
        errors.append((l2,curl_l2,np.hypot(l2,curl_l2)))
    errors=np.asarray(errors)
    assert np.all(errors[1:]<errors[:-1])
    rates=np.log(errors[:-1]/errors[1:])/np.log(2.)
    assert np.all(rates>.8),rates
