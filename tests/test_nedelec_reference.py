import numpy as np
import pytest
from skfem.element import ElementTriN1

from skfemntv._nedelec_reference import (
    tri_n1_basis,
    tri_n1_curl,
    tri_n1_edge_moments,
)


def test_tri_n1_edge_moments_are_kronecker_dual():
    np.testing.assert_allclose(tri_n1_edge_moments(),np.eye(3),atol=2e-15)


@pytest.mark.parametrize("edge",range(3))
def test_reversing_one_edge_reverses_only_its_moment_row(edge):
    expected=np.eye(3)
    expected[edge]*=-1.
    np.testing.assert_allclose(
        tri_n1_edge_moments(reversed_edges=(edge,)),expected,atol=2e-15
    )


def test_tri_n1_basis_and_curl_match_scikit_fem_with_documented_signs():
    points=np.array(((0.1,0.6,0.2),(0.2,0.1,0.5)))
    actual_basis=tri_n1_basis(points)
    actual_curl=tri_n1_curl(points)
    reference=ElementTriN1()
    signs=np.array((-1.,-1.,1.))

    for basis_index,sign in enumerate(signs):
        values,curl=reference.lbasis(points,basis_index)
        np.testing.assert_allclose(actual_basis[basis_index],sign*values)
        np.testing.assert_allclose(actual_curl[basis_index],sign*curl)


def test_tri_n1_rejects_invalid_point_and_edge_inputs():
    with pytest.raises(ValueError,match="shape"):
        tri_n1_basis(np.zeros((3,2)))
    with pytest.raises(ValueError,match="shape"):
        tri_n1_curl(np.zeros(4))
    with pytest.raises(ValueError,match="edge indices"):
        tri_n1_edge_moments(reversed_edges=(3,))
