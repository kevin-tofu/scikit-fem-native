import numpy as np
import pytest
from skfem import ElementTetN1

from skfemntv._nedelec_reference import (
    tet_n1_basis,tet_n1_curl,tet_n1_edge_moments,
)


def test_tet_n1_edge_moments_are_dual_to_directed_edges():
    np.testing.assert_allclose(tet_n1_edge_moments(),np.eye(6),atol=2e-15)


def test_reversing_tet_edge_reverses_only_its_moment_row():
    expected=np.eye(6)
    expected[[1,4]]*=-1.
    np.testing.assert_allclose(
        tet_n1_edge_moments(reversed_edges=(1,4)),expected,atol=2e-15
    )


def test_tet_n1_values_and_curls_match_scikit_fem_with_explicit_edge_sign():
    points=np.array(((.1,.2,.15),(.2,.1,.25),(.3,.15,.1)))
    values=tet_n1_basis(points)
    curls=tet_n1_curl(points)
    element=ElementTetN1()
    signs=np.array((1.,1.,-1.,1.,1.,1.))
    for basis in range(6):
        reference_value,reference_curl=element.lbasis(points,basis)
        np.testing.assert_allclose(values[basis],signs[basis]*reference_value)
        np.testing.assert_allclose(curls[basis],signs[basis]*reference_curl)


def test_tet_n1_reference_input_validation():
    with pytest.raises(ValueError,match="tetrahedron reference points"):
        tet_n1_basis(np.zeros((2,3)))
    with pytest.raises(ValueError,match="tetrahedron reference points"):
        tet_n1_curl(np.zeros((2,3)))
    with pytest.raises(ValueError,match="0 through 5"):
        tet_n1_edge_moments(reversed_edges=(6,))
