import numpy as np
import pytest

import skfemntv


def _matching_triangle():
    points=np.asarray(((0.,1.,0.),(0.,0.,1.),(0.,0.,0.)))
    facets=np.asarray(((0,),(1,),(2,)),dtype=np.int64)
    return skfemntv.TriangleSupermesh(
        points,facets,points,facets,components=3
    )


def _tabulation(trace,size):
    return skfemntv.CrossTabulation(
        trace.dofs,trace.shape_values,size
    )


def test_cross_tabulation_matches_standard_supermesh_assembly():
    integration=_matching_triangle()
    master=_tabulation(integration.master_trace,integration.master_size)
    slave=_tabulation(integration.slave_trace,integration.slave_size)

    actual=integration.assemble_cross_tabulation(master,slave)
    expected=integration.assemble()

    np.testing.assert_allclose(actual.toarray(),expected.toarray(),atol=2.e-14)


def test_cross_tabulation_accepts_caller_defined_test_space():
    integration=_matching_triangle()
    master=integration.master_trace
    slave=_tabulation(integration.slave_trace,integration.slave_size)
    transform=np.asarray(((3.,-1.,-1.),(-1.,3.,-1.),(-1.,-1.,3.)))
    test=skfemntv.CrossTabulation(
        master.dofs,master.shape_values@transform.T,integration.master_size
    )

    matrix=integration.assemble_cross_tabulation(test,slave).toarray()
    diagonal=np.diag(matrix)
    np.testing.assert_allclose(
        matrix-np.diag(diagonal),0.,atol=2.e-14
    )
    np.testing.assert_allclose(diagonal,np.full(9,1./6.),atol=2.e-14)


def test_cross_tabulation_rejects_incompatible_tables():
    integration=_matching_triangle()
    trace=integration.master_trace
    with pytest.raises(ValueError,match="local basis"):
        skfemntv.CrossTabulation(
            trace.dofs[:,:2],trace.shape_values,integration.master_size
        )

    with pytest.raises(ValueError,match="entity counts"):
        skfemntv.CrossTabulation(
            trace.dofs,trace.shape_values[:-1],integration.master_size
        )


def test_skfemntv_exposes_no_mortar_formulation_api():
    assert not hasattr(skfemntv.TriangleSupermesh,"assemble_mortar")
    assert not hasattr(skfemntv,"MortarCouplingResult")
    assert not hasattr(skfemntv,"MortarMultiplierMetadata")
