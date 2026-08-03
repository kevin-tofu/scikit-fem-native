import numpy as np
from scipy.sparse import csr_matrix

import skfemntv


def _nonmatching_surface():
    master=np.array([[0.,1.,0.],[0.,0.,1.],[0.,0.,0.]])
    slave=np.array([[0.,1.,0.,.5],[0.,0.,1.,.5],[0.,0.,0.,0.]])
    return master,np.array([[0],[1],[2]]),slave,np.array([[0,0],[1,3],[3,2]])


def test_public_mortar_result_contains_only_sparse_global_blocks():
    mp,mt,sp,st=_nonmatching_surface()
    result=skfemntv.TriangleSupermesh(mp,mt,sp,st).assemble_mortar()
    assert isinstance(result,skfemntv.MortarCouplingResult)
    assert all(isinstance(matrix,csr_matrix) for matrix in (
        result.master_matrix,result.slave_matrix,result.coupling_matrix
    ))
    assert result.coupling_matrix.shape==(4,7)
    assert np.isclose(result.overlap_area,.5)
    expected=result.master_matrix@np.ones(3)-result.slave_matrix@np.ones(4)
    np.testing.assert_allclose(expected,0.,atol=2e-14)


def test_all_multiplier_spaces_assemble_without_global_dense_storage():
    mp,mt,sp,st=_nonmatching_surface()
    supermesh=skfemntv.TriangleSupermesh(mp,mt,sp,st)
    expected_rows={
        "slave":4,"master":3,"overlap_p0":2,"slave_facet_p0":2,
        "master_facet_p0":1,"dual":4,
    }
    for multiplier,rows in expected_rows.items():
        result=supermesh.assemble_mortar(multiplier)
        assert result.master_matrix.shape==(rows,3)
        assert result.slave_matrix.shape==(rows,4)
        assert result.coupling_matrix.shape==(rows,7)


def test_facet_p0_collects_overlap_cells_without_changing_integrals():
    mp,mt,sp,st=_nonmatching_surface()
    supermesh=skfemntv.TriangleSupermesh(mp,mt,sp,st)
    overlap=supermesh.assemble_mortar("overlap_p0")
    slave=supermesh.assemble_mortar("slave_facet_p0")
    master=supermesh.assemble_mortar("master_p0")

    assert overlap.coupling_matrix.shape[0]==2
    assert slave.coupling_matrix.shape[0]==2
    assert master.coupling_matrix.shape[0]==1
    np.testing.assert_allclose(
        master.master_matrix.toarray(),
        np.asarray(overlap.master_matrix.sum(axis=0)),
    )
    np.testing.assert_allclose(
        master.slave_matrix.toarray(),
        np.asarray(overlap.slave_matrix.sum(axis=0)),
    )
    np.testing.assert_allclose(
        slave.coupling_matrix@np.ones(7),0.,atol=2.e-14
    )


def test_local_dual_basis_is_biorthogonal_on_one_facet():
    points=np.array([[0.,1.,0.],[0.,0.,1.],[0.,0.,0.]])
    triangles=np.array([[0],[1],[2]])
    result=skfemntv.TriangleSupermesh(
        points,triangles,points,triangles
    ).assemble_mortar("dual")
    diagonal=result.slave_matrix.diagonal()
    offdiagonal=result.slave_matrix.toarray()-np.diag(diagonal)
    np.testing.assert_allclose(offdiagonal,0.,atol=2e-14)
    np.testing.assert_allclose(diagonal,np.full(3,1./6.),atol=2e-14)


def test_trace_data_shares_points_and_has_opposing_normals_and_gradients():
    mp,mt,sp,st=_nonmatching_surface()
    supermesh=skfemntv.TriangleSupermesh(mp,mt,sp,st)
    master=supermesh.master_trace
    slave=supermesh.slave_trace
    assert master.coordinates is slave.coordinates
    assert master.quadrature_weights is slave.quadrature_weights
    assert master.physical_gradients.shape==(2,6,3,3)
    assert slave.physical_gradients.shape==(2,6,3,3)
    np.testing.assert_allclose(master.outward_normals,-slave.outward_normals)
    assert master.parent_facets.shape==(2,)
    assert slave.parent_elements.shape==(2,)
    assert supermesh.diagnostics.maximum_normal_opposition_error==0.


def test_independent_outward_orientation_mismatch_is_diagnosed():
    mesh=skfemntv.MeshTet()
    facets=skfemntv.FacetBasis(
        mesh,skfemntv.ElementVector(skfemntv.ElementTetP1(),dim=1),intorder=2
    )
    supermesh=skfemntv.TriangleSupermesh.from_facets(facets,facets)
    np.testing.assert_allclose(
        supermesh.master_trace.outward_normals,
        -supermesh.slave_trace.outward_normals,
    )
    assert supermesh.diagnostics.orientation_mismatch_count>0
    assert supermesh.diagnostics.maximum_normal_opposition_error>1.
