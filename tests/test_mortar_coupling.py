import numpy as np
import pytest
from scipy.sparse import csr_matrix,eye
from skfem.supermeshing import intersect

import skfemntv


def _nonmatching_surface():
    master=np.array([[0.,1.,0.],[0.,0.,1.],[0.,0.,0.]])
    slave=np.array([[0.,1.,0.,.5],[0.,0.,1.,.5],[0.,0.,0.,0.]])
    return master,np.array([[0],[1],[2]]),slave,np.array([[0,0],[1,3],[3,2]])


def _grid_surface(cells,reverse=False):
    axis=np.linspace(0.,1.,cells+1)
    points=np.asarray([(x,y,0.) for y in axis for x in axis]).T
    triangles=[]
    for j in range(cells):
        for i in range(cells):
            a=j*(cells+1)+i;b=a+1;c=a+cells+1;d=c+1
            triangles.extend(
                ((a,b,c),(b,d,c)) if reverse else ((a,b,d),(a,d,c))
            )
    return points,np.asarray(triangles,dtype=np.int64).T


def _skfem_slave_facet_p0(master_points,master_triangles,slave_points,slave_triangles):
    mesh,master_parents,slave_parents=intersect(
        (master_points[:2],master_triangles),(slave_points[:2],slave_triangles)
    )
    matrix=np.zeros((slave_triangles.shape[1],master_points.shape[1]+slave_points.shape[1]))
    for element,(master_parent,slave_parent) in enumerate(zip(
        master_parents,slave_parents,strict=True
    )):
        overlap=mesh.p[:,mesh.t[:,element]]
        area=.5*abs(np.linalg.det(np.column_stack((
            overlap[:,1]-overlap[:,0],overlap[:,2]-overlap[:,0]
        ))))
        for points,triangles,parent,offset,sign in (
            (master_points,master_triangles,master_parent,0,1.),
            (slave_points,slave_triangles,slave_parent,master_points.shape[1],-1.),
        ):
            nodes=triangles[:,int(parent)]
            barycentric=np.linalg.solve(
                np.vstack((points[:2,nodes],np.ones(3))),
                np.vstack((overlap,np.ones(3))),
            )
            matrix[int(slave_parent),offset+nodes]+=sign*area*barycentric.mean(axis=1)
    return matrix


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


def test_native_slave_facet_p0_matches_skfem_supermeshing_reference():
    master_points,master_triangles=_grid_surface(2)
    slave_points,slave_triangles=_grid_surface(3,reverse=True)
    native=skfemntv.TriangleSupermesh(
        master_points,master_triangles,slave_points,slave_triangles
    ).assemble_mortar("slave_facet_p0").coupling_matrix.toarray()
    reference=_skfem_slave_facet_p0(
        master_points,master_triangles,slave_points,slave_triangles
    )

    np.testing.assert_allclose(native,reference,rtol=2.e-13,atol=2.e-14)


def test_multiplier_metadata_maps_compact_rows_to_parent_facets():
    mp,mt,sp,st=_nonmatching_surface()
    result=skfemntv.TriangleSupermesh(
        mp,mt,sp,st
    ).assemble_mortar("slave_facet_p0")
    metadata=result.multiplier

    assert isinstance(metadata,skfemntv.MortarMultiplierMetadata)
    assert metadata.space=="slave_facet_p0"
    assert metadata.entity_kind=="parent_facet"
    assert metadata.side=="slave"
    np.testing.assert_array_equal(metadata.entity_ids,[0,1])
    np.testing.assert_array_equal(metadata.row_entities,[0,1])
    np.testing.assert_array_equal(metadata.row_components,[0,0])
    np.testing.assert_array_equal(metadata.supported_rows,[0,1])
    np.testing.assert_array_equal(metadata.rows_for([1]),[1])
    for array in (
        metadata.entity_ids,metadata.row_entities,metadata.row_components,
        metadata.supported_rows,metadata.rows_for([0]),
    ):
        assert not array.flags.writeable


def test_vector_facet_p0_metadata_can_select_entity_and_component():
    mesh=skfemntv.MeshTet()
    basis=skfemntv.FacetBasis(
        mesh,skfemntv.ElementVector(skfemntv.ElementTetP1(),dim=2),
        intorder=2,
    )
    result=skfemntv.TriangleSupermesh.from_facets(
        basis,basis
    ).assemble_mortar("master_facet_p0")
    metadata=result.multiplier

    assert metadata.row_count==2*len(metadata.entity_ids)
    entity=int(metadata.entity_ids[1])
    rows=metadata.rows_for([entity],components=[1])
    assert len(rows)==1
    assert metadata.row_components[rows[0]]==1
    assert metadata.entity_ids[metadata.row_entities[rows[0]]]==entity


def test_kkt_blocks_keep_sparse_blocks_separate_and_select_active_rows():
    mp,mt,sp,st=_nonmatching_surface()
    result=skfemntv.TriangleSupermesh(
        mp,mt,sp,st
    ).assemble_mortar("slave_facet_p0")
    stiffness=eye(7,format="csr")
    force=np.arange(7,dtype=np.float64)

    all_blocks=result.kkt_blocks(stiffness,force)
    assert isinstance(all_blocks,skfemntv.MortarKKTBlocks)
    assert all_blocks.primal_matrix is stiffness
    assert all_blocks.coupling_matrix is result.coupling_matrix
    assert all_blocks.primal_size==7
    assert all_blocks.multiplier_size==2
    assert all_blocks.multiplier_metadata is result.multiplier
    np.testing.assert_array_equal(all_blocks.multiplier_rows,[0,1])
    np.testing.assert_array_equal(all_blocks.constraint_rhs,[0.,0.])

    rows=result.multiplier.rows_for([1])
    active=result.kkt_blocks(
        stiffness,force,constraint_rhs=np.array([3.,4.]),rows=rows
    )
    assert active.coupling_matrix.shape==(1,7)
    np.testing.assert_allclose(
        active.coupling_matrix.toarray(),
        result.coupling_matrix[[1]].toarray(),
    )
    np.testing.assert_array_equal(active.constraint_rhs,[4.])
    np.testing.assert_array_equal(active.multiplier_rows,[1])
    for array in (
        all_blocks.primal_rhs,all_blocks.constraint_rhs,
        all_blocks.multiplier_rows,active.constraint_rhs,
    ):
        assert not array.flags.writeable


def test_kkt_blocks_validate_sparse_shapes_and_row_selection():
    mp,mt,sp,st=_nonmatching_surface()
    result=skfemntv.TriangleSupermesh(mp,mt,sp,st).assemble_mortar()
    stiffness=eye(7,format="csr")

    with pytest.raises(TypeError,match="CSR"):
        result.kkt_blocks(np.eye(7),np.ones(7))
    with pytest.raises(ValueError,match="unique"):
        result.kkt_blocks(stiffness,np.ones(7),rows=[0,0])
    with pytest.raises(IndexError,match="bounds"):
        result.kkt_blocks(stiffness,np.ones(7),rows=[99])
    with pytest.raises(ValueError,match="primal_rhs"):
        result.kkt_blocks(stiffness,np.ones(6))


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


@pytest.mark.parametrize("space",[
    "overlap_p0","slave_facet_p0","master_facet_p0",
    "slave_p1","master_p1","dual",
])
def test_vector_nonmatching_mortar_spaces_reproduce_affine_trace(space):
    master_points,master_triangles=_grid_surface(2)
    slave_points,slave_triangles=_grid_surface(1,reverse=True)
    result=skfemntv.TriangleSupermesh(
        master_points,master_triangles,slave_points,slave_triangles,
        components=3,
    ).assemble_mortar(space)
    gradient=np.array([
        [.2,-.1,.3],[.1,.4,-.2],[-.3,.2,.1],
    ])
    offset=np.array([.4,-.3,.2])

    def affine(points):
        return (points.T@gradient.T+offset).reshape(-1)

    displacement=np.concatenate((affine(master_points),affine(slave_points)))
    np.testing.assert_allclose(
        result.coupling_matrix@displacement,0.,atol=3.e-14
    )


def test_nonmatching_dual_multiplier_has_full_supported_row_rank():
    master_points,master_triangles=_grid_surface(2)
    slave_points,slave_triangles=_grid_surface(1,reverse=True)
    result=skfemntv.TriangleSupermesh(
        master_points,master_triangles,slave_points,slave_triangles,
        components=3,
    ).assemble_mortar("dual")
    supported=result.multiplier.supported_rows
    matrix=result.coupling_matrix[supported].toarray()

    assert matrix.shape==(12,39)
    assert np.linalg.matrix_rank(matrix,tol=1.e-12)==matrix.shape[0]


def test_global_qr_removes_cross_facet_overlap_dependencies():
    points,triangles=_grid_surface(2)
    supermesh=skfemntv.TriangleSupermesh(
        points,triangles,points,triangles,components=3
    )
    raw=supermesh.assemble_mortar("overlap_p0")
    reduced=supermesh.assemble_mortar(
        "overlap_p0",reduction="global_qr",rank_tolerance=1.e-10
    )

    assert raw.coupling_matrix.shape==(24,54)
    assert reduced.coupling_matrix.shape==(21,54)
    assert reduced.multiplier.row_count==21
    np.testing.assert_array_equal(reduced.multiplier.supported_rows,np.arange(21))
    assert reduced.reduction.method=="global_qr"
    assert reduced.reduction.raw_row_count==24
    assert reduced.reduction.supported_row_count==24
    assert reduced.reduction.independent_row_count==21
    assert reduced.reduction.numerical_rank==21
    assert not reduced.reduction.selected_raw_rows.flags.writeable
    assert np.linalg.matrix_rank(
        reduced.coupling_matrix.toarray(),tol=1.e-10
    )==21

    gradient=np.array([
        [.2,-.1,.3],[.1,.4,-.2],[-.3,.2,.1],
    ])
    displacement=(points.T@gradient.T+np.array([.4,-.3,.2])).reshape(-1)
    np.testing.assert_allclose(
        reduced.coupling_matrix@np.concatenate((displacement,displacement)),
        0.,atol=3.e-14,
    )


def test_global_qr_dense_reference_requires_explicit_size_guard():
    points,triangles=_grid_surface(2)
    supermesh=skfemntv.TriangleSupermesh(
        points,triangles,points,triangles,components=3
    )
    with pytest.raises(ValueError,match="dense_reduction_max_rows"):
        supermesh.assemble_mortar(
            "overlap_p0",reduction="global_qr",dense_reduction_max_rows=23
        )


@pytest.mark.parametrize("space",["slave_p1","master_p1","dual"])
def test_nodal_multiplier_metadata_excludes_unused_surface_nodes(space):
    master_points,master_triangles=_grid_surface(1)
    slave_points,slave_triangles=_grid_surface(1,reverse=True)
    master_points=np.column_stack((master_points,(2.,2.,2.)))
    slave_points=np.column_stack((slave_points,(-2.,-2.,-2.)))
    result=skfemntv.TriangleSupermesh(
        master_points,master_triangles,slave_points,slave_triangles,
        components=3,
    ).assemble_mortar(space)
    supported=result.multiplier.supported_rows
    unsupported=np.setdiff1d(np.arange(result.coupling_matrix.shape[0]),supported)

    np.testing.assert_array_equal(supported,np.arange(12))
    np.testing.assert_array_equal(unsupported,np.arange(12,15))
    assert result.coupling_matrix[supported].shape==(12,30)
    assert result.coupling_matrix[unsupported].nnz==0


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
