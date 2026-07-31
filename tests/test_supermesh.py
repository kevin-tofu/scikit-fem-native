import numpy as np

import skfn


def master_surface():
    points=np.array([[0.,1.,0.],[0.,0.,1.],[0.,0.,0.]])
    triangles=np.array([[0],[1],[2]])
    return points,triangles


def split_slave_surface():
    points=np.array([[0.,1.,0.,.5],[0.,0.,1.,.5],[0.,0.,0.,0.]])
    triangles=np.array([[0,0],[1,3],[3,2]])
    return points,triangles


def test_identical_triangle_has_analytical_p1_mass_matrix():
    points,triangles=master_surface()
    supermesh=skfn.TriangleSupermesh(
        points,triangles,points,triangles
    )
    actual=supermesh.assemble().toarray()
    expected=np.full((3,3),1./24.)
    np.fill_diagonal(expected,1./12.)
    np.testing.assert_allclose(actual,expected,rtol=2e-14,atol=2e-14)
    assert np.isclose(supermesh.diagnostics.overlap_area,.5)


def test_nonmatching_split_preserves_constant_field():
    master_points,master_triangles=master_surface()
    slave_points,slave_triangles=split_slave_surface()
    supermesh=skfn.TriangleSupermesh(
        master_points,master_triangles,slave_points,slave_triangles
    )
    coupling=supermesh.assemble()
    np.testing.assert_allclose(
        coupling@np.ones(4),np.full(3,1./6.),rtol=2e-14,atol=2e-14
    )
    assert np.isclose(coupling.sum(),.5)
    assert supermesh.diagnostics.overlap_pair_count==2
    assert supermesh.diagnostics.integration_triangle_count==2


def test_vector_coupling_has_no_cross_component_terms():
    master_points,master_triangles=master_surface()
    slave_points,slave_triangles=split_slave_surface()
    supermesh=skfn.TriangleSupermesh(
        master_points,master_triangles,slave_points,slave_triangles,
        components=3,
    )
    coupling=supermesh.assemble().toarray()
    for row in range(coupling.shape[0]):
        for column in range(coupling.shape[1]):
            if row%3!=column%3:
                assert coupling[row,column]==0.


def test_repeated_supermesh_assembly_reuses_csr():
    master_points,master_triangles=master_surface()
    slave_points,slave_triangles=split_slave_surface()
    supermesh=skfn.TriangleSupermesh(
        master_points,master_triangles,slave_points,slave_triangles
    )
    first=supermesh.assemble(1.)
    values=first.toarray().copy();data_id=id(first.data)
    second=supermesh.assemble(2.)
    assert id(second.data)==data_id
    np.testing.assert_allclose(second.toarray(),2*values)


def test_planar_update_reuses_pattern_and_matches_fresh_build():
    master_points,master_triangles=master_surface()
    slave_points,slave_triangles=split_slave_surface()
    supermesh=skfn.TriangleSupermesh(
        master_points,master_triangles,slave_points,slave_triangles
    )
    native_id=id(supermesh._native)
    translation=np.array([[.2],[-.1],[.0]])
    updated_master=master_points+translation
    updated_slave=slave_points+translation
    supermesh.update(updated_master,updated_slave)
    fresh=skfn.TriangleSupermesh(
        updated_master,master_triangles,
        updated_slave,slave_triangles,
    )
    assert id(supermesh._native)==native_id
    assert supermesh.diagnostics.pattern_reused
    assert supermesh.diagnostics.update_count==1
    np.testing.assert_allclose(
        supermesh.assemble().toarray(),
        fresh.assemble().toarray(),
        rtol=2e-14,atol=2e-14,
    )
    np.testing.assert_allclose(
        supermesh.global_coordinates,
        fresh.global_coordinates,
        rtol=0.,atol=2e-15,
    )


def test_planar_sliding_update_rebuilds_changed_pair_pattern():
    first_points,triangle=master_surface()
    second_points=first_points+np.array([[2.],[0.],[0.]])
    master_points=np.concatenate((first_points,second_points),axis=1)
    master_triangles=np.concatenate((triangle,triangle+3),axis=1)
    slave_points=first_points.copy()
    supermesh=skfn.TriangleSupermesh(
        master_points,master_triangles,slave_points,triangle
    )
    native_id=id(supermesh._native)
    moved_slave=slave_points+np.array([[2.],[0.],[0.]])
    supermesh.update(master_points,moved_slave)
    fresh=skfn.TriangleSupermesh(
        master_points,master_triangles,moved_slave,triangle
    )
    assert id(supermesh._native)!=native_id
    assert not supermesh.diagnostics.pattern_reused
    assert supermesh.diagnostics.created_overlap_pair_count==1
    assert supermesh.diagnostics.disappeared_overlap_pair_count==1
    assert supermesh.assemble().shape==(6,3)
    np.testing.assert_allclose(
        supermesh.assemble().toarray(),
        fresh.assemble().toarray(),
        rtol=2e-14,atol=2e-14,
    )


def test_planar_update_supports_opening_and_closing():
    points,triangles=master_surface()
    supermesh=skfn.TriangleSupermesh(
        points,triangles,points,triangles
    )
    opened=points+np.array([[0.],[0.],[1.]])
    supermesh.update(points,opened)
    assert supermesh.diagnostics.integration_triangle_count==0
    assert supermesh.diagnostics.disappeared_overlap_pair_count==1
    assert supermesh.assemble().shape==(3,3)
    assert supermesh.assemble().nnz==0

    supermesh.update(points,points)
    assert supermesh.diagnostics.integration_triangle_count==1
    assert supermesh.diagnostics.created_overlap_pair_count==1
    assert supermesh.diagnostics.update_count==2
    expected=skfn.TriangleSupermesh(
        points,triangles,points,triangles
    ).assemble()
    np.testing.assert_allclose(
        supermesh.assemble().toarray(),expected.toarray(),
        rtol=2e-14,atol=2e-14,
    )


def test_supermesh_search_reuses_topology_and_stable_pattern():
    master_points,master_triangles=master_surface()
    slave_points,slave_triangles=split_slave_surface()
    search=skfn.SupermeshSearch(
        master_triangles,slave_triangles
    )
    integration=search.build(master_points,slave_points)
    native_id=id(integration._native)
    matrix_id=id(integration._matrix)
    for step in range(1,6):
        translation=np.array([[.01*step],[0.],[0.]])
        updated=search.update(
            master_points+translation,slave_points+translation
        )
        assert updated is integration
        assert id(updated._native)==native_id
        assert id(updated._matrix)==matrix_id
        assert updated.diagnostics.pattern_reused
        assert updated.diagnostics.update_count==step


def test_contraction_profiler_matches_assembled_matrix_sum():
    master_points,master_triangles=master_surface()
    slave_points,slave_triangles=split_slave_surface()
    supermesh=skfn.TriangleSupermesh(
        master_points,master_triangles,slave_points,slave_triangles
    )
    matrix=supermesh.assemble(1.)
    coefficient=np.ones(supermesh._coefficient_shape)
    checksum,_=supermesh._native.contract_only(coefficient)
    np.testing.assert_allclose(
        checksum,matrix.sum(),rtol=2e-14,atol=2e-14
    )


def test_aabb_broad_phase_rejects_distant_pairs():
    master_points,master_triangles=master_surface()
    slave_points,slave_triangles=split_slave_surface()
    distant=slave_points.copy()
    distant[0]+=10.
    combined_points=np.concatenate((slave_points,distant),axis=1)
    combined_triangles=np.concatenate(
        (slave_triangles,slave_triangles+slave_points.shape[1]),axis=1
    )
    supermesh=skfn.TriangleSupermesh(
        master_points,master_triangles,combined_points,combined_triangles
    )
    assert supermesh.diagnostics.total_pair_count==4
    assert supermesh.diagnostics.candidate_pair_count==2


def test_projection_tolerance_and_gap_diagnostics():
    points,triangles=master_surface()
    offset=points.copy();offset[2]+=1e-4
    supermesh=skfn.TriangleSupermesh(
        points,triangles,offset,triangles,projection_tolerance=2e-4
    )
    assert np.isclose(supermesh.diagnostics.maximum_plane_gap,1e-4)
    assert supermesh.diagnostics.noncoplanar_rejection_count==0


def test_tet10_to_tet4_coupling_preserves_constant_trace():
    linear_mesh=skfn.MeshTet()
    quadratic_mesh=skfn.MeshTet2.from_mesh(linear_mesh)
    master=skfn.FacetBasis(
        quadratic_mesh,skfn.ElementVector(skfn.ElementTetP2()),intorder=4
    )
    slave=skfn.FacetBasis(
        linear_mesh,skfn.ElementVector(skfn.ElementTetP1()),intorder=4
    )
    supermesh=skfn.TriangleSupermesh.from_facets(master,slave)
    coupling=supermesh.assemble()
    expected,_=skfn.NativeLinearForm(master).assemble(value=np.ones(3))
    np.testing.assert_allclose(
        coupling@np.ones(slave.N),expected,rtol=3e-12,atol=3e-12
    )


def test_hex27_to_hex8_coupling_preserves_constant_trace():
    linear_mesh=skfn.MeshHex()
    quadratic_mesh=skfn.MeshHex2.from_mesh(linear_mesh)
    master=skfn.FacetBasis(
        quadratic_mesh,skfn.ElementVector(skfn.ElementHex2()),intorder=4
    )
    slave=skfn.FacetBasis(
        linear_mesh,skfn.ElementVector(skfn.ElementHex1()),intorder=4
    )
    supermesh=skfn.TriangleSupermesh.from_facets(master,slave)
    coupling=supermesh.assemble()
    expected,_=skfn.NativeLinearForm(master).assemble(value=np.ones(3))
    np.testing.assert_allclose(
        coupling@np.ones(slave.N),expected,rtol=4e-12,atol=4e-12
    )


def test_high_order_nonmatching_coupling_reproduces_linear_vector_field():
    linear_mesh=skfn.MeshTet()
    quadratic_mesh=skfn.MeshTet2.from_mesh(linear_mesh)
    master=skfn.FacetBasis(
        quadratic_mesh,skfn.ElementVector(skfn.ElementTetP2()),intorder=4
    )
    slave=skfn.FacetBasis(
        linear_mesh,skfn.ElementVector(skfn.ElementTetP1()),intorder=4
    )
    coupling=skfn.TriangleSupermesh.from_facets(master,slave).assemble()
    slave_linear_field=linear_mesh.p.T.reshape(-1)
    element_coordinates=np.stack([
        quadratic_mesh.p[:,quadratic_mesh.t[:,element]].T
        for element in master.parent_elements
    ])
    physical_points=np.einsum(
        "eqn,end->eqd",master.tabulated_shape,element_coordinates
    )
    expected,_=skfn.NativeLinearForm(master).assemble(
        value=physical_points
    )
    np.testing.assert_allclose(
        coupling@slave_linear_field,expected,rtol=5e-12,atol=5e-12
    )


def test_master_slave_exchange_transposes_coupling():
    linear_mesh=skfn.MeshTet()
    quadratic_mesh=skfn.MeshTet2.from_mesh(linear_mesh)
    high=skfn.FacetBasis(
        quadratic_mesh,skfn.ElementVector(skfn.ElementTetP2()),intorder=4
    )
    low=skfn.FacetBasis(
        linear_mesh,skfn.ElementVector(skfn.ElementTetP1()),intorder=4
    )
    high_low=skfn.TriangleSupermesh.from_facets(high,low).assemble()
    low_high=skfn.TriangleSupermesh.from_facets(low,high).assemble()
    np.testing.assert_allclose(
        high_low.toarray(),low_high.toarray().T,rtol=6e-12,atol=6e-12
    )


def test_adaptive_curved_surface_tessellation_converges():
    mesh=skfn.MeshTet2()
    points=mesh.p.copy()
    points[:,4]+=np.array([0.,0.,.25])
    curved=skfn.MeshTet2(points,mesh.t)
    facets=skfn.FacetBasis(
        curved,skfn.ElementVector(skfn.ElementTetP2()),intorder=4
    )
    expected,_=skfn.NativeLinearForm(facets).assemble(value=np.ones(3))
    errors=[];triangle_counts=[]
    for tolerance in (.05,.01):
        supermesh=skfn.TriangleSupermesh.from_facets(
            facets,facets,geometry_tolerance=tolerance,
            max_subdivision_level=4,
        )
        reproduced=supermesh.assemble()@np.ones(facets.N)
        errors.append(np.linalg.norm(reproduced-expected))
        triangle_counts.append(
            supermesh.diagnostics.master_search_triangle_count
        )
    assert triangle_counts[1]>triangle_counts[0]
    assert errors[1]<.3*errors[0]


def test_curved_supermesh_normals_are_evaluated_at_quadrature_points():
    mesh=skfn.MeshTet2()
    points=mesh.p.copy()
    points[:,4]+=np.array([0.,0.,.25])
    curved=skfn.MeshTet2(points,mesh.t)
    facets=skfn.FacetBasis(
        curved,skfn.ElementVector(skfn.ElementTetP2()),intorder=4
    )
    supermesh=skfn.TriangleSupermesh.from_facets(
        facets,facets,geometry_tolerance=.01,
        max_subdivision_level=4,
    )
    np.testing.assert_allclose(
        np.linalg.norm(supermesh.master_normals,axis=2),1.,
        rtol=3e-14,atol=3e-14,
    )
    variation=np.linalg.norm(
        supermesh.master_normals-supermesh.master_normals[:,:1],axis=2
    )
    assert variation.max()>1e-3


def test_scalar_multiplier_to_vector_tensor_coupling():
    mesh=skfn.MeshTet()
    multiplier=skfn.FacetBasis(
        mesh,skfn.ElementVector(skfn.ElementTetP1(),dim=1),intorder=4
    )
    displacement=skfn.FacetBasis(
        mesh,skfn.ElementVector(skfn.ElementTetP1(),dim=3),intorder=4
    )
    supermesh=skfn.TriangleSupermesh.from_facets(
        multiplier,displacement
    )
    component_tensor=np.array([[1.,2.,3.]])
    coupling=supermesh.assemble_tensor(component_tensor)
    constant_vector=np.tile([1.,2.,3.],mesh.p.shape[1])
    expected,_=skfn.NativeLinearForm(multiplier).assemble(
        value=np.array([14.])
    )
    assert coupling.shape==(multiplier.N,displacement.N)
    np.testing.assert_allclose(
        coupling@constant_vector,expected,rtol=3e-12,atol=3e-12
    )


def test_nonsymmetric_tensor_reverses_by_transpose():
    mesh=skfn.MeshTet()
    scalar=skfn.FacetBasis(
        mesh,skfn.ElementVector(skfn.ElementTetP1(),dim=1),intorder=4
    )
    vector=skfn.FacetBasis(
        mesh,skfn.ElementVector(skfn.ElementTetP1(),dim=3),intorder=4
    )
    tensor=np.array([[.2,-.7,1.3]])
    forward=skfn.TriangleSupermesh.from_facets(
        scalar,vector
    ).assemble_tensor(tensor)
    reverse=skfn.TriangleSupermesh.from_facets(
        vector,scalar
    ).assemble_tensor(tensor.T)
    np.testing.assert_allclose(
        forward.toarray(),reverse.toarray().T,rtol=4e-12,atol=4e-12
    )


def test_value_gradient_cross_coupling_reproduces_linear_field():
    mesh=skfn.MeshTet()
    scalar=skfn.FacetBasis(
        mesh,skfn.ElementVector(skfn.ElementTetP1(),dim=1),intorder=4
    )
    supermesh=skfn.TriangleSupermesh.from_facets(scalar,scalar)
    direction=np.array([[[2.,-3.,.5]]])
    coupling=supermesh.assemble_cross(
        direction,row_kind="value",column_kind="gradient"
    ).copy()
    reverse=supermesh.assemble_cross(
        np.transpose(direction,(1,2,0)),
        row_kind="gradient",column_kind="value",
    ).copy()
    linear_x=mesh.p[0]
    expected,_=skfn.NativeLinearForm(scalar).assemble(value=np.array([2.]))
    np.testing.assert_allclose(
        coupling@linear_x,expected,rtol=4e-12,atol=4e-12
    )
    np.testing.assert_allclose(
        coupling.toarray(),reverse.toarray().T,rtol=4e-12,atol=4e-12
    )


def test_scalar_gradient_gradient_cross_coupling_reproduces_flux():
    mesh=skfn.MeshTet()
    scalar=skfn.FacetBasis(
        mesh,skfn.ElementVector(skfn.ElementTetP1(),dim=1),intorder=4
    )
    supermesh=skfn.TriangleSupermesh.from_facets(scalar,scalar)
    coupling=supermesh.assemble_cross(
        1.,row_kind="gradient",column_kind="gradient"
    )
    expected,_=skfn.NativeLinearForm(scalar).assemble(
        gradient=np.array([[1.,0.,0.]])
    )
    np.testing.assert_allclose(
        coupling@mesh.p[0],expected,rtol=4e-12,atol=4e-12
    )
    np.testing.assert_allclose(
        coupling.toarray(),coupling.toarray().T,rtol=4e-12,atol=4e-12
    )
