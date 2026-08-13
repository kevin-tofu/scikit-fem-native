import numpy as np

import skfemntv


def _meshes():
    axis=np.linspace(0.,1.,3)
    original=skfemntv.MeshTet.init_tensor(axis,axis,axis)
    connectivity=original.t.copy()
    selected=np.arange(original.nelements)%2==0
    first=connectivity[0,selected].copy()
    connectivity[0,selected]=connectivity[1,selected]
    connectivity[1,selected]=first
    mixed=skfemntv.MeshTet(original.p,connectivity)
    return original,mixed


def _edge_permutation(source,target):
    lookup={tuple(edge):index for index,edge in enumerate(target.T)}
    return np.array([lookup[tuple(edge)] for edge in source.T])


def _field(x):
    return np.array((1.+x[1],x[0]*x[2],-.25+x[0]*x[1]))


def _exact_curl(x):
    return np.array((x[0],-x[1],-1.+0.*x[2]))


def test_mixed_tet_orientation_preserves_operator_load_and_interpolant():
    original_mesh,mixed_mesh=_meshes()
    original=skfemntv.AffineTetN1Basis(original_mesh,intorder=4)
    mixed=skfemntv.AffineTetN1Basis(mixed_mesh,intorder=4)
    assert original.geometry_diagnostics.inverted_cell_count==0
    assert 0<mixed.geometry_diagnostics.inverted_cell_count<mixed_mesh.nelements

    permutation=_edge_permutation(
        original.dof_map.topology.edges,mixed.dof_map.topology.edges
    )
    original_matrix=skfemntv.TetN1Assembler(original).assemble_maxwell(
        mass_coefficient=1.3,curl_coefficient=.2
    ).toarray()
    mixed_matrix=skfemntv.TetN1Assembler(mixed).assemble_maxwell(
        mass_coefficient=1.3,curl_coefficient=.2
    ).toarray()[permutation][:,permutation]
    np.testing.assert_allclose(original_matrix,mixed_matrix,atol=4e-13)

    original_load=skfemntv.TetN1LinearAssembler(
        original
    ).assemble_vector_load(_field)
    mixed_load=skfemntv.TetN1LinearAssembler(
        mixed
    ).assemble_vector_load(_field)[permutation]
    np.testing.assert_allclose(original_load,mixed_load,atol=4e-14)

    original_coefficients=original.interpolate_edge_moments(_field)
    mixed_coefficients=mixed.interpolate_edge_moments(_field)[permutation]
    np.testing.assert_allclose(
        original_coefficients,mixed_coefficients,atol=3e-15
    )


def test_mixed_tet_orientation_preserves_interpolation_error_norms():
    norms=[]
    for mesh in _meshes():
        basis=skfemntv.AffineTetN1Basis(mesh,intorder=4)
        coefficients=basis.interpolate_edge_moments(_field)
        value_error=basis.evaluate(coefficients)-_field(basis.global_coordinates)
        curl_error=(
            basis.evaluate_curl(coefficients)-_exact_curl(basis.global_coordinates)
        )
        norms.append((
            np.einsum("ieq,ieq,eq->",value_error,value_error,basis.dx),
            np.einsum("ieq,ieq,eq->",curl_error,curl_error,basis.dx),
        ))
    np.testing.assert_allclose(norms[0],norms[1],atol=3e-14)
