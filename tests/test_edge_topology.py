import numpy as np
import pytest

import skfemntv
from skfemntv.edge_topology import (
    boundary_edge_ids,
    build_oriented_edge_topology,
)


def _assert_local_orientation_contract(mesh,topology):
    corners=mesh.t[:3 if mesh.dim()==2 else 4]
    for cell in range(mesh.nelements):
        for local,(start,end) in enumerate(topology.local_edges):
            directed=(int(corners[start,cell]),int(corners[end,cell]))
            edge=tuple(topology.edges[:,topology.element_edges[local,cell]])
            expected=edge if topology.element_edge_signs[local,cell]==1 else edge[::-1]
            assert directed==expected


def test_triangle_edges_have_deterministic_ids_and_orientation_signs():
    mesh=skfemntv.MeshTri.init_tensor(
        np.linspace(0.0,1.0,3),np.linspace(0.0,1.0,2)
    )
    topology=build_oriented_edge_topology(mesh)

    assert topology.element_edges.shape==(3,mesh.nelements)
    assert topology.element_edge_signs.shape==(3,mesh.nelements)
    assert set(np.unique(topology.element_edge_signs)) <= {-1,1}
    assert np.all(topology.edges[0] < topology.edges[1])
    _assert_local_orientation_contract(mesh,topology)


def test_triangle_vertex_reordering_preserves_global_edges():
    mesh=skfemntv.MeshTri.init_tensor(
        np.linspace(0.0,1.0,3),np.linspace(0.0,1.0,2)
    )
    reordered=skfemntv.MeshTri(mesh.p,mesh.t[[1,0,2]])
    original=build_oriented_edge_topology(mesh)
    changed=build_oriented_edge_topology(reordered)

    assert set(map(tuple,original.edges.T))==set(map(tuple,changed.edges.T))
    _assert_local_orientation_contract(reordered,changed)


def test_tetrahedron_edges_and_boundary_selection_cover_all_single_cell_edges():
    mesh=skfemntv.MeshTet()
    topology=build_oriented_edge_topology(mesh)

    assert topology.element_edges.shape==(6,mesh.nelements)
    assert len(set(map(tuple,topology.edges.T)))==topology.edges.shape[1]
    _assert_local_orientation_contract(mesh,topology)
    np.testing.assert_array_equal(
        boundary_edge_ids(mesh,topology),np.arange(topology.edges.shape[1])
    )


def test_interior_triangle_edge_is_not_selected_as_boundary():
    mesh=skfemntv.MeshTri.init_tensor(
        np.linspace(0.0,1.0,2),np.linspace(0.0,1.0,2)
    )
    topology=build_oriented_edge_topology(mesh)
    boundary=set(map(int,boundary_edge_ids(mesh,topology)))
    interior={
        edge for edge in range(topology.edges.shape[1]) if edge not in boundary
    }
    assert len(interior)==1


def test_repeated_triangle_corner_is_rejected_with_cell_id():
    class InvalidTriangleMesh:
        t=np.array([[0],[0],[1]],dtype=np.int64)

        @staticmethod
        def dim():
            return 2

    with pytest.raises(ValueError,match="cell 0 repeats a corner vertex"):
        build_oriented_edge_topology(InvalidTriangleMesh())
