import numpy as np
import pytest

import skfemntv
from skfemntv.edge_dof_map import build_oriented_edge_dof_map


def _edge_values(dof_map):
    return np.array([
        10.*first+second for first,second in dof_map.topology.edges.T
    ])


def test_one_global_dof_is_assigned_to_each_global_edge():
    mesh=skfemntv.MeshTri.init_tensor(
        np.linspace(0.,1.,3),np.linspace(0.,1.,2)
    )
    dofs=build_oriented_edge_dof_map(mesh)

    assert dofs.ndofs==dofs.topology.edges.shape[1]
    np.testing.assert_array_equal(dofs.element_dofs,dofs.topology.element_edges)
    np.testing.assert_array_equal(dofs.basis_signs,dofs.topology.element_edge_signs)


def test_global_coefficients_round_trip_through_oriented_local_bases():
    mesh=skfemntv.MeshTri.init_tensor(
        np.linspace(0.,1.,3),np.linspace(0.,1.,2)
    )
    dofs=build_oriented_edge_dof_map(mesh)
    global_values=_edge_values(dofs)
    local=dofs.local_coefficients(global_values)
    recovered=dofs.global_moments_from_local(local)

    np.testing.assert_array_equal(recovered,global_values[dofs.element_dofs])


def test_shared_edge_uses_one_dof_and_one_global_oriented_moment():
    mesh=skfemntv.MeshTri.init_tensor(
        np.linspace(0.,1.,2),np.linspace(0.,1.,2)
    )
    dofs=build_oriented_edge_dof_map(mesh)
    counts=np.bincount(dofs.element_dofs.ravel(),minlength=dofs.ndofs)
    shared=int(np.flatnonzero(counts==2)[0])
    locations=np.argwhere(dofs.element_dofs==shared)
    global_values=np.arange(1.,dofs.ndofs+1.)
    recovered=dofs.global_moments_from_local(
        dofs.local_coefficients(global_values)
    )

    assert len(locations)==2
    for local,cell in locations:
        assert recovered[local,cell]==global_values[shared]


def test_cell_vertex_reordering_preserves_global_edge_field():
    mesh=skfemntv.MeshTri.init_tensor(
        np.linspace(0.,1.,3),np.linspace(0.,1.,2)
    )
    reordered=skfemntv.MeshTri(mesh.p,mesh.t[[1,0,2]])
    for candidate in (mesh,reordered):
        dofs=build_oriented_edge_dof_map(candidate)
        global_values=_edge_values(dofs)
        recovered=dofs.global_moments_from_local(
            dofs.local_coefficients(global_values)
        )
        np.testing.assert_array_equal(
            recovered,global_values[dofs.element_dofs]
        )


def test_edge_dof_map_rejects_wrong_coefficient_shapes():
    dofs=build_oriented_edge_dof_map(skfemntv.MeshTri())
    with pytest.raises(ValueError,match="global edge coefficients"):
        dofs.local_coefficients(np.zeros((dofs.ndofs,1)))
    with pytest.raises(ValueError,match="local edge coefficients"):
        dofs.global_moments_from_local(np.zeros(dofs.ndofs))
