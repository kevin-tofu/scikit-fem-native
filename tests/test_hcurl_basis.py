import numpy as np
import pytest
import skfem
from skfem.helpers import curl,dot

import skfemntv
from skfemntv.hcurl_basis import AffineTriN1Basis


def _scatter(basis,element_matrices):
    result=np.zeros((basis.N,basis.N))
    for cell,matrix in enumerate(element_matrices):
        dofs=basis.element_dofs[:,cell]
        result[np.ix_(dofs,dofs)]+=matrix
    return result


def _reference(mesh):
    reference_mesh=skfem.MeshTri(mesh.p,mesh.t[:3])
    return skfem.Basis(reference_mesh,skfem.ElementTriN1(),intorder=2)


def test_minimal_hcurl_basis_shapes_and_integration_weights():
    mesh=skfemntv.MeshTri.init_tensor(
        np.linspace(0.,1.,3),np.linspace(0.,1.,2)
    )
    basis=AffineTriN1Basis(mesh)
    assert basis.values.shape==(3,2,mesh.nelements,len(basis.W))
    assert basis.curls.shape==(3,mesh.nelements,len(basis.W))
    np.testing.assert_array_equal(
        basis.values,basis._element_values.transpose(1,2,0,3)
    )
    np.testing.assert_array_equal(
        basis.curls,basis._element_curls.transpose(1,0,2)
    )
    np.testing.assert_allclose(basis.dx.sum(axis=1),.5*np.abs(basis.detJ))


def test_mass_and_curl_curl_matrices_match_scikit_fem():
    mesh=skfemntv.MeshTri.init_tensor(
        np.array((-.2,.4,1.3)),np.array((.1,.8))
    )
    basis=AffineTriN1Basis(mesh)
    reference=_reference(mesh)

    @skfem.BilinearForm
    def mass(u,v,w):
        return dot(u,v)

    @skfem.BilinearForm
    def curl_curl(u,v,w):
        return curl(u)*curl(v)

    # Both spaces identify a global DOF by its ascending vertex pair, but the
    # numeric edge IDs are assigned in different traversal orders.
    reference_edges={tuple(edge):index for index,edge in enumerate(reference.mesh.facets.T)}
    permutation=np.array([
        reference_edges[tuple(edge)] for edge in basis.dof_map.topology.edges.T
    ])
    expected_mass=skfem.asm(mass,reference).toarray()[permutation][:,permutation]
    expected_curl=skfem.asm(curl_curl,reference).toarray()[permutation][:,permutation]

    np.testing.assert_allclose(
        _scatter(basis,basis.element_mass_matrices()),
        expected_mass,rtol=3e-14,atol=3e-14,
    )
    np.testing.assert_allclose(
        _scatter(basis,basis.element_curl_curl_matrices()),
        expected_curl,rtol=3e-14,atol=3e-14,
    )


def test_vertex_reordering_preserves_assembled_operators():
    mesh=skfemntv.MeshTri.init_tensor(
        np.linspace(0.,1.,3),np.linspace(0.,1.,2)
    )
    reordered=skfemntv.MeshTri(mesh.p,mesh.t[[1,0,2]])
    original=AffineTriN1Basis(mesh)
    changed=AffineTriN1Basis(reordered)

    def by_edge(basis,matrix):
        lookup={tuple(edge):i for i,edge in enumerate(basis.dof_map.topology.edges.T)}
        order=np.array([lookup[edge] for edge in sorted(lookup)])
        return matrix[np.ix_(order,order)]

    np.testing.assert_allclose(
        by_edge(original,_scatter(original,original.element_mass_matrices())),
        by_edge(changed,_scatter(changed,changed.element_mass_matrices())),
        atol=3e-14,
    )
    np.testing.assert_allclose(
        by_edge(original,_scatter(original,original.element_curl_curl_matrices())),
        by_edge(changed,_scatter(changed,changed.element_curl_curl_matrices())),
        atol=3e-14,
    )


def test_minimal_hcurl_basis_rejects_nontriangle_and_singular_meshes():
    with pytest.raises(TypeError,match="triangular"):
        AffineTriN1Basis(skfemntv.MeshTet())
    singular=skfemntv.MeshTri(
        np.array(((0.,1.,2.),(0.,0.,0.))),np.array(((0,),(1,),(2,)))
    )
    with pytest.raises(ValueError,match="nonsingular"):
        AffineTriN1Basis(singular)
