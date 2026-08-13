import numpy as np
import pytest
import skfem
from scipy.sparse import coo_matrix
from skfem.helpers import curl,dot

import skfemntv


def _assemble_local(basis,elements):
    dofs=basis.element_dofs.T
    rows=np.repeat(dofs,6,axis=1).ravel()
    columns=np.tile(dofs,(1,6)).ravel()
    return coo_matrix(
        (elements.ravel(),(rows,columns)),shape=(basis.N,basis.N)
    ).tocsr()


def _mesh():
    axis=np.linspace(0.,1.,2)
    return skfemntv.MeshTet.init_tensor(axis,axis,axis)


def test_affine_tet_n1_basis_layout_and_geometry_contract():
    basis=skfemntv.AffineTetN1Basis(_mesh(),intorder=3)
    cells=basis.mesh.nelements
    quadrature=len(basis.W)
    assert basis.values.shape==(6,3,cells,quadrature)
    assert basis.curls.shape==(6,3,cells,quadrature)
    assert basis.dx.shape==(cells,quadrature)
    assert basis.element_dofs.shape==(6,cells)
    assert basis.geometry_diagnostics.minimum_volume>0.
    assert basis.geometry_diagnostics.inverted_cell_count>=0
    np.testing.assert_allclose(np.sum(basis.dx),1.)


def test_affine_tet_n1_element_operators_match_scikit_fem():
    mesh=_mesh()
    basis=skfemntv.AffineTetN1Basis(mesh,intorder=3)
    reference_mesh=skfem.MeshTet(mesh.p,mesh.t[:4])
    reference=skfem.Basis(reference_mesh,skfem.ElementTetN1(),intorder=3)

    @skfem.BilinearForm
    def mass(u,v,w):
        return dot(u,v)

    @skfem.BilinearForm
    def curl_curl(u,v,w):
        return dot(curl(u),curl(v))

    lookup={tuple(edge):index for index,edge in enumerate(reference_mesh.edges.T)}
    permutation=np.array([
        lookup[tuple(edge)] for edge in basis.dof_map.topology.edges.T
    ])
    actual_mass=_assemble_local(basis,basis.element_mass_matrices()).toarray()
    actual_curl=_assemble_local(
        basis,basis.element_curl_curl_matrices()
    ).toarray()
    expected_mass=skfem.asm(mass,reference).toarray()[permutation][:,permutation]
    expected_curl=skfem.asm(
        curl_curl,reference
    ).toarray()[permutation][:,permutation]
    np.testing.assert_allclose(actual_mass,expected_mass,atol=2e-14)
    np.testing.assert_allclose(actual_curl,expected_curl,atol=2e-13)


def test_affine_tet_n1_rejects_wrong_or_singular_mesh():
    with pytest.raises(TypeError,match="tetrahedral mesh"):
        skfemntv.AffineTetN1Basis(skfemntv.MeshTri())
    singular=skfemntv.MeshTet(
        np.array(((0.,1.,0.,1.),(0.,0.,1.,1.),(0.,0.,0.,0.))),
        np.array(((0,),(1,),(2,),(3,))),
    )
    with pytest.raises(ValueError,match="nonsingular tetrahedra"):
        skfemntv.AffineTetN1Basis(singular)


def test_affine_tet_n1_aspect_threshold_is_explicit():
    basis=skfemntv.AffineTetN1Basis(_mesh())
    measured=basis.geometry_diagnostics.maximum_aspect_ratio
    with pytest.raises(ValueError,match="aspect ratio exceeds"):
        skfemntv.AffineTetN1Basis(_mesh(),max_aspect_ratio=.99*measured)
    with pytest.raises(ValueError,match="positive finite scalar"):
        skfemntv.AffineTetN1Basis(_mesh(),max_aspect_ratio=np.inf)
