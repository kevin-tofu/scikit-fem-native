import numpy as np
import pytest
import skfem
from skfem.helpers import curl,dot

import skfemntv


def _bases():
    axis=np.linspace(0.,1.,3)
    mesh=skfemntv.MeshTet.init_tensor(axis,axis,axis)
    native=skfemntv.AffineTetN1Basis(mesh,intorder=3)
    reference_mesh=skfem.MeshTet(mesh.p,mesh.t[:4])
    reference=skfem.Basis(reference_mesh,skfem.ElementTetN1(),intorder=3)
    lookup={tuple(edge):index for index,edge in enumerate(reference_mesh.edges.T)}
    permutation=np.array([
        lookup[tuple(edge)] for edge in native.dof_map.topology.edges.T
    ])
    return native,reference,permutation


def test_tet_n1_mass_curl_and_maxwell_match_scikit_fem():
    basis,reference,permutation=_bases()
    assembler=skfemntv.TetN1Assembler(basis)

    @skfem.BilinearForm
    def mass(u,v,w):
        return 2.3*dot(u,v)

    @skfem.BilinearForm
    def curl_curl(u,v,w):
        return .7*dot(curl(u),curl(v))

    expected_mass=skfem.asm(mass,reference).toarray()[permutation][:,permutation]
    expected_curl=skfem.asm(
        curl_curl,reference
    ).toarray()[permutation][:,permutation]
    np.testing.assert_allclose(
        assembler.assemble_mass(2.3).toarray(),expected_mass,atol=3e-14
    )
    np.testing.assert_allclose(
        assembler.assemble_curl_curl(.7).toarray(),expected_curl,atol=3e-13
    )
    np.testing.assert_allclose(
        assembler.assemble_maxwell(
            mass_coefficient=2.3,curl_coefficient=.7
        ).toarray(),expected_mass+expected_curl,atol=3e-13
    )


def test_tet_n1_quadrature_coefficients_and_repeated_structure():
    basis,_,_=_bases()
    assembler=skfemntv.TetN1Assembler(basis)
    coefficient=np.linspace(.5,1.5,basis.dx.size).reshape(basis.dx.shape)
    matrix=assembler.assemble_mass(coefficient)
    assert np.all(np.isfinite(matrix.data))
    identity=(id(matrix),id(matrix.data),id(matrix.indices),id(matrix.indptr))
    repeated=assembler.assemble_curl_curl(coefficient)
    assert identity==(
        id(repeated),id(repeated.data),id(repeated.indices),id(repeated.indptr)
    )


def test_tet_n1_assembler_type_coefficient_and_memory_contract():
    basis,_,_=_bases()
    estimate=skfemntv.estimate_tet_n1_assembly_memory(basis)
    assert estimate.kind=="hcurl_tet_n1"
    assert estimate.row_local_dofs==estimate.column_local_dofs==6
    with pytest.raises(TypeError,match="AffineTetN1Basis"):
        skfemntv.TetN1Assembler(skfemntv.AffineTriN1Basis(skfemntv.MeshTri()))
    with pytest.raises(skfemntv.AssemblyMemoryBudgetError):
        skfemntv.TetN1Assembler(basis,memory_limit_bytes=1)
    assembler=skfemntv.TetN1Assembler(basis)
    with pytest.raises(ValueError,match="mass coefficient"):
        assembler.assemble_mass(np.zeros((2,2,2)))
