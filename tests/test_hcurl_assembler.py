import numpy as np
import pytest
import skfem
from skfem.helpers import curl,dot

import skfemntv
from skfemntv.hcurl_assembler import (
    TriN1Assembler,estimate_tri_n1_assembly_memory,
)
from skfemntv.hcurl_basis import AffineTriN1Basis


def _bases():
    mesh=skfemntv.MeshTri.init_tensor(
        np.array((-.2,.4,1.3)),np.array((.1,.8))
    )
    native=AffineTriN1Basis(mesh,intorder=3)
    reference=skfem.Basis(
        skfem.MeshTri(mesh.p,mesh.t[:3]),skfem.ElementTriN1(),intorder=3
    )
    edge_lookup={tuple(edge):i for i,edge in enumerate(reference.mesh.facets.T)}
    permutation=np.array([
        edge_lookup[tuple(edge)] for edge in native.dof_map.topology.edges.T
    ])
    return native,reference,permutation


def _reference_matrices(reference,mass_coefficient=1.,curl_coefficient=1.):
    @skfem.BilinearForm
    def mass(u,v,w):
        return w.mass*dot(u,v)

    @skfem.BilinearForm
    def curl_curl(u,v,w):
        return w.curl_value*curl(u)*curl(v)

    return (
        skfem.asm(mass,reference,mass=mass_coefficient).toarray(),
        skfem.asm(
            curl_curl,reference,curl_value=curl_coefficient
        ).toarray(),
    )


def test_dedicated_mass_curl_and_maxwell_match_scikit_fem():
    basis,reference,permutation=_bases()
    assembler=TriN1Assembler(basis)
    expected_mass,expected_curl=_reference_matrices(reference,2.3,.7)
    expected_mass=expected_mass[permutation][:,permutation]
    expected_curl=expected_curl[permutation][:,permutation]

    np.testing.assert_allclose(
        assembler.assemble_mass(2.3).toarray(),expected_mass,atol=4e-14
    )
    np.testing.assert_allclose(
        assembler.assemble_curl_curl(.7).toarray(),expected_curl,atol=4e-14
    )
    np.testing.assert_allclose(
        assembler.assemble_maxwell(
            mass_coefficient=2.3,curl_coefficient=.7
        ).toarray(),expected_mass+expected_curl,atol=5e-14
    )


def test_quadrature_dependent_coefficients_match_scikit_fem():
    basis,reference,permutation=_bases()
    assembler=TriN1Assembler(basis)
    x=np.empty((2,)+basis.dx.shape)
    for cell in range(basis.mesh.nelements):
        vertices=basis.mesh.p[:,basis.mesh.t[:3,cell]]
        x[:,cell]=vertices[:,0,None]+basis.jacobians[...,cell]@basis.X
    mass_value=1.+.2*x[0]
    curl_value=.5+.3*x[1]
    @skfem.BilinearForm
    def reference_maxwell(u,v,w):
        return (1.+.2*w.x[0])*dot(u,v)+(.5+.3*w.x[1])*curl(u)*curl(v)

    expected_mass=skfem.asm(reference_maxwell,reference).toarray()
    expected_curl=np.zeros_like(expected_mass)
    expected=(expected_mass+expected_curl)[permutation][:,permutation]
    np.testing.assert_allclose(
        assembler.assemble_maxwell(
            mass_coefficient=mass_value,curl_coefficient=curl_value
        ).toarray(),expected,atol=5e-14,
    )


def test_repeated_assembly_reuses_csr_structure_and_matrix_object():
    basis,_,_=_bases()
    assembler=TriN1Assembler(basis)
    matrix=assembler.assemble_mass()
    mass_values=matrix.toarray().copy()
    identity=(id(matrix),id(matrix.data),id(matrix.indices),id(matrix.indptr))
    repeated=assembler.assemble_curl_curl()
    assert identity==(
        id(repeated),id(repeated.data),id(repeated.indices),id(repeated.indptr)
    )
    assert not np.allclose(mass_values,repeated.toarray())


def test_coefficient_validation_and_memory_preflight():
    basis,_,_=_bases()
    estimate=estimate_tri_n1_assembly_memory(basis)
    assert estimate.kind=="hcurl_tri_n1"
    assert estimate.rows==estimate.columns==basis.N
    with pytest.raises(skfemntv.AssemblyMemoryBudgetError):
        TriN1Assembler(basis,memory_limit_bytes=1)
    assembler=TriN1Assembler(basis)
    with pytest.raises(ValueError,match="mass coefficient"):
        assembler.assemble_mass(np.zeros((2,2,2)))
