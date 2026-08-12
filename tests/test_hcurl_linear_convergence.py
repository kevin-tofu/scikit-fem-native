import numpy as np
import pytest
import skfem
from scipy.sparse.linalg import spsolve
from skfem.helpers import dot

import skfemntv


def _edge_permutation(native,reference):
    lookup={tuple(edge):i for i,edge in enumerate(reference.mesh.facets.T)}
    return np.array([
        lookup[tuple(edge)] for edge in native.dof_map.topology.edges.T
    ])


def test_vector_load_matches_scikit_fem_and_reuses_result():
    mesh=skfemntv.MeshTri.init_tensor(
        np.linspace(0.,1.,5),np.linspace(0.,1.,4)
    )
    basis=skfemntv.AffineTriN1Basis(mesh,intorder=4)
    assembler=skfemntv.TriN1LinearAssembler(basis)

    def load(points):
        x,y=points
        return np.array((1.+x,-.3+.5*y))

    actual=assembler.assemble_vector_load(load)
    saved=actual.copy()
    reference=skfem.Basis(
        skfem.MeshTri(mesh.p,mesh.t[:3]),skfem.ElementTriN1(),
        quadrature=(basis.X,basis.W),
    )

    @skfem.LinearForm
    def reference_load(v,w):
        return dot(load(w.x),v)

    expected=skfem.asm(reference_load,reference)
    permutation=_edge_permutation(basis,reference)
    # scikit-fem's global basis has the opposite sign from the ascending-edge
    # moment convention used by AffineTriN1Basis.
    np.testing.assert_allclose(saved,-expected[permutation],atol=4e-14)
    repeated=assembler.assemble_vector_load(np.array((2.,-1.))[:,None,None])
    assert repeated is actual
    assert not np.allclose(repeated,saved)


def test_vector_load_validates_basis_and_shape():
    with pytest.raises(TypeError,match="AffineTriN1Basis"):
        skfemntv.TriN1LinearAssembler(object())
    basis=skfemntv.AffineTriN1Basis(skfemntv.MeshTri())
    assembler=skfemntv.TriN1LinearAssembler(basis)
    with pytest.raises(ValueError,match="vector load"):
        assembler.assemble_vector_load(np.ones(3))


def _exact(points):
    x,y=points
    return np.array((np.sin(np.pi*y),np.sin(np.pi*x)))


def _exact_curl(points):
    x,y=points
    return np.pi*(np.cos(np.pi*x)-np.cos(np.pi*y))


def _solve_and_error(divisions,beta=.2):
    mesh=skfemntv.MeshTri.init_tensor(
        np.linspace(0.,1.,divisions+1),np.linspace(0.,1.,divisions+1)
    )
    basis=skfemntv.AffineTriN1Basis(mesh,intorder=5)
    matrix=skfemntv.TriN1Assembler(basis).assemble_maxwell(
        mass_coefficient=1.,curl_coefficient=beta
    ).copy()
    factor=1.+beta*np.pi**2
    rhs=skfemntv.TriN1LinearAssembler(basis).assemble_vector_load(
        lambda points:factor*_exact(points)
    ).copy()
    boundary=basis.boundary_dofs()
    free=np.setdiff1d(np.arange(basis.N),boundary)
    solution=np.zeros(basis.N)
    solution[free]=spsolve(matrix[free][:,free],rhs[free])
    points=basis.global_coordinates
    value_error=basis.evaluate(solution)-_exact(points)
    curl_error=basis.evaluate_curl(solution)-_exact_curl(points)
    l2=np.sqrt(np.einsum("ieq,ieq,eq->",value_error,value_error,basis.dx))
    curl_l2=np.sqrt(np.einsum("eq,eq,eq->",curl_error,curl_error,basis.dx))
    return l2,curl_l2,np.hypot(l2,curl_l2)


def test_manufactured_maxwell_solution_converges_in_hcurl_norm():
    history=np.array([_solve_and_error(n) for n in (4,8,16)])
    assert np.all(history[1:]<history[:-1])
    rates=np.log2(history[:-1]/history[1:])
    assert np.all(rates>.8),rates
