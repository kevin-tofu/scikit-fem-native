import numpy as np
import pytest
import skfem
from skfem.helpers import curl,dot
from scipy.sparse.linalg import spsolve

import skfemntv
from skfemntv.hcurl_assembler import TriN1Assembler
from skfemntv.hcurl_basis import AffineTriN1Basis


def _mesh():
    return skfemntv.MeshTri.init_tensor(
        np.linspace(0.,1.,5),np.linspace(0.,1.,4)
    ).with_boundaries({
        "left":lambda x:np.isclose(x[0],0.),
        "bottom":lambda x:np.isclose(x[1],0.),
    })


def test_all_and_named_boundary_edge_dofs_match_scikit_fem():
    mesh=_mesh()
    basis=AffineTriN1Basis(mesh)
    reference_mesh=skfem.MeshTri(mesh.p,mesh.t[:3]).with_boundaries({
        "left":lambda x:np.isclose(x[0],0.),
        "bottom":lambda x:np.isclose(x[1],0.),
    })
    reference=skfem.Basis(reference_mesh,skfem.ElementTriN1())
    reference_edges={tuple(edge):i for i,edge in enumerate(reference.mesh.facets.T)}
    permutation=np.array([
        reference_edges[tuple(edge)] for edge in basis.dof_map.topology.edges.T
    ])
    inverse=np.empty_like(permutation)
    inverse[permutation]=np.arange(len(permutation))

    for boundary in (None,"left","bottom"):
        expected=reference.get_dofs(boundary).all()
        np.testing.assert_array_equal(
            basis.boundary_dofs(boundary),np.sort(inverse[expected])
        )


def test_predicate_and_named_union_are_unique():
    basis=AffineTriN1Basis(_mesh())
    predicate=basis.boundary_dofs(lambda x:np.isclose(x[0],0.))
    np.testing.assert_array_equal(predicate,basis.boundary_dofs("left"))
    union=basis.boundary_dofs(("left","bottom"))
    expected=np.union1d(
        basis.boundary_dofs("left"),basis.boundary_dofs("bottom")
    )
    np.testing.assert_array_equal(union,expected)
    assert len(union)==len(np.unique(union))


def test_boundary_selection_reports_invalid_requests():
    basis=AffineTriN1Basis(_mesh())
    with pytest.raises(KeyError,match="unknown boundary"):
        basis.boundary_dofs("missing")
    with pytest.raises(ValueError,match="one boolean"):
        basis.boundary_dofs(lambda x:np.ones(2,dtype=bool))
    interior=int(basis.mesh.interior_facets()[0])
    with pytest.raises(ValueError,match="boundary facet"):
        basis.boundary_dofs([interior])


def test_constrained_maxwell_solve_matches_scikit_fem_matrix_solution():
    mesh=_mesh()
    basis=AffineTriN1Basis(mesh,intorder=3)
    matrix=TriN1Assembler(basis).assemble_maxwell(
        mass_coefficient=1.,curl_coefficient=.2
    ).copy()
    constrained=basis.boundary_dofs()
    free=np.setdiff1d(np.arange(basis.N),constrained)
    rhs=np.sin(np.arange(basis.N)+.5)
    solution=np.zeros(basis.N)
    solution[free]=spsolve(matrix[free][:,free],rhs[free])

    residual=matrix@solution-rhs
    np.testing.assert_allclose(residual[free],0.,atol=2e-13)
    np.testing.assert_array_equal(solution[constrained],0.)

    reference_mesh=skfem.MeshTri(mesh.p,mesh.t[:3])
    reference=skfem.Basis(reference_mesh,skfem.ElementTriN1(),intorder=3)

    @skfem.BilinearForm
    def maxwell(u,v,w):
        return dot(u,v)+.2*curl(u)*curl(v)

    reference_edges={tuple(edge):i for i,edge in enumerate(reference.mesh.facets.T)}
    permutation=np.array([
        reference_edges[tuple(edge)] for edge in basis.dof_map.topology.edges.T
    ])
    reference_matrix=skfem.asm(maxwell,reference).toarray()
    reference_matrix=reference_matrix[permutation][:,permutation]
    expected=np.zeros(basis.N)
    expected[free]=np.linalg.solve(
        reference_matrix[np.ix_(free,free)],rhs[free]
    )
    np.testing.assert_allclose(solution,expected,atol=2e-13)
