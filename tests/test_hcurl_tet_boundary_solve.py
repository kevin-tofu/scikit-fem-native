import numpy as np
import pytest
import skfem
from scipy.sparse.linalg import spsolve
from skfem.helpers import curl,dot

import skfemntv


def _mesh():
    axis=np.linspace(0.,1.,3)
    return skfemntv.MeshTet.init_tensor(axis,axis,axis).with_boundaries({
        "left":lambda x:np.isclose(x[0],0.),
        "bottom":lambda x:np.isclose(x[1],0.),
    })


def _reference(mesh,intorder=3):
    reference_mesh=skfem.MeshTet(mesh.p,mesh.t[:4]).with_boundaries({
        "left":lambda x:np.isclose(x[0],0.),
        "bottom":lambda x:np.isclose(x[1],0.),
    })
    reference=skfem.Basis(
        reference_mesh,skfem.ElementTetN1(),intorder=intorder
    )
    return reference_mesh,reference


def _permutation(basis,reference_mesh):
    lookup={tuple(edge):index for index,edge in enumerate(reference_mesh.edges.T)}
    return np.array([
        lookup[tuple(edge)] for edge in basis.dof_map.topology.edges.T
    ])


def test_tet_all_named_predicate_and_union_boundary_dofs_match_scikit_fem():
    mesh=_mesh()
    basis=skfemntv.AffineTetN1Basis(mesh)
    reference_mesh,reference=_reference(mesh)
    permutation=_permutation(basis,reference_mesh)
    inverse=np.empty_like(permutation)
    inverse[permutation]=np.arange(len(permutation))
    for boundary in (None,"left","bottom"):
        expected=reference.get_dofs(boundary).all()
        np.testing.assert_array_equal(
            basis.boundary_dofs(boundary),np.sort(inverse[expected])
        )
    np.testing.assert_array_equal(
        basis.boundary_dofs(lambda x:np.isclose(x[0],0.)),
        basis.boundary_dofs("left"),
    )
    np.testing.assert_array_equal(
        basis.boundary_dofs(("left","bottom")),
        np.union1d(basis.boundary_dofs("left"),basis.boundary_dofs("bottom")),
    )


def test_tet_boundary_selection_rejects_invalid_requests():
    basis=skfemntv.AffineTetN1Basis(_mesh())
    with pytest.raises(KeyError,match="unknown boundary"):
        basis.boundary_dofs("missing")
    with pytest.raises(ValueError,match="one boolean"):
        basis.boundary_dofs(lambda x:np.ones(2,dtype=bool))
    with pytest.raises(ValueError,match="boundary facet"):
        basis.boundary_dofs([int(basis.mesh.interior_facets()[0])])


def test_tet_vector_load_and_constrained_maxwell_solve_match_scikit_fem():
    mesh=_mesh()
    basis=skfemntv.AffineTetN1Basis(mesh,intorder=3)
    field=lambda x:np.array((
        1.+x[1],.5+x[2],-.25+x[0]
    ))
    load=skfemntv.TetN1LinearAssembler(basis).assemble_vector_load(field).copy()
    matrix=skfemntv.TetN1Assembler(basis).assemble_maxwell(
        mass_coefficient=1.,curl_coefficient=.2
    ).copy()
    constrained=basis.boundary_dofs()
    free=np.setdiff1d(np.arange(basis.N),constrained)
    solution=np.zeros(basis.N)
    solution[free]=spsolve(matrix[free][:,free],load[free])

    reference_mesh,reference=_reference(mesh)
    permutation=_permutation(basis,reference_mesh)

    @skfem.LinearForm
    def reference_load(v,w):
        return dot(np.array((1.+w.x[1],.5+w.x[2],-.25+w.x[0])),v)

    @skfem.BilinearForm
    def reference_maxwell(u,v,w):
        return dot(u,v)+.2*dot(curl(u),curl(v))

    expected_load=skfem.asm(reference_load,reference)[permutation]
    expected_matrix=skfem.asm(
        reference_maxwell,reference
    ).toarray()[permutation][:,permutation]
    np.testing.assert_allclose(load,expected_load,atol=3e-14)
    expected=np.zeros(basis.N)
    expected[free]=np.linalg.solve(
        expected_matrix[np.ix_(free,free)],expected_load[free]
    )
    np.testing.assert_allclose(solution,expected,atol=3e-13)
    np.testing.assert_allclose((matrix@solution-load)[free],0.,atol=3e-13)
    np.testing.assert_array_equal(solution[constrained],0.)
