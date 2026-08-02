import numpy as np
import pytest
import skfem

import skfemntv


def _linear_mesh(kind):
    if kind=="tri":
        mesh=skfemntv.MeshTri.init_tensor(
            np.linspace(0.,1.,4),np.linspace(0.,1.,3)
        )
        return mesh,skfem.MeshTri(mesh.p,mesh.t)
    if kind=="quad":
        mesh=skfemntv.MeshQuad.init_tensor(
            np.linspace(0.,1.,4),np.linspace(0.,1.,3)
        )
        return mesh,skfem.MeshQuad(mesh.p,mesh.t)
    if kind=="tet":
        mesh=skfemntv.MeshTet.init_tensor(
            [0.,1.],[0.,1.],[0.,1.]
        )
        return mesh,skfem.MeshTet(mesh.p,mesh.t)
    mesh=skfemntv.MeshHex.init_tensor(
        [0.,.5,1.],[0.,1.],[0.,1.]
    )
    return mesh,skfem.MeshHex(mesh.p,mesh.t)


def _facet_coordinate_signatures(mesh):
    signatures=[]
    for facet in mesh.facets.T:
        points=np.round(mesh.p[:,facet].T,14)
        signatures.append(tuple(map(tuple,points[np.lexsort(points.T[::-1])])))
    return sorted(signatures)


@pytest.mark.parametrize("kind",["tri","quad","tet","hex"])
def test_linear_mesh_topology_matches_skfem(kind):
    mesh,reference=_linear_mesh(kind)
    np.testing.assert_array_equal(mesh.facets,reference.facets)
    np.testing.assert_array_equal(mesh.t2f,reference.t2f)
    np.testing.assert_array_equal(mesh.f2t,reference.f2t)
    np.testing.assert_array_equal(
        mesh.boundary_facets(),reference.boundary_facets()
    )
    np.testing.assert_array_equal(
        mesh.interior_facets(),np.flatnonzero(reference.f2t[1]!=-1)
    )
    assert mesh.facets is mesh.facets
    assert mesh.t2f is mesh.t2f
    assert mesh.f2t is mesh.f2t


@pytest.mark.parametrize("kind",["tri","quad","tet","hex"])
def test_quadratic_mesh_retains_linear_topology(kind):
    linear,_=_linear_mesh(kind)
    quadratic=(
        skfemntv.MeshTri2.from_mesh(linear) if kind=="tri" else
        skfemntv.MeshQuad2.from_mesh(linear) if kind=="quad" else
        skfemntv.MeshTet2.from_mesh(linear) if kind=="tet" else
        skfemntv.MeshHex2.from_mesh(linear)
    )
    if kind=="hex":
        assert _facet_coordinate_signatures(quadratic)==(
            _facet_coordinate_signatures(linear)
        )
        def signatures_by_id(mesh):
            result=[]
            for facet in mesh.facets.T:
                points=np.round(mesh.p[:,facet].T,14)
                points=points[np.lexsort(points.T[::-1])]
                result.append(tuple(map(tuple,points)))
            return result
        linear_signatures=signatures_by_id(linear)
        quadratic_signatures=signatures_by_id(quadratic)
        lookup={signature:i for i,signature in enumerate(linear_signatures)}
        remap=np.array([lookup[value] for value in quadratic_signatures])
        np.testing.assert_array_equal(remap[quadratic.t2f],linear.t2f)
        np.testing.assert_array_equal(
            quadratic.f2t[:,np.argsort(remap)],linear.f2t
        )
    else:
        np.testing.assert_array_equal(quadratic.facets,linear.facets)
        np.testing.assert_array_equal(quadratic.t2f,linear.t2f)
        np.testing.assert_array_equal(quadratic.f2t,linear.f2t)
    full=quadratic._facet_connectivity(
        quadratic.boundary_facets(),full=True
    )
    expected_width={"tri":3,"quad":3,"tet":6,"hex":9}[kind]
    assert full.shape==(expected_width,len(quadratic.boundary_facets()))


@pytest.mark.parametrize("kind",["tri","quad","tet","hex"])
def test_topology_queries_match_skfem(kind):
    mesh,reference=_linear_mesh(kind)
    predicate=lambda x:np.isclose(x[0],0.)
    np.testing.assert_array_equal(
        mesh.facets_satisfying(predicate),
        reference.facets_satisfying(predicate),
    )
    np.testing.assert_array_equal(
        mesh.facets_satisfying(predicate,boundaries_only=True),
        reference.facets_satisfying(predicate,boundaries_only=True),
    )
    element_predicate=lambda x:x[0]<.6
    np.testing.assert_array_equal(
        mesh.elements_satisfying(element_predicate),
        reference.elements_satisfying(element_predicate),
    )


@pytest.mark.parametrize("kind",["tri","quad","tet","hex"])
def test_facet_ids_select_basis_and_dofs(kind):
    mesh,_=_linear_mesh(kind)
    element=(
        skfemntv.ElementTriP1() if kind=="tri" else
        skfemntv.ElementQuad1() if kind=="quad" else
        skfemntv.ElementTetP1() if kind=="tet" else
        skfemntv.ElementHex1()
    )
    selected=mesh.facets_satisfying(
        lambda x:np.isclose(x[0],0.),boundaries_only=True
    )
    facet_basis=skfemntv.FacetBasis(
        mesh,skfemntv.ElementVector(element),facets=selected
    )
    basis=skfemntv.Basis(mesh,skfemntv.ElementVector(element))
    expected_nodes=np.unique(mesh.facets[:,selected])
    expected_dofs=np.unique(basis.nodal_dofs[:,expected_nodes])
    np.testing.assert_array_equal(
        basis.get_dofs(facets=selected).all(),expected_dofs
    )
    assert facet_basis.dx.shape[0]==len(selected)
