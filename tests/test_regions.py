import numpy as np
import pytest
import skfem

import skfemntv
from skfemntv.helpers import ddot,grad


def test_region_normalizes_ids_and_exposes_diagnostics():
    region=skfemntv.CellRegion([4,1,4,2],entity_count=6)

    np.testing.assert_array_equal(region,[1,2,4])
    assert region.diagnostics==skfemntv.RegionDiagnostics(
        entity_count=6,selected_count=3,is_empty=False,
        minimum_id=1,maximum_id=4,
    )
    assert not region.ids.flags.writeable
    with pytest.raises(ValueError):
        region.ids[0]=0


def test_region_algebra_and_complement():
    left=skfemntv.CellRegion([0,1,3],5)
    right=skfemntv.CellRegion([1,2,3],5)

    np.testing.assert_array_equal(left|right,[0,1,2,3])
    np.testing.assert_array_equal(left&right,[1,3])
    np.testing.assert_array_equal(left-right,[0])
    np.testing.assert_array_equal(~left,[2,4])
    assert isinstance(left|right,skfemntv.CellRegion)


def test_region_algebra_rejects_incompatible_entities():
    cells=skfemntv.CellRegion([0],2)
    facets=skfemntv.FacetRegion([0],2)

    with pytest.raises(TypeError,match="same entity kind"):
        cells|facets
    with pytest.raises(ValueError,match="counts"):
        cells|skfemntv.CellRegion([0],3)
    with pytest.raises(ValueError,match="requires entity_count"):
        ~skfemntv.NodeRegion([0])


def test_facet_region_algebra_preserves_orientation_metadata():
    left=skfemntv.FacetRegion(
        [3,1],5,sides=[1,0],normal_signs=[1,-1]
    )
    right=skfemntv.FacetRegion([1,2],5,sides=[0,1],normal_signs=[-1,1])
    combined=left|right

    np.testing.assert_array_equal(combined,[1,2,3])
    np.testing.assert_array_equal(combined.sides,[0,1,1])
    np.testing.assert_array_equal(combined.normal_signs,[-1,1,1])
    intersection=left&right
    np.testing.assert_array_equal(intersection,[1])
    np.testing.assert_array_equal(intersection.normal_signs,[-1])

    conflict=skfemntv.FacetRegion([1],5,sides=[1])
    with pytest.raises(ValueError,match="orientation metadata conflicts"):
        left|conflict


def test_mesh_predicates_return_first_class_regions():
    mesh=skfemntv.MeshTri.init_tensor(
        np.linspace(0.,1.,4),np.linspace(0.,1.,3)
    )
    cells=mesh.elements_satisfying(lambda x:x[0]<.5)
    facets=mesh.facets_satisfying(
        lambda x:np.isclose(x[0],0.),boundaries_only=True
    )

    assert isinstance(cells,skfemntv.CellRegion)
    assert cells.entity_count==mesh.nelements
    assert isinstance(facets,skfemntv.FacetRegion)
    assert facets.entity_count==mesh.facets.shape[1]
    assert len(cells)>0
    assert len(facets)>0


def test_named_subdomain_selects_basis_without_renumbering():
    base=skfemntv.MeshTri.init_tensor(
        np.linspace(0.,1.,5),np.linspace(0.,1.,4)
    )
    last=base.nelements-1
    mesh=base.with_subdomains({
        "left":lambda x:x[0]<.5,
        "last":[last],
    })
    element=skfemntv.ElementVector(skfemntv.ElementTriP1(),dim=1)
    full=skfemntv.Basis(mesh,element)
    left=skfemntv.Basis(mesh,element,elements="left")

    assert isinstance(mesh.subdomains["left"],skfemntv.CellRegion)
    np.testing.assert_array_equal(left.tind,mesh.subdomains["left"])
    assert left.N==full.N
    assert last in mesh.subdomains["last"]

    @skfemntv.BilinearForm
    def stiffness(u,v,w):
        return ddot(grad(u),grad(v))

    direct=skfemntv.asm(stiffness,left)
    selected=skfemntv.asm(
        stiffness,full.with_elements(mesh.subdomains["left"])
    )
    np.testing.assert_allclose(direct.toarray(),selected.toarray())


def test_named_boundary_is_a_region_and_constructs_facet_basis():
    mesh=skfemntv.MeshQuad.init_tensor(
        np.linspace(0.,1.,4),np.linspace(0.,1.,3)
    ).with_boundaries({"left":lambda x:np.isclose(x[0],0.)})
    element=skfemntv.ElementVector(skfemntv.ElementQuad1(),dim=1)

    assert isinstance(mesh.boundaries["left"],skfemntv.FacetRegion)
    named=skfemntv.FacetBasis(mesh,element,facets="left")
    direct=skfemntv.FacetBasis(
        mesh,element,facets=mesh.boundaries["left"]
    )
    np.testing.assert_array_equal(named.element_dofs,direct.element_dofs)
    np.testing.assert_allclose(
        named.global_coordinates,direct.global_coordinates
    )
    np.testing.assert_allclose(named.dx,direct.dx)


def test_exterior_normal_selection_flips_normal_without_changing_parent():
    mesh=skfemntv.MeshQuad.init_tensor(
        np.linspace(0.,1.,4),np.linspace(0.,1.,3)
    )
    right_facing=mesh.facets_satisfying(
        lambda x:np.isclose(x[0],0.),
        boundaries_only=True,normal=np.array([1.,0.]),
    )
    assert isinstance(right_facing,skfemntv.FacetRegion)
    np.testing.assert_array_equal(right_facing.sides,0)
    np.testing.assert_array_equal(right_facing.normal_signs,-1)

    basis=skfemntv.FacetBasis(
        mesh,skfemntv.ElementVector(skfemntv.ElementQuad1(),dim=1),
        facets=right_facing,
    )
    assert np.all(basis.normals[...,0]>.999999)
    np.testing.assert_array_equal(
        basis.parent_elements,mesh.f2t[0,right_facing]
    )


@pytest.mark.parametrize("dimension",[2,3])
def test_interior_normal_selection_chooses_parent_side(dimension):
    if dimension==2:
        mesh=skfemntv.MeshTri.init_tensor(
            np.linspace(0.,1.,3),np.linspace(0.,1.,3)
        )
        element=skfemntv.ElementTriP1()
    else:
        mesh=skfemntv.MeshTet.init_tensor(
            np.linspace(0.,1.,3),np.linspace(0.,1.,2),
            np.linspace(0.,1.,2),
        )
        element=skfemntv.ElementTetP1()
    requested=np.zeros(dimension);requested[0]=1.
    oriented=mesh.facets_satisfying(
        lambda x:np.isclose(x[0],.5),normal=requested
    )
    assert len(oriented)>0
    assert np.all(mesh.f2t[1,oriented]>=0)

    basis=skfemntv.FacetBasis(
        mesh,skfemntv.ElementVector(element,dim=1),facets=oriented
    )
    np.testing.assert_array_equal(
        basis.parent_elements,
        mesh.f2t[oriented.sides,np.asarray(oriented)],
    )
    assert np.all(np.einsum("eqd,d->eq",basis.normals,requested)>.999999)


def test_normal_oriented_side_matches_scikit_fem():
    mesh=skfemntv.MeshTri.init_tensor(
        np.linspace(0.,1.,3),np.linspace(0.,1.,3)
    )
    predicate=lambda x:np.isclose(x[0],.5)
    native=mesh.facets_satisfying(predicate,normal=np.array([1.,0.]))
    reference_mesh=skfem.MeshTri(mesh.p,mesh.t)
    reference=reference_mesh.facets_satisfying(
        predicate,normal=np.array([1.,0.])
    )

    np.testing.assert_array_equal(native,reference)
    np.testing.assert_array_equal(native.sides,reference.ori)


def test_normal_selection_validation():
    mesh=skfemntv.MeshTri()
    with pytest.raises(ValueError,match="shape"):
        mesh.facets_satisfying(lambda x:x[0]>=0.,normal=[1.,0.,0.])
    with pytest.raises(ValueError,match="nonzero"):
        mesh.facets_satisfying(lambda x:x[0]>=0.,normal=[0.,0.])


def test_regions_are_accepted_by_get_dofs():
    mesh=skfemntv.MeshTet()
    basis=skfemntv.Basis(
        mesh,skfemntv.ElementVector(skfemntv.ElementTetP1())
    )
    facets=skfemntv.FacetRegion(mesh.boundary_facets(),mesh.facets.shape[1])
    nodes=skfemntv.NodeRegion([0,2],mesh.p.shape[1])

    np.testing.assert_array_equal(
        basis.get_dofs(facets=facets).all(),basis.get_dofs().all()
    )
    assert len(basis.get_dofs(nodes=nodes))==6


def test_empty_region_and_selector_validation():
    empty=skfemntv.CellRegion([],3)
    assert empty.diagnostics.is_empty
    assert empty.diagnostics.minimum_id is None

    mesh=skfemntv.MeshTri()
    with pytest.raises(ValueError,match="one boolean per cell"):
        mesh.with_subdomains({"bad":lambda x:np.array([True,False])})
    with pytest.raises(KeyError,match="unknown subdomain"):
        skfemntv.Basis(
            mesh,
            skfemntv.ElementVector(skfemntv.ElementTriP1(),dim=1),
            elements="missing",
        )
