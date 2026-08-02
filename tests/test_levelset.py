import numpy as np
import pytest

import skfemntv


def test_levelset_classifies_tensor_cells_and_returns_regions():
    mesh=skfemntv.MeshQuad.init_tensor(
        np.array([-1.,-.5,0.,.5,1.]),np.array([0.,1.])
    )
    result=skfemntv.LevelSet(lambda x:x[0]-.25).classify(mesh)

    np.testing.assert_array_equal(result.inside,[0,1])
    np.testing.assert_array_equal(result.cut,[2])
    np.testing.assert_array_equal(result.outside,[3])
    assert len(result.touching)==0
    np.testing.assert_array_equal(
        result.active,result.inside|result.cut|result.touching
    )
    assert result.region("cut") is result.cut
    assert result.region(skfemntv.CellClassification.INSIDE) is result.inside
    assert not result.labels.flags.writeable
    assert result.diagnostics.cut_count==1


def test_levelset_distinguishes_touching_from_crossing():
    mesh=skfemntv.MeshTri.init_tensor(
        np.array([-1.,0.,1.]),np.array([0.,1.])
    )
    result=skfemntv.LevelSet(lambda x:x[0]).classify(mesh)

    assert len(result.cut)==0
    assert len(result.touching)==4
    assert len(result.inside)==0
    assert len(result.outside)==0


@pytest.mark.parametrize("mesh",[
    skfemntv.MeshTri(),skfemntv.MeshTri2(),
    skfemntv.MeshQuad(),skfemntv.MeshQuad2(),
    skfemntv.MeshTet(),skfemntv.MeshTet2(),
    skfemntv.MeshWedge1(),skfemntv.MeshPyramid1(),
    skfemntv.MeshHex(),skfemntv.MeshHex2(),
])
def test_levelset_supports_every_mesh_topology(mesh):
    result=skfemntv.LevelSet(lambda x:x[0]-np.mean(x[0])).classify(mesh)

    assert result.diagnostics.cell_count==mesh.nelements
    assert (
        len(result.inside)+len(result.outside)+len(result.cut)
        +len(result.touching)==mesh.nelements
    )


def test_high_order_nodes_participate_in_classification():
    mesh=skfemntv.MeshTri2()
    values=np.ones(mesh.p.shape[1])
    values[3]=-1.
    result=skfemntv.LevelSet(values,tolerance=0.).classify(mesh)

    np.testing.assert_array_equal(result.cut,[0])


def test_levelset_tolerance_controls_touching_cells():
    mesh=skfemntv.MeshTri()
    result=skfemntv.LevelSet(
        np.array([-1.,1.,1.e-9]),tolerance=1.e-8
    ).classify(mesh)

    np.testing.assert_array_equal(result.cut,[0])
    assert result.diagnostics.tolerance==pytest.approx(1.e-8)


def test_levelset_result_selects_basis_without_renumbering():
    mesh=skfemntv.MeshQuad.init_tensor(
        np.linspace(-1.,1.,5),np.linspace(0.,1.,3)
    )
    selected=skfemntv.LevelSet(lambda x:x[0]).classify(mesh).active
    basis=skfemntv.Basis(
        mesh,skfemntv.ElementVector(skfemntv.ElementQuad1(),dim=1),
        elements=selected,
    )

    np.testing.assert_array_equal(basis.tind,selected)
    assert basis.N==mesh.p.shape[1]


def test_levelset_extracts_active_topology_and_global_dofs():
    mesh=skfemntv.MeshQuad.init_tensor(
        np.array([-1.,0.,1.,2.]),np.array([0.,1.])
    )
    result=skfemntv.LevelSet(lambda x:x[0]-.5).classify(mesh)
    active=set(map(int,result.active))

    active_facets=result.active_facets(mesh)
    boundary=result.active_boundary_facets(mesh)
    interior=result.active_interior_facets(mesh)
    ghost=result.ghost_facets(mesh)
    assert set(active_facets)==set(boundary)|set(interior)
    assert set(boundary).isdisjoint(interior)
    assert len(ghost)>0
    assert set(ghost)<=set(interior)
    for facet,side in zip(boundary,boundary.sides):
        assert int(mesh.f2t[side,facet]) in active
        other=1-int(side)
        assert (
            mesh.f2t[other,facet]<0
            or int(mesh.f2t[other,facet]) not in active
        )

    basis=skfemntv.Basis(
        mesh,skfemntv.ElementVector(skfemntv.ElementQuad1(),dim=2)
    )
    dofs=result.active_dofs(basis)
    expected=basis.get_dofs(elements=result.active)
    np.testing.assert_array_equal(dofs.all(),expected.all())
    np.testing.assert_array_equal(
        result.active_dofs(basis,components=0).all(),
        expected.all("u^1"),
    )


def test_active_topology_rejects_a_different_cell_count():
    result=skfemntv.LevelSet(lambda x:x[0]).classify(skfemntv.MeshTri())
    other=skfemntv.MeshTri.init_tensor(
        np.linspace(0.,1.,3),np.linspace(0.,1.,3)
    )
    with pytest.raises(ValueError,match="different cell counts"):
        result.active_facets(other)


@pytest.mark.parametrize("field,match",[
    (np.ones((2,2)),"one-dimensional"),
    (np.ones(2),"expected"),
    (np.array([0.,1.,np.nan]),"node 2"),
])
def test_levelset_rejects_invalid_nodal_fields(field,match):
    if np.asarray(field).ndim!=1:
        with pytest.raises(ValueError,match=match):
            skfemntv.LevelSet(field)
    else:
        with pytest.raises(ValueError,match=match):
            skfemntv.LevelSet(field).classify(skfemntv.MeshTri())


def test_levelset_rejects_unknown_region_name():
    result=skfemntv.LevelSet(lambda x:x[0]).classify(skfemntv.MeshTri())
    with pytest.raises(ValueError,match="unknown cell classification"):
        result.region("near")
