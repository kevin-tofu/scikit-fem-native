import numpy as np
import pytest
import skfem

import skfemntv


@pytest.mark.parametrize(
    "mesh,element,reference_mesh,reference_element,intorder",
    [
        (
            skfemntv.MeshTet(),
            skfemntv.ElementTetP1(),
            skfem.MeshTet,
            skfem.ElementTetP1,
            2,
        ),
        (
            skfemntv.MeshTet2(),
            skfemntv.ElementTetP2(),
            skfem.MeshTet2,
            skfem.ElementTetP2,
            4,
        ),
        (
            skfemntv.MeshHex(),
            skfemntv.ElementHex1(),
            skfem.MeshHex,
            skfem.ElementHex1,
            2,
        ),
        (
            skfemntv.MeshHex2(),
            skfemntv.ElementHex2(),
            skfem.MeshHex2,
            skfem.ElementHex2,
            4,
        ),
    ],
)
def test_coordinate_boundary_dofs_match_skfem(
    mesh,element,reference_mesh,reference_element,intorder
):
    predicate=lambda x:np.isclose(x[0],0.)
    basis=skfemntv.Basis(
        mesh,skfemntv.ElementVector(element),intorder=intorder
    )
    actual=basis.get_dofs(predicate).all()

    reference_geometry=(
        skfem.MeshHex2.from_mesh(skfem.MeshHex())
        if reference_mesh is skfem.MeshHex2
        else reference_mesh(mesh.p,mesh.t)
    )
    reference=skfem.Basis(
        reference_geometry,
        skfem.ElementVector(reference_element()),
        intorder=intorder,
    )
    expected=reference.get_dofs(predicate).all()
    actual_coordinates=np.sort(
        np.round(basis.doflocs[:,actual],14),axis=1
    )
    expected_coordinates=np.sort(
        np.round(reference.doflocs[:,expected],14),axis=1
    )
    np.testing.assert_array_equal(actual_coordinates,expected_coordinates)


@pytest.mark.parametrize(
    "mesh,element,intorder",
    [
        (skfemntv.MeshTet(),skfemntv.ElementTetP1(),2),
        (skfemntv.MeshTet2(),skfemntv.ElementTetP2(),4),
        (skfemntv.MeshHex(),skfemntv.ElementHex1(),2),
        (skfemntv.MeshHex2(),skfemntv.ElementHex2(),4),
    ],
)
def test_named_boundary_matches_coordinate_predicate(
    mesh,element,intorder
):
    predicate=lambda x:np.isclose(x[0],0.)
    named=mesh.with_boundaries({"left":predicate})
    basis=skfemntv.Basis(
        named,skfemntv.ElementVector(element),intorder=intorder
    )
    np.testing.assert_array_equal(
        basis.get_dofs("left").all(),
        basis.get_dofs(predicate).all(),
    )
    assert "left" in named.boundaries
    assert not mesh.boundaries


def test_composite_named_boundary_maps_each_field():
    linear=skfemntv.MeshTet.init_tensor(
        np.linspace(0.,1.,3),
        np.linspace(0.,1.,2),
        np.linspace(0.,1.,2),
    )
    mesh=skfemntv.MeshTet2.from_mesh(linear).with_boundaries({
        "left":lambda x:np.isclose(x[0],0.),
    })
    basis=skfemntv.Basis(
        mesh,
        skfemntv.ElementVector(skfemntv.ElementTetP2())
        *skfemntv.ElementTetP1(),
        intorder=4,
    )
    split=basis.split_bases()
    indices=basis.split_indices()
    expected=np.sort(np.concatenate([
        mapping[field_basis.get_dofs("left").all()]
        for field_basis,mapping in zip(split,indices)
    ]))
    np.testing.assert_array_equal(
        basis.get_dofs("left").all(),expected
    )
