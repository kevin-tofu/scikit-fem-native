import numpy as np
import pytest

import skfemntv


def _integrate_linear(quadrature,constant,gradient):
    values=constant+quadrature.points@np.asarray(gradient)
    return float(quadrature.weights@values)


def test_tri3_half_cut_integrates_constant_and_linear_exactly():
    mesh=skfemntv.MeshTri(
        np.array([[0.,1.,0.],[0.,0.,1.]]),np.array([[0],[1],[2]])
    )
    level_set=skfemntv.LevelSet(lambda x:x[0]+x[1]-.5,tolerance=0.)
    inside=level_set.cut_quadrature(mesh)
    outside=level_set.cut_quadrature(mesh,side="outside")

    assert inside.weights.sum()==pytest.approx(.125)
    assert outside.weights.sum()==pytest.approx(.375)
    assert inside.weights.sum()+outside.weights.sum()==pytest.approx(.5)
    # Integral over the small right triangle: area * centroid coordinates.
    assert _integrate_linear(inside,0.,[1.,2.])==pytest.approx(.0625)
    assert np.all(inside.weights>0.)
    np.testing.assert_allclose(inside.normals,[np.array([1.,1.])/np.sqrt(2.)])


def test_tet4_planar_cut_partitions_volume_and_integrates_linear_field():
    mesh=skfemntv.MeshTet()
    level_set=skfemntv.LevelSet(lambda x:x.sum(axis=0)-.5,tolerance=0.)
    inside=level_set.cut_quadrature(mesh)
    outside=level_set.cut_quadrature(mesh,side="outside")

    expected_inside=(.5**3)/6.
    assert inside.weights.sum()==pytest.approx(expected_inside)
    assert inside.weights.sum()+outside.weights.sum()==pytest.approx(1./6.)
    # The clipped tetrahedron has centroid (.125, .125, .125).
    assert _integrate_linear(inside,0.,[1.,2.,3.])==pytest.approx(
        expected_inside*.75
    )
    assert np.all(inside.weights>0.)


def test_cut_quadrature_has_csr_cell_offsets_without_padding():
    mesh=skfemntv.MeshTri.init_tensor(
        np.linspace(-1.,1.,5),np.linspace(0.,1.,3)
    )
    quadrature=skfemntv.LevelSet(lambda x:x[0]-.1).cut_quadrature(mesh)

    assert quadrature.cell_offsets.shape==(mesh.nelements+1,)
    assert quadrature.cell_offsets[-1]==len(quadrature.weights)
    assert np.all(np.diff(quadrature.cell_offsets)>=0)
    for cell in range(mesh.nelements):
        selection=quadrature.cell_slice(cell)
        np.testing.assert_array_equal(quadrature.cells[selection],cell)
    for array in (
        quadrature.cell_offsets,quadrature.points,
        quadrature.reference_points,quadrature.weights,
        quadrature.cells,quadrature.normals,
    ):
        assert not array.flags.writeable


def test_cut_quadrature_rejects_high_order_and_non_simplex_meshes():
    for mesh in (skfemntv.MeshTri2(),skfemntv.MeshQuad(),skfemntv.MeshTet2()):
        with pytest.raises(NotImplementedError,match="Tri3 and Tet4"):
            skfemntv.LevelSet(lambda x:x[0]).cut_quadrature(mesh)


def test_cut_quadrature_validates_side_and_classification():
    level_set=skfemntv.LevelSet(lambda x:x[0])
    with pytest.raises(ValueError,match="side"):
        level_set.cut_quadrature(skfemntv.MeshTri(),side="both")
    classification=level_set.classify(skfemntv.MeshTri())
    other=skfemntv.MeshTri.init_tensor(
        np.linspace(0.,1.,3),np.linspace(0.,1.,3)
    )
    with pytest.raises(ValueError,match="different cell counts"):
        level_set.cut_quadrature(other,classification=classification)
