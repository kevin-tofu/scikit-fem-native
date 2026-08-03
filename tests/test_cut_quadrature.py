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


def test_cut_quadrature_rejects_unsupported_simplex_and_non_simplex_meshes():
    for mesh in (skfemntv.MeshQuad(),skfemntv.MeshTet2()):
        with pytest.raises(NotImplementedError,match="Tri6"):
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


@pytest.mark.parametrize("mesh,field,exact",[
    (
        skfemntv.MeshTri(),
        lambda x:x[:,0]**2+2.*x[:,0]*x[:,1]+3.*x[:,1]**2,
        5./12.,
    ),
    (
        skfemntv.MeshTet(),
        lambda x:x[:,0]**2+x[:,1]**2+x[:,2]**2,
        1./20.,
    ),
])
def test_order_two_integrates_quadratics_on_full_simplex(mesh,field,exact):
    quadrature=skfemntv.LevelSet(
        -np.ones(mesh.p.shape[1]),tolerance=0.
    ).cut_quadrature(mesh,intorder=2)

    assert quadrature.diagnostics.integration_order==2
    assert quadrature.weights@field(quadrature.points)==pytest.approx(exact)
    assert np.all(quadrature.weights>0.)


def test_higher_order_duffy_rule_integrates_cubic_on_cut_triangle():
    mesh=skfemntv.MeshTri()
    level_set=skfemntv.LevelSet(lambda x:x[0]+x[1]-.5,tolerance=0.)
    order_one=level_set.cut_quadrature(mesh,intorder=1)
    order_three=level_set.cut_quadrature(mesh,intorder=3)
    field=lambda x:(x[:,0]+x[:,1])**3
    # Integral on x+y <= a is a^5 / 5 for a=.5.
    exact=.5**5/5.

    assert abs(order_three.weights@field(order_three.points)-exact)<1.e-14
    assert abs(order_one.weights@field(order_one.points)-exact)>1.e-4
    assert len(order_three.weights)>len(order_one.weights)


@pytest.mark.parametrize("intorder,error",[
    (0,ValueError),(1.5,TypeError),(True,TypeError),
])
def test_cut_quadrature_validates_integration_order(intorder,error):
    with pytest.raises(error,match="intorder"):
        skfemntv.LevelSet(lambda x:x[0]).cut_quadrature(
            skfemntv.MeshTri(),intorder=intorder
        )
