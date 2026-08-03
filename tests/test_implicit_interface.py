import numpy as np
import pytest

import skfemntv
from skfemntv.helpers import dot


def test_tri3_interface_integrates_length_and_linear_field():
    mesh=skfemntv.MeshTri()
    level_set=skfemntv.LevelSet(lambda x:x[0]+x[1]-.5,tolerance=0.)
    rule=level_set.interface_quadrature(mesh,intorder=2)

    assert rule.weights.sum()==pytest.approx(np.sqrt(.5))
    assert rule.weights@rule.points[:,0]==pytest.approx(np.sqrt(.5)*.25)
    np.testing.assert_allclose(
        rule.normals,
        np.broadcast_to(
            np.array([1.,1.])[None,:]/np.sqrt(2.),rule.normals.shape
        ),
    )
    assert np.all(rule.weights>0.)
    assert rule.side=="interface"


def test_tet4_interface_integrates_area_and_linear_field():
    mesh=skfemntv.MeshTet()
    level_set=skfemntv.LevelSet(lambda x:x.sum(axis=0)-.5,tolerance=0.)
    rule=level_set.interface_quadrature(mesh,intorder=2)
    area=np.sqrt(3.)/8.

    assert rule.weights.sum()==pytest.approx(area)
    assert rule.weights@rule.points[:,0]==pytest.approx(area/6.)
    np.testing.assert_allclose(
        rule.normals,np.ones((len(rule.weights),3))/np.sqrt(3.)
    )


def test_tet4_quadrilateral_interface_is_partitioned_without_padding():
    mesh=skfemntv.MeshTet()
    # Plane x + y = .5 cuts four tetrahedron edges and forms a quadrilateral.
    rule=skfemntv.LevelSet(
        lambda x:x[0]+x[1]-.5,tolerance=0.
    ).interface_quadrature(mesh,intorder=2)

    assert len(rule.weights)==6
    assert rule.cell_offsets.tolist()==[0,6]
    assert np.all(rule.weights>0.)
    assert not rule.points.flags.writeable


def test_implicit_facet_basis_interpolates_and_assembles_surface_forms():
    mesh=skfemntv.MeshTri()
    level_set=skfemntv.LevelSet(lambda x:x[0]+x[1]-.5,tolerance=0.)
    rule=level_set.interface_quadrature(mesh,intorder=2)
    parent=skfemntv.Basis(
        mesh,skfemntv.ElementVector(skfemntv.ElementTriP1(),dim=1)
    )
    basis=skfemntv.ImplicitFacetBasis(parent,rule)
    field=basis.interpolate(1.+2.*mesh.p[0]-mesh.p[1])
    np.testing.assert_allclose(
        field.value,1.+2.*rule.points[:,0]-rule.points[:,1]
    )

    @skfemntv.Functional
    def normal_moment(w):
        return w.x[0]*w.n[0]

    @skfemntv.BilinearForm
    def mass(u,v,w):
        return dot(u,v)

    assert skfemntv.asm(normal_moment,basis)==pytest.approx(
        rule.weights@(rule.points[:,0]*rule.normals[:,0])
    )
    matrix=skfemntv.asm(mass,basis,num_threads=2)
    np.testing.assert_allclose(matrix.toarray(),matrix.toarray().T)
    assert matrix.sum()==pytest.approx(rule.weights.sum())


def test_interface_quadrature_rejects_ambiguous_and_unsupported_geometry():
    with pytest.raises(ValueError,match="not unique in cell 0"):
        skfemntv.LevelSet(np.zeros(3),tolerance=0.).interface_quadrature(
            skfemntv.MeshTri()
        )
    with pytest.raises(NotImplementedError,match="Tri3 and Tet4"):
        skfemntv.LevelSet(lambda x:x[0]).interface_quadrature(
            skfemntv.MeshQuad()
        )
    with pytest.raises(TypeError,match="ImplicitInterfaceQuadrature"):
        skfemntv.ImplicitFacetBasis(
            skfemntv.Basis(
                skfemntv.MeshTri(),
                skfemntv.ElementVector(skfemntv.ElementTriP1(),dim=1),
            ),
            skfemntv.LevelSet(lambda x:x[0]).cut_quadrature(
                skfemntv.MeshTri()
            ),
        )
