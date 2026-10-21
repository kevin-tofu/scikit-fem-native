import numpy as np
import pytest
import skfem
import skfemntv


@pytest.mark.parametrize("dimension,order",[(2,1),(2,2),(3,1)])
def test_point_source_matches_scikit_fem(dimension,order):
    if dimension==2:
        mesh=skfemntv.MeshTri.init_tensor([0.,.5,1.],[0.,.5,1.])
        reference_mesh=skfem.MeshTri(mesh.p,mesh.t)
        native_element=(skfemntv.ElementTriP1() if order==1
                        else skfemntv.ElementTriP2())
        reference_element=(skfem.ElementTriP1() if order==1
                           else skfem.ElementTriP2())
        point=np.array([.2,.3])
    else:
        mesh=skfemntv.MeshTet.init_tensor([0.,1.],[0.,1.],[0.,1.])
        reference_mesh=skfem.MeshTet(mesh.p,mesh.t)
        native_element=skfemntv.ElementTetP1()
        reference_element=skfem.ElementTetP1()
        point=np.array([.1,.2,.3])
    native=skfemntv.Basis(mesh,native_element,intorder=4)
    reference=skfem.Basis(reference_mesh,reference_element,intorder=4)
    np.testing.assert_allclose(
        native.point_source(point),reference.point_source(point),
        rtol=2e-12,atol=2e-12,
    )


def test_vector_point_source_matches_first_probe_row_from_scikit_fem():
    mesh=skfemntv.MeshTri.init_tensor([0.,1.],[0.,1.])
    reference_mesh=skfem.MeshTri(mesh.p,mesh.t)
    native=skfemntv.Basis(
        mesh,skfemntv.ElementVector(skfemntv.ElementTriP1(),dim=2)
    )
    reference=skfem.Basis(
        reference_mesh,skfem.ElementVector(skfem.ElementTriP1(),dim=2)
    )
    point=np.array([.3,.4])
    np.testing.assert_array_equal(
        native.point_source(point),reference.point_source(point)
    )


def test_point_source_matches_probe_action_and_validates_point_shape():
    basis=skfemntv.Basis(skfemntv.MeshTri(),skfemntv.ElementTriP1())
    point=np.array([.2,.3])
    coefficients=np.array([1.,2.,3.])
    np.testing.assert_allclose(
        basis.point_source(point)@coefficients,
        (basis.probes(point[:,None])@coefficients)[0],
    )
    with pytest.raises(ValueError,match="shape"):
        basis.point_source(np.array([.2,.3,.4]))
