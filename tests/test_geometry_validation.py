import numpy as np
import pytest

import skfemntv


def _scalar(element):
    return skfemntv.ElementVector(element,dim=1)


@pytest.mark.parametrize(
    "mesh,element",
    [
        (skfemntv.MeshTri(),skfemntv.ElementTriP1()),
        (skfemntv.MeshQuad(),skfemntv.ElementQuad1()),
        (skfemntv.MeshTet(),skfemntv.ElementTetP1()),
        (skfemntv.MeshWedge1(),skfemntv.ElementWedge1()),
        (skfemntv.MeshPyramid1(),skfemntv.ElementPyramid1()),
        (skfemntv.MeshHex(),skfemntv.ElementHex1()),
    ],
)
def test_valid_geometry_exposes_diagnostics(mesh,element):
    basis=skfemntv.Basis(mesh,_scalar(element))
    diagnostics=basis.geometry_diagnostics

    assert isinstance(diagnostics,skfemntv.GeometryDiagnostics)
    assert diagnostics.element_count==mesh.nelements
    assert diagnostics.quadrature_points_per_element==basis.X.shape[1]
    assert diagnostics.minimum_determinant>0.
    assert diagnostics.maximum_determinant>=diagnostics.minimum_determinant
    assert diagnostics.minimum_scaled_determinant>0.
    assert 0<=diagnostics.worst_element<mesh.nelements
    assert 0<=diagnostics.worst_quadrature_point<basis.X.shape[1]
    assert diagnostics.determinant_tolerance>0.
    assert diagnostics.maximum_condition_number>=mesh.dim()
    assert 0<=diagnostics.worst_condition_element<mesh.nelements
    assert (
        0<=diagnostics.worst_condition_quadrature_point<basis.X.shape[1]
    )


def test_uniform_negative_orientation_is_allowed_and_reported():
    mesh=skfemntv.MeshTet(
        np.array([
            [0.,0.,1.,0.],
            [0.,1.,0.,0.],
            [0.,0.,0.,1.],
        ]),
        np.array([[0],[1],[2],[3]]),
    )
    basis=skfemntv.Basis(mesh,_scalar(skfemntv.ElementTetP1()))

    assert basis.geometry_diagnostics.maximum_determinant<0.
    assert basis.geometry_diagnostics.minimum_scaled_determinant==pytest.approx(
        1.
    )
    assert basis.geometry_diagnostics.negative_orientation_elements==1
    assert np.all(basis.dx>0.)


def test_scale_aware_check_accepts_uniformly_small_valid_geometry():
    scale=1e-50
    mesh=skfemntv.MeshTet(
        scale*skfemntv.MeshTet().p,
        skfemntv.MeshTet().t,
    )
    basis=skfemntv.Basis(mesh,_scalar(skfemntv.ElementTetP1()))

    np.testing.assert_allclose(
        basis.geometry_diagnostics.minimum_determinant,
        scale**3,rtol=2e-15,
    )
    assert basis.geometry_diagnostics.minimum_scaled_determinant==pytest.approx(
        1.
    )


def test_near_singular_geometry_is_rejected_relative_to_jacobian_scale():
    mesh=skfemntv.MeshTri(
        np.array([[0.,1.,0.],[0.,0.,1e-16]]),
        np.array([[0],[1],[2]]),
    )

    with pytest.raises(ValueError,match="reason=near_singular_or_non_finite"):
        skfemntv.Basis(mesh,_scalar(skfemntv.ElementTriP1()))


def test_curved_tet10_internal_inversion_is_checked_at_quadrature_points():
    mesh=skfemntv.MeshTet2.from_mesh(skfemntv.MeshTet())
    points=mesh.p.copy()
    points[:,4]=-2.
    curved=skfemntv.MeshTet2(points,mesh.t)

    # The four corner nodes still define a positively oriented tetrahedron;
    # the displaced midside node inverts the isoparametric map internally.
    corner=points[:,mesh.t[:4,0]]
    assert np.linalg.det(corner[:,1:]-corner[:,:1])>0.
    with pytest.raises(
        ValueError,
        match=(
            r"cell=0, quadrature_point=2, determinant=-.*"
            r"reason=orientation_change"
        ),
    ):
        skfemntv.Basis(
            curved,_scalar(skfemntv.ElementTetP2()),intorder=4
        )


def test_curved_hex27_internal_inversion_is_checked_at_quadrature_points():
    mesh=skfemntv.MeshHex2.from_mesh(skfemntv.MeshHex())
    points=mesh.p.copy()
    points[:,1]=-3.
    curved=skfemntv.MeshHex2(points,mesh.t)

    # Node 1 is an edge node; all eight corner nodes retain the valid cube.
    corner_ids=mesh.t[[0,2,6,18,8,20,24,26],0]
    np.testing.assert_array_equal(points[:,corner_ids],mesh.p[:,corner_ids])
    with pytest.raises(ValueError,match="reason=orientation_change"):
        skfemntv.Basis(
            curved,_scalar(skfemntv.ElementHex2()),intorder=4
        )


def test_condition_number_reports_anisotropic_but_valid_geometry():
    mesh=skfemntv.MeshTri(
        np.array([[0.,1.,0.],[0.,0.,1e-8]]),
        np.array([[0],[1],[2]]),
    )
    diagnostics=skfemntv.Basis(
        mesh,_scalar(skfemntv.ElementTriP1())
    ).geometry_diagnostics

    assert diagnostics.maximum_condition_number==pytest.approx(1e8)
    assert diagnostics.worst_condition_element==0
    assert diagnostics.worst_condition_quadrature_point==0


def test_restricted_basis_diagnostics_keep_global_element_id():
    mesh=skfemntv.MeshTri.init_tensor(
        np.linspace(0.,1.,4),np.linspace(0.,1.,3)
    )
    selected=np.array([2,5,8])
    basis=skfemntv.Basis(
        mesh,_scalar(skfemntv.ElementTriP1()),elements=selected
    )

    assert basis.geometry_diagnostics.element_count==len(selected)
    assert basis.geometry_diagnostics.worst_element in selected


def test_empty_restricted_basis_has_explicit_diagnostics():
    basis=skfemntv.Basis(
        skfemntv.MeshTri(),_scalar(skfemntv.ElementTriP1()),elements=[]
    )
    diagnostics=basis.geometry_diagnostics

    assert diagnostics.element_count==0
    assert diagnostics.worst_element==-1
    assert diagnostics.worst_quadrature_point==-1
    assert diagnostics.minimum_determinant==float("inf")
    assert diagnostics.maximum_determinant==float("-inf")
    assert diagnostics.maximum_condition_number==0.
    assert diagnostics.worst_condition_element==-1
