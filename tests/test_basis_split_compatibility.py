import numpy as np
import pytest
import skfem
import skfemntv


@pytest.mark.parametrize("kind",["scalar","vector","composite"])
def test_basis_split_indices_bases_and_coefficients_match_scikit_fem(kind):
    mesh=skfemntv.MeshTri.init_tensor([0.,.5,1.],[0.,.5,1.]).with_boundaries({
        "left":lambda x:np.isclose(x[0],0.),
        "right":lambda x:np.isclose(x[0],1.),
    })
    reference_mesh=skfem.MeshTri(mesh.p,mesh.t).with_boundaries({
        "left":lambda x:np.isclose(x[0],0.),
        "right":lambda x:np.isclose(x[0],1.),
    })
    if kind=="scalar":
        element=skfemntv.ElementTriP2()
        reference_element=skfem.ElementTriP2()
    elif kind=="vector":
        element=skfemntv.ElementVector(skfemntv.ElementTriP1(),dim=2)
        reference_element=skfem.ElementVector(skfem.ElementTriP1(),dim=2)
    else:
        element=skfemntv.ElementComposite(
            skfemntv.ElementTriP2(),
            skfemntv.ElementVector(skfemntv.ElementTriP1(),dim=2),
        )
        reference_element=skfem.ElementComposite(
            skfem.ElementTriP2(),
            skfem.ElementVector(skfem.ElementTriP1(),dim=2),
        )
    native=skfemntv.Basis(mesh,element,intorder=4)
    reference=skfem.Basis(reference_mesh,reference_element,intorder=4)

    actual_indices=native.split_indices()
    expected_indices=reference.split_indices()
    assert len(actual_indices)==len(expected_indices)
    for actual,expected in zip(actual_indices,expected_indices):
        np.testing.assert_array_equal(actual,expected)

    actual_bases=native.split_bases()
    expected_bases=reference.split_bases()
    assert len(actual_bases)==len(expected_bases)
    for actual,expected in zip(actual_bases,expected_bases):
        assert actual.N==expected.N
        np.testing.assert_allclose(actual.doflocs,expected.doflocs)

    coefficients=np.arange(native.N,dtype=np.float64)
    actual_split=native.split(coefficients)
    expected_split=reference.split(coefficients)
    for (actual_values,actual_basis),(expected_values,expected_basis) in zip(
        actual_split,expected_split
    ):
        np.testing.assert_array_equal(actual_values,expected_values)
        assert actual_basis.N==expected_basis.N


def test_basis_convenience_vectors_and_with_element_match_scikit_fem():
    mesh=skfemntv.MeshTri.init_tensor([0.,1.],[0.,1.])
    reference_mesh=skfem.MeshTri(mesh.p,mesh.t)
    reference=skfem.Basis(reference_mesh,skfem.ElementTriP1(),intorder=3)
    native=skfemntv.Basis(
        mesh,skfemntv.ElementTriP1(),quadrature=reference.quadrature
    )
    np.testing.assert_array_equal(native.zeros(),reference.zeros())
    np.testing.assert_array_equal(native.ones(),reference.ones())
    np.testing.assert_array_equal(native.zero_w(),reference.zero_w())
    actual=native.with_element(skfemntv.ElementTriP2())
    expected=reference.with_element(skfem.ElementTriP2())
    np.testing.assert_array_equal(actual.element_dofs,expected.element_dofs)
    np.testing.assert_allclose(actual.doflocs,expected.doflocs)


def test_complement_dofs_matches_scikit_fem_selection_forms():
    mesh=skfemntv.MeshTri.init_tensor([0.,.5,1.],[0.,.5,1.]).with_boundaries({
        "left":lambda x:np.isclose(x[0],0.),
        "right":lambda x:np.isclose(x[0],1.),
    })
    reference_mesh=skfem.MeshTri(mesh.p,mesh.t).with_boundaries({
        "left":lambda x:np.isclose(x[0],0.),
        "right":lambda x:np.isclose(x[0],1.),
    })
    for native_element,reference_element in (
        (skfemntv.ElementTriP1(),skfem.ElementTriP1()),
        (skfemntv.ElementVector(skfemntv.ElementTriP1(),dim=2),
         skfem.ElementVector(skfem.ElementTriP1(),dim=2)),
    ):
        native=skfemntv.Basis(mesh,native_element)
        reference=skfem.Basis(reference_mesh,reference_element)
        left=native.get_dofs(facets="left")
        right=native.get_dofs(facets="right")
        left_ref=reference.get_dofs(facets="left")
        right_ref=reference.get_dofs(facets="right")
        for actual,expected in (
            (native.complement_dofs(left),reference.complement_dofs(left_ref)),
            (native.complement_dofs(left,right),reference.complement_dofs(left_ref,right_ref)),
            (native.complement_dofs({"left":left,"right":right}),
             reference.complement_dofs({"left":left_ref,"right":right_ref})),
            (native.complement_dofs(np.array([],dtype=np.int64)),
             reference.complement_dofs(np.array([],dtype=np.int64))),
        ):
            np.testing.assert_array_equal(actual,expected)
