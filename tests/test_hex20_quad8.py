import numpy as np

import skfemntv


def test_hex20_basis_partition_and_affine_gradient():
    mesh=skfemntv.MeshHex20.from_mesh(skfemntv.MeshHex())
    basis=skfemntv.Basis(
        mesh,skfemntv.ElementVector(skfemntv.ElementHex20(),dim=3),intorder=4
    )
    np.testing.assert_allclose(basis.tabulated_shape.sum(axis=2),1.,atol=2e-14)
    np.testing.assert_allclose(
        basis.tabulated_gradients.sum(axis=2),0.,atol=2e-13
    )
    matrix=np.array([[.1,.2,-.1],[-.3,.4,.05],[.2,-.15,.3]])
    values=(matrix@mesh.p).T.reshape(-1)
    field=basis.interpolate(values)
    expected=np.broadcast_to(matrix[:,:,None,None],field.grad.shape)
    np.testing.assert_allclose(field.grad,expected,rtol=2e-13,atol=2e-13)


def test_hex20_boundary_facets_are_quad8():
    mesh=skfemntv.MeshHex20.from_mesh(skfemntv.MeshHex())
    facets=mesh._facet_connectivity(mesh.boundary_facets(),full=True)
    assert facets.shape==(8,6)
    assert np.unique(facets).size==20


def test_hex20_native_elasticity_rigid_translation():
    mesh=skfemntv.MeshHex20.from_mesh(skfemntv.MeshHex())
    basis=skfemntv.Basis(
        mesh,skfemntv.ElementVector(skfemntv.ElementHex20(),dim=3),intorder=4
    )
    assembler=skfemntv.NativeAssembler.from_basis(
        basis,skfemntv.LinearElasticity(100.,.25)
    )
    translation=np.tile(np.array([.2,-.1,.3]),mesh.p.shape[1])
    evaluation=assembler.assemble(translation,None)
    np.testing.assert_allclose(evaluation.residual,0.,atol=3e-12)
    np.testing.assert_allclose(
        evaluation.tangent.toarray(),evaluation.tangent.toarray().T,atol=3e-12
    )
    assert basis.geometry_diagnostics.minimum_determinant>0.


def test_quad8_basis_partition_and_affine_gradient():
    points=skfemntv.ElementQuad8().doflocs.T
    mesh=skfemntv.MeshQuad8(points,np.arange(8)[:,None])
    basis=skfemntv.Basis(
        mesh,skfemntv.ElementVector(skfemntv.ElementQuad8(),dim=2),intorder=4
    )
    np.testing.assert_allclose(basis.tabulated_shape.sum(axis=2),1.,atol=2e-14)
    np.testing.assert_allclose(
        basis.tabulated_gradients.sum(axis=2),0.,atol=2e-13
    )
