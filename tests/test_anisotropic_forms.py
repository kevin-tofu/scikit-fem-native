import numpy as np
import pytest
import skfem
from skfem.helpers import dot as reference_dot
from skfem.helpers import grad as reference_grad
from skfem.helpers import mul as reference_mul

import skfemntv
from skfemntv.helpers import dot, grad, mul


def _bases():
    mesh = skfemntv.MeshTri.init_tensor(
        np.linspace(0.0, 1.0, 5), np.linspace(0.0, 1.0, 4)
    )
    native = skfemntv.Basis(
        mesh,
        skfemntv.ElementVector(skfemntv.ElementTriP1(), dim=1),
    )
    reference = skfem.Basis(
        skfem.MeshTri(mesh.p, mesh.t), skfem.ElementTriP1()
    )
    return native, reference


def _forms():
    @skfemntv.BilinearForm
    def native(u, v, w):
        return dot(mul(w.diffusion, grad(u)), grad(v))

    @skfem.BilinearForm
    def reference(u, v, w):
        return reference_dot(
            reference_mul(w.diffusion, reference_grad(u)),
            reference_grad(v),
        )

    return native, reference


def test_constant_anisotropic_diffusion_matches_skfem():
    native_basis, reference_basis = _bases()
    native_form, reference_form = _forms()
    tensor = np.array([[2.0, 0.4], [0.4, 0.8]])

    actual = skfemntv.asm(native_form, native_basis, diffusion=tensor)
    expected = skfem.asm(reference_form, reference_basis, diffusion=tensor)

    np.testing.assert_allclose(actual.toarray(), expected.toarray(), rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(actual.toarray(), actual.toarray().T, atol=2e-14)


def test_recommended_scikit_fem_layout_and_nonsymmetric_operator_match():
    native_basis, reference_basis = _bases()
    native_form, reference_form = _forms()
    x = np.moveaxis(native_basis.global_coordinates, -1, 0)
    tensor = np.empty((2, 2) + native_basis.dx.shape)
    tensor[0, 0] = 1.0 + x[0]
    tensor[0, 1] = 0.3 + 0.1 * x[1]
    tensor[1, 0] = -0.2
    tensor[1, 1] = 2.0 - 0.4 * x[0]

    actual = skfemntv.asm(native_form, native_basis, diffusion=tensor)
    expected = skfem.asm(reference_form, reference_basis, diffusion=tensor)

    np.testing.assert_allclose(actual.toarray(), expected.toarray(), rtol=3e-14, atol=3e-14)
    assert not np.allclose(actual.toarray(), actual.toarray().T)


def test_low_level_native_layout_matches_recommended_component_first_layout():
    native_basis, _ = _bases()
    native_form, _ = _forms()
    x = np.moveaxis(native_basis.global_coordinates, -1, 0)
    recommended = np.empty((2, 2) + native_basis.dx.shape)
    recommended[0, 0] = 1.0 + x[0]
    recommended[0, 1] = 0.2
    recommended[1, 0] = -0.1
    recommended[1, 1] = 0.8 + x[1]
    # This mirrors the documented normalization: public component axes move
    # behind entity/quadrature only at the native-kernel boundary.
    native_layout = np.moveaxis(recommended, (0, 1), (-2, -1))

    public_matrix = skfemntv.asm(
        native_form, native_basis, diffusion=recommended
    )
    native_matrix = skfemntv.asm(
        native_form, native_basis, diffusion=native_layout
    )
    np.testing.assert_allclose(
        native_matrix.toarray(), public_matrix.toarray(), atol=2e-14
    )


def test_anisotropic_tensor_can_be_combined_with_isotropic_terms():
    native_basis, reference_basis = _bases()
    tensor = np.array([[1.4, 0.2], [-0.1, 0.7]])

    @skfemntv.BilinearForm
    def native(u, v, w):
        return dot(mul(w.diffusion, grad(u)), grad(v)) + 0.25 * dot(u, v)

    @skfem.BilinearForm
    def reference(u, v, w):
        return reference_dot(
            reference_mul(w.diffusion, reference_grad(u)), reference_grad(v)
        ) + 0.25 * u * v

    actual = skfemntv.asm(native, native_basis, diffusion=tensor)
    expected = skfem.asm(reference, reference_basis, diffusion=tensor)
    np.testing.assert_allclose(actual.toarray(), expected.toarray(), rtol=3e-14, atol=3e-14)


def test_anisotropic_diffusion_rejects_wrong_shape_and_vector_field():
    native_basis, _ = _bases()
    native_form, _ = _forms()
    with pytest.raises(ValueError, match="recommended scikit-fem shape"):
        skfemntv.asm(native_form, native_basis, diffusion=np.eye(3))

    vector = skfemntv.Basis(
        native_basis.mesh,
        skfemntv.ElementVector(skfemntv.ElementTriP1(), dim=2),
    )
    with pytest.raises(skfemntv.UnsupportedNativeForm, match="scalar field"):
        skfemntv.asm(native_form, vector, diffusion=np.eye(2))
