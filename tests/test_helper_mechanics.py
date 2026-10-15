import numpy as np
import pytest
import skfem

import skfemntv
from skfem.helpers import cross as reference_cross
from skfem.helpers import det as reference_det
from skfem.helpers import eye as reference_eye
from skfem.helpers import inv as reference_inv
from skfemntv.helpers import cross, det, eye, grad, inv


@pytest.mark.parametrize("dimension", [2, 3])
def test_determinant_and_inverse_match_scikit_fem_for_quadrature_arrays(dimension):
    rng = np.random.default_rng(813 + dimension)
    matrices = rng.normal(size=(dimension, dimension, 5, 7))
    matrices += 3.0 * np.eye(dimension)[:, :, None, None]
    np.testing.assert_allclose(det(matrices), reference_det(matrices))
    np.testing.assert_allclose(inv(matrices), reference_inv(matrices))


@pytest.mark.parametrize("dimension", [2, 3])
def test_eye_matches_scikit_fem_for_scalar_and_quadrature_weights(dimension):
    weights = np.linspace(0.3, 1.7, 12).reshape(3, 4)
    np.testing.assert_allclose(eye(weights, dimension), reference_eye(weights, dimension))
    np.testing.assert_allclose(eye(2.5, dimension), reference_eye(2.5, dimension))


@pytest.mark.parametrize("dimension", [2, 3])
def test_cross_matches_scikit_fem_with_component_first_arrays(dimension):
    rng = np.random.default_rng(190 + dimension)
    left = rng.normal(size=(dimension, 4, 6))
    right = rng.normal(size=(dimension, 4, 6))
    np.testing.assert_allclose(cross(left, right), reference_cross(left, right))


@pytest.mark.parametrize("helper", [det, inv, eye, cross])
def test_mechanics_helpers_reject_symbolic_form_expressions(helper):
    mesh = skfemntv.MeshTri.init_tensor([0.0, 1.0], [0.0, 1.0])
    basis = skfemntv.Basis(mesh, skfemntv.ElementTriP1())

    @skfemntv.BilinearForm
    def form(u, v, w):
        if helper is det:
            return det(grad(u)) * v
        if helper is inv:
            return inv(grad(u))[0, 0] * v
        if helper is eye:
            return eye(u, 2)[0, 0] * v
        return cross(u, v)

    with pytest.raises(skfemntv.UnsupportedNativeForm, match="numeric arrays"):
        skfemntv.asm(form, basis)


@pytest.mark.parametrize("helper,args", [(det, (np.ones((4, 4)),)),
                                           (inv, (np.ones((4, 4)),)),
                                           (cross, (np.ones(4), np.ones(4)))])
def test_mechanics_helpers_reject_unsupported_dimensions(helper, args):
    with pytest.raises(ValueError):
        helper(*args)


def test_eye_rejects_noninteger_dimension():
    with pytest.raises(ValueError, match="nonnegative integer"):
        eye(1.0, 2.5)
