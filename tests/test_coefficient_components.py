import numpy as np
import pytest
import skfem
from skfem.helpers import dot as reference_dot
from skfem.helpers import grad as reference_grad
from skfem.helpers import mul as reference_mul

import skfemntv
from skfemntv.helpers import ddot, dot, grad, mul


def _bases():
    mesh = skfemntv.MeshTri.init_tensor(
        np.linspace(0.0, 1.0, 5), np.linspace(0.0, 1.0, 4)
    )
    native = skfemntv.Basis(
        mesh, skfemntv.ElementVector(skfemntv.ElementTriP1(), dim=1)
    )
    reference = skfem.Basis(
        skfem.MeshTri(mesh.p, mesh.t), skfem.ElementTriP1()
    )
    return native, reference


def test_multiple_named_fields_and_first_axis_components_match_skfem():
    native_basis, reference_basis = _bases()
    x = np.moveaxis(native_basis.global_coordinates, -1, 0)
    material = np.empty((2,) + native_basis.dx.shape)
    material[0] = 0.7 + 0.2 * x[0]
    material[1] = 1.1 + 0.3 * x[1]
    diffusion = np.array([[1.4, 0.2], [-0.1, 0.8]])

    @skfemntv.BilinearForm
    def native(u, v, w):
        # Coefficient components use the first axis, matching scikit-fem.
        return (
            w.material[0] * dot(u, v)
            + w.material[1] * ddot(grad(u), grad(v))
            + dot(mul(w.diffusion, grad(u)), grad(v))
        )

    @skfem.BilinearForm
    def reference(u, v, w):
        return (
            w.material[0] * u * v
            + w.material[1] * reference_dot(
                reference_grad(u), reference_grad(v)
            )
            + reference_dot(
                reference_mul(w.diffusion, reference_grad(u)),
                reference_grad(v),
            )
        )

    actual = skfemntv.asm(
        native, native_basis, material=material, diffusion=diffusion
    )
    expected = skfem.asm(
        reference, reference_basis, material=material, diffusion=diffusion
    )
    np.testing.assert_allclose(
        actual.toarray(), expected.toarray(), rtol=4e-14, atol=4e-14
    )


def test_linear_form_component_access_uses_first_axis():
    native_basis, reference_basis = _bases()
    loads = np.empty((2,) + native_basis.dx.shape)
    loads[0] = 0.5
    loads[1] = 1.0 + np.moveaxis(
        native_basis.global_coordinates, -1, 0
    )[0]

    @skfemntv.LinearForm
    def native(v, w):
        return dot(w.loads[1], v)

    @skfem.LinearForm
    def reference(v, w):
        return w.loads[1] * v

    actual = skfemntv.asm(native, native_basis, loads=loads)
    expected = skfem.asm(reference, reference_basis, loads=loads)
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)


def test_composite_form_component_access_matches_skfem():
    mesh = skfemntv.MeshTet.init_tensor(
        np.linspace(0.0, 1.0, 3),
        np.linspace(0.0, 1.0, 2),
        np.linspace(0.0, 1.0, 2),
    )
    native_basis = skfemntv.Basis(
        mesh,
        skfemntv.ElementTetP1() * skfemntv.ElementTetP1(),
        intorder=2,
    )
    reference_basis = skfem.Basis(
        skfem.MeshTet(mesh.p, mesh.t),
        skfem.ElementTetP1() * skfem.ElementTetP1(),
        intorder=2,
    )
    material = np.empty((2,) + native_basis.dx.shape)
    material[0] = 0.8
    material[1] = 1.3

    @skfemntv.BilinearForm
    def native(u1, u2, v1, v2, w):
        return w.material[0] * u1 * v1 + w.material[1] * u2 * v2

    @skfem.BilinearForm
    def reference(u1, u2, v1, v2, w):
        return w.material[0] * u1 * v1 + w.material[1] * u2 * v2

    actual = skfemntv.asm(native, native_basis, material=material)
    expected = skfem.asm(reference, reference_basis, material=material)
    np.testing.assert_allclose(
        actual.toarray(), expected.toarray(), rtol=3e-14, atol=3e-14
    )


def test_component_access_reports_missing_invalid_and_out_of_range():
    basis, _ = _bases()

    @skfemntv.BilinearForm
    def indexed(u, v, w):
        return w.material[2] * dot(u, v)

    with pytest.raises(ValueError, match="missing form parameter 'material'"):
        skfemntv.asm(indexed, basis)
    with pytest.raises(ValueError, match="has no component 2"):
        skfemntv.asm(indexed, basis, material=np.ones((2,) + basis.dx.shape))

    @skfemntv.BilinearForm
    def noninteger(u, v, w):
        return w.material["stiffness"] * dot(u, v)

    with pytest.raises(ValueError, match="component index must be an integer"):
        skfemntv.asm(noninteger, basis)
