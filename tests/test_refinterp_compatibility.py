"""Reference-element visualization interpolation parity tests."""

import numpy as np
import pytest
import skfem
import skfemntv


@pytest.mark.parametrize(
    "dimension,order", [(2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2)]
)
@pytest.mark.parametrize("vector", [False, True])
def test_refinterp_matches_skfem(dimension, order, vector):
    if dimension == 2:
        native_mesh = skfemntv.MeshTri.init_tensor([0., 1.], [0., 1.])
        reference_mesh = skfem.MeshTri(native_mesh.p, native_mesh.t)
        native_scalar = (skfemntv.ElementTriP0() if order == 0 else
                         skfemntv.ElementTriP1() if order == 1 else skfemntv.ElementTriP2())
        reference_scalar = (skfem.ElementTriP0() if order == 0 else
                            skfem.ElementTriP1() if order == 1 else skfem.ElementTriP2())
    else:
        native_mesh = skfemntv.MeshTet.init_tensor([0., 1.], [0., 1.], [0., 1.])
        reference_mesh = skfem.MeshTet(native_mesh.p, native_mesh.t)
        native_scalar = (skfemntv.ElementTetP0() if order == 0 else
                         skfemntv.ElementTetP1() if order == 1 else skfemntv.ElementTetP2())
        reference_scalar = (skfem.ElementTetP0() if order == 0 else
                            skfem.ElementTetP1() if order == 1 else skfem.ElementTetP2())
    if vector:
        native_element = skfemntv.ElementVector(native_scalar, dim=dimension)
        reference_element = skfem.ElementVector(reference_scalar, dim=dimension)
    else:
        native_element, reference_element = native_scalar, reference_scalar
    native = skfemntv.Basis(native_mesh, native_element)
    reference = skfem.Basis(reference_mesh, reference_element)
    native_coefficients = np.arange(native.N, dtype=float) / (native.N + 1.)
    reference_coefficients = np.arange(reference.N, dtype=float) / (reference.N + 1.)

    result_mesh, result_values = native.refinterp(native_coefficients, nrefs=1)
    expected_mesh, expected_values = reference.refinterp(
        reference_coefficients, Nrefs=1
    )
    np.testing.assert_allclose(result_mesh.p, expected_mesh.p, rtol=0., atol=0.)
    np.testing.assert_array_equal(result_mesh.t, expected_mesh.t)
    np.testing.assert_allclose(result_values, expected_values, rtol=2e-13, atol=2e-13)


def test_refinterp_validates_coefficients_and_reference_count():
    basis = skfemntv.Basis(
        skfemntv.MeshTri(), skfemntv.ElementTriP1()
    )
    with pytest.raises(ValueError, match="y must have shape"):
        basis.refinterp(np.zeros(basis.N + 1))
    with pytest.raises(ValueError, match="nrefs"):
        basis.refinterp(np.zeros(basis.N), nrefs=-1)


@pytest.mark.parametrize("dimension", [2, 3])
def test_refinterp_maps_curved_quadratic_geometry(dimension):
    if dimension == 2:
        linear = skfemntv.MeshTri.init_tensor([0., 1.], [0., 1.])
        reference_linear = skfem.MeshTri(linear.p, linear.t)
        native_mesh = skfemntv.MeshTri2.from_mesh(linear)
        reference_mesh = skfem.MeshTri2.from_mesh(reference_linear)
        native_element, reference_element = skfemntv.ElementTriP2(), skfem.ElementTriP2()
        axis = 1
    else:
        linear = skfemntv.MeshTet.init_tensor([0., 1.], [0., 1.], [0., 1.])
        reference_linear = skfem.MeshTet(linear.p, linear.t)
        native_mesh = skfemntv.MeshTet2.from_mesh(linear)
        reference_mesh = skfem.MeshTet2.from_mesh(reference_linear)
        native_element, reference_element = skfemntv.ElementTetP2(), skfem.ElementTetP2()
        axis = 2

    def warp(points):
        vertices = np.any(np.all(
            np.isclose(points.T[:, None, :], linear.p.T[None, :, :]), axis=2
        ), axis=1)
        points[axis, ~vertices] += .01 * np.sin(np.pi * points[0, ~vertices])
        return points

    native_mesh = skfemntv.MeshTri2(warp(native_mesh.p.copy()), native_mesh.t.copy()) if dimension == 2 else skfemntv.MeshTet2(warp(native_mesh.p.copy()), native_mesh.t.copy())
    reference_mesh = skfem.MeshTri2(doflocs=warp(reference_mesh.doflocs.copy()), t=reference_mesh.t.copy()) if dimension == 2 else skfem.MeshTet2(doflocs=warp(reference_mesh.doflocs.copy()), t=reference_mesh.t.copy())
    native = skfemntv.Basis(native_mesh, native_element)
    reference = skfem.Basis(reference_mesh, reference_element)
    native_coefficients = native.doflocs[0] ** 2 + native.doflocs[axis]
    reference_coefficients = reference.doflocs[0] ** 2 + reference.doflocs[axis]
    native_plot, native_values = native.refinterp(native_coefficients, nrefs=1)
    if dimension == 2:
        reference_plot, reference_values = reference.refinterp(
            reference_coefficients, nrefs=1
        )
        np.testing.assert_allclose(native_plot.p, reference_plot.p, rtol=0., atol=2e-15)
        np.testing.assert_array_equal(native_plot.t, reference_plot.t)
        np.testing.assert_allclose(native_values, reference_values, rtol=2e-12, atol=2e-12)
    else:
        # scikit-fem 12.0.2 fails in MeshTet2.init_refdom().refined(); compare
        # native interpolation against its independent arbitrary-point path.
        evaluated = native.interpolator(native_coefficients)(native_plot.p)
        np.testing.assert_allclose(native_values, evaluated, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize(
    "dimension,element_name",
    [(2, "ElementQuad0"), (2, "ElementQuad1"), (2, "ElementQuad2"),
     (2, "ElementQuad8"), (3, "ElementHex0"), (3, "ElementHex1"),
     (3, "ElementHex2"), (3, "ElementHex20")],
)
@pytest.mark.parametrize("vector", [False, True])
def test_tensor_refinterp_matches_skfem(dimension, element_name, vector):
    if dimension == 2:
        native_mesh = skfemntv.MeshQuad.init_tensor([0., 1.], [0., 1.])
        reference_mesh = skfem.MeshQuad(native_mesh.p, native_mesh.t)
    else:
        native_mesh = skfemntv.MeshHex.init_tensor([0., 1.], [0., 1.], [0., 1.])
        reference_mesh = skfem.MeshHex(native_mesh.p, native_mesh.t)
    native_scalar = getattr(skfemntv, element_name)()
    reference_scalar = (
        None if element_name in ("ElementQuad8", "ElementHex2", "ElementHex20")
        else getattr(skfem, element_name)()
    )
    if element_name == "ElementQuad2":
        native_mesh = skfemntv.MeshQuad2.from_mesh(native_mesh)
        reference_mesh = skfem.MeshQuad2.from_mesh(reference_mesh)
    elif element_name == "ElementQuad8":
        native_mesh = skfemntv.MeshQuad8.from_mesh(native_mesh)
    elif element_name == "ElementHex2":
        native_mesh = skfemntv.MeshHex2.from_mesh(native_mesh)
    elif element_name == "ElementHex20":
        native_mesh = skfemntv.MeshHex20.from_mesh(native_mesh)
    if vector:
        native_element = skfemntv.ElementVector(native_scalar, dim=dimension)
        reference_element = (
            None if reference_scalar is None else
            skfem.ElementVector(reference_scalar, dim=dimension)
        )
    else:
        native_element, reference_element = native_scalar, reference_scalar
    native = skfemntv.Basis(native_mesh, native_element)
    native_coefficients = np.arange(native.N, dtype=float) / (native.N + 1.)
    result_mesh, result_values = native.refinterp(native_coefficients, nrefs=1)
    if reference_element is None:
        # scikit-fem has no matching Quad8/Hex20 element, and its Hex2
        # refinterp fails while refining MeshHex2's reference domain.
        expected_values = native.interpolator(native_coefficients)(result_mesh.p)
        np.testing.assert_allclose(
            result_values, np.asarray(expected_values).reshape(-1),
            rtol=3e-13, atol=3e-13,
        )
        return
    reference = skfem.Basis(reference_mesh, reference_element)
    reference_coefficients = np.arange(reference.N, dtype=float) / (reference.N + 1.)
    expected_mesh, expected_values = reference.refinterp(
        reference_coefficients, nrefs=1
    )
    reference_by_point = {
        tuple(np.round(point, 13)): index
        for index, point in enumerate(expected_mesh.p.T)
    }
    permutation = np.asarray([
        reference_by_point[tuple(np.round(point, 13))]
        for point in result_mesh.p.T
    ])
    np.testing.assert_allclose(
        result_values.reshape(-1, result_mesh.p.shape[1]),
        expected_values.reshape(-1, expected_mesh.p.shape[1])[:, permutation],
        rtol=3e-13, atol=3e-13,
    )
    def cell_keys(mesh):
        return sorted(
            tuple(sorted(tuple(np.round(mesh.p[:, node], 13)) for node in cell))
            for cell in mesh.t.T
        )
    assert cell_keys(result_mesh) == cell_keys(expected_mesh)


@pytest.mark.parametrize("vector_first", [False, True])
def test_composite_refinterp_matches_first_field_contract(vector_first):
    native_mesh = skfemntv.MeshTri.init_tensor([0., 1.], [0., 1.])
    reference_mesh = skfem.MeshTri(native_mesh.p, native_mesh.t)
    if vector_first:
        native_first = skfemntv.ElementVector(skfemntv.ElementTriP1(), dim=2)
        reference_first = skfem.ElementVector(skfem.ElementTriP1(), dim=2)
        native_second = skfemntv.ElementTriP1()
        reference_second = skfem.ElementTriP1()
    else:
        native_first = skfemntv.ElementTriP1()
        reference_first = skfem.ElementTriP1()
        native_second = skfemntv.ElementDG(skfemntv.ElementTriP1())
        reference_second = skfem.ElementDG(skfem.ElementTriP1())
    native = skfemntv.Basis(
        native_mesh,
        skfemntv.ElementComposite(native_first, native_second),
    )
    reference = skfem.Basis(
        reference_mesh,
        skfem.ElementComposite(reference_first, reference_second),
    )
    native_coefficients = np.arange(native.N, dtype=float) / (native.N + 1.)
    reference_coefficients = np.arange(reference.N, dtype=float) / (reference.N + 1.)
    native_plot, native_values = native.refinterp(native_coefficients, nrefs=1)
    reference_plot, reference_values = reference.refinterp(
        reference_coefficients, nrefs=1
    )
    np.testing.assert_allclose(native_plot.p, reference_plot.p, rtol=0., atol=0.)
    np.testing.assert_array_equal(native_plot.t, reference_plot.t)
    np.testing.assert_allclose(native_values, reference_values, rtol=2e-13, atol=2e-13)
