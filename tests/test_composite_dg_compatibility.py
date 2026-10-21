"""Composite continuous/discontinuous DOF and projection parity."""

import numpy as np
import pytest
import skfem
import skfemntv
import skfemntv.basis as basis_module


@pytest.mark.parametrize("python_fallback", [False, True])
@pytest.mark.parametrize("dg_order", [0, 1])
@pytest.mark.parametrize("all_discontinuous", [False, True])
def test_composite_dg_matches_skfem(
    monkeypatch, python_fallback, dg_order, all_discontinuous
):
    if python_fallback:
        monkeypatch.setattr(basis_module, "_assemble_composite_dofs_native", None)
    mesh = skfemntv.MeshTri.init_tensor([0., 1., 2.], [0., 1.])
    reference_mesh = skfem.MeshTri(mesh.p, mesh.t)
    native_dg = (
        skfemntv.ElementTriP0() if dg_order == 0
        else skfemntv.ElementTriP1()
    )
    reference_dg = (
        skfem.ElementTriP0() if dg_order == 0 else skfem.ElementTriP1()
    )
    native_first = (
        skfemntv.ElementDG(skfemntv.ElementTriP0())
        if all_discontinuous else skfemntv.ElementTriP1()
    )
    reference_first = (
        skfem.ElementDG(skfem.ElementTriP0())
        if all_discontinuous else skfem.ElementTriP1()
    )
    native_element = skfemntv.ElementComposite(
        native_first, skfemntv.ElementDG(native_dg)
    )
    reference_element = skfem.ElementComposite(
        reference_first, skfem.ElementDG(reference_dg)
    )
    native = skfemntv.Basis(mesh, native_element, intorder=5)
    reference = skfem.Basis(reference_mesh, reference_element, intorder=5)

    assert native.N == reference.N
    np.testing.assert_allclose(native.doflocs, reference.doflocs)
    np.testing.assert_array_equal(native.element_dofs, reference.element_dofs)
    np.testing.assert_array_equal(native.get_dofs().all(), reference.get_dofs().all())
    np.testing.assert_array_equal(
        native.get_dofs(elements=[0]).all(), reference.get_dofs(elements=[0]).all()
    )

    def continuous_field(x):
        return x[0] + 2. * x[1]

    def discontinuous_field(x):
        return x[0] ** 2 - x[1]

    coefficients = native.project((continuous_field, discontinuous_field))
    expected = reference.project((continuous_field, discontinuous_field))
    np.testing.assert_allclose(coefficients, expected, rtol=2e-11, atol=2e-11)

    points = np.array([[.2, .8, 1.4], [.2, .3, .6]])
    probe_matrices = native.probes(points)
    assert len(probe_matrices) == 2
    evaluated = native.evaluate_at(points, coefficients)
    for matrix, result in zip(probe_matrices, evaluated):
        np.testing.assert_allclose(matrix @ coefficients, result.value)


def test_composite_vector_dg_matches_skfem():
    mesh = skfemntv.MeshTri.init_tensor([0., 1., 2.], [0., 1.])
    reference_mesh = skfem.MeshTri(mesh.p, mesh.t)
    native_element = skfemntv.ElementComposite(
        skfemntv.ElementTriP1(),
        skfemntv.ElementVector(
            skfemntv.ElementDG(skfemntv.ElementTriP1()), dim=2
        ),
    )
    reference_element = skfem.ElementComposite(
        skfem.ElementTriP1(),
        skfem.ElementVector(skfem.ElementDG(skfem.ElementTriP1()), dim=2),
    )
    native = skfemntv.Basis(mesh, native_element, intorder=5)
    reference = skfem.Basis(reference_mesh, reference_element, intorder=5)
    assert native.N == reference.N
    np.testing.assert_allclose(native.doflocs, reference.doflocs)
    np.testing.assert_array_equal(native.element_dofs, reference.element_dofs)

    functions = (
        lambda x: x[0] + x[1],
        lambda x: np.stack((x[0] ** 2, 2. * x[1] - x[0])),
    )
    np.testing.assert_allclose(
        native.project(functions), reference.project(functions),
        rtol=2e-11, atol=2e-11,
    )
