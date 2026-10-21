import numpy as np
import pytest

import skfemntv
import skfemntv.basis as basis_module


@pytest.mark.parametrize("vector_components", [1, 2])
def test_native_composite_dofs_match_python_numbering(
    monkeypatch, vector_components
):
    if basis_module._assemble_composite_dofs_native is None:
        pytest.skip("native composite DOF kernel is unavailable")
    mesh = skfemntv.MeshTri2.from_mesh(
        skfemntv.MeshTri.init_tensor(
            np.linspace(0., 1., 4), np.linspace(0., 1., 3)
        )
    )
    first = skfemntv.ElementTriP2()
    if vector_components > 1:
        first = skfemntv.ElementVector(first, dim=vector_components)
    element = first * skfemntv.ElementTriP1()

    native = skfemntv.Basis(mesh, element, intorder=4)
    monkeypatch.setattr(basis_module, "_assemble_composite_dofs_native", None)
    python = skfemntv.Basis(mesh, element, intorder=4)

    assert native.N == python.N
    np.testing.assert_array_equal(native.nodal_dofs, python.nodal_dofs)
    np.testing.assert_array_equal(native.element_dofs, python.element_dofs)
    np.testing.assert_array_equal(native.doflocs, python.doflocs)
    for native_field, python_field in zip(native.subbases, python.subbases):
        assert native_field.N == python_field.N
        np.testing.assert_array_equal(native_field.nodal_dofs, python_field.nodal_dofs)
        np.testing.assert_array_equal(native_field.element_dofs, python_field.element_dofs)
