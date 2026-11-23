"""Check local node and mesh connectivity order against scikit-fem."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import skfem
import skfem.element as skfem_elements
import skfemntv


def _shared_nodal_element_names() -> tuple[str, ...]:
    names: list[str] = []
    for name in dir(skfemntv):
        if not name.startswith("Element") or name in {
            "ElementComposite",
            "ElementDG",
            "ElementVector",
        }:
            continue
        native_type = getattr(skfemntv, name)
        reference_type = getattr(skfem_elements, name, None)
        if (
            not inspect.isclass(native_type)
            or not inspect.isclass(reference_type)
        ):
            continue
        try:
            native_locations = native_type().doflocs
            reference_locations = reference_type().doflocs
        except (TypeError, ValueError, AttributeError):
            continue
        if native_locations.ndim == 2 and reference_locations.ndim == 2:
            names.append(name)
    return tuple(sorted(names))


SHARED_NODAL_ELEMENTS = _shared_nodal_element_names()


@pytest.mark.parametrize("element_name", SHARED_NODAL_ELEMENTS)
def test_every_shared_element_has_the_same_local_node_order(element_name: str):
    native_element = getattr(skfemntv, element_name)()
    reference_element = getattr(skfem_elements, element_name)()

    assert native_element.doflocs.shape == reference_element.doflocs.shape
    np.testing.assert_allclose(
        native_element.doflocs,
        reference_element.doflocs,
        rtol=0.0,
        atol=1.0e-14,
    )


def _mesh_and_basis(element_name: str):
    if element_name.startswith("ElementTri"):
        native_mesh = skfemntv.MeshTri.init_tensor(
            [0.0, 0.4, 1.0], [0.0, 0.6, 1.0]
        )
        mesh_name = "MeshTri"
    elif element_name.startswith("ElementQuad"):
        native_mesh = skfemntv.MeshQuad.init_tensor(
            [0.0, 0.4, 1.0], [0.0, 0.6, 1.0]
        )
        mesh_name = "MeshQuad"
    elif element_name.startswith("ElementTet"):
        native_mesh = skfemntv.MeshTet.init_tensor(
            [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]
        )
        mesh_name = "MeshTet"
    elif element_name.startswith("ElementHex"):
        native_mesh = skfemntv.MeshHex.init_tensor(
            [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]
        )
        mesh_name = "MeshHex"
    else:
        pytest.skip("line element basis parity is covered by line compatibility tests")

    reference_mesh = getattr(skfem, mesh_name)(
        native_mesh.p.copy(), native_mesh.t.copy()
    )
    if element_name in {"ElementQuad2", "ElementHex2"}:
        native_mesh = getattr(skfemntv, f"{mesh_name}2").from_mesh(native_mesh)
        reference_mesh = getattr(skfem, f"{mesh_name}2").from_mesh(reference_mesh)

    native_basis = skfemntv.Basis(
        native_mesh, getattr(skfemntv, element_name)(), intorder=4
    )
    reference_basis = skfem.Basis(
        reference_mesh, getattr(skfem_elements, element_name)(), intorder=4
    )
    return native_mesh, reference_mesh, native_basis, reference_basis


MESH_BASIS_ELEMENTS = tuple(
    name
    for name in SHARED_NODAL_ELEMENTS
    if name.startswith(("ElementTri", "ElementQuad", "ElementTet", "ElementHex"))
)


@pytest.mark.parametrize("element_name", MESH_BASIS_ELEMENTS)
def test_shared_element_dofs_select_the_same_physical_nodes(element_name: str):
    _native_mesh, _reference_mesh, native_basis, reference_basis = _mesh_and_basis(
        element_name
    )

    assert native_basis.element_dofs.shape == reference_basis.element_dofs.shape
    native_coordinates = native_basis.doflocs[:, native_basis.element_dofs]
    reference_coordinates = reference_basis.doflocs[:, reference_basis.element_dofs]
    np.testing.assert_allclose(
        native_coordinates,
        reference_coordinates,
        rtol=0.0,
        atol=1.0e-13,
    )


@pytest.mark.parametrize(
    ("mesh_name", "native_mesh"),
    (
        ("MeshTri", lambda: skfemntv.MeshTri.init_tensor([0.0, 0.4, 1.0], [0.0, 0.6, 1.0])),
        ("MeshQuad", lambda: skfemntv.MeshQuad.init_tensor([0.0, 0.4, 1.0], [0.0, 0.6, 1.0])),
        ("MeshTet", lambda: skfemntv.MeshTet.init_tensor([0.0, 1.0], [0.0, 1.0], [0.0, 1.0])),
        ("MeshHex", lambda: skfemntv.MeshHex.init_tensor([0.0, 1.0], [0.0, 1.0], [0.0, 1.0])),
    ),
)
def test_mesh_boundary_facets_keep_the_same_connectivity_and_orientation(
    mesh_name: str, native_mesh
):
    native = native_mesh()
    reference = getattr(skfem, mesh_name)(native.p.copy(), native.t.copy())

    np.testing.assert_array_equal(native.t, reference.t)
    np.testing.assert_array_equal(native.facets, reference.facets)
    np.testing.assert_array_equal(native.t2f, reference.t2f)
    np.testing.assert_array_equal(native.f2t, reference.f2t)
    np.testing.assert_array_equal(
        native.boundary_facets(), reference.boundary_facets()
    )


def test_hex20_local_order_is_the_scikit_fem_hex2_corner_and_edge_prefix():
    native = skfemntv.ElementHex20().doflocs
    reference = skfem.ElementHex2().doflocs[:20]

    np.testing.assert_allclose(native, reference, rtol=0.0, atol=1.0e-14)
