import importlib

import pytest
import skfem
import numpy as np
import skfemntv


def _resolve(path):
    parts = path.split(".")
    for split in range(len(parts), 0, -1):
        try:
            value = importlib.import_module(".".join(parts[:split]))
        except ModuleNotFoundError:
            continue
        for part in parts[split:]:
            value = getattr(value, part)
        return value
    raise ModuleNotFoundError(path)


def test_compatibility_manifest_has_unique_names_and_known_statuses():
    entries = skfemntv.compatibility_items()

    assert len(entries) == len(skfemntv.COMPATIBILITY_MANIFEST)
    assert tuple(entry.upstream_name for entry in entries) == tuple(
        sorted(skfemntv.COMPATIBILITY_MANIFEST)
    )
    assert all(entry.category and entry.detail for entry in entries)


def test_supported_and_experimental_entries_resolve_native_symbols():
    for entry in skfemntv.compatibility_items():
        if entry.status in (
            skfemntv.CapabilityStatus.SUPPORTED,
            skfemntv.CapabilityStatus.EXPERIMENTAL,
        ):
            assert entry.native_name is not None, entry.upstream_name
            assert _resolve(entry.native_name) is not None, entry.native_name
        else:
            assert entry.native_name is None, entry.upstream_name


def test_manifest_names_exist_in_reference_scikit_fem_api():
    assert skfem.__version__ == skfemntv.UPSTREAM_REFERENCE_VERSION
    for entry in skfemntv.compatibility_items():
        assert _resolve(entry.upstream_name) is not None, entry.upstream_name


@pytest.mark.parametrize(
    "name,status",
    [
        ("skfem.Basis", skfemntv.CapabilityStatus.SUPPORTED),
        ("skfem.ElementTriP2", skfemntv.CapabilityStatus.SUPPORTED),
        ("skfem.ElementTriN1", skfemntv.CapabilityStatus.EXPERIMENTAL),
        ("skfem.MeshLine", skfemntv.CapabilityStatus.PLANNED),
        ("skfem.Basis.probes", skfemntv.CapabilityStatus.EXPERIMENTAL),
        ("skfem.Basis.interpolator", skfemntv.CapabilityStatus.EXPERIMENTAL),
        ("skfem.Basis.project", skfemntv.CapabilityStatus.EXPERIMENTAL),
        ("skfem.helpers.curl", skfemntv.CapabilityStatus.EXPERIMENTAL),
        ("skfem.helpers.det", skfemntv.CapabilityStatus.EXPERIMENTAL),
        ("skfem.helpers.inv", skfemntv.CapabilityStatus.EXPERIMENTAL),
        ("skfem.helpers.eye", skfemntv.CapabilityStatus.EXPERIMENTAL),
        ("skfem.helpers.cross", skfemntv.CapabilityStatus.EXPERIMENTAL),
        ("skfem.condense", skfemntv.CapabilityStatus.SUPPORTED),
        ("skfem.enforce", skfemntv.CapabilityStatus.SUPPORTED),
        ("skfem.penalize", skfemntv.CapabilityStatus.SUPPORTED),
        ("skfem.solve", skfemntv.CapabilityStatus.SUPPORTED),
        ("skfem.Mesh.refined", skfemntv.CapabilityStatus.EXPERIMENTAL),
        ("skfem.Mesh.remove_elements", skfemntv.CapabilityStatus.EXPERIMENTAL),
        ("skfem.Mesh.restrict", skfemntv.CapabilityStatus.EXPERIMENTAL),
        ("skfem.Mesh.load", skfemntv.CapabilityStatus.EXPERIMENTAL),
        ("skfem.Mesh.save", skfemntv.CapabilityStatus.EXPERIMENTAL),
    ],
)
def test_compatibility_status_for_workflow_landmarks(name, status):
    assert skfemntv.get_compatibility(name).status is status


@pytest.mark.parametrize(
    "mesh_type,upstream_type",
    [
        (skfemntv.MeshTri, skfem.MeshTri),
        (skfemntv.MeshQuad, skfem.MeshQuad),
        (skfemntv.MeshTet, skfem.MeshTet),
    ],
)
def test_mesh_lifecycle_manifest_matches_native_and_reference_methods(
    mesh_type, upstream_type
):
    for method in ("refined", "restrict", "remove_elements"):
        assert callable(getattr(mesh_type, method))
        assert callable(getattr(upstream_type, method))
        entry = skfemntv.get_compatibility(f"skfem.Mesh.{method}")
        assert entry.status is skfemntv.CapabilityStatus.EXPERIMENTAL


@pytest.mark.parametrize(
    "name",
    [
        "mesh.refine_tri",
        "mesh.refine_quad",
        "mesh.refine_tet",
        "mesh.restrict_tri",
        "mesh.restrict_quad",
        "mesh.restrict_tet",
    ],
)
def test_mesh_lifecycle_capabilities_are_declared_experimental(name):
    capability = skfemntv.get_capability(name)
    assert capability.category == "mesh"
    assert capability.status is skfemntv.CapabilityStatus.EXPERIMENTAL


def test_compatibility_status_filter_accepts_strings():
    planned = skfemntv.compatibility_items(status="planned")

    assert planned
    assert all(entry.status is skfemntv.CapabilityStatus.PLANNED for entry in planned)


def test_unknown_compatibility_symbol_has_a_precise_error():
    with pytest.raises(KeyError, match="unknown scikit-fem API name"):
        skfemntv.get_compatibility("skfem.not_a_public_symbol")


def test_scalar_element_can_be_passed_directly_to_basis_and_facet_basis():
    mesh = skfemntv.MeshTri.init_tensor(
        np.array((0.0, 1.0)), np.array((0.0, 1.0))
    )
    basis = skfemntv.Basis(mesh, skfemntv.ElementTriP1())
    facet_basis = skfemntv.FacetBasis(mesh, skfemntv.ElementTriP1())

    assert basis.N == mesh.p.shape[1]
    assert facet_basis.N == basis.N
