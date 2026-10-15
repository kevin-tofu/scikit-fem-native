"""Optional meshio interoperability checks against scikit-fem."""

import importlib.util

import numpy as np
import pytest
import skfem

import skfemntv
from skfemntv import mesh_io


def _cell_keys(mesh):
    return sorted(
        tuple(sorted(map(tuple, np.round(mesh.p[:, cell].T, 13))))
        for cell in mesh.t.T
    )


def _boundary_keys(mesh, name):
    return sorted(
        tuple(sorted(map(tuple, np.round(mesh.p[:, mesh.facets[:, facet]].T, 13))))
        for facet in mesh.boundaries[name]
    )


def test_mesh_io_dependency_is_optional_and_error_is_actionable():
    if importlib.util.find_spec("meshio") is not None:
        pytest.skip("meshio is installed")
    mesh = skfemntv.MeshTri.init_tensor(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    with pytest.raises(ImportError, match=r"scikit-fem-native\[io\]"):
        mesh.save("unused.vtu")


def test_adapter_roundtrip_preserves_tags_and_fields_without_file_backend(monkeypatch):
    class FakeMesh:
        def __init__(self, points, cells, point_data=None, cell_data=None):
            self.points = points
            self.cells = cells
            self.point_data = point_data or {}
            self.cell_data = cell_data or {}
            self.cells_dict = {name: values for name, values in cells.items()}
            self.cell_data_dict = {
                name: {cell_type: arrays[0] for cell_type in cells}
                for name, arrays in self.cell_data.items()
            }
            self.cell_sets_dict = {}

    class FakeMeshIo:
        Mesh = FakeMesh

        def write(self, path, mesh, **kwargs):
            self.saved = mesh
            return None

        def read(self, path, **kwargs):
            return self.saved

    fake = FakeMeshIo()
    monkeypatch.setattr(mesh_io, "_meshio", lambda: fake)
    native = skfemntv.MeshTri.init_tensor(
        np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0])
    ).with_boundaries(
        {"left": lambda x: np.isclose(x[0], 0.0)}
    ).with_subdomains({"right": lambda x: x[0] > 1.0})
    point_values = np.arange(native.p.shape[1], dtype=np.float64)
    cell_values = np.arange(native.nelements, dtype=np.float64)
    native.save("mesh.vtu", point_data={"u": point_values}, cell_data={"q": cell_values})
    out = ["point_data", "cell_data"]
    loaded = skfemntv.MeshTri.load("mesh.vtu", out=out)

    assert _cell_keys(loaded) == _cell_keys(native)
    assert _boundary_keys(loaded, "left") == _boundary_keys(native, "left")
    assert len(loaded.subdomains["right"]) == len(native.subdomains["right"])
    np.testing.assert_array_equal(out[0]["u"], point_values)
    np.testing.assert_array_equal(out[1]["q"][0], cell_values)


def test_meshio_roundtrip_matches_scikit_fem(tmp_path):
    pytest.importorskip("meshio")
    native = skfemntv.MeshTri.init_tensor(
        np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0])
    ).with_boundaries(
        {
            "left": lambda x: np.isclose(x[0], 0.0),
            "right": lambda x: np.isclose(x[0], 2.0),
        }
    ).with_subdomains({"upper": lambda x: x[1] > 0.5})
    path = tmp_path / "mesh.vtu"
    point_values = np.arange(native.p.shape[1], dtype=np.float64)
    cell_values = np.arange(native.nelements, dtype=np.float64) * 2.0
    native.save(
        path,
        point_data={"u": point_values},
        cell_data={"energy": cell_values},
    )

    native_out = ["point_data", "cell_data"]
    reference_out = ["point_data", "cell_data"]
    actual = skfemntv.MeshTri.load(path, out=native_out)
    expected = skfem.MeshTri.load(path, out=reference_out)
    assert _cell_keys(actual) == _cell_keys(expected)
    assert set(actual.boundaries) == set(expected.boundaries)
    assert set(actual.subdomains) == set(expected.subdomains)
    for name in actual.boundaries:
        assert _boundary_keys(actual, name) == _boundary_keys(expected, name)
    np.testing.assert_array_equal(native_out[0]["u"], reference_out[0]["u"])
    np.testing.assert_array_equal(native_out[1]["energy"][0], reference_out[1]["energy"][0])


def test_meshio_import_accepts_upstream_encoded_tags(tmp_path):
    pytest.importorskip("meshio")
    reference = skfem.MeshTri.init_tensor(
        np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0])
    ).with_boundaries(
        {"left": lambda x: np.isclose(x[0], 0.0)}
    ).with_subdomains({"right": lambda x: x[0] > 1.0})
    path = tmp_path / "reference.vtu"
    reference.save(path)
    actual = skfemntv.MeshTri.load(path)
    assert _cell_keys(actual) == _cell_keys(reference)
    assert _boundary_keys(actual, "left") == _boundary_keys(reference, "left")
    assert len(actual.subdomains["right"]) == len(reference.subdomains["right"])
