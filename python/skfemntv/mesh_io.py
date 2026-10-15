"""Optional meshio-based mesh import and export."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .regions import CellRegion, FacetRegion


_FORMATS = {
    "triangle": ("MeshTri", 2, "line"),
    "triangle6": ("MeshTri2", 2, "line"),
    "quad": ("MeshQuad", 2, "line"),
    "quad9": ("MeshQuad2", 2, "line"),
    "tetra": ("MeshTet", 3, "triangle"),
    "tetra10": ("MeshTet2", 3, "triangle"),
    "hexahedron": ("MeshHex", 3, "quad"),
    "hexahedron27": ("MeshHex2", 3, "quad"),
    "wedge": ("MeshWedge1", 3, None),
}


def _meshio():
    try:
        import meshio
    except ImportError as error:
        raise ImportError(
            "mesh I/O requires the optional 'meshio' dependency; "
            "install scikit-fem-native[io]"
        ) from error
    return meshio


def _choose_type(cells, force=None):
    if force is not None:
        if force not in _FORMATS:
            raise NotImplementedError(f"meshio cell type {force!r} is not supported")
        if force not in cells:
            raise ValueError(f"meshio data does not contain cell type {force!r}")
        return force
    for cell_type in ("tetra", "tetra10", "hexahedron", "hexahedron27", "wedge",
                      "triangle", "triangle6", "quad", "quad9"):
        if cell_type in cells:
            return cell_type
    raise NotImplementedError(f"meshio cell types are not supported: {tuple(cells)}")


def _facet_key(nodes):
    return tuple(sorted(set(map(int, nodes))))


def from_meshio(mesh, out=None, *, force_meshio_type=None, int_data_to_sets=False,
                **kwargs):
    """Convert a meshio Mesh to the matching native mesh family."""
    del kwargs
    if int_data_to_sets and hasattr(mesh, "int_data_to_sets"):
        mesh.int_data_to_sets()
    cells = mesh.cells_dict
    cell_type = _choose_type(cells, force_meshio_type)
    class_name, dim, boundary_type = _FORMATS[cell_type]
    from . import basis as mesh_basis

    mesh_class = getattr(mesh_basis, class_name)
    points = np.ascontiguousarray(np.asarray(mesh.points)[:, :dim].T)
    connectivity = np.ascontiguousarray(np.asarray(cells[cell_type]).T)
    if cell_type in ("hexahedron", "hexahedron27"):
        hex_mapping = np.array(
            [0, 3, 6, 2, 1, 5, 7, 4, 10, 16, 14, 9, 12, 18, 17, 11,
             8, 15, 19, 13, 20, 25, 22, 23, 21, 24, 26],
            dtype=np.int64,
        )
        inverse = np.argsort(hex_mapping)
        connectivity = connectivity[inverse[: connectivity.shape[0]]]
    result = mesh_class(points, connectivity)
    main_count = result.nelements

    cell_sets = getattr(mesh, "cell_sets_dict", {}) or {}
    result._subdomains = {
        name: CellRegion(np.asarray(by_type[cell_type], dtype=np.int64), main_count)
        for name, by_type in cell_sets.items()
        if cell_type in by_type and not name.startswith("gmsh:")
    }

    result._boundaries = {}
    if boundary_type is not None and boundary_type in cells:
        facet_lookup = {}
        for facet_id in range(result.facets.shape[1]):
            if hasattr(result, "_facet_sizes"):
                nodes = result.facets[:result._facet_sizes[facet_id], facet_id]
            else:
                nodes = result.facets[:, facet_id]
            facet_lookup[_facet_key(nodes)] = facet_id
        for name, by_type in cell_sets.items():
            if boundary_type not in by_type or name.startswith("gmsh:"):
                continue
            selected_cells = np.asarray(by_type[boundary_type], dtype=np.int64)
            selected = []
            for cell_id in selected_cells:
                key = _facet_key(cells[boundary_type][cell_id])
                if key in facet_lookup:
                    selected.append(facet_lookup[key])
            if selected:
                result._boundaries[name] = FacetRegion(
                    np.asarray(selected, dtype=np.int64), result.facets.shape[1]
                )

    # Decode the compact tags written by scikit-fem's meshio adapter.
    cell_data = getattr(mesh, "cell_data_dict", {}) or {}
    for tag_name, by_type in cell_data.items():
        if cell_type not in by_type or not tag_name.startswith("skfem:"):
            continue
        _, kind, name = tag_name.split(":", 2)
        tags = np.asarray(by_type[cell_type], dtype=np.int64)
        if kind == "s":
            result._subdomains[name] = CellRegion(
                np.flatnonzero(tags), main_count
            )
        elif kind == "b":
            selected = []
            for local_facet in range(result.t2f.shape[0]):
                selected.extend(
                    result.t2f[local_facet, np.flatnonzero(tags & (1 << local_facet))]
                )
            if selected:
                result._boundaries[name] = FacetRegion(
                    np.unique(selected), result.facets.shape[1]
                )
    if not result._boundaries:
        result._boundaries = None
    if not result._subdomains:
        result._subdomains = None

    if isinstance(out, list):
        for index, field in enumerate(out):
            out[index] = getattr(mesh, field)
    return result


def load_mesh(filename, out=None, **kwargs):
    return from_meshio(_meshio().read(filename, **kwargs), out=out)


def _encode_tags(mesh):
    encoded = {}
    for name, elements in (mesh._subdomains or {}).items():
        values = np.zeros(mesh.nelements, dtype=np.int32)
        values[np.asarray(elements, dtype=np.int64)] = 1
        encoded[f"skfem:s:{name}"] = values
    for name, facets in (mesh._boundaries or {}).items():
        mask = np.zeros(mesh.t2f.shape, dtype=np.int32)
        for local_facet in range(mesh.t2f.shape[0]):
            mask[local_facet] = np.isin(mesh.t2f[local_facet], facets)
        encoded[f"skfem:b:{name}"] = (
            (1 << np.arange(mesh.t2f.shape[0], dtype=np.int32)) @ mask
        )
    return encoded


def save_mesh(mesh, filename, point_data=None, cell_data=None, *,
              encode_cell_data=True, encode_point_data=False, **kwargs):
    """Write a supported mesh and optional fields through meshio."""
    meshio = _meshio()
    type_by_class = {name: cell_type for cell_type, (name, _, _) in _FORMATS.items()}
    mesh_type = type(mesh).__name__
    if mesh_type not in type_by_class:
        raise NotImplementedError(f"mesh export is not supported for {mesh_type}")
    cell_type = type_by_class[mesh_type]
    connectivity = np.asarray(mesh.t.T, dtype=np.int64)
    if cell_type in ("hexahedron", "hexahedron27"):
        hex_mapping = np.array(
            [0, 3, 6, 2, 1, 5, 7, 4, 10, 16, 14, 9, 12, 18, 17, 11,
             8, 15, 19, 13, 20, 25, 22, 23, 21, 24, 26],
            dtype=np.int64,
        )
        connectivity = connectivity[hex_mapping[: connectivity.shape[1]]]
    cells = {cell_type: connectivity}
    data = {} if cell_data is None else dict(cell_data)
    if encode_cell_data:
        data.update(_encode_tags(mesh))
    if encode_point_data:
        point_data = {} if point_data is None else dict(point_data)
        for name, region in (mesh._subdomains or {}).items():
            values = np.zeros(mesh.p.shape[1], dtype=np.int32)
            values[np.unique(mesh.t[:, np.asarray(region, dtype=np.int64)])] = 1
            point_data[f"skfem:s:{name}"] = values
        for name, region in (mesh._boundaries or {}).items():
            values = np.zeros(mesh.p.shape[1], dtype=np.int32)
            values[np.unique(mesh.facets[:, np.asarray(region, dtype=np.int64)])] = 1
            point_data[f"skfem:b:{name}"] = values
    mio = meshio.Mesh(
        np.asarray(mesh.p.T),
        cells,
        point_data=point_data,
        cell_data={name: [np.asarray(values)] for name, values in data.items()},
    )
    path = Path(filename)
    if "file_format" not in kwargs and path.suffix == ".msh":
        kwargs["file_format"] = "gmsh"
    return meshio.write(path, mio, **kwargs)
