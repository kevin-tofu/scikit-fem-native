"""Oriented edge topology primitives for the future H(curl) space.

This module contains no basis functions or finite-element mappings.  It fixes
only the global edge identity and the sign relating each directed local edge
to that global orientation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._skfn import build_oriented_edge_topology as _build_native_edge_topology


_LOCAL_EDGES = {
    (2, 3): ((0, 1), (1, 2), (2, 0)),
    (3, 4): ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)),
}


@dataclass(frozen=True)
class OrientedEdgeTopology:
    """Global edges and the local-to-global orientation of every cell edge."""

    edges: np.ndarray
    element_edges: np.ndarray
    element_edge_signs: np.ndarray
    local_edges: tuple[tuple[int, int], ...]


def build_oriented_edge_topology(mesh) -> OrientedEdgeTopology:
    """Build deterministic edge IDs using ascending global vertex pairs.

    ``element_edge_signs[local_edge, cell]`` is ``+1`` when the directed
    local pair agrees with the ascending global pair and ``-1`` otherwise.
    The current contract intentionally covers linear triangle/tetrahedron
    corner topology only; higher-order edge nodes do not change edge identity.
    """
    dimension=int(mesh.dim())
    corners=3 if dimension==2 else 4 if dimension==3 else None
    key=(dimension,corners)
    if corners is None or key not in _LOCAL_EDGES:
        raise NotImplementedError(
            "oriented edge topology currently supports triangles and tetrahedra"
        )
    if mesh.t.shape[0] < corners:
        raise ValueError("mesh connectivity does not contain all corner vertices")

    local_edges=_LOCAL_EDGES[key]
    connectivity=np.ascontiguousarray(mesh.t[:corners],dtype=np.int64)
    # The native routine preserves the historical cell-major/local-edge-major
    # first-seen numbering, so IDs and orientation arrays remain API-compatible.
    edges,element_edges,signs=_build_native_edge_topology(
        connectivity,dimension
    )
    return OrientedEdgeTopology(
        np.ascontiguousarray(edges),
        np.ascontiguousarray(element_edges),
        np.ascontiguousarray(signs),
        local_edges,
    )


def boundary_edge_ids(mesh,topology=None) -> np.ndarray:
    """Return sorted global edge IDs belonging to at least one boundary facet."""
    topology=(
        build_oriented_edge_topology(mesh) if topology is None else topology
    )
    lookup={tuple(edge):index for index,edge in enumerate(topology.edges.T)}
    selected=set()
    for facet in mesh.boundary_facets():
        nodes=np.unique(mesh._facet_connectivity([facet],full=False)[:,0])
        for first_index,first in enumerate(nodes):
            for second in nodes[first_index+1:]:
                edge=(min(int(first),int(second)),max(int(first),int(second)))
                edge_id=lookup.get(edge)
                if edge_id is not None:
                    selected.add(edge_id)
    return np.asarray(sorted(selected),dtype=np.int64)


__all__=[
    "OrientedEdgeTopology",
    "boundary_edge_ids",
    "build_oriented_edge_topology",
]
