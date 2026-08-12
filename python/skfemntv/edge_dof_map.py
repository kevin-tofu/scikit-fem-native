"""Discrete local/global orientation map for one edge-owned scalar DOF."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .edge_topology import OrientedEdgeTopology,build_oriented_edge_topology


@dataclass(frozen=True)
class OrientedEdgeDofMap:
    """Map local edge basis functions to globally oriented edge unknowns.

    This is not an H(curl) finite-element space.  It only expresses the
    algebraic rule for one scalar moment per edge.
    """

    topology: OrientedEdgeTopology

    @property
    def ndofs(self) -> int:
        return int(self.topology.edges.shape[1])

    @property
    def element_dofs(self) -> np.ndarray:
        return self.topology.element_edges

    @property
    def basis_signs(self) -> np.ndarray:
        return self.topology.element_edge_signs

    def local_coefficients(self,global_coefficients) -> np.ndarray:
        """Convert global edge moments to coefficients of local edge bases."""
        values=np.asarray(global_coefficients)
        if values.ndim!=1 or values.shape[0]!=self.ndofs:
            raise ValueError(
                f"global edge coefficients must have shape ({self.ndofs},)"
            )
        return self.basis_signs*values[self.element_dofs]

    def global_moments_from_local(self,local_coefficients) -> np.ndarray:
        """Recover per-cell global-oriented moments without merging cells."""
        values=np.asarray(local_coefficients)
        if values.shape!=self.element_dofs.shape:
            raise ValueError(
                "local edge coefficients must have shape "
                f"{self.element_dofs.shape}"
            )
        return self.basis_signs*values


def build_oriented_edge_dof_map(mesh) -> OrientedEdgeDofMap:
    return OrientedEdgeDofMap(build_oriented_edge_topology(mesh))


__all__=["OrientedEdgeDofMap","build_oriented_edge_dof_map"]
