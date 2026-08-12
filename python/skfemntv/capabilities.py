"""Machine-readable mathematical compatibility contract for skfemntv."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CapabilityStatus(str, Enum):
    SUPPORTED = "supported"
    EXPERIMENTAL = "experimental"
    PLANNED = "planned"
    EXTERNAL = "external"


@dataclass(frozen=True)
class Capability:
    name: str
    category: str
    status: CapabilityStatus
    detail: str


class UnsupportedCapabilityError(NotImplementedError):
    """Raised when a requested mathematical capability is unavailable."""


_ENTRIES = (
    Capability("space.h1", "space", CapabilityStatus.SUPPORTED, "Continuous nodal H1 spaces."),
    Capability("space.l2_dg", "space", CapabilityStatus.EXPERIMENTAL, "Scalar discontinuous cell-local spaces."),
    Capability("space.hcurl", "space", CapabilityStatus.PLANNED, "Requires oriented edge DOFs and covariant Piola mapping."),
    Capability("space.hcurl_tri_n1_basis", "space", CapabilityStatus.EXPERIMENTAL, "Minimal affine-triangle Nedelec basis with oriented global edge numbering and element mass/curl-curl integration; not accepted by public asm."),
    Capability("space.hdiv", "space", CapabilityStatus.PLANNED, "Requires oriented facet DOFs and contravariant Piola mapping."),
    Capability("dof.vertex", "dof", CapabilityStatus.SUPPORTED, "Vertex and mesh-node-associated DOFs."),
    Capability("dof.edge", "dof", CapabilityStatus.PLANNED, "Oriented edge-owned functionals."),
    Capability("dof.edge_orientation_map", "dof", CapabilityStatus.EXPERIMENTAL, "Local/global numbering and sign map for one scalar moment per triangle/tetrahedron edge; not an H(curl) space."),
    Capability("dof.edge_boundary_tri", "dof", CapabilityStatus.EXPERIMENTAL, "All, named, predicate, and explicit boundary-edge selection for AffineTriN1Basis."),
    Capability("dof.facet", "dof", CapabilityStatus.PLANNED, "Oriented facet-owned functionals."),
    Capability("dof.cell", "dof", CapabilityStatus.EXPERIMENTAL, "Cell-local DG DOFs."),
    Capability("topology.oriented_edge", "topology", CapabilityStatus.EXPERIMENTAL, "Deterministic triangle/tetrahedron edge IDs and local orientation signs; no edge DOF space yet."),
    Capability("element.tri_n1_reference", "element", CapabilityStatus.EXPERIMENTAL, "Reference-triangle lowest-order Nedelec basis, curl, and edge moments; no physical mapping or global space yet."),
    Capability("mapping.h1", "mapping", CapabilityStatus.SUPPORTED, "Scalar pullback and physical gradients."),
    Capability("mapping.covariant_piola", "mapping", CapabilityStatus.PLANNED, "H(curl) vector and curl transformation."),
    Capability("mapping.covariant_piola_tri_affine", "mapping", CapabilityStatus.EXPERIMENTAL, "Reference-to-physical value and scalar-curl transformation for affine triangles; not connected to Basis."),
    Capability("mapping.contravariant_piola", "mapping", CapabilityStatus.PLANNED, "H(div) vector and divergence transformation."),
    Capability("form.value", "form", CapabilityStatus.SUPPORTED, "Value contractions."),
    Capability("form.grad", "form", CapabilityStatus.SUPPORTED, "Physical-gradient contractions."),
    Capability("form.coefficient_components", "form", CapabilityStatus.SUPPORTED, "Integer component access along the first coefficient axis."),
    Capability("form.multiple_coefficients", "form", CapabilityStatus.SUPPORTED, "Multiple independently named coefficient fields in one form."),
    Capability("form.anisotropic_gradient", "form", CapabilityStatus.SUPPORTED, "Scalar H1 gradient contraction with constant or quadrature-dependent rank-2 tensors."),
    Capability("form.div", "form", CapabilityStatus.SUPPORTED, "Divergence of nodal vector fields."),
    Capability("form.curl", "form", CapabilityStatus.PLANNED, "Curl of H(curl) fields."),
    Capability("assembly.hcurl_tri_n1", "assembly", CapabilityStatus.EXPERIMENTAL, "Dedicated reusable CSR mass/curl-curl/Maxwell assembly for AffineTriN1Basis; separate from public asm."),
    Capability("evaluation.quadrature", "evaluation", CapabilityStatus.SUPPORTED, "Interpolation at basis quadrature points."),
    Capability("evaluation.hcurl_tri_n1", "evaluation", CapabilityStatus.EXPERIMENTAL, "Edge-moment interpolation and value/curl evaluation for AffineTriN1Basis."),
    Capability("evaluation.arbitrary_point", "evaluation", CapabilityStatus.PLANNED, "Physical point location and field evaluation."),
    Capability("preflight.bilinear_memory", "preflight", CapabilityStatus.SUPPORTED, "Allocation-free conservative estimate for standard and cross bilinear assemblers."),
    Capability("preflight.cut_memory", "preflight", CapabilityStatus.SUPPORTED, "Memory estimate for segmented cut-cell bilinear and cross assemblers."),
    Capability("solver.linear", "solver", CapabilityStatus.EXTERNAL, "Use SciPy, PETSc, or scikit-fem solver utilities."),
)

CAPABILITY_REGISTRY = {entry.name: entry for entry in _ENTRIES}


def get_capability(name: str) -> Capability:
    """Return one declared capability, rejecting unknown capability names."""
    try:
        return CAPABILITY_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"unknown skfemntv capability {name!r}") from exc


def supports(name: str, *, include_experimental: bool = False) -> bool:
    status = get_capability(name).status
    return status is CapabilityStatus.SUPPORTED or (
        include_experimental and status is CapabilityStatus.EXPERIMENTAL
    )


def require_capability(name: str, *, include_experimental: bool = False) -> Capability:
    """Return a usable capability or raise with its declared status and reason."""
    capability = get_capability(name)
    if supports(name, include_experimental=include_experimental):
        return capability
    raise UnsupportedCapabilityError(
        f"skfemntv capability {name!r} is {capability.status.value}: "
        f"{capability.detail}"
    )


def capabilities(*, category: str | None = None) -> tuple[Capability, ...]:
    """List declared capabilities in stable name order."""
    return tuple(
        entry for entry in sorted(_ENTRIES, key=lambda item: item.name)
        if category is None or entry.category == category
    )


__all__ = [
    "CAPABILITY_REGISTRY",
    "Capability",
    "CapabilityStatus",
    "UnsupportedCapabilityError",
    "capabilities",
    "get_capability",
    "require_capability",
    "supports",
]
