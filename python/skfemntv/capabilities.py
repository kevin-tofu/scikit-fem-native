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
    Capability("mesh.line", "mesh", CapabilityStatus.EXPERIMENTAL, "Line mesh construction, topology, named boundary/subdomain tags, refinement, restriction, and element removal; one-dimensional Basis/assembly parity remains incomplete."),
    Capability("mesh.refine_line", "mesh", CapabilityStatus.EXPERIMENTAL, "Uniform and element-marked refinement for MeshLine, preserving named boundary and subdomain tags."),
    Capability("mesh.restrict_line", "mesh", CapabilityStatus.EXPERIMENTAL, "Cell restriction and removal for MeshLine, with compacted vertices and remapped named regions."),
    Capability("mesh.refine_tri", "mesh", CapabilityStatus.EXPERIMENTAL, "Uniform and conforming adaptive refinement for linear MeshTri; named boundaries and subdomains are preserved."),
    Capability("mesh.refine_quad", "mesh", CapabilityStatus.EXPERIMENTAL, "Uniform refinement for linear MeshQuad; named boundaries and subdomains are preserved."),
    Capability("mesh.refine_tet", "mesh", CapabilityStatus.EXPERIMENTAL, "Uniform and conforming adaptive refinement for linear MeshTet; named boundaries and subdomains are preserved."),
    Capability("mesh.restrict_tri", "mesh", CapabilityStatus.EXPERIMENTAL, "Cell restriction and removal for linear MeshTri, with compacted vertices and remapped named regions."),
    Capability("mesh.restrict_quad", "mesh", CapabilityStatus.EXPERIMENTAL, "Cell restriction and removal for linear MeshQuad, with compacted vertices and remapped named regions."),
    Capability("mesh.restrict_tet", "mesh", CapabilityStatus.EXPERIMENTAL, "Cell restriction and removal for linear MeshTet, with compacted vertices and remapped named regions."),
    Capability("space.l2_dg", "space", CapabilityStatus.EXPERIMENTAL, "Scalar discontinuous cell-local spaces."),
    Capability("space.hcurl", "space", CapabilityStatus.EXPERIMENTAL, "affine triangle/tetrahedron Nedelec spaces are accepted by public Basis/asm, through N3 on triangles and tetrahedra; curved mappings and higher orders remain unavailable."),
    Capability("space.hcurl_tri_n1_basis", "space", CapabilityStatus.EXPERIMENTAL, "Affine-triangle Nedelec basis with oriented global edge DOFs, interpolation, point evaluation, projection, and public mass/curl assembly."),
    Capability("space.hcurl_tri_n2_basis", "space", CapabilityStatus.EXPERIMENTAL, "Second-order affine-triangle Nedelec basis with two oriented edge DOFs, cell DOFs, interpolation, point evaluation, projection, and public mass/curl assembly."),
    Capability("space.hcurl_tri_n3_basis", "space", CapabilityStatus.EXPERIMENTAL, "Third-order affine-triangle Nedelec basis with three oriented edge DOFs, six cell DOFs, interpolation, point evaluation, projection, and public mass/curl assembly."),
    Capability("space.hcurl_tet_n1_basis", "space", CapabilityStatus.EXPERIMENTAL, "Affine-tetrahedron Nedelec basis with oriented global edge DOFs, interpolation, point evaluation, projection, and public mass/curl assembly."),
    Capability("space.hcurl_tet_n2_basis", "space", CapabilityStatus.EXPERIMENTAL, "Second-order affine-tetrahedron Nedelec basis with oriented edge and face DOFs, moment interpolation, point evaluation, and public mass/curl assembly."),
    Capability("space.hcurl_tet_n3_basis", "space", CapabilityStatus.EXPERIMENTAL, "Third-order affine-tetrahedron Nedelec basis with oriented edge, face, and cell DOFs, moment interpolation, point evaluation, and public mass/curl assembly."),
    Capability("space.hdiv", "space", CapabilityStatus.EXPERIMENTAL, "Affine simplex RT0 elements on triangles and tetrahedra with oriented facet-flux DOFs, contravariant Piola mapping, interpolation/divergence, and selected public Basis/asm forms."),
    Capability("dof.vertex", "dof", CapabilityStatus.SUPPORTED, "Vertex and mesh-node-associated DOFs."),
    Capability("dof.edge", "dof", CapabilityStatus.EXPERIMENTAL, "Globally shared tangential edge DOFs for affine TriN1/TetN1, two per edge for TriN2/TetN2, and three per edge for TriN3/TetN3."),
    Capability("dof.edge_orientation_map", "dof", CapabilityStatus.EXPERIMENTAL, "Local/global edge numbering for affine N1/N2 triangle/tetrahedron and signed-permutation N3 triangle spaces, using scikit-fem-compatible edge IDs."),
    Capability("dof.edge_boundary_tri", "dof", CapabilityStatus.EXPERIMENTAL, "All, named, predicate, and explicit boundary-edge selection for affine TriN1/N2/N3 bases."),
    Capability("dof.edge_boundary_tet", "dof", CapabilityStatus.EXPERIMENTAL, "All, named, predicate, and explicit boundary-facet edge selection for AffineTetN1/N2/N3 bases."),
    Capability("dof.facet", "dof", CapabilityStatus.EXPERIMENTAL, "Two orientation-transformed face DOFs per triangular face for affine TetN2 and six per face for TetN3."),
    Capability("dof.cell", "dof", CapabilityStatus.EXPERIMENTAL, "Cell-local DG DOFs and interior H(curl) DOFs for affine TriN2/N3 and TetN3."),
    Capability("topology.oriented_edge", "topology", CapabilityStatus.EXPERIMENTAL, "Lexicographically numbered triangle/tetrahedron edges and local orientation signs or permutations used by affine H(curl) spaces."),
    Capability("element.tri_n1_reference", "element", CapabilityStatus.EXPERIMENTAL, "Reference-triangle lowest-order Nedelec basis, curl, and edge moments, mapped by affine-triangle Basis."),
    Capability("element.tri_n2_reference", "element", CapabilityStatus.EXPERIMENTAL, "Reference-triangle second-order Nedelec basis and curl, mapped by affine-triangle Basis."),
    Capability("element.tri_n3_reference", "element", CapabilityStatus.EXPERIMENTAL, "Reference-triangle third-order Nedelec basis and curl, mapped by affine-triangle Basis."),
    Capability("element.tet_n1_reference", "element", CapabilityStatus.EXPERIMENTAL, "Reference-tetrahedron lowest-order Nedelec basis, vector curl, and edge moments, mapped by affine-tetrahedron Basis."),
    Capability("element.tet_n2_reference", "element", CapabilityStatus.EXPERIMENTAL, "Reference-tetrahedron second-order Nedelec basis, gradients, and vector curl, mapped by affine-tetrahedron Basis with entity transformations."),
    Capability("element.tet_n3_reference", "element", CapabilityStatus.EXPERIMENTAL, "Reference-tetrahedron third-order Nedelec basis, gradients, and vector curl, mapped by affine-tetrahedron Basis with entity transformations."),
    Capability("mapping.h1", "mapping", CapabilityStatus.SUPPORTED, "Scalar pullback and physical gradients."),
    Capability("mapping.covariant_piola", "mapping", CapabilityStatus.EXPERIMENTAL, "Affine H(curl) vector and curl transformation for triangle and tetrahedron N1/N2/N3 Nedelec elements."),
    Capability("mapping.covariant_piola_tri_affine", "mapping", CapabilityStatus.EXPERIMENTAL, "Reference-to-physical value and scalar-curl transformation used by affine-triangle N1/N2/N3 Basis."),
    Capability("mapping.covariant_piola_tet_affine", "mapping", CapabilityStatus.EXPERIMENTAL, "Reference-to-physical value and vector-curl transformation used by affine-tetrahedron N1/N2/N3 Basis."),
    Capability("mapping.contravariant_piola", "mapping", CapabilityStatus.EXPERIMENTAL, "Affine simplex RT0 vector and divergence transformation."),
    Capability("form.value", "form", CapabilityStatus.SUPPORTED, "Value contractions."),
    Capability("form.grad", "form", CapabilityStatus.SUPPORTED, "Physical-gradient contractions."),
    Capability("helper.det", "helper", CapabilityStatus.EXPERIMENTAL, "Numeric component-first 2x2/3x3 determinants; symbolic nonlinear form expressions are unsupported."),
    Capability("helper.inv", "helper", CapabilityStatus.EXPERIMENTAL, "Numeric component-first 2x2/3x3 inverses; symbolic nonlinear form expressions are unsupported."),
    Capability("helper.eye", "helper", CapabilityStatus.EXPERIMENTAL, "Diagonal matrices scaled by scalar or quadrature-array coefficients; symbolic form expressions are unsupported."),
    Capability("helper.cross", "helper", CapabilityStatus.EXPERIMENTAL, "Numeric component-first 2D/3D cross products; symbolic bilinear contractions are unsupported."),
    Capability("form.coefficient_components", "form", CapabilityStatus.SUPPORTED, "Integer component access along the first coefficient axis."),
    Capability("form.multiple_coefficients", "form", CapabilityStatus.SUPPORTED, "Multiple independently named coefficient fields in one form."),
    Capability("form.anisotropic_gradient", "form", CapabilityStatus.SUPPORTED, "Scalar H1 gradient contraction with constant or quadrature-dependent rank-2 tensors."),
    Capability("form.div", "form", CapabilityStatus.SUPPORTED, "Divergence of nodal vector fields."),
    Capability("form.curl", "form", CapabilityStatus.EXPERIMENTAL, "Curl-curl forms on vector H1 spaces and affine N1/N2/N3 triangle/tetrahedron H(curl) spaces."),
    Capability("assembly.hcurl_tri_n1", "assembly", CapabilityStatus.EXPERIMENTAL, "Public asm mass and scalar-curl forms plus reusable CSR Maxwell assembly for affine TriN1 Basis."),
    Capability("assembly.hcurl_tri_n2", "assembly", CapabilityStatus.EXPERIMENTAL, "Public asm mass, scalar-curl, and vector-load forms plus reusable CSR Maxwell assembly for affine TriN2 Basis."),
    Capability("assembly.hcurl_tri_n3", "assembly", CapabilityStatus.EXPERIMENTAL, "Public asm mass, scalar-curl, and vector-load forms plus reusable CSR Maxwell assembly for affine TriN3 Basis."),
    Capability("assembly.hcurl_tet_n1", "assembly", CapabilityStatus.EXPERIMENTAL, "Public asm mass and vector-curl forms plus reusable CSR Maxwell assembly for affine TetN1 Basis."),
    Capability("assembly.hcurl_tet_n1_load", "assembly", CapabilityStatus.EXPERIMENTAL, "Vector-load terms in public LinearForm/asm and reusable assembly for affine TetN1."),
    Capability("assembly.hcurl_tet_n2", "assembly", CapabilityStatus.EXPERIMENTAL, "Public asm mass, vector-curl, and vector-load forms plus reusable CSR assembly for affine TetN2."),
    Capability("assembly.hcurl_tet_n3", "assembly", CapabilityStatus.EXPERIMENTAL, "Public asm mass, vector-curl, and vector-load forms plus reusable CSR assembly for affine TetN3."),
    Capability("assembly.hcurl_tri_n1_load", "assembly", CapabilityStatus.EXPERIMENTAL, "Vector-load terms in public LinearForm/asm and reusable assembly for affine TriN1."),
    Capability("assembly.hdiv_rt0", "assembly", CapabilityStatus.EXPERIMENTAL, "Public asm scalar/tensor mass, divergence-divergence, vector-load, rectangular RT0-DG0 divergence coupling, and RT0 facet normal-trace linear/bilinear forms for affine simplex RT0; C++ integration and Python/SciPy sparse scatter."),
    Capability("evaluation.quadrature", "evaluation", CapabilityStatus.SUPPORTED, "Interpolation at basis quadrature points."),
    Capability("evaluation.l2_projection", "evaluation", CapabilityStatus.EXPERIMENTAL, "L2 projection on full affine triangular/tetrahedral scalar/vector P0/P1/P2 bases; subset projection, composites, and curved geometry remain unavailable."),
    Capability("evaluation.hcurl_tri_n1", "evaluation", CapabilityStatus.EXPERIMENTAL, "Interpolation, quadrature and arbitrary-point value/curl evaluation, refinterp, and L2 projection for affine TriN1."),
    Capability("evaluation.hcurl_tri_n2", "evaluation", CapabilityStatus.EXPERIMENTAL, "Interpolation, quadrature and arbitrary-point value/curl/gradient evaluation, refinterp, and L2 projection for affine TriN2."),
    Capability("evaluation.hcurl_tri_n3", "evaluation", CapabilityStatus.EXPERIMENTAL, "Interpolation, quadrature and arbitrary-point value/curl/gradient evaluation, refinterp, and L2 projection for affine TriN3."),
    Capability("evaluation.hcurl_tet_n1", "evaluation", CapabilityStatus.EXPERIMENTAL, "Interpolation, quadrature and arbitrary-point value/curl evaluation, refinterp, and L2 projection for affine TetN1."),
    Capability("evaluation.hcurl_tet_n2", "evaluation", CapabilityStatus.EXPERIMENTAL, "Edge/face moment interpolation, quadrature and arbitrary-point value/curl/gradient evaluation, refinterp, and L2 projection for affine TetN2."),
    Capability("evaluation.hcurl_tet_n3", "evaluation", CapabilityStatus.EXPERIMENTAL, "Edge/face/cell moment interpolation, quadrature and arbitrary-point value/curl/gradient evaluation, refinterp, and L2 projection for affine TetN3."),
    Capability("evaluation.arbitrary_point", "evaluation", CapabilityStatus.EXPERIMENTAL, "Physical-point value/gradient evaluation for affine triangular/tetrahedral P0/P1/P2 bases; other cells and curved simplex geometry remain unavailable."),
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
