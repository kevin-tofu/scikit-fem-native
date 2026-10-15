"""Machine-readable compatibility inventory for the scikit-fem public API.

The manifest is intentionally independent of importing scikit-fem at runtime.
It describes the upstream API names that callers commonly use and their
nearest native equivalent or current disposition.
"""

from __future__ import annotations

from dataclasses import dataclass

from .capabilities import CapabilityStatus


UPSTREAM_REFERENCE_VERSION = "12.0.2"


@dataclass(frozen=True)
class CompatibilityEntry:
    """Disposition of one public scikit-fem API symbol or workflow."""

    upstream_name: str
    category: str
    status: CapabilityStatus
    native_name: str | None
    detail: str


_ENTRIES = (
    CompatibilityEntry(
        'skfem.asm',
        'assembly',
        CapabilityStatus.SUPPORTED,
        'skfemntv.asm',
        'Supported for the documented native form and basis subset.',
    ),
    CompatibilityEntry(
        'skfem.BilinearForm',
        'form',
        CapabilityStatus.SUPPORTED,
        'skfemntv.BilinearForm',
        'Typed native form vocabulary; not arbitrary Python/NumPy tracing.',
    ),
    CompatibilityEntry(
        'skfem.LinearForm',
        'form',
        CapabilityStatus.SUPPORTED,
        'skfemntv.LinearForm',
        'Typed native form vocabulary; selected volume, facet, composite, and cut paths.',
    ),
    CompatibilityEntry(
        'skfem.Functional',
        'form',
        CapabilityStatus.SUPPORTED,
        'skfemntv.Functional',
        'Functional integration on supported basis providers.',
    ),
    CompatibilityEntry(
        'skfem.Basis',
        'basis',
        CapabilityStatus.SUPPORTED,
        'skfemntv.Basis',
        'Selected H1/DG elements, geometry, regions, and quadrature evaluation.',
    ),
    CompatibilityEntry(
        'skfem.FacetBasis',
        'basis',
        CapabilityStatus.SUPPORTED,
        'skfemntv.FacetBasis',
        'Selected exterior-facet topologies and element families.',
    ),
    CompatibilityEntry(
        'skfem.InteriorFacetBasis',
        'basis',
        CapabilityStatus.SUPPORTED,
        'skfemntv.InteriorFacetBasis',
        'Selected interior-facet topologies and trace operations.',
    ),
    CompatibilityEntry(
        'skfem.MeshTri',
        'mesh',
        CapabilityStatus.SUPPORTED,
        'skfemntv.MeshTri',
        'Core construction and selection, restriction/removal, uniform and conforming adaptive refinement for linear triangles.',
    ),
    CompatibilityEntry(
        'skfem.MeshTri2',
        'mesh',
        CapabilityStatus.SUPPORTED,
        'skfemntv.MeshTri2',
        'Quadratic triangle geometry for selected P2 assembly paths.',
    ),
    CompatibilityEntry(
        'skfem.MeshQuad',
        'mesh',
        CapabilityStatus.SUPPORTED,
        'skfemntv.MeshQuad',
        'Core construction/selection, restriction/removal, and uniform refinement for linear quadrilateral meshes.',
    ),
    CompatibilityEntry(
        'skfem.MeshQuad2',
        'mesh',
        CapabilityStatus.SUPPORTED,
        'skfemntv.MeshQuad2',
        'Quadratic quadrilateral geometry for selected Q2 assembly paths.',
    ),
    CompatibilityEntry(
        'skfem.MeshTet',
        'mesh',
        CapabilityStatus.SUPPORTED,
        'skfemntv.MeshTet',
        'Core construction, selection, restriction/removal, and refinement APIs for linear tetrahedra.',
    ),
    CompatibilityEntry(
        'skfem.MeshTet2',
        'mesh',
        CapabilityStatus.SUPPORTED,
        'skfemntv.MeshTet2',
        'Quadratic tetrahedron geometry for selected P2 assembly paths.',
    ),
    CompatibilityEntry(
        'skfem.MeshHex',
        'mesh',
        CapabilityStatus.SUPPORTED,
        'skfemntv.MeshHex',
        'Core construction and selection APIs; refinement and file I/O are absent.',
    ),
    CompatibilityEntry(
        'skfem.MeshHex2',
        'mesh',
        CapabilityStatus.SUPPORTED,
        'skfemntv.MeshHex2',
        'Quadratic hexahedron geometry for selected Q2 assembly paths.',
    ),
    CompatibilityEntry(
        'skfem.MeshWedge1',
        'mesh',
        CapabilityStatus.SUPPORTED,
        'skfemntv.MeshWedge1',
        'Linear wedge mesh and selected H1 assembly.',
    ),
    CompatibilityEntry(
        'skfem.MeshLine',
        'mesh',
        CapabilityStatus.PLANNED,
        None,
        'No line mesh or 1D assembly family.',
    ),
    CompatibilityEntry(
        'skfem.Mesh.refined',
        'mesh',
        CapabilityStatus.EXPERIMENTAL,
        'skfemntv.MeshQuad.refined',
        'Uniform refinement is available for linear MeshTri, MeshQuad, and MeshTet; conforming adaptive refinement is available for MeshTri and MeshTet.',
    ),
    CompatibilityEntry(
        'skfem.Mesh.load',
        'mesh_io',
        CapabilityStatus.EXPERIMENTAL,
        'skfemntv.MeshTri.load',
        'Optional meshio import for supported linear and selected quadratic cell families.',
    ),
    CompatibilityEntry(
        'skfem.Mesh.save',
        'mesh_io',
        CapabilityStatus.EXPERIMENTAL,
        'skfemntv.MeshTri.save',
        'Optional meshio export with point/cell data and native boundary/subdomain tags.',
    ),
    CompatibilityEntry(
        'skfem.Mesh.remove_elements',
        'mesh',
        CapabilityStatus.EXPERIMENTAL,
        'skfemntv.MeshQuad.remove_elements',
        'Removes selected cells and compacts unused vertices for linear MeshTri, MeshQuad, and MeshTet.',
    ),
    CompatibilityEntry(
        'skfem.Mesh.restrict',
        'mesh',
        CapabilityStatus.EXPERIMENTAL,
        'skfemntv.MeshQuad.restrict',
        'Extracts selected cells, remaps vertex IDs, and carries selected boundary/subdomain labels for linear MeshTri, MeshQuad, and MeshTet.',
    ),
    CompatibilityEntry(
        'skfem.Mesh.element_finder',
        'mesh',
        CapabilityStatus.PLANNED,
        None,
        'Spatial point location is not provided.',
    ),
    CompatibilityEntry(
        'skfem.ElementTriP0',
        'element',
        CapabilityStatus.SUPPORTED,
        'skfemntv.ElementTriP0',
        'Scalar cell-local P0.',
    ),
    CompatibilityEntry(
        'skfem.ElementTriP1',
        'element',
        CapabilityStatus.SUPPORTED,
        'skfemntv.ElementTriP1',
        'Scalar/vector H1 P1.',
    ),
    CompatibilityEntry(
        'skfem.ElementTriP2',
        'element',
        CapabilityStatus.SUPPORTED,
        'skfemntv.ElementTriP2',
        'Scalar/vector H1 P2.',
    ),
    CompatibilityEntry(
        'skfem.ElementTriP4',
        'element',
        CapabilityStatus.PLANNED,
        None,
        'Higher-order simplex family is not implemented.',
    ),
    CompatibilityEntry(
        'skfem.ElementTriP3',
        'element',
        CapabilityStatus.PLANNED,
        None,
        'Higher-order simplex family is not implemented.',
    ),
    CompatibilityEntry(
        'skfem.ElementQuad1',
        'element',
        CapabilityStatus.SUPPORTED,
        'skfemntv.ElementQuad1',
        'Scalar/vector H1 Q1.',
    ),
    CompatibilityEntry(
        'skfem.ElementQuad2',
        'element',
        CapabilityStatus.SUPPORTED,
        'skfemntv.ElementQuad2',
        'Scalar/vector H1 Q2.',
    ),
    CompatibilityEntry(
        'skfem.ElementQuadS2',
        'element',
        CapabilityStatus.SUPPORTED,
        'skfemntv.ElementQuad8',
        'Serendipity Q8 equivalent is available under the explicit ElementQuad8 name.',
    ),
    CompatibilityEntry(
        'skfem.ElementTetP1',
        'element',
        CapabilityStatus.SUPPORTED,
        'skfemntv.ElementTetP1',
        'Scalar/vector H1 P1.',
    ),
    CompatibilityEntry(
        'skfem.ElementTetP2',
        'element',
        CapabilityStatus.SUPPORTED,
        'skfemntv.ElementTetP2',
        'Scalar/vector H1 P2.',
    ),
    CompatibilityEntry(
        'skfem.ElementHex1',
        'element',
        CapabilityStatus.SUPPORTED,
        'skfemntv.ElementHex1',
        'Scalar/vector H1 Q1.',
    ),
    CompatibilityEntry(
        'skfem.ElementHex2',
        'element',
        CapabilityStatus.SUPPORTED,
        'skfemntv.ElementHex2',
        'Scalar/vector H1 Q2.',
    ),
    CompatibilityEntry(
        'skfem.ElementHexS2',
        'element',
        CapabilityStatus.SUPPORTED,
        'skfemntv.ElementHex20',
        'Serendipity Hex20 equivalent under the explicit ElementHex20 name.',
    ),
    CompatibilityEntry(
        'skfem.ElementDG',
        'space',
        CapabilityStatus.EXPERIMENTAL,
        'skfemntv.ElementDG',
        'Scalar cell-local DG elements; composite DG combinations are limited.',
    ),
    CompatibilityEntry(
        'skfem.ElementHexRT1',
        'space',
        CapabilityStatus.PLANNED,
        None,
        'Compatible H(div) families are not implemented.',
    ),
    CompatibilityEntry(
        'skfem.ElementVector',
        'space',
        CapabilityStatus.SUPPORTED,
        'skfemntv.ElementVector',
        'Vectorized supported nodal H1 elements.',
    ),
    CompatibilityEntry(
        'skfem.ElementComposite',
        'space',
        CapabilityStatus.SUPPORTED,
        'skfemntv.ElementComposite',
        'Selected mixed H1/composite assembly paths.',
    ),
    CompatibilityEntry(
        'skfem.ElementTriN1',
        'space',
        CapabilityStatus.EXPERIMENTAL,
        'skfemntv.AffineTriN1Basis',
        'Dedicated affine N1 API; not accepted by general Basis/asm.',
    ),
    CompatibilityEntry(
        'skfem.ElementTetN1',
        'space',
        CapabilityStatus.EXPERIMENTAL,
        'skfemntv.AffineTetN1Basis',
        'Dedicated affine N1 API; not accepted by general Basis/asm.',
    ),
    CompatibilityEntry(
        'skfem.ElementTriRT0',
        'space',
        CapabilityStatus.PLANNED,
        None,
        'H(div) spaces and contravariant Piola mapping are not implemented.',
    ),
    CompatibilityEntry(
        'skfem.helpers.grad',
        'helper',
        CapabilityStatus.SUPPORTED,
        'skfemntv.helpers.grad',
        'Typed H1 gradient operation.',
    ),
    CompatibilityEntry(
        'skfem.helpers.div',
        'helper',
        CapabilityStatus.SUPPORTED,
        'skfemntv.helpers.div',
        'Typed nodal-vector divergence.',
    ),
    CompatibilityEntry(
        'skfem.helpers.dot',
        'helper',
        CapabilityStatus.SUPPORTED,
        'skfemntv.helpers.dot',
        'Supported typed dot contractions.',
    ),
    CompatibilityEntry(
        'skfem.helpers.ddot',
        'helper',
        CapabilityStatus.SUPPORTED,
        'skfemntv.helpers.ddot',
        'Supported typed double contractions.',
    ),
    CompatibilityEntry(
        'skfem.helpers.mul',
        'helper',
        CapabilityStatus.SUPPORTED,
        'skfemntv.helpers.mul',
        'Typed coefficient/tensor multiplication for selected contractions.',
    ),
    CompatibilityEntry(
        'skfem.helpers.sym_grad',
        'helper',
        CapabilityStatus.SUPPORTED,
        'skfemntv.helpers.sym_grad',
        'Symmetric gradient for supported H1 vector fields.',
    ),
    CompatibilityEntry(
        'skfem.helpers.transpose',
        'helper',
        CapabilityStatus.SUPPORTED,
        'skfemntv.helpers.transpose',
        'Supported typed tensor transpose.',
    ),
    CompatibilityEntry(
        'skfem.helpers.trace',
        'helper',
        CapabilityStatus.SUPPORTED,
        'skfemntv.helpers.trace',
        'Supported tensor trace.',
    ),
    CompatibilityEntry(
        'skfem.helpers.curl',
        'helper',
        CapabilityStatus.EXPERIMENTAL,
        'skfemntv.helpers.curl',
        'Discrete scalar 2D and vector 2D/3D fields, plus vector H1 curl-curl bilinear forms; H(curl) DOFs are not supported.',
    ),
    CompatibilityEntry(
        'skfem.helpers.det',
        'helper',
        CapabilityStatus.EXPERIMENTAL,
        'skfemntv.helpers.det',
        'Numerical 2x2/3x3 determinant for arrays with component axes first; nonlinear symbolic form use is unsupported.',
    ),
    CompatibilityEntry(
        'skfem.helpers.inv',
        'helper',
        CapabilityStatus.EXPERIMENTAL,
        'skfemntv.helpers.inv',
        'Numerical 2x2/3x3 inverse for arrays with component axes first; nonlinear symbolic form use is unsupported.',
    ),
    CompatibilityEntry(
        'skfem.helpers.eye',
        'helper',
        CapabilityStatus.EXPERIMENTAL,
        'skfemntv.helpers.eye',
        'Diagonal matrix construction for scalar or quadrature-array weights; symbolic form use is unsupported.',
    ),
    CompatibilityEntry(
        'skfem.helpers.cross',
        'helper',
        CapabilityStatus.EXPERIMENTAL,
        'skfemntv.helpers.cross',
        'Numerical 2D/3D cross product for component-first arrays; symbolic bilinear form contraction is unsupported.',
    ),
    CompatibilityEntry(
        'skfem.Basis.interpolate',
        'evaluation',
        CapabilityStatus.SUPPORTED,
        'skfemntv.Basis.interpolate',
        'Evaluates at the basis quadrature points.',
    ),
    CompatibilityEntry(
        'skfem.Basis.get_dofs',
        'dof',
        CapabilityStatus.SUPPORTED,
        'skfemntv.Basis.get_dofs',
        'Boundary, region, component, and field selection on supported spaces.',
    ),
    CompatibilityEntry(
        'skfem.Basis.with_elements',
        'basis',
        CapabilityStatus.SUPPORTED,
        'skfemntv.Basis.with_elements',
        'Restricts a basis to selected global cell IDs.',
    ),
    CompatibilityEntry(
        'skfem.Basis.probes',
        'evaluation',
        CapabilityStatus.EXPERIMENTAL,
        'skfemntv.Basis.probes',
        'Sparse point-evaluation matrices for affine simplex P0/P1/P2 bases, including vector fields.',
    ),
    CompatibilityEntry(
        'skfem.Basis.interpolator',
        'evaluation',
        CapabilityStatus.EXPERIMENTAL,
        'skfemntv.Basis.interpolator',
        'Callable point evaluation for affine simplex P0/P1/P2 bases, including vector fields.',
    ),
    CompatibilityEntry(
        'skfem.Basis.project',
        'evaluation',
        CapabilityStatus.EXPERIMENTAL,
        'skfemntv.Basis.project',
        'L2 projection on affine MeshTri/MeshTet scalar or vector P0/P1/P2 bases, including DOF restriction by element selection; composite fields and curved geometry are unsupported.',
    ),
    CompatibilityEntry(
        'skfem.Basis.refinterp',
        'evaluation',
        CapabilityStatus.PLANNED,
        None,
        'Reference-element visualization interpolation is not provided.',
    ),
    CompatibilityEntry(
        'skfem.condense',
        'solver',
        CapabilityStatus.SUPPORTED,
        'skfemntv.utils.condense',
        'Sparse linear and generalized eigenvalue systems; ndarray, DofsView, and dictionary DOF selections.',
    ),
    CompatibilityEntry(
        'skfem.enforce',
        'solver',
        CapabilityStatus.SUPPORTED,
        'skfemntv.utils.enforce',
        'Sparse matrices and vector or sparse right-hand sides.',
    ),
    CompatibilityEntry(
        'skfem.penalize',
        'solver',
        CapabilityStatus.SUPPORTED,
        'skfemntv.utils.penalize',
        'Sparse matrices and vector or sparse right-hand sides.',
    ),
    CompatibilityEntry(
        'skfem.solve',
        'solver',
        CapabilityStatus.SUPPORTED,
        'skfemntv.utils.solve',
        'Sparse linear systems and generalized eigenvalue problems; custom solvers are accepted.',
    ),
    CompatibilityEntry(
        'skfem.visuals.matplotlib',
        'visualization',
        CapabilityStatus.EXTERNAL,
        None,
        'Use a visualization package such as Matplotlib or PyVista.',
    ),
)

COMPATIBILITY_MANIFEST = {entry.upstream_name: entry for entry in _ENTRIES}


def compatibility_items(
    *, category: str | None = None, status: CapabilityStatus | str | None = None
) -> tuple[CompatibilityEntry, ...]:
    """Return the upstream compatibility inventory in stable name order."""
    if status is not None:
        status = CapabilityStatus(status)
    return tuple(
        entry
        for entry in sorted(_ENTRIES, key=lambda item: item.upstream_name)
        if (category is None or entry.category == category)
        and (status is None or entry.status is status)
    )


def get_compatibility(upstream_name: str) -> CompatibilityEntry:
    """Return the declared disposition for one upstream API name."""
    try:
        return COMPATIBILITY_MANIFEST[upstream_name]
    except KeyError as exc:
        raise KeyError(f"unknown scikit-fem API name {upstream_name!r}") from exc


__all__ = [
    "COMPATIBILITY_MANIFEST",
    "UPSTREAM_REFERENCE_VERSION",
    "CompatibilityEntry",
    "compatibility_items",
    "get_compatibility",
]
