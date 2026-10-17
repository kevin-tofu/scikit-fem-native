# scikit-fem parity inventory — 2026-10-15

## Scope and baseline

The compatibility target is now explicit: `skfemntv` should align with the
public `scikit-fem` API and preserve results for supported workflows. This is
an inventory and prioritization, not a promise to reproduce every specialized
element immediately.

The local reference installation is scikit-fem 12.0.2. The comparison uses its
top-level API, helpers, mesh/basis methods, and getting-started workflow. The
upstream API is documented at <https://scikit-fem.readthedocs.io/en/stable/api.html>.

Three levels of parity are useful:

1. **Import/API parity:** familiar names, arguments, return shapes, and DOF
   selection semantics.
2. **Workflow parity:** users can build common spaces, assemble forms, apply
   boundary data, solve, evaluate, and export results with equivalent calls.
3. **Numerical parity:** equivalent supported workflows match upstream
   matrices, vectors, fields, and convergence behavior.

The current project has useful numerical parity slices, but does not yet offer
workflow parity across the normal scikit-fem lifecycle.

## Inventory

| Area | Current `skfemntv` state | Parity gap |
|---|---|---|
| Form assembly | Functional, linear, bilinear, cross-bilinear; selected volume, boundary, interior-facet, composite, cut, and interface assembly | `det`, `inv`, `eye`, and `cross` now match upstream on numeric component-first arrays. Their nonlinear symbolic form use, `prod`, and higher derivatives remain unsupported in general `asm`. |
| Mesh families | Tri, Quad, Tet, Hex, Wedge, Pyramid; linear and selected quadratic/serendipity variants | No 1D Line mesh family; substantially fewer element types and constructors than upstream. |
| Mesh lifecycle | Tensor initialization, boundary/subdomain labels, geometric regions, topology queries, experimental `MeshTri` / `MeshTet` uniform and adaptive refinement, `MeshQuad` uniform refinement, restriction/removal for all three linear families, and optional meshio `load` / `save` | Adaptive element layout may differ from upstream. Other mesh-family refinement/editing and visualization methods remain unavailable in the general public mesh API; I/O supports selected cell types. |
| Basis and DOFs | H1 nodal spaces, vector/composite spaces, selected DG, facet bases, component-aware DOF selection, geometry diagnostics, and experimental affine-simplex point evaluation/probes/interpolators/L2 projection | Curved geometry, tensor-product cells, composite-field sampling, `refinterp`, and multi-field interpolation remain unavailable. |
| Element spaces | H1 families include P0/P1/P2 and selected Q1/Q2, Quad8, Hex20, Wedge/Pyramid; composite and DG support | No general H(div). H(curl) is isolated experimental N1 triangle/tetrahedron API rather than `ElementTriN1`/`ElementTetN1` accepted by `Basis` and `asm`. Many upstream high-order, bubble, plate, RT/BDM, Hermite, and skeleton elements are absent. |
| Solver utilities | Top-level `solve`, `condense`, `enforce`, and `penalize` use SciPy sparse solvers and have parity tests for common sparse systems, nonzero boundary values, DOF views/dictionaries, callbacks, and eigenvalue expansion | Additional uncommon solver callbacks and sparse matrix formats may need parity cases; projection onto curved, composite, or partial bases remains unavailable. |
| Results and visualization | Quadrature evaluation and dedicated examples; rendering is external | No general plotting, mesh/result export, arbitrary-point sampling, or documented post-processing API. |
| Compatibility evidence | Gallery comparisons for Poisson 2D/3D and transient heat; many direct form/basis equivalence tests | No single versioned matrix states which upstream constructors, methods, elements, and helpers pass, fail explicitly, or are intentionally external. |

## Assessment

The most consequential gap is not a missing assembly kernel: it is the missing
workflow surface around assembly. A typical upstream program can use mesh
construction/refinement, `Basis`, DOF selection, assembly, boundary
elimination, solving, interpolation/projection, and visualization/export. A
native backend swap currently reaches only a selected part of that path.

The second major gap is that the original compatibility contract was too
coarse. The runtime capability registry describes mathematical capabilities,
including the implemented mesh lifecycle slices, but does not enumerate every
mesh/element constructor, helper operation, or basis method.
`skfemntv.compatibility_items()` and `get_compatibility()` expose
the first versioned inventory of common API landmarks, checked against the
reference installation. It is not yet exhaustive over every upstream element
and method.

Full parity with every scikit-fem element is a large target. The useful first
target is the common public workflow, while marking specialized spaces as
explicitly unsupported until they are integrated and validated.

## Recommended order

### P0 — Define and continuously verify the compatibility matrix

- Pin the baseline to a supported scikit-fem release range.
- Enumerate public mesh constructors, common element families, basis methods,
  helper operations, and solver utilities.
- For each entry record `supported`, `experimental`, `planned`, or `external`,
  plus a direct parity test or an expected explicit error.
- Include both API/shape checks and numerical comparisons; use full gallery
  workflows where possible.
- Keep the runtime registry and user documentation generated from or checked
  against this matrix.

### P1 — Close common workflow gaps

1. **Partially implemented:** `Basis.evaluate_at`, sparse `Basis.probes`, and
   callable `Basis.interpolator` cover affine simplex P0/P1/P2, including
   vector fields for probes/interpolation. Extend to curved and tensor-product
   cells, composite fields, and projection workflows.
2. **Implemented:** match scikit-fem's `condense`, `enforce`, `penalize`, and
   `solve` public call shapes using SciPy sparse solvers. Pytest compares
   boundary handling, DOF views/dictionaries, callbacks, and generalized
   eigenproblem expansion against the reference package.
3. **Partially implemented:** linear `MeshTri.refined` supports uniform and
   conforming adaptive refinement and preserves named boundaries/subdomains.
   `MeshTri.restrict` and `remove_elements` also compact vertices and remap
   boundary/subdomain tags. Optional meshio adapters now load/save selected
   linear and quadratic cell types with point/cell data and mesh tags. Linear
   `MeshTet.refined` supports uniform and conforming adaptive refinement and
   carries named boundaries/subdomains forward. `MeshTet.restrict` and
   `remove_elements` now compact vertices and remap named regions like the
   triangle equivalents. `MeshQuad.refined` supports uniform refinement with
   boundary/subdomain preservation. `MeshQuad.restrict` and
   `remove_elements` now compact vertices and remap named regions as well.
4. **Implemented experimentally:** `Basis.project` computes L2 projections
   for scalar/vector P0/P1/P2 on full affine triangle/tetrahedron bases and
   matches scikit-fem in parity tests. Subset, composite, and curved-basis
   projection remain open.
5. **Implemented experimentally:** numeric `det`, `inv`, `eye`, and `cross`
   match scikit-fem on component-first arrays. Supporting them in symbolic
   nonlinear form expressions requires extending the typed expression model.
6. Extend the typed form vocabulary from concrete upstream weak forms, starting
   with the operations needed for mechanics and multiphysics. Every operation
   needs a parity test and a precise unsupported-path test.
7. Add a supported line-mesh family and common P3/P4 triangle elements if the
   target includes the normal scikit-fem teaching/research workflows.

### P2 — Broaden spaces by coherent families

- Integrate the existing N1 H(curl) slices into the general basis/form
  lifecycle before adding more element orders.
- Add H(div) through oriented facet DOFs and contravariant Piola mapping.
- Expand DG/composite-DG and additional high-order/specialized element
  families based on upstream use cases and test demand.

### External by design, with parity recipes

Keep nonlinear constitutive laws, nonlinear iteration policy, time stepping,
preconditioning, meshing algorithms, and visualization libraries outside the
native assembly kernels. Provide adapters/examples so common scikit-fem
workflows can still be reproduced without implying that all these policies
belong in `skfemntv`.

## Existing documents to reconcile

`notes/assembly-gap-audit-20260803.md`,
`notes/20260812-priorities-and-compatible-elements.md`, and
`notes/dev-status-20260803.md` contain useful rationale, but some status lists
predate implemented regions, level-set classification, cut quadrature,
interface assembly, tensor coefficients, and memory preflight. Treat those
lists as historical until their checkboxes and recommended-next-work sections
are reconciled against this inventory.
