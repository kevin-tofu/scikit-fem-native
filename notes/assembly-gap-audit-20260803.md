# Assembly capability gap audit — 2026-08-03

## Scope

This audit asks whether a user can perform the first operations normally needed
in an FEM assembly project.  The target is a practical native assembly engine,
not complete scikit-fem compatibility and not a solver framework.

The audit is based on the public `skfemntv` API, explicit unsupported paths,
the test suite, and `notes/todo-20260801.md`.  scikit-fem remains the primary
unit-level numerical reference.  A CalculiX comparison is deferred because it
would duplicate tests that can currently be made more precisely at matrix and
vector level with scikit-fem.

## What is already usable

| User operation | Current state | Evidence |
|---|---|---|
| Construct common 2D/3D H1 spaces | Available | Tri, Quad, Tet, Hex, Wedge, and Pyramid; scalar/vector P0/P1/P2 or Q1/Q2 where documented |
| Assemble volume bilinear/linear forms | Available | `BilinearForm`, `LinearForm`, `asm`; comparison tests for every principal topology |
| Integrate a scalar result | Available | `Functional`, including coefficient fields and restricted bases |
| Select a subset of cells | Available | `Basis(..., elements=...)`, `with_elements`, callable/mask/ID selection |
| Select geometric boundaries | Available, basic | `facets_satisfying`, `with_boundaries`, `get_dofs`, named boundaries |
| Assemble boundary loads | Available | `FacetBasis`, physical normals, high-order geometry, mixed triangle/quad exterior faces |
| Assemble interior-facet terms | Available for the documented topologies | `InteriorFacetBasis`, two sides, `jump`, `avg`, `normal_grad` |
| Use spatially varying data | Available for common contractions | `w.x`, scalars, arrays, callables, `DiscreteField` |
| Use mixed and rectangular spaces | Available | `ElementComposite`, split indices/bases, cross-basis assembly |
| Use discontinuous fields | Available with limits | `ElementDG`, P0, element-local DOFs; no composite containing DG |
| Reuse nonlinear sparsity and state | Available | `NativeAssembler`, `MaterialAssembler`, residual/tangent modes, commit-by-returned-state |
| Assemble nonmatching surfaces | Available | supermesh, trace values/gradients/normals, mortar blocks, diagnostics |
| Control native parallelism | Available | process-visible CPU cap, global/context/per-call thread controls |

This is enough for Poisson, linear elasticity, Stokes-type mixed assembly,
standard DG/interface terms on supported meshes, hyperelasticity, and selected
history-dependent materials.  Solving, condensation, and nonlinear iteration
policy intentionally remain with SciPy, PETSc, or the application.

## Priority gaps

### P0 — Reject invalid geometry before assembly

Geometry validation is now implemented in the common native volume-tabulation
path.  A uniformly negative local orientation is valid and is reported in
diagnostics.  Scale-aware near-singular points and determinant sign changes
within a curved element are rejected before assembly.

Required behavior:

- [x] reject scale-aware near-singular cells;
- [x] reject internal orientation changes;
- [x] report cell ID, quadrature-point ID, determinant, and threshold;
- [x] expose determinant extrema, scaled determinant, worst cell, and
  orientation count as diagnostics;
- [x] test valid, reversed, badly scaled, and internally inverted Tet10
  geometry;
- [x] add an internally inverted Hex27 case and condition diagnostics;
- [ ] add further near-collapse cases;
- [x] perform the check during basis construction, before assembly.

This precedes CutFEM work: cut-cell quadrature magnifies geometry-conditioning
problems and needs trustworthy diagnostics.

### P0 — Promote selection from predicates to explicit regions

The first region layer is implemented.  Geometric predicates now return
immutable region values; named cell subdomains and named boundary regions can
be passed directly to Basis, FacetBasis, and DOF selection.  Region algebra
provides union, intersection, difference, and complement without renumbering
global entities.  `facets_satisfying(..., normal=...)` and
`get_dofs(skip=...)` remain the next gaps.

The public region results are:

```python
CellRegion(ids, entity_count=None)
FacetRegion(ids, entity_count=None)
NodeRegion(ids, entity_count=None)
```

Region objects should remain accepted anywhere an ID array is accepted.  They
must preserve global entity IDs and be cheap views; they must not copy meshes or
renumber global DOFs.  First implement:

1. [x] named cell subdomains (`mesh.with_subdomains`);
2. [x] union, intersection, difference, and complement;
3. [x] deterministic selection and empty-region diagnostics;
4. [ ] normal-oriented exterior-facet selection;
5. [ ] component-aware DOF selection for vector/composite spaces;
6. [ ] classification/orientation metadata for level-set results.

These operations are useful in ordinary multi-material and load-selection
workflows even if CutFEM is never used.

### P1 — Arbitrary-point field evaluation

`Basis.interpolate` evaluates at the basis integration points.  Research and
production workflows also need values and physical gradients at user points
for probes, transfer, error estimation, inverse problems, and coupling.

A focused API should provide:

```python
evaluation = basis.evaluate_at(points, coefficients, outside="error")
```

The result should contain values, physical gradients, containing cell IDs,
reference coordinates, and diagnostics for points outside or on ambiguous
boundaries.  It requires a spatial broad phase and robust inverse mapping but
does not require adding plotting or file I/O to the core.

### P1 — Complete common form algebra deliberately

The tracer handles the documented linear/bilinear contractions, but some
ordinary Python expressions remain unsupported: coefficient indexing, multiple
coefficient combinations in selected lowering paths, and operations outside
the native term vocabulary.  Unsupported expressions already fail instead of
falling back, which is the correct policy.

Next work should be driven by a machine-readable compatibility table.  Add one
success and one expected-failure test per operation.  Prioritize operations
needed by concrete weak forms:

- scalar/vector coefficient component access;
- outer product and transpose needed by anisotropic diffusion and mechanics;
- explicit symmetric/skew tensor construction;
- coefficient-dependent facet and interior-facet contractions;
- multiple independently named material/coefficient fields in one form.

Do not implement arbitrary NumPy tracing.  Extend a small typed intermediate
form representation so native dispatch and diagnostics remain predictable.

### P1 — Reentrancy and memory preflight

Repeated native assembly is useful today, but the contract for concurrent calls
on one assembler instance is not explicit.  Large high-order problems also need
a memory estimate before CSR scatter maps and geometry tables are allocated.
Both items are already identified in `todo-20260801.md` and remain production
gates.

### P2 — Useful but not immediate assembly gaps

- composite spaces containing DG fields;
- mixed-face Wedge/Pyramid interior-facet coverage where still unsupported;
- Line P1/P2 meshes and elements;
- custom mapping/DOF objects for facet bases;
- mesh refinement, import/export, visualization, and solver utilities.

The last group should usually remain external unless an assembly feature needs
it.  H(div) and H(curl) spaces are substantial future tracks, not small API
compatibility patches.

## CutFEM and level-set direction

### Separate selection from integration

A level-set boundary selector is not just a more complicated
`facets_satisfying` predicate.  Given nodal or callable level set `phi`, the
system must distinguish:

- cells wholly inside (`phi < 0`);
- cells wholly outside (`phi > 0`);
- cut cells;
- background facets in the active mesh;
- implicit interface pieces inside cut cells.

Selection returns topology and classification.  A separate cut-quadrature
stage creates integration points.  Keeping these stages separate allows the
same classification to drive volume forms, interface forms, ghost penalties,
active DOFs, and diagnostics.

Proposed public concepts:

```python
levelset = LevelSet.from_callable(mesh, phi)
cut = classify_level_set(mesh, levelset, tolerance=...)

inside = CutCellBasis(basis, cut, side="inside", intorder=...)
interface = ImplicitFacetBasis(basis, cut, intorder=...)
active = basis.get_dofs(elements=cut.active_cells)
```

Names are provisional; the important part is ownership and data flow.

### Required cut integration data

Native assembly should receive quadrature-local arrays, not construct dense
global objects:

- background cell ID and optional subcell ID;
- physical and reference quadrature coordinates;
- positive quadrature weight;
- shape values and physical gradients;
- level-set value and physical gradient when requested;
- oriented interface normal;
- volume/area fractions and conditioning diagnostics.

The existing `Basis` assumes one rectangular quadrature rule shared across
selected cells.  Cut cells require cell-local rules with different point
counts.  A practical native representation is CSR-like:

```text
cell_offsets[ncells + 1]
points[ncut_qp, dim]
weights[ncut_qp]
```

Assembly loops then process `cell_offsets[e]:cell_offsets[e + 1]`.  Padding
every cell to the worst cut-cell point count would waste memory and weaken the
large-scale design.

### Orientation and topology changes

Define the implicit normal as `grad(phi) / |grad(phi)|`, pointing from negative
to positive level-set values.  Report cells where the gradient is too small to
orient the interface.  When the level set moves, distinguish:

- value-only update with unchanged cut topology;
- local quadrature rebuild;
- active-set/CSR-pattern rebuild.

Never silently reuse a sparsity pattern after the active DOF set changes.

### Validation ladder

1. Exact line/plane cuts of Tri, Quad, Tet, and Hex reference cells.
2. Constant integration equals cut volume/area.
3. Linear integration and normal orientation are exact to tolerance.
4. Circle/sphere volume and interface-measure convergence under refinement.
5. Serial/parallel equality and deterministic classification near tolerance.
6. Unfitted Poisson manufactured solution with Nitsche boundary terms.
7. Ghost-penalty conditioning across cuts approaching a cell vertex.
8. Moving level set with topology-change and cache-invalidation checks.

scikit-fem comparison should be used wherever an equivalent fitted or custom
quadrature calculation can be expressed.  Analytic geometry and manufactured
solutions cover the genuinely CutFEM-specific parts.

## Recommended execution order

1. Implement normal-oriented facet selection and component-aware DOF queries.
2. Add classification metadata needed by level-set regions.
3. Add arbitrary-point value/gradient evaluation.
4. Close form-algebra gaps required by anisotropic and multi-coefficient forms.
5. Introduce level-set classification without cut integration.
6. Add CSR-like cell-local cut quadrature and constant/linear exactness tests.
7. Build `CutCellBasis` and `ImplicitFacetBasis`, then Nitsche/ghost-penalty
   examples as user-defined forms.

The common Jacobian validation path is implemented.  After its full regression
suite passes, the next feature track is item 1: first-class regions.
