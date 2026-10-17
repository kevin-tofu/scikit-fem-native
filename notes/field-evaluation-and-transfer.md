# Field evaluation and transfer

This note summarizes the purpose, current scope, and intended use of the field
evaluation and transfer operations in `skfemntv`. These operations turn finite
element coefficient vectors into values that applications can sample, compare,
visualize, or transfer between representations.

## Key concepts

A **composite basis** combines multiple finite element fields into one global
unknown vector. A mixed displacement-pressure problem is a common example: the
displacement may use vector P2 while pressure uses scalar P1. Each field has
its own value shape and may have its own element order, while the composite
basis assigns all fields a shared global DOF numbering. Field-wise sampling
therefore returns a tuple in the same order as the fields in
`ElementComposite`.

A **curved mesh** represents curved cell boundaries with higher-order geometry
nodes and a nonlinear map from reference cells to physical cells. A P2 field
does not by itself imply a curved mesh: if geometry nodes stay at edge
midpoints, the geometry remains straight-sided even though the field has
quadratic DOFs. Curved geometry needs a nonlinear physical-to-reference point
search, which is a separate implementation task from adding P2 field values.

## Operations

| Operation | Purpose | Result |
|---|---|---|
| `Basis.interpolate(coefficients)` | Evaluate a coefficient vector at the basis quadrature points, for integration and form assembly | A `DiscreteField`; a tuple of fields for an `ElementComposite` basis |
| `Basis.evaluate_at(points, coefficients)` | Evaluate values and physical gradients at arbitrary physical points, with point-location diagnostics | `PointEvaluation`; a tuple of these for a composite basis |
| `Basis.probes(points)` | Build a sparse linear operator that evaluates the finite element field at points | A sparse matrix; for a composite basis, one matrix per field, all using the composite global coefficient vector |
| `Basis.interpolator(coefficients)` | Reuse point evaluation as a callable, including inputs with trailing point-grid dimensions | A callable; for a composite basis, it returns one array per field |
| `Basis.project(interpolant)` | Compute an L2 projection of a callable, constant, or quadrature field into the basis | A coefficient vector in the basis's global DOF order |

## When to use them

- Use `interpolate` when the integration points are already defined by a
  `Basis`, such as inside a form or a quadrature-based post-processing step.
- Use `evaluate_at` when physical point locations and diagnostics matter,
  including outside points and points shared by multiple cells.
- Use `probes` when the same point set will be evaluated repeatedly for many
  coefficient vectors; its sparse matrix can be reused.
- Use `interpolator` for a convenient callable interface, including regular
  grids used by plotting or visualization code.
- Use `project` to transfer a function or field into a finite element space,
  for example when initializing a new solve from an analytic field or a
  different discretization.

## Current scope

Point evaluation, probes, and interpolators support P0/P1/P2 fields on affine
triangles and tetrahedra and on quadratic isoparametric triangles. This includes
vector fields and composite continuous P1/P2 scalar/vector fields on affine
geometry. `evaluate_at` reports
cell IDs, reference coordinates, outside points, boundary points, and
ambiguous points on shared facets. `outside="nan"` returns NaN values for
outside points; the default raises an error.

`project` supports scalar/vector P0/P1/P2 fields on affine triangles and
tetrahedra, and continuous P1/P2 scalar/vector fields within an
`ElementComposite`. Its `elements=` selection constrains the solution DOFs to
those associated with the selected elements and zeros all other DOFs. Named
subdomains, boolean masks, element IDs, and predicates are accepted.

For composite bases, `evaluate_at` returns a tuple of `PointEvaluation`
objects, `probes` returns a tuple of sparse matrices, and `interpolator`
returns a callable whose result is a tuple. Each tuple position follows the
field order in `ElementComposite`; every probe matrix acts on the full global
composite coefficient vector. This field-wise interface accommodates mixed
scalar and vector fields without flattening different value shapes into one
ambiguous array. The current upstream scikit-fem release does not provide the
same composite point-sampling workflow, so these composite results are checked
against independently constructed per-field reference bases and the native
global DOF maps.

Curved geometry, composite P0/DG fields, composite projection of arbitrary
field containers beyond one interpolant per field, and point sampling outside
the basis's selected elements remain unsupported. These restrictions should
stay explicit until each path has direct numerical comparisons and a clear
result-shape contract.

## Next parity goals

1. Extend curved geometry support to tetrahedra, then add L2 projection on
   curved simplex geometry.
2. Composite sampling on tensor-product cells and a documented post-processing
   path for the result arrays.

Tests for the current work live in `tests/test_point_evaluation.py`,
`tests/test_basis_probes_compatibility.py`,
`tests/test_composite_point_evaluation.py`, and
`tests/test_basis_project_compatibility.py`.
