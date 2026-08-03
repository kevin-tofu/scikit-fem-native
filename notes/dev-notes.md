# skfem-native development notes

## Objective

Move reusable finite-element kernels out of kktkit while keeping dataset,
workflow, solver-policy, ROM, and reporting concerns in kktkit.  skfemntv
should return solver-ready sparse blocks with explicit mesh/DOF provenance and
enough diagnostics to reject invalid interfaces before factorization.

## Priority 1: global Mortar rank reduction

Facet-local QR is insufficient for overlap-P0 because dependencies can span
parent facets.  A representative nonmatching vector interface produces 24 raw
rows with numerical rank 21.  kktkit currently performs the missing global QR
after mapping the native matrix to global DOFs.

Required API and guarantees:

- `reduction="none" | "global_qr" | "global_svd"` on Mortar assembly.
- Report raw, supported, and independent row counts, numerical rank, tolerance,
  and reduction backend.
- Preserve row-to-entity/component provenance for selected QR rows.
- Do not mix vector components or unilateral contact groups accidentally.
- Prefer a sparse rank-revealing implementation; a guarded dense implementation
  is acceptable only as an initial correctness reference.
- Test constant and affine patch reproduction before and after reduction.

Initial implementation on `dev`:

- `reduction="global_qr"` is available as a correctness-reference backend.
- It reports immutable `MortarReductionDiagnostics` and preserves selected
  row entity/component metadata.
- Dense allocation is guarded by `dense_reduction_max_rows`; exceeding it is
  an explicit error until a sparse rank-revealing backend replaces it.
- The matching 2x2 vector overlap-P0 regression reduces 24 raw rows to rank 21
  while preserving the affine patch field.
- Backend selection now prefers `sksparse.spqr`, then `sparseqr`, and uses the
  guarded SciPy dense reference only when neither optional sparse backend is
  available.  Diagnostics record the selected backend, fallback reason, and
  reduction time.  Sparse-backend tests verify that the dense path is not
  entered when the dense row guard is exceeded.

## Priority 2: compact trace DOFs

Nodal multiplier spaces currently use the component-wide nodal space and
describe exact zero rows through `supported_rows`.  Add an option returning a
compact matrix directly, together with immutable maps to original master,
slave, and multiplier DOFs.  Solver-ready results must guarantee that no
unsupported or exact-zero constraint rows remain.

## Priority 3: typed facet identity and diagnostics

The canonical identity inside skfemntv is `facet_indices_skfem`.  Never expose
an unqualified integer array as though Gmsh element IDs, Gmsh physical groups,
surface-local triangles, parent volume elements, and overlap cells were
interchangeable.  Metadata should name the ID space explicitly.

Native diagnostics should include source facet counts and areas, overlap area
and ratio, discarded duplicate/layer counts, partition-of-unity and affine
patch errors, zero/raw/supported/independent row counts, numerical rank,
normal-opposition error, and search/quadrature/assembly/reduction timings.

## Priority 4: element and geometry coverage

- Tet10 and Hex20 traces.
- Triangle, quadrilateral, and mixed interfaces.
- Curved isoparametric surfaces with controlled subdivision error.
- Scalar, 2-D vector, and 3-D vector fields without assumed DOF ordering.
- Reusable search topology, overlap pattern, and CSR sparsity across load cases,
  snapshots, and nonlinear geometry updates.

## Priority 5: contact kernels

After tied Mortar is stable, add gap/normal projection, frictionless active-row
residual and Jacobian, tangential grouping, and Coulomb history kernels.
kktkit should continue to own load stepping and active-set policy; skfemntv
should own numerical contact evaluation and its derivatives.

## Validation policy

Every native replacement needs two levels of regression:

1. skfem/scikit-fem reference versus skfemntv for matrices, row spaces,
   overlap measures, patch fields, and rank.
2. kktkit public API with `assembly_backend="skfem"` and `"skfemntv"` for
   constraint row space, RBM removal, displacement, compliance, and residual.

Performance reports must separate cold import, search, Mortar assembly,
reduction, full kktkit assembly, factorization, and solve time.  A speedup claim
must use repeated warm measurements and retain numerical-equivalence metrics.
