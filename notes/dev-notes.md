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

## Nonlinear assembly roadmap

The tied-Mortar work covers only one part of assembly.  Replacing kktkit's
nonlinear FEM paths requires a stable stateful assembly contract before adding
more materials or elements.

### Unified residual and consistent tangent

One element traversal should evaluate internal residual, consistent tangent,
strain/stress fields, energy, trial material state, and diagnostics from the
same displacement and integration-point state.  Residual and tangent must not
perform independent material updates.

A target interface is:

```python
result = assembler.assemble_nonlinear(
    displacement=u,
    committed_state=state,
    compute_tangent=True,
    dt=dt,
)
```

The result should contain `residual`, `tangent`, `trial_state`, field outputs,
energy, and immutable diagnostics.

### Trial, commit, and rollback

Plasticity, viscoelasticity, damage, and friction require explicit state
ownership:

```text
committed state -> trial update -> Newton convergence -> commit
                                             failure -> rollback/cutback
```

Integration-point ordering must remain stable across assembly calls and
geometry updates.  kktkit may own load-step and cutback policy, while skfemntv
owns committed/trial material-state representation and update kernels.

### Updated geometry and follower loads

Large-deformation assembly needs current coordinates, deformation gradients,
current normals, geometric stiffness, current-area integration, follower-load
residuals, and their external-force tangent.  Supermesh/contact updates should
reuse search topology and rebuild only changed overlap pairs where possible.

### Nonlinear contact tangent

Contact assembly must eventually provide gap residual, normal variation,
contact geometric tangent, active-row data, normal/tangential traction,
friction return mapping, and a consistent friction tangent.  Generic QR must
not destroy normal/tangent row grouping or active-set identity.

### Element, material, and formulation coverage

Target element coverage is Tet4/Tet10, Hex8/Hex20, wedge, and pyramid for
linear and geometric-nonlinear elasticity, hyperelasticity, plasticity, mass,
and damping.  The material interface should return stress, algorithmic tangent,
and trial state for a common deformation/state input.

Near-incompressible problems additionally require mixed displacement-pressure
or controlled alternatives such as selective integration, B-bar/F-bar, and
appropriate stabilization/hourglass treatment.

### Transient nonlinear assembly

Dynamic replacement requires consistent/lumped mass, damping,
Newmark/generalized-alpha residual and effective tangent, state-history reuse,
multiple RHS support, and factorization/preconditioner reuse.

### Nonlinear diagnostics

Report element and integration-point counts, residual/tangent/material-update
timings, state memory, inverted elements, minimum Jacobian determinant,
maximum strain/stress, plastic-point count, local return-map iterations,
non-finite element IDs, and tangent symmetry/consistency errors.

### Recommended nonlinear implementation order

1. Unified residual+tangent+trial-state result.
2. Explicit commit/rollback state API.
3. Neo-Hookean equivalence through the kktkit public API.
4. J2 plasticity equivalence over load steps and cutbacks.
5. Updated geometry, geometric stiffness, and follower loads.
6. Nonlinear Mortar/contact residual and consistent tangent.
7. Mixed near-incompressible formulations.

## Validation policy

Every native replacement needs two levels of regression:

1. skfem/scikit-fem reference versus skfemntv for matrices, row spaces,
   overlap measures, patch fields, and rank.
2. kktkit public API with `assembly_backend="skfem"` and `"skfemntv"` for
   constraint row space, RBM removal, displacement, compliance, and residual.

Performance reports must separate cold import, search, Mortar assembly,
reduction, full kktkit assembly, factorization, and solve time.  A speedup claim
must use repeated warm measurements and retain numerical-equivalence metrics.
