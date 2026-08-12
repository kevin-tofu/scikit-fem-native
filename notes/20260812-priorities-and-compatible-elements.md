# skfemntv priorities and compatible-element roadmap

Date: 2026-08-12

## Decision

The immediate priority is to make the existing H1 assembly subset safer at
production scale.  H(curl) support is valuable, but it must be built on a
general topological-entity DOF model rather than added as a special case to the
current nodal `Basis`.

The recommended order is:

1. memory preflight and large-problem diagnostics;
2. a machine-readable compatibility contract and early capability errors;
3. form-algebra coverage needed by anisotropic and coupled problems;
4. arbitrary-point value and gradient evaluation;
5. reentrancy and native-thread ownership contracts;
6. deeper numerical validation;
7. entity DOFs and first-order H(curl);
8. H(div), mixed/DG generalization, and higher-order compatible elements.

The compatibility contract is the first implementation task because it makes
the boundary explicit before memory and DOF APIs grow.

## Priority tracks

### P0: memory preflight

Before allocating CSR structure, scatter maps, basis geometry, or material
state, report estimated bytes by category.  Establish measurements at roughly
10k, 100k, and 1M DOFs.  An estimate must state its assumptions and must not be
presented as an allocation guarantee.

Initial implementation status: `estimate_bilinear_memory` now reports retained
basis bytes, native tabulation copies, DOF maps, CSR upper bounds, scatter maps,
coloring storage, and temporary pattern construction.  Standard, rectangular,
restricted, vector, and composite spaces are covered.  Cut-cell assemblers,
coefficient peak memory, allocator calibration, and measured 10k/100k/1M-DOF
reference tables remain follow-up work.

Calibration tooling is available as
`benchmarks/memory_preflight_calibration.py`.  It runs every family/component/
target combination in a fresh process, records estimated and measured memory,
actual CSR nonzeros, setup/assembly time, and writes a stable CSV schema.  Its
default matrix covers Tet4, Tet10, Hex8, and Hex20 with one and three components
at target sizes of 10k, 100k, and 1M DOFs.  An 8 GiB default budget skips native
assembler construction when the preflight estimate is too large.

Example:

```bash
python benchmarks/memory_preflight_calibration.py \
  --memory-budget-gib 16 \
  --output benchmarks/results/memory-preflight.csv
```

The RSS columns are process measurements, while the estimate intentionally
excludes interpreter/library baseline RSS, allocator fragmentation, and Python
object overhead.  Compare deltas and ratios rather than absolute process RSS.

The first 10k-DOF calibration is recorded in
`benchmarks/results/memory-preflight-10k.csv`.  All eight cases completed.  The
allocation-free CSR upper bound was 1.60--5.99 times the actual nnz depending
on topology, order, and component count.  Estimated construction-peak total
divided by measured process RSS delta was 1.01--1.37, so this sample supports
the estimate's intended safety-side behavior.  These are machine-specific
observations, not universal correction factors; larger runs are still needed
before selecting a default enforcement margin.

The 100k-DOF calibration is recorded in
`benchmarks/results/memory-preflight-100k.csv`.  All eight cases completed.
The CSR upper-bound ratio was 1.66--6.21 and estimated peak divided by measured
RSS delta was 1.06--1.40.  Together with the 10k sample, this supports keeping
the estimate allocation-free and conservative rather than fitting a
machine-specific correction factor.

`NativeBilinearForm` and `NativeCrossBilinearForm` now accept
`memory_limit_bytes` and `memory_safety_factor`.  The default safety factor is
1.25 and is applied on top of the estimated construction peak.  Budget failure
raises `AssemblyMemoryBudgetError` before native CSR/scatter allocation and
reports the dominant estimated allocation.  No implicit machine-memory lookup
or process-wide limit is imposed; applications own the budget policy.

Cut-cell bilinear and cross assemblers are now covered as well.  Their estimate
uses compact active-cell DOFs and offsets, actual flattened cut-point counts,
variable-quadrature shape/gradient storage, and active-cell scatter structure.
Cut linear forms do not build CSR/scatter patterns and are outside this
bilinear-memory budget API.

### P0: compatibility and failure contract

Publish a machine-readable registry for mesh families, element families,
spaces, mappings, form operations, and workflows.  Each entry is supported,
experimental, planned, or intentionally external.  Unsupported operations
must fail before assembly with a precise capability error; they must never be
silently lowered to a form with different mathematical meaning.

### P0: form algebra

Prioritize concrete weak-form requirements:

- coefficient component access (completed for a component-first axis 0);
- multiple independently named coefficient fields (completed for typed
  linear, bilinear, and composite assembly paths);
- outer product and transpose;
- explicit symmetric and skew tensor construction;
- anisotropic tensor contractions;
- coefficient-dependent boundary and interior-facet contractions.

Avoid arbitrary NumPy tracing.  Extend the typed form representation in small,
testable operations.

Initial anisotropic-gradient support is complete for scalar H1 fields.  The
typed form recognizes `dot(mul(A, grad(u)), grad(v))` and lowers it to the
native tensor cross-contraction kernel.  Constant, quadrature-dependent, and
nonsymmetric rank-2 tensors are covered and can be combined with existing
isotropic value/gradient terms.  Vector-field fourth-order tensors and general
outer products/transposes are still follow-up work.  Independently named
scalar/tensor fields can now be combined in one form, and coefficient
components use the scikit-fem-compatible first axis.  Missing, noninteger, and
out-of-range component access has an explicit diagnostic.

The public quadrature-dependent tensor convention is scikit-fem's
component-first `(dim, dim, entity, quadrature)` layout.  The entity-first
`(entity, quadrature, dim, dim)` representation is retained only as a
low-level/native compatibility input.  Conversion is explicit and commented in
the form normalizer because the native order is chosen for contiguous
quadrature-local tensor access, not because it represents different indices.

### P1: arbitrary-point evaluation

Add physical-point location and value/gradient evaluation, with containing-cell
IDs and explicit outside/ambiguous diagnostics.  This supports probes,
post-processing, mesh transfer, inverse problems, ROM, and contact workflows.

### P1: concurrency and validation

Define whether an assembler instance is reentrant and how native threads
interact with OpenMP, BLAS, and PETSc pools.  Add manufactured-solution
convergence, distorted-element tests, nonlinear tangent checks, objectivity,
serial/parallel equivalence, and comparisons with an independent FEM code.

## Why an edge midpoint is not an edge DOF

`TetP2` and `Hex20` contain nodes geometrically located on edges, but those are
still H1 nodal values.  An H(curl) edge DOF is an oriented line functional,
typically

```text
integral_edge(E dot tangent ds)
```

Shared cells must agree on its global edge number and apply opposite signs when
their local edge orientations disagree.  Treating this as a vector value at an
edge midpoint breaks tangential conformity.

## Required entity-DOF foundation

The future `Basis` must distinguish topological ownership from visualization
locations:

```text
entity DOFs
|- vertex DOFs: H1 nodal elements
|- edge DOFs:   H(curl), higher-order H1
|- facet DOFs:  H(div), higher-order elements
`- cell DOFs:   DG, bubbles, higher-order elements
```

Required data include:

- canonical global edge and facet numbering;
- local-entity to global-entity maps;
- orientation/permutation data per cell;
- multiple DOFs per entity and polynomial moment metadata;
- element-local DOF descriptors independent of mesh nodes;
- boundary selection by owned entity;
- interpolation functionals, not only point sampling;
- mixed spaces containing different entity ownership types.

`doflocs` may remain as a plotting aid, but it must not define DOF identity.

## First H(curl) milestone

Use first-kind, first-order Nedelec elements on tetrahedra:

- one oriented DOF per edge;
- reference basis `lambda_i grad(lambda_j) - lambda_j grad(lambda_i)`;
- covariant Piola mapping `J^-T E_hat`;
- physical curl transformation;
- `curl(u)` in the typed form vocabulary;
- `curl-curl`, vector mass, and source linear forms;
- PEC boundary selection by boundary edges;
- interpolation by edge line integrals;
- comparison against scikit-fem `ElementTetN1`;
- exact-sequence and orientation tests;
- a frequency-domain Maxwell example.

Do not start with Hex, higher-order Nedelec, absorbing boundaries, or a full
eigensolver API.  Those should follow after the tetrahedral kernel and DOF
contract are stable.

## H(curl) validation gates

The milestone is complete only if it demonstrates:

1. invariant assembly under global node relabeling;
2. correct sign changes when local cell connectivity is permuted;
3. tangential continuity across an interior facet;
4. gradient fields lying in the discrete curl kernel;
5. matching mass and curl-curl matrices against an independent implementation;
6. expected convergence on a manufactured Maxwell problem;
7. PEC boundary elimination selecting edges, not merely boundary vertices.

## Follow-on work

Once entity DOFs and mappings are generic, add Raviart-Thomas H(div) support.
It reuses entity ownership and orientation while adding contravariant Piola
mapping, divergence tabulation, and normal-trace boundary conditions.  Only
after both H(curl) and H(div) work should the library claim a compatible finite
element sequence rather than isolated specialized elements.
