# Development status — 2026-08-03

## Project identity

- GitHub repository: `kevin-tofu/skfem-native`
- Distribution name: `skfem-native`
- Python import name: `skfemntv`
- Current branch: `main`
- Current version and tag: `0.1.2` / `v0.1.2`
- Current commit: `1143df0` (`Bump version to 0.1.2`)
- Python requirement: CPython 3.10 or newer
- Runtime policy: no scikit-fem import, fallback, or Python element assembly

The intended user-facing compatibility pattern is:

```python
import skfemntv as skfem
```

This works for the documented compatible subset.  Native-only extensions and
unsupported operations are explicit; unsupported forms raise
`UnsupportedNativeForm` rather than silently switching implementation.

## Release status

`v0.1.2` was published as a GitHub Release and uploaded to PyPI through Trusted
Publishing.  The complete release workflow succeeded:

- source distribution;
- Linux x86_64 wheels;
- Windows AMD64 wheels;
- macOS arm64 wheels;
- macOS x86_64 wheels;
- final PyPI publish job.

PyPI currently provides CPython 3.10--3.14 wheels.  Users on a supported
platform can install without a local C++ compiler:

```bash
python -m pip install skfem-native
```

The wheel coverage is:

- Linux x86_64 with glibc 2.27 or newer;
- Windows 64-bit x86;
- macOS Apple Silicon;
- macOS Intel.

Linux arm64, Alpine/musl, Windows 32-bit, and PyPy do not yet have prebuilt
wheels and may fall back to an sdist build.

## Implemented assembly scope

### Meshes and elements

- Tri3 / Tri6;
- Quad4 / Quad9;
- Tet4 / Tet10;
- Wedge6;
- Pyramid5 as an explicit skfemntv extension;
- Hex8 / Hex27;
- scalar P0 and nodal H1 P1/P2 or Q1/Q2 where applicable;
- vector, discontinuous, and composite elements with documented limits.

Mesh topology includes cached facets, `t2f`, `f2t`, boundary/interior facet
queries, tensor-mesh constructors, high-order conversion, named boundaries,
and cell/facet predicates.

### Forms and bases

- `BilinearForm`, `LinearForm`, `Functional`, and `asm`;
- `Basis`, `FacetBasis`, and `InteriorFacetBasis`;
- volume, exterior-facet, and interior-facet integration;
- `dot`, `ddot`, `grad`, `div`, `sym_grad`, and `trace`;
- `jump`, `avg`, and `normal_grad` for interface terms;
- physical coordinates through `w.x` and normals through `w.n`;
- scalar, array, callable, and interpolated-field coefficients;
- mixed signatures such as `u, p, v, q, w`;
- rectangular trial/test spaces and mixed-order coupling;
- element-restricted bases without global DOF renumbering;
- interpolation of a global coefficient vector at basis quadrature points.

scikit-fem is used only in tests as a numerical and API reference.  Matrix,
vector, functional, interpolation, boundary, mixed-space, high-order, and
interior-facet results are compared directly where an equivalent scikit-fem
operation exists.

### Nonlinear and material assembly

- reusable CSR structure and native residual/tangent evaluation;
- linear elasticity and Neo-Hookean kernels;
- consistent tangent assembly;
- J2 plasticity with committed/trial state separation;
- Standard Linear Solid viscoelasticity;
- adaptive-step `time_step` override without rebuilding geometry or sparsity;
- residual-only and residual-plus-tangent modes;
- caller-owned output buffers through the low-level evaluation path.

Solver policy remains outside the package.  End-to-end tests use SciPy where a
solve is needed to validate assembly, but `solve` and `condense` are not public
skfemntv APIs.

### Nonmatching interfaces and mortar

- planar triangle supermesh construction;
- AABB broad phase and batched native overlap construction;
- parallel planar supermesh processing;
- curved Tet10/Hex27 surface tessellation and projection diagnostics;
- trace shape values, physical gradients, normals, weights, and parent IDs;
- paired master/slave orientation diagnostics;
- sparse cross/value/gradient blocks;
- `MortarCouplingResult` with sparse master, slave, and coupling matrices;
- slave P1, master P1, overlap-cell P0, and facet-local dual multiplier bases;
- composable Poisson and elasticity Nitsche flux terms;
- constant/linear reproduction, action-reaction, refinement, and scikit-fem
  comparison tests.

Native kernels return sparse blocks, COO/CSR data, or quadrature-local arrays;
they do not construct a multiplier-sized global dense matrix.

## Parallelism and performance

Native thread controls are available globally, through a context manager, and
per assembly call:

```python
skfemntv.set_num_threads(4)

with skfemntv.thread_limit(4):
    matrix = skfemntv.asm(form, basis)

result = assembler.assemble(u, state, num_threads=4)
```

The requested count is capped by CPU affinity visible to the process.  Parallel
paths include colored CSR scatter for volume/nonlinear assembly, interface and
mortar CSR assembly, and planar supermesh construction.  Serial/parallel
numerical agreement is covered by tests.

Benchmarks distinguish:

- mesh construction;
- basis construction;
- sparsity/assembler preparation;
- first assembly;
- repeated assembly;
- residual-only versus residual-plus-tangent;
- one-thread and multi-thread native execution;
- peak memory where the platform exposes it.

The comparison suite includes scikit-fem's performance-style Poisson problem,
large DoF sweeps, nonlinear mesh/order sweeps, and native scaling.  Basis
construction was optimized by removing eager duplicate work and caching
triangle frames.  Performance claims should continue to report setup and
repeated assembly separately rather than hiding native preparation cost.

## Packaging and workflow

- `scripts/upgrade_version.py` updates the project version and supports Python
  3.10 without relying unconditionally on `tomllib`;
- `tools/check_release_version.py` verifies `v<version>` release tags;
- local package checks build, inspect, install, import, and optionally test a
  wheel in an isolated environment;
- `tools/local_ci.py` mirrors fast, package, wheel, and full validation stages;
- `tools/build_wheels.py` wraps cibuildwheel for native platform builds;
- GitHub Actions cover ordinary CI, manually dispatched full validation, and
  release publication;
- `notes/gh-usage.md` documents CLI inspection, failure logs, release creation,
  monitoring, and PyPI verification.

## Current production gaps

The project is a useful focused assembly engine, but the following remain the
highest-value gaps.

### Geometry validity

Basis construction now applies a scale-aware Jacobian policy in the common
native tabulation path.  It rejects near-singular points and determinant sign
changes within a curved element before assembly.  Errors identify cell and
quadrature-point IDs and report determinant/tolerance data.  A uniformly
negative local orientation remains valid and is counted in
`GeometryDiagnostics`.  Tet10/Hex27 internal inversions and the maximum
Jacobian condition number are covered.

### Geometric regions

The first region layer is implemented: immutable `CellRegion`, `FacetRegion`,
and `NodeRegion` values, union/intersection/difference/complement, selection and
empty-region diagnostics, named cell subdomains, named boundary regions, and
direct use by Basis, FacetBasis, and DOF selection.  Global IDs are preserved
and restricted bases do not renumber global DOFs.

Normal-oriented facet queries are implemented for exterior and interior
facets.  Each `FacetRegion` carries immutable parent-side and normal-sign
metadata, and every FacetBasis topology path consumes mixed orientations.  The
Component-aware DOF selection is implemented through `DofsView` groups,
scikit-fem-style `all/keep/drop`, numeric vector components, and composite
field/component selectors.

Level-set classification metadata is implemented by `LevelSet`,
`CellClassificationResult`, and `CellClassification`.  Callable and nodal
fields produce immutable global labels plus inside, outside, cut, touching,
and active `CellRegion` values.  All mesh topologies use every connectivity
node, including high-order nodes.  Field-scale-aware tolerance, non-finite
value diagnostics, and direct restricted-Basis use are tested.  Active facets,
active-boundary facets with parent-side metadata, active-interior facets,
cut-adjacent ghost-penalty candidates, and component-aware active global DOFs
are available without imposing a CutFEM formulation.  Cut-volume and
implicit-interface quadrature remain separate stages.

The first cut-volume stage is implemented for affine Tri3 and Tet4 cells.
`CutCellQuadrature` stores CSR-like cell offsets, physical/reference points,
positive physical weights, background cell IDs, and oriented level-set normals
with memory proportional to generated quadrature points.  Inside and outside
rules partition the parent-cell measure, and analytic tests cover exact
constant/linear integration.  Positive order-two simplex rules and general
higher-order Duffy rules cover polynomial volume integration without changing
the CSR storage.  High-order/curved reconstruction and the implicit interface
remain intentionally unsupported rather than silently approximated.

`CutCellBasis` is implemented as the first variable-quadrature assembly
geometry provider.  It tabulates affine TriP1/TetP1 shape values and physical
gradients at flattened cut points, maps each point to global parent-element
DOFs, supports restricted parent bases, and interpolates scalar/vector fields.
It deliberately does not emulate the rectangular `Basis.dx` contract; native
form assembly must consume its CSR offsets and flattened arrays explicitly.

### P1 — Arbitrary-point field evaluation

`Basis.interpolate` evaluates at existing quadrature points.  Probes, transfer,
inverse problems, and post-processing need value and physical-gradient
evaluation at arbitrary physical points, including containing-cell IDs and
outside/ambiguous-point diagnostics.

### P1 — Form algebra gaps

The native form vocabulary should be extended deliberately for concrete weak
forms, especially coefficient component access, multiple independent
coefficient fields, outer products/transposes, anisotropic tensors, and richer
facet coefficients.  Arbitrary NumPy tracing and runtime fallback should not be
introduced.

### P1 — Large-problem and concurrency contracts

- preflight memory estimates for CSR patterns, scatter maps, geometry, and
  material state;
- 10k/100k/1M-DoF memory measurements by topology;
- an explicit reentrancy contract for concurrent calls on one assembler;
- deterministic handling of surrounding OpenMP/BLAS/PETSc thread pools.

Lower-priority gaps include composite DG spaces, complete mixed-face interior
facets, Line elements, custom facet mappings/DOFs, and future H(div)/H(curl)
tracks.

## Research direction: CutFEM and level sets

CutFEM should build on first-class regions but must not be implemented as only
a more complex facet predicate.  The design separates:

1. level-set sampling;
2. inside/outside/cut cell classification;
3. active cells, facets, and DOFs;
4. cell-local cut-volume quadrature;
5. implicit-interface quadrature;
6. user-defined Nitsche and ghost-penalty forms.

Cut cells have different quadrature-point counts.  The intended storage is a
CSR-like local representation rather than padding every cell:

```text
cell_offsets[ncells + 1]
points[ncut_qp, dim]
weights[ncut_qp]
```

Each point also needs its background cell, reference coordinate, shape values,
physical gradients, level-set gradient, and consistently oriented interface
normal where applicable.  Moving level sets must distinguish value updates,
local quadrature rebuilds, and active-set/CSR-pattern rebuilds.

Validation should progress from exact planar cuts and constant/linear
integration to circle/sphere convergence, moving interfaces, unfitted Poisson
with Nitsche terms, and ghost-penalty conditioning.  scikit-fem remains the
comparison reference wherever the same custom quadrature can be expressed;
analytic geometry and manufactured solutions validate CutFEM-specific pieces.

See `notes/assembly-gap-audit-20260803.md` for the detailed capability matrix,
proposed region APIs, cut-quadrature data, and validation ladder.

## Deferred directions

CalculiX comparison is intentionally deferred.  For current assembly features,
scikit-fem gives more precise matrix/vector comparisons and makes failures much
easier to localize.  CalculiX becomes valuable later for `.inp`
interoperability, independent large-model validation, and migration from an
existing industrial solver, not as a duplicate unit-test oracle.

The following also remain outside core scope unless a concrete assembly use
case requires them:

- linear/nonlinear solver policy;
- pressure pinning and condensation policy;
- automatic contact/mortar/Nitsche formulation selection;
- mandatory PETSc or external C++ tensor dependencies;
- plotting and general mesh-file management.

## Recommended next work

1. Add arbitrary-point value/gradient evaluation.
2. Close form-algebra gaps needed by anisotropic and multi-coefficient forms.
3. Connect LinearForm, BilinearForm, and Functional assembly to the flattened
   `CutCellBasis` contract without padding.
4. Build `ImplicitFacetBasis` as an assembly geometry provider.
5. Add curved/high-order interface reconstruction with convergence tests.

## Branch checkpoints

- `fae02db`: geometry validation and initial development-status documents;
- `57adef0`: first-class regions and named subdomains;
- `e99d17d`: normal-oriented facet regions;
- `2efecb7`: component-aware DOF selection;
- level-set classification is implemented on `feature/levelset`.
