# Changelog

## Unreleased

- Add a runnable electrostatic potential visualization with electric-field
  arrows and a regression test for the parallel-plate example.
- Include example PNG assets in source distributions.
- Clarify the boundary between scalar H1 electrostatics and experimental
  H(curl) edge-element APIs.

## 0.3.5 — 2026-10-17

- Add native H(div), nonlinear material, point-evaluation, mesh-topology, and
  extended assembly bindings and compatibility coverage.
- Expand the public Python exports and runtime capability metadata for the
  newly supported native paths.

## 0.3.4 — 2026-10-16

- Add native acceleration coverage for basis tabulation, composite DOFs,
  curved surfaces, cut quadrature, mesh promotion/topology, and point
  evaluation.
- Expand scikit-fem compatibility tests and backend benchmarks.
- Keep solver utilities under `skfemntv.utils` instead of exporting them from
  the package root.

## 0.3.3 — 2026-10-15

- Add the machine-readable compatibility manifest and parity helpers for the
  supported scikit-fem API surface.
- Add point evaluation, mesh I/O, refinement/restriction compatibility, and
  expanded P2 assembly coverage.

## 0.3.2 — 2026-09-26

- Update the project license metadata and license text to Apache-2.0.
- Align the standalone `scikit-fem-native` repository with the packaged
  `optimizer/packages/skfem-native` copy.

## 0.3.1 — 2026-08-21

- Align package distribution rename to `scikit-fem-native` with canonical
  documentation and repository metadata references.
- Fix remaining project metadata and docs links to avoid inconsistent
  `skfem-native` / `scikit-fem-native` navigation during user onboarding.

## 0.3.0 — 2026-08-13

### Added

- Experimental lowest-order Nedelec H(curl) vertical slices for affine
  triangles and tetrahedra, including oriented edge DOFs, covariant Piola
  mapping, interpolation, quadrature evaluation, boundary constraints, vector
  loads, and reusable mass/curl-curl/Maxwell assembly.
- A threaded native TetN1 integration kernel and native edge-topology/CSR setup
  kernels, with explicit application-owned thread control.
- Conservative memory preflight for the dedicated H(curl) assemblers and
  recorded TetN1 performance and memory calibration benchmarks.
- Typed-form support for multiple coefficients, coefficient components, and
  scalar anisotropic gradient contractions.
- Machine-readable capability declarations and explicit errors for unsupported
  mathematical operations.
- Documentation and compatibility examples for scikit-fem-style workflows.

### Changed

- Vectorized TriN1 basis mapping and moved size-dependent edge numbering and
  CSR setup loops to native code.
- Stabilized public tensor layouts: component-first public coefficient/basis
  views and entity-first private native storage are documented explicitly.
- Expanded geometry, orientation, convergence, distorted-mesh, memory-budget,
  and cross-platform validation.

### Fixed

- Made memory calibration portable to Windows by using the Win32 process
  working-set API when the Unix-only `resource` module is unavailable.

### Scope

H(curl) support remains experimental and is limited to dedicated affine TriN1
and TetN1 APIs.  General `Basis`/`asm` integration, curved H(curl) mappings,
higher-order Nedelec elements, and solver policy are not included.

## 0.2.1

- Previous PyPI release.  This changelog begins with the consolidated 0.3.0
  release notes; older history remains available in the Git repository.
