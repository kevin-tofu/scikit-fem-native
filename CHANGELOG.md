# Changelog

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
