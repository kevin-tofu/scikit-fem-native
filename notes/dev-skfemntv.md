# skfemntv development status

This note tracks the optimizer vendored copy and the independent
`skfem-native` checkout. A passing supported test suite is **not** a claim of
full scikit-fem compatibility.

## Current test boundary

On 2026-10-22, the optimizer copy collected 829 native tests. Before skips,
171 failed in 22 newly added parity modules; the other tests passed except one
pre-existing skip. `tests/conftest.py` now marks only those 171 cases as skipped
with an explicit reason. Tests in partly implemented modules that already pass
remain active. Re-run without the skip marks when implementing a listed API;
remove each mark once its case passes on the supported Python/OS matrix.

These are real compatibility and numerical-regression tests, not disposable
demos. The skip only keeps CI honest about the current supported surface.

## Implemented and exercised

- Existing static FEM, geometry, contact, quadrature, and dedicated H(curl)/
  H(div) paths remain in the blocking suite.
- RT0 triangle/tetrahedron `Basis` and selected public `asm` mass,
  divergence, vector-load, and RT0-DG0 mixed forms are implemented.
- The merged source also retains the standalone copy's line elements,
  topology helpers, and C++ bindings. `MeshLine` FEM parity is still
  incomplete; merely exporting `MeshLine` is not sufficient.
- Hex27 basis tabulation and the C++ binding set are checked in wheel smoke.
- Nonlinear CSR scatter accepts NumPy signed 32/64-bit index dtypes on both
  Linux and Windows. Its earlier buffer-format-string comparison failed on
  Windows despite the index widths being correct; dtype and contiguity are
  checked directly now.
- Memory-calibration peak RSS is bounded by the sampled current RSS at each
  phase; Linux `ru_maxrss` can briefly report less than `/proc/self/status`.

## Open parity work

1. Complete mesh APIs first: `MeshLine.with_boundaries` and refinement,
   coordinate transforms, edge topology, Hex refinement, native P2 promotion,
   element finder, and point source. Keep region and numbering semantics
   aligned with scikit-fem.
2. Complete `Basis` APIs: standard parameters, split/composite and DG paths,
   point evaluation, `refinterp`, and projection. Validate scalar/vector,
   element subset, and tensor-product cases separately.
3. Integrate the dedicated H(curl) basis with public `Basis`/`asm`; preserve
   edge orientation and Piola mapping. Do not describe dedicated Nedelec
   functions as general H(curl) support.
4. Extend nonlinear multi-cell J2/Neo-Hookean assembly and objectivity checks.
   Several failing cases currently hit `split_indices()` on a vector field;
   determine whether that is a missing vector dispatch or a true numerical
   regression before implementing a workaround. The material-point tests that
   already pass remain blocking.
5. Add native composite DOF, curved-surface, and tensor point-evaluation
   frontends with parity checks for both Linux and macOS wheels.

## Sync and release discipline

Use the standalone package as the release source, then synchronize the tested
implementation and relevant tests to the optimizer vendored copy. Compare
source files before copying: the optimizer RT0 additions and the standalone
line/C++ APIs diverged, so replacing either tree wholesale would erase working
features. Run the native suite in both locations, build and import a wheel,
and run the kktkit integration tests before pushing. A source sync does not
update an already published PyPI wheel; bump and publish the package version
separately when the added APIs are intended for distribution.

The optimizer currently declares `scikit-fem-native` 0.3.2; the standalone
checkout declares 0.3.6. Check the imported `skfemntv.__file__`, distribution
version, and compiled `_skfn` location when comparing runs. Do not infer
implementation parity from a version string alone.
