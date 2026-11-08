# `skfem-native` copy integration

The package is maintained in two independent Git repositories:

- `optimizer/packages/skfem-native` is a local path dependency of the optimizer.
- `kktkit/packages/skfem-native` is the standalone package repository.

Their Git histories are unrelated, so integration is done by reviewing and
applying feature patches rather than merging repository histories.

## Synchronized package features

Both working trees now contain the same Python package sources and tests for:

- affine RT0 H(div), including tensor-weighted mass, boundary normal traces,
  facet linear loads, and normal-trace bilinear forms;
- affine triangular and tetrahedral Nédélec H(curl) spaces through N3;
- line-mesh boundary labels, refinement, restriction, and removal;
- the matching experimental capability registry and parity documentation.

The standalone CI workflows also check that an installed package can import
the native extension and representative public APIs.

Both package `pyproject.toml` files are at version `0.4.0`, set with each
copy's `scripts/upgrade_version.py`. The optimizer root lockfile was refreshed
to record its local package dependency at `0.4.0`.

## Differences kept local

| Difference | Kept with | Reason |
| --- | --- | --- |
| Backend and topology benchmarks; `tests/roadmap` | optimizer | These measure or plan optimizer-specific integration work. |
| `.agents`, `.aws`, and `.codex` directories | optimizer | Local tooling and environment configuration. |
| `examples/electrostatic_potential.png` and its package include rule | kktkit | The standalone example owns this image asset. |
| `notes/dev-skfemntv.md` | kktkit | Standalone repository development history. |
| `poetry.lock` files | each repository | They resolve different development dependency snapshots and are not runtime source. |

The package `pyproject.toml` files otherwise match except for the PNG include
rule needed by the standalone image. The optimizer root `poetry.lock` is
separate from the package-level development locks.

## Ongoing sync rule

Treat the standalone `kktkit` package as the source for shared finite-element
features. When porting a feature, keep optimizer-specific benchmarks and
application code local, update both package versions only when a shared
feature set is ready, and check the runtime sources and tests for parity.
