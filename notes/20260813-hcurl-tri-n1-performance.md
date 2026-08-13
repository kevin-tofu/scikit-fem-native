# Experimental TriN1 assembly performance checkpoint

## Question

Should the dedicated Python/SciPy TriN1 assembler immediately be replaced by a
C++ integration or scatter kernel?

## Benchmark

`benchmarks/hcurl_tri_n1_assembly.py` compares repeated assembly of

```text
integral u dot v + 0.2 curl(u) curl(v)
```

against scikit-fem `ElementTriN1`.  Basis and sparse-pattern construction are
reported separately and excluded from repeated assembly timing.  The benchmark
also times element integration and fixed-CSR scatter independently.  Raw output
is stored in `benchmarks/results/hcurl-tri-n1-assembly.csv`.

One local run produced:

| resolution | DOFs | elements | TriN1 ms | scikit-fem ms | speedup | integration | scatter |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 800 | 512 | 0.493 | 0.693 | 1.41x | 91.6% | 2.3% |
| 32 | 3,136 | 2,048 | 1.205 | 1.481 | 1.23x | 95.1% | 3.2% |
| 64 | 12,416 | 8,192 | 4.188 | 4.516 | 1.08x | 97.4% | 3.4% |
| 128 | 49,408 | 32,768 | 17.441 | 18.510 | 1.06x | 98.2% | 3.4% |

These are environment-specific medians, not release guarantees.

## Finding and optimization

The initial implementation was 1.7--4 times slower than scikit-fem.  Profiling
showed that CSR scatter consumed less than one percent; the bottleneck was a
generic four-input `numpy.einsum` used for coefficient-weighted integration.

Multiplying `dx * coefficient` first and using a three-input contraction with
`optimize=True` reduced the 128-resolution repeated assembly from roughly
75 ms to 17 ms.  Numerical comparison and convergence tests remain unchanged.

## Decision

Do not add a native C++ scatter kernel now.  Scatter is only about three percent
of optimized repeated assembly, so even eliminating it cannot materially
improve the total.  The optimized NumPy implementation is approximately at
parity with or modestly faster than scikit-fem over the measured range.

Future native work should require a new profile demonstrating a meaningful
bottleneck, ideally on larger production-shaped meshes and coefficient fields.
Basis construction, memory traffic, and batched element integration are more
plausible targets than fixed-CSR scatter.

## Basis-construction follow-up

A later profile at resolution 64 showed that basis construction spent most of
its time calling the Piola mapping once per cell from a Python loop.  This was
call overhead rather than a slow numerical kernel: mapping all cells in one
batched NumPy operation reduced the resolution-128 basis time from about
4.92 s to 0.365 s (about 13.5x) without changing the stored entity-first
assembler layout or the public component-first view.

After that change, the remaining construction profile is dominated by two
Python dictionary loops:

- canonical global-edge numbering in `build_oriented_edge_topology`;
- CSR scatter-position construction in `TriN1Assembler.__init__`.

These setup paths, rather than repeated integration or scatter, are now the
credible native C++ candidates.  Moving them should be justified against
large-mesh setup measurements and should return the existing arrays so the
Python public API and orientation tests remain unchanged.

Both setup paths were subsequently moved behind two coarse-grained native
functions while retaining the Python topology and assembler objects.  At
resolution 128 (32,768 cells), one local run measured:

| stage | before native setup | after native setup | improvement |
|---|---:|---:|---:|
| basis construction | 0.365 s | 0.055 s | 6.6x |
| assembler construction | 0.414 s | 0.0095 s | 43.6x |

The basis time is now approximately equal to the scikit-fem reference basis
time in that run.  Repeated assembly remains NumPy-based because its profile
still shows integration dominating and performance remains near parity with
scikit-fem.  This split keeps orientation/pattern loops native without moving
the public finite-element model or coefficient policy into C++.
