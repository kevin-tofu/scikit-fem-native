# Nonlinear fused assembly

This benchmark compares fused native Tet4 or Hex8 Neo-Hookean residual/tangent
assembly against equivalent scikit-fem forms.  Setup and linear solve are
excluded; the measured operation is one warm-cache nonlinear assembly call.

```bash
python benchmarks/nonlinear-assembly/neo_hookean.py \
  --repeat 7 --native-threads 4 \
  --output benchmarks/nonlinear-assembly/results/neo-hookean.csv \
  --plot-output benchmarks/nonlinear-assembly/results/neo-hookean.png
```

The native Tet4 path exploits the constant deformation gradient of an affine
element and evaluates its constitutive law at one integration point.  The
script verifies residual and tangent against the scikit-fem forms before
timing unless `--no-check` is passed.

Select topology, integration order, and warped geometry explicitly:

```bash
python benchmarks/nonlinear-assembly/neo_hookean.py \
  --topology hex --intorder 4 --distorted --points 3 4 5 6
```

Hex8 uses the tabulated fused path because its physical gradients vary across
quadrature points, particularly for distorted elements.  Tet4 accepts the same
explicit quadrature sweep even though its affine deformation gradient is
constant and one-point integration is sufficient for this material kernel.

The primary plot compares residual-plus-tangent work only.  Residual-only
timings remain in the CSV and table for line search, modified Newton, and
matrix-free use cases, but are not plotted as though they were equivalent to
residual-plus-tangent assembly.

## J2 plasticity

The J2 benchmark separates the constitutive material-point update from fused
global residual and tangent assembly, and reports native thread scaling:

```bash
python benchmarks/nonlinear-assembly/j2_plasticity.py \
  --topology tet --intorder 2 --repeat 3 --native-threads 4 \
  --output benchmarks/nonlinear-assembly/results/j2-tet4.csv \
  --plot-output benchmarks/nonlinear-assembly/results/j2-tet4.png
```

The displacement is deliberately plastic.  Serial and parallel residual,
tangent, and trial state are checked before timing.  The scikit-fem reference
performs the same vectorized radial return once per integration point and then
passes its stress and algorithmic tangent to the residual and tangent forms.
The timed operation includes that update and both assembly calls from the same
committed zero state.  This is a benchmark reference supplied by this
repository, not a built-in scikit-fem material model.

The history benchmark prescribes five strain states covering elastic loading,
plastic loading, elastic unloading, reverse plasticity, and reloading.  Every
accepted trial state is committed before the next assembly; no linear solve is
performed:

```bash
python benchmarks/nonlinear-assembly/j2_history.py \
  --repeat 3 --native-threads 4 \
  --output benchmarks/nonlinear-assembly/results/j2-history-tet4.csv \
  --plot-output benchmarks/nonlinear-assembly/results/j2-history-tet4.png
```

## Standard Linear Solid

The viscoelastic benchmark prescribes loading, two held-displacement steps,
and two unloaded steps with a different time step at every stage.  It compares
the complete five-step material update and R+K assembly history without a
solve or assembler reconstruction:

```bash
python benchmarks/nonlinear-assembly/standard_linear_solid.py \
  --repeat 1 --native-threads 4 \
  --output benchmarks/nonlinear-assembly/results/standard-linear-solid-tet4.csv \
  --plot-output benchmarks/nonlinear-assembly/results/standard-linear-solid-tet4.png
```
