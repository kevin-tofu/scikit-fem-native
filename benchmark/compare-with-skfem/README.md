# Poisson assembly scaling

This benchmark compares native `skfn` assembly with scikit-fem while growing a
structured triangular mesh.  Both implementations receive exactly the same
coordinates, connectivity, P1 space, and integration order.  It measures:

- basis construction;
- Poisson stiffness-matrix assembly;
- constant right-hand-side assembly.

Linear-system solution is deliberately excluded: `skfn` is an assembly engine,
and both packages can pass the resulting CSR matrix to the same solver.

Run the default DoF sweep:

```bash
python benchmark/compare-with-skfem/poisson_assembly.py
```

Write machine-readable results or choose a smaller/larger sweep:

```bash
python benchmark/compare-with-skfem/poisson_assembly.py \
  --sizes 32 64 128 256 512 \
  --repeat 7 \
  --output benchmark-results/poisson.csv
```

Each form is assembled before timing to populate both libraries' caches.  The
reported values are medians, not single measurements.  Before timing, the
script also checks that both assembled matrices and vectors agree numerically.
For reproducible comparisons, record the printed Python/package versions and
run on an otherwise idle machine with a fixed CPU power policy.
