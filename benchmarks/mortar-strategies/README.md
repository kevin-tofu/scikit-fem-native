# Tied mortar strategy benchmark

This benchmark verifies that four modelling/algebraic choices can be built
from the public skfemntv assembly API:

| strategy | construction |
|---|---|
| `fine` | native `overlap_p0` assembly |
| `coarse-p0` | native `slave_facet_p0` assembly |
| `algebraic-qr` | `overlap_p0`, followed by facet-local SciPy QR |
| `algebraic-svd` | `overlap_p0`, followed by facet-local NumPy SVD |

QR and SVD intentionally live in this benchmark, outside `skfemntv`.  They are
limited to a tied interface: applying arbitrary row combinations to unilateral
or frictional constraints may not preserve multiplier positivity or friction
cones.  Each algebraic reduction materializes only one slave-facet block as a
dense array; the global constraint matrix is never converted to dense storage.

Run a sweep and optionally produce CSV/PNG output:

```bash
python benchmarks/mortar-strategies/mortar_strategies.py \
  --cells 4,8,16,32 --repeat 3 --threads 4 \
  --output benchmarks/mortar-strategies/results.csv \
  --plot-output benchmarks/mortar-strategies/results.png
```

The two surfaces use different resolutions and opposite cell diagonals.  Every
strategy checks the tied constant-field null mode.  Reported timings separate
native assembly from external algebraic reduction.  `maximum_local_rows`
documents the largest dense block used by QR/SVD.

## Reference run

The committed reference uses `cells=2,4,8`, three timing samples, and four
requested native threads.  At the largest point there are 1,216 overlap cells:

| strategy | multiplier rows | observation |
|---|---:|---|
| fine | 1,216 | maximum overlap-local freedom |
| coarse-p0 | 162 | strongest geometric row reduction |
| algebraic-qr | 1,030 | preserves selected sparse rows |
| algebraic-svd | 1,030 | same rank, but denser reduced rows |

These results are deliberately not presented as universal timings.  They show
that all four paths are constructible, while also showing that algebraic
reduction is not automatically beneficial.  On this mesh, facet-P0 reduces
rows much more cheaply; QR/SVD remain modelling-dependent external options.
