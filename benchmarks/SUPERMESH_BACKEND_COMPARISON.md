# Supermesh backend comparison

`supermesh_backend_comparison.py` compares the public skfemntv Mortar API
against a scikit-fem/Shapely reference implementation. It measures matching,
nonmatching, and strongly imbalanced master/slave surface meshes.

The benchmark intentionally stops after sparse Mortar coupling assembly.
QR/SVD reduction, contact state, KKT construction, and linear solvers belong to
kktkit and are not included here.

Run with the benchmark extras installed:

```bash
python benchmarks/supermesh_backend_comparison.py \
  --cells 12 \
  --repeat 3 \
  --output benchmark-results/supermesh-backends
```

Outputs:

- `results.csv`
- `results.json`
- `comparison.png`

Compare overlap area and constant-field residual before interpreting timing.
Different overlap-cell counts are acceptable because triangulation and
small-overlap handling differ, but the integrated area and reproduced fields
must agree within numerical tolerance.
