# CutFEM assembly benchmarks

`cut_assembly.py` separates native assembler setup from repeated assembly and
compares the segmented `cell_offsets` kernel with the former flattened native
adapter.  The vertical level set gives an exact active-volume fraction.

```bash
python benchmarks/cutfem/cut_assembly.py \
  --resolution 128 --fractions .1 .5 .9 --intorders 1 2 4 \
  --threads 4 --repeat 3 \
  --output benchmarks/cutfem/cut-assembly.csv \
  --plot-output benchmarks/cutfem/cut-assembly.png
```

The numerical equality check is enabled by default.  Use `--no-check` only for
profiling after correctness has already been established.

## Reference run (2026-08-03)

The committed CSV and PNG use a 64 x 64 background grid, three repetitions,
and four requested threads.  Segmented/flattened repeated bilinear assembly
speedups were:

| integration order | active 10% | active 50% | active 90% |
|---:|---:|---:|---:|
| 1 | 0.89x | 0.97x | 0.96x |
| 2 | 2.50x | 2.26x | 1.94x |
| 4 | 5.95x | 3.85x | 2.95x |

Order one has too few points for segmentation to amortize its fixed overhead.
At orders two and four, avoiding quadrature-point duplication in CSR setup and
cell metadata gives a clear benefit.  These are local-machine measurements,
not universal performance guarantees; regenerate the CSV for target hardware.

`implicit_cross.py` benchmarks two-sided value and full-gradient trace blocks.
Every configuration first checks an independent NumPy element-loop oracle,
serial/parallel agreement, the former flattened native path, and the
constant-jump null mode before timing:

```bash
python benchmarks/cutfem/implicit_cross.py \
  --resolution 128 --intorders 1 2 4 6 --threads 4 --repeat 3 \
  --output benchmarks/cutfem/implicit-cross.csv \
  --plot-output benchmarks/cutfem/implicit-cross.png
```

The 2026-08-03 reference run uses a 128 x 128 background grid.  Compacting to
nonempty interface cells changed segmented setup from substantially slower than
the flattened path to 1.14x, 1.29x, and 1.42x faster at orders 2, 4, and 6.
Repeated assembly is approximately equal: 0.90--0.93x for value blocks and
0.98--1.01x for gradient blocks.  The main cross-kernel benefit is therefore
bounded metadata growth at higher order, not a large contraction speedup.
The committed CSV retains oracle and serial/parallel errors for every row.
