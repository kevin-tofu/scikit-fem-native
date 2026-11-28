# GridapTopOpt comparison record

These checked-in artifacts compare the native wheel level-set run with the
reference history from GridapTopOpt.jl `v0.2.0`,
`scripts/Examples/Unfitted/Wheel3D_CutFEM_b.jl`.

- `gridap-comparison.csv`: per-iteration compliance, volume, level-set, and
  interface-crossing metrics.
- `gridap-comparison.png`: the same metrics plotted over the 30-iteration run.
- `gridap-comparison-final.vtu`: common background mesh with
  `level_set_gridap` and `level_set_native` point fields.

The compliance histories remain close (mean relative error 1.67%, maximum
7.29%), but the final interfaces do not: at iteration 30 the volume-fraction
difference is 0.0192, interface-crossing RMS difference is 0.171, and 252
nodes have different level-set signs. This is a comparison result, not evidence
that the two optimizers produce the same shape.

Reference implementation:
[GridapTopOpt.jl v0.2.0](https://github.com/zjwegert/GridapTopOpt.jl/tree/v0.2.0).
