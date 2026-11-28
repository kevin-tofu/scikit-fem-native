# Level-set wheel reference

The wheel benchmark follows GridapTopOpt.jl `v0.2.0`,
`scripts/Examples/Unfitted/Wheel3D_CutFEM_b.jl`. The pinned geometry is
[`Wheel_3d.geo`](Wheel_3d.geo). Its inner shell receives torsional traction
`100 * (-y, x, 0)`, the outer shell is fixed, and compliance is minimized
under a 50% material-volume target.

The executable Python tutorial is in
[`tutorials/wheel-levelset/`](../../tutorials/wheel-levelset/README.md). It
contains the native state solver, the level-set optimization driver, a
background mesh, and a per-iteration comparison with the Gridap reference.
The comparison shows close compliance values but increasing interface
divergence; the two implementations do not produce the same final shape.

The full paper-scale mesh has about 1.32 million tetrahedra. The bundled mesh
is a smaller background mesh intended for reproduction and development, not a
scale-equivalent performance benchmark.
