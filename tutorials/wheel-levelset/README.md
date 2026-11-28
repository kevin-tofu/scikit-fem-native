# Wheel level-set optimization tutorial

This tutorial ports the 3-D elastic-wheel optimization workflow from
`GridapTopOpt.jl/scripts/Examples/Unfitted/Wheel3D_CutFEM_b.jl` to Python and
uses the skfemntv CutFEM wheel state solve. It uses the reference case's
initial level set, torsional traction, `vf = 0.5`, compliance functional,
volume derivative, augmented-Lagrangian update, Hilbertian velocity
extension, and level-set transport parameters.

The update is no longer a four-mode design-space update. It forms a nodal
compliance shape gradient on the reconstructed interface and an analytic
volume gradient,

\[
  dL = dJ + (-\lambda + \Lambda C)\,dC,
  \qquad
  (M + A_\alpha) v = dL,
\]

where `A_alpha` discretizes the reference regularizer
`alpha(h) grad(p) . grad(q) + p*q`, with
`alpha(h) = (gamma * max_steps * h)^2 + 1e-6`. The level set is then advanced
with the reference normalized transport field and a Crank–Nicolson P1
advection solve stabilized on the full background-mesh skeleton, matching
GridapTopOpt's `SkeletonTriangulation(model)`. The augmented
Lagrangian uses the reference initialization and multiplier/penalty updates.

The compliance gradient now includes the state adjoint, following
GridapTopOpt's `StateParamMap` pullback. For the physical stiffness
`K_phys`, stabilized state operator `K_total`, displacement `u`, and adjoint
`p`, the script solves

\[
  K_{\mathrm{total}}^T p = 2K_{\mathrm{phys}}u,
  \qquad
  \frac{dJ}{d\phi}
  = \frac{\epsilon(p):\sigma(\epsilon(u))
          -\epsilon(u):\sigma(\epsilon(u))}
         {|\nabla\phi|}
\]

on the reconstructed interface. The volume derivative matches the analytic
expression in the reference script. The shape derivative differentiates the
physical-domain integrals; discrete changes to the stabilization-facet and isolated
cell selectors are not differentiable and are treated as fixed during each
gradient evaluation. A central finite-difference check of the initial native
discrete compliance along the computed gradient gave `432.827766`, versus
`432.831147` from the adjoint gradient (relative difference `7.8e-6`).
Reinitialization uses the reference artificial-viscosity residual and its
specified Newton tangent. The transport operator uses the P1 velocity field,
the plus trace in the interior-facet stabilization, the same `gamma_g = 0.01`
and ten time steps. For this frozen linear transport operator,
`DIRK_CrankNicolson_2_2` has the trapezoidal tableau and is algebraically the
Crank–Nicolson update used here. The state, volume, interface, and
reinitialization integration order defaults to 4, matching the reference
driver's `degree = 2 * (order + 1)` for P1 fields.

This mirrors GridapTopOpt's state-map pullback: differentiate the functional
with respect to `u`, solve the state adjoint, then combine its pullback with
the direct interface derivative. The finite-difference check validates this
gradient against the native state solve; it does not prove identical
derivatives or trajectories between the two frameworks, whose cut-cell
differentiation and assembly may differ. The augmented-Lagrangian oscillation
check also uses the reference 26-value window, down-crossing rule, start
iteration, and cooldown.

An update-only comparison on the reference wheel mesh isolated the remaining
level-set evolution difference. With `gamma_g = 0`, the native and Gridap
transport results agree to relative L2 `5.0e-15`; using only cut-neighbor
facets with stabilization gave `7.09e-2`. Switching to all interior facets
reduced that difference to `1.96e-2` (37 nodal sign differences out of 7,974).
Gridap defines the plus trace as the first incident cell in global cell order.
The reference-mesh VTU preserves that order, so the native update now selects
the lower cell index as plus even when `MeshTet` reports the other cell first.
With this trace convention and degree-four facet quadrature, the relative L2
difference is `3.24e-4` (2 nodal sign differences out of 7,974). A seven-point
rule gave `3.69e-4`; the remaining update difference is small and is not due
to the stabilization domain or plus-side selection. The geometry verifier
therefore preserves the source cell order while checking duplicates.

An earlier three-iteration run, before the quadrature correction documented
below, showed compliance relative errors of `0.031%`, `0.465%`, `0.090%`, and
`0.714%` for iterations 0–3. Those trajectory statistics are superseded by
the corrected run below.

The first three saved updates were replayed from their saved level sets and
state/adjoint fields. The volume fractions before evolution, after transport,
and after reinitialisation were respectively `0.783295 -> 0.803038 -> 0.804743`,
`0.804743 -> 0.811700 -> 0.813239`, and
`0.813239 -> 0.720602 -> 0.721393`. Reinitialisation therefore changes volume
by only `0.00170`, `0.00154`, and `0.00079`; it does not explain the iteration-3
gap to Gridap's volume fraction `0.73348`. That gap is already present after
transport, before reinitialisation.

At iteration 2, central finite differences of the actual native CutFEM solves
validated both assembled shape derivatives along the normalized compliance
gradient. The compliance directional derivative was `13.97927` by finite
differences and `13.97973` by the adjoint (relative error `3.3e-5`). The volume
derivative was `-4.334769` and `-4.334847` (relative error `1.8e-5`). This also
confirms the negative volume-gradient sign for the negative-`phi` material
region.

A one-iteration Gridap diagnostic was run on the same reference mesh. The
7,974 Gridap points were matched to native points by coordinates (maximum
coordinate mismatch `1.15e-16`). At the initial design, the Gridap and native
level sets differed by RMS `0.0284` with only 3 nodal sign differences. After
best-fit scalar rescaling, the compliance and volume sensitivity vectors
differed by `1.90%` and `5.40%` in relative L2. The evolution input after
Gridap's Hilbertian projection, compared with the native projected velocity,
differed by `1.12%` after normalizing both vectors; their unnormalized norms
differed by a factor of about `4.86`. Since the transport update normalizes
the velocity by its maximum magnitude, the global scale difference alone does
not explain the trajectory split. These measurements make a large initial
adjoint or projection error unlikely, but do not identify why the paths diverge
later. The next useful comparison is the sensitivity and projected evolution
input at the same later level set, especially iteration 2, before changing the
transport or reinitialization algorithms again.

That same-level-set check was completed at native iteration 2. Gridap was
given the native `levelset-002.vtu` values mapped by node coordinates, then
solved its own state and adjoint. The compliance sensitivity differed from
native by relative L2 `2.21e-8`; the volume sensitivity differed by `5.71e-15`.
Using the multiplier carried into the third update, Gridap then projected its
augmented-Lagrangian gradient. That exact Gridap evolution input was passed to
both transport implementations on the same level set. The transported fields
differed by relative L2 `5.63e-5` (maximum nodal difference `2.09e-5`), and
their volume fractions were `0.7206024622` and `0.7206024096`. This rules out
a material sensitivity or transport-operator discrepancy at that state.

The first direct reinitialisation comparison exposed a volume-quadrature bug in
the native implementation. The Duffy rule weights sum to `1/6`, while
`cell_volumes` already included the reference-tetrahedron factor `1/6`; their
product therefore underweighted both volume terms by another factor of six.
The interface penalty was unaffected. Native assembly now uses
`6 * cell_volume * reference_weight`, equivalent to `|det J|` times the
reference rule.

With this correction, independently assembled native and Gridap reinitialiser
residuals on the same post-transport field agree to relative L2 `2.05e-5`; the
maximum residual-entry difference is `8.05e-9`. Their tangent matrices agree
to relative Frobenius `1.28e-7`. The interface-penalty tangent alone agrees to
`4.48e-16`, confirming both the common physical interface rows and penalty
scaling. This localizes the previous large discrepancy to the incorrect
reference-to-physical volume scaling, rather than to a different weak form.

After solving reinitialisation again from that same field, the native and
Gridap level sets have norms `4.37128` and `4.30057`, correlation `0.999766`,
and best-scaled relative L2 difference `1.80%`. Their volume fractions are
`0.722295` and `0.721422`, with 8 nodal sign differences. On the 11,157 common
crossed background edges, the RMS difference in edge crossing fraction is
`0.0148`. The reinitialised geometry is therefore close, though not yet
identical. The earlier reinitialisation replay statistics above were generated
before correcting this quadrature factor; use the corrected run below for
trajectory comparison.

A corrected three-iteration run was compared with GridapTopOpt's saved
`Omega_act_0.vtu` through `Omega_act_3.vtu` and `history.txt`. The Gridap
level-set values were reconstructed from duplicated per-cell vertices and
matched to native nodes by coordinates; duplicate values at shared nodes
agreed within `1.1e-16`. Compliance relative differences at iterations 0–3
were `0.006%`, `0.190%`, `0.025%`, and `0.155%`. Volume fractions were
`0.783335`, `0.805085`, `0.812762`, and `0.737670` for native, compared with
Gridap's rounded `0.78331`, `0.80530`, `0.81239`, and `0.73348`. Level-set sign
mismatches were `0`, `1`, `7`, and `20` nodes. RMS interface crossing fraction
differences on common background edges were `0.00457`, `0.00643`, `0.00470`,
and `0.02553`. The correction substantially improves the short optimization
trajectory.

The corrected native optimization was then run for all 30 iterations. Its
per-iteration comparison is saved in
[`results/gridap-comparison/`](results/gridap-comparison/).
The plot `gridap-comparison.png` shows compliance, volume fraction, interface
crossing RMS, and nodal sign mismatches. The final VTU contains both
`level_set_gridap` and `level_set_native` point fields on the same background
mesh for direct ParaView comparison. Across all 31 saved states, compliance
relative error averaged `1.67%`, peaked at `7.29%` at iteration 15, and was
`0.79%` at iteration 30. Volume-fraction difference peaked at `0.0538` at
iteration 15 and ended at `0.0192`. Interface crossing RMS averaged `0.108`,
peaked at `0.224` at iteration 21, and was `0.171` at the final iteration;
nodal sign mismatches peaked at 370 and ended at 252. Thus the compliance
history remains fairly close, while the optimized interface increasingly
diverges after roughly iteration 10. The quadrature bug is fixed, but this
comparison does not support claiming that the two implementations produce the
same final shape. The remaining trajectory difference is not isolated to one
specific update detail yet.

From the skfemntv repository root, run:

```bash
PYTHONPATH=python python \
  tutorials/wheel-levelset/optimize_wheel_levelset.py \
  --iterations 30 --quadrature-order 4
```

The bundled background mesh is `tutorials/wheel-levelset/data/Wheel_3d-background.vtu`.
Results default to `tutorial-output/wheel-levelset-optimization/`:

- `optimization-history.csv`: compliance, volume fraction, multiplier,
  penalty, and maximum velocity for each iteration.
- `levelset-NNN.vtu`: the nodal level set and inside/cut/outside cell labels.
- `state-*.vtu`: the displacement field for each iteration's baseline solve
  and the final state.
- `adjoint-*.npz`: the compliance adjoint field used to assemble the
  interface shape gradient.
- `solver-logs/`: complete state-solver logs.

Each iteration performs one CutFEM state solve and one adjoint solve. The
verified adjoint-based 30-iteration run ended at compliance `2.365334` and
volume fraction `0.469552` (target `0.5`, residual `-0.030448`); it did not
satisfy the reference volume tolerance of `0.01`. The saved Gridap history at
its iteration 30 reports compliance `2.3357` and residual `-0.01682`. The
adjoint differs from `2u` by `3.0e-2` relative L2 at native iteration 30,
so the adjoint correction is no longer negligible as the design evolves. The
earlier 30-iteration four-mode diagnostic is retained at
`tutorial-output/wheel-levelset-optimization-30/`; it is
not the result of this shape-gradient update. The current 30-step run is at
`tutorial-output/wheel-levelset-optimization/`.

The state assembly comparison is documented in
[`results/gridap-comparison/README.md`](results/gridap-comparison/README.md).
