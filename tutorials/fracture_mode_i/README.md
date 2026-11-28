# Load-controlled Mode-I crack growth with XFEM

This tutorial extends the stationary edge-crack example to a sequence of
equilibrium solves. A vertical displacement is prescribed on the top and
bottom boundaries of a plane-strain plate. The crack is represented by a
level set and shifted-Heaviside plus Mode-I crack-tip enrichment; the triangle
background mesh remains unchanged as the tip advances.

After each solve, the tutorial fits the near-tip `sigma_yy` field to the
Mode-I Williams stress field and estimates `K_I`. The propagation rule is
`K_I >= K_IC`. When it is met, the crack tip advances by the configured
`growth_step`, then the cut-cell quadrature and enriched basis are rebuilt for
the next state. This intentionally models only straight, symmetric Mode-I
growth; it is not a mixed-mode direction law or a general-purpose fracture
solver. The stress-fit estimate is a teaching diagnostic and should be
replaced by a contour-independent interaction integral before quantitative
engineering use.

## Run

From the scikit-fem-native repository root:

```bash
PYTHONPATH=python python tutorials/fracture_mode_i/grow_edge_crack.py \
  --resolution 24 \
  --initial-tip-x -0.15 \
  --growth-step 0.12 \
  --toughness 0.012 \
  --maximum-steps 4 \
  --output-dir tutorials/fracture_mode_i/results
```

The CSV records each tip position, fitted `K_I`, toughness threshold, whether
the criterion was met, and whether the run advanced to another state. A VTU
file is written for each solved state so the crack extension can be inspected
in ParaView; the PNG summarizes `K_I` and tip position over the run.

## Recorded demonstration

The checked-in run uses the command above. With these dimensionless settings,
the fitted `K_I` stays between `0.01349` and `0.01388`, above `K_IC=0.012`,
through four tip advances from `x=-0.15` to `x=0.33`. The free-equation
residual is about `2e-16` at each step. `maximum_steps=4` stops the demonstration
after that fourth advance; the final CSV row therefore still meets the
criterion but does not schedule a fifth advance.

`K_IC` and `K_I` use the same dimensionless unit system as the example's
`E=1` material and imposed strain. The default values are chosen to show a few
growth increments on the bundled mesh, not to represent a specific material.

## Scope and next step

The benchmark checks that the XFEM crack representation, equilibrium solve,
stress extraction, and level-set tip update work together. It does not include
plasticity, crack branching, remeshing, or a mixed-mode path law. The companion
[Paris-law fatigue tutorial](fatigue/README.md) reuses these equilibrium states
with cyclic load range and a cycle counter. Both examples remain educational
demonstrations until the stress-fit estimator has mesh and fit-window
convergence against an interaction integral.
