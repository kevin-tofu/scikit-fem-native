# Paris-law fatigue crack-growth demonstration

This reuses the load-controlled XFEM equilibrium solve and advances the crack
according to Paris' law

$$
\frac{da}{dN}=C(\Delta K)^m, \qquad
\Delta K=K_{\max}-K_{\min}=(1-R)K_{\max},
$$

where `R` is the minimum-to-maximum load ratio. For a prescribed crack
increment `da`, the script reports the corresponding cycle increment
`dN = da / (C (Delta K)^m)`. A threshold `Delta K_th` suppresses growth below
the selected range.

Run from the repository root:

```bash
PYTHONPATH=python python tutorials/fracture_mode_i/grow_edge_crack_fatigue.py \
  --resolution 24 \
  --initial-tip-x -0.15 \
  --growth-step 0.12 \
  --maximum-steps 4 \
  --applied-peak-strain 0.01 \
  --load-ratio 0.1 \
  --paris-c 1.0 \
  --paris-m 3.0 \
  --threshold-delta-k 0.005 \
  --output-dir tutorials/fracture_mode_i/fatigue/results
```

The `E=1` example uses dimensionless values. Its Paris coefficient is chosen
for a visible demonstration and is not a material calibration. Outputs are a
cycle-history CSV, a PNG, and one VTU per solved crack length.

This is a quasistatic, linear-elastic Paris-law example. It does not model
plasticity, overload retardation, closure, mixed-mode direction changes, or
variable-amplitude cycle counting. The stress-fit `K_I` estimator should be
replaced with an interaction integral and subjected to mesh/fit-window
convergence before quantitative use.
