#!/usr/bin/env bash
set -euo pipefail

repeat="${REPEAT:-1}"
points="${POINTS:-2 3 4}"
intorder="${INTORDER:-4}"
results="benchmarks/nonlinear-assembly/results"

for topology in tet4 tet10 hex8 hex27 wedge6; do
  python benchmarks/nonlinear-assembly/mesh_order_sweep.py \
    --topology "${topology}" \
    --intorder "${intorder}" \
    --points ${points} \
    --repeat "${repeat}" \
    --distorted \
    --output "${results}/neo-hookean-${topology}-order-sweep.csv" \
    --plot-output "${results}/neo-hookean-${topology}-order-sweep.png"
done
