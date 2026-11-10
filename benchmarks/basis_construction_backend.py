"""Measure the Hex27 tabulation contribution to full Basis construction."""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np

import skfemntv.basis as basis_module
from skfemntv import Basis, ElementHex2, MeshHex, MeshHex2
from skfemntv._skfn import tabulate_hex27_shapes


def python_hex27(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shape = np.empty((points.shape[1], 27))
    gradient = np.empty((points.shape[1], 27, 3))
    for q, (x, y, z) in enumerate(points.T):
        values = tuple(
            (2. * (v - .5) * (v - 1.), 4. * v * (1. - v),
                2. * v * (v - .5)) for v in (x, y, z)
        )
        derivatives = tuple(
            (4. * v - 3., 4. - 8. * v, 4. * v - 1.) for v in (x, y, z)
        )
        node = 0
        for k in range(3):
            for j in range(3):
                for i in range(3):
                    shape[q, node] = values[0][i] * values[1][j] * values[2][k]
                    gradient[q, node] = (
                        derivatives[0][i] * values[1][j] * values[2][k],
                        values[0][i] * derivatives[1][j] * values[2][k],
                        values[0][i] * values[1][j] * derivatives[2][k],
                    )
                    node += 1
    return shape, gradient


def basis_volume(basis: Basis) -> float:
    return float(np.sum(basis._geometry_determinants * basis.W[None, :]))


def median_build(mesh: MeshHex2, order: int, repeats: int, native: bool):
    basis_module.tabulate_hex27_shapes = (
        tabulate_hex27_shapes if native else python_hex27
    )
    elapsed = []
    reference_volume = None
    for _ in range(repeats):
        start = time.perf_counter()
        basis = Basis(mesh, ElementHex2(), intorder=order)
        elapsed.append(time.perf_counter() - start)
        volume = basis_volume(basis)
        if reference_volume is None:
            reference_volume = volume
        else:
            np.testing.assert_allclose(volume, reference_volume, rtol=2e-14, atol=2e-14)
    return statistics.median(elapsed), reference_volume


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    print("hexes  intorder  qpoints  python_basis_ms  native_basis_ms  speedup  volume")
    for divisions, order in ((1, 4), (1, 8), (1, 12), (4, 8)):
        linear = MeshHex.init_tensor(
            np.linspace(0., 1., divisions + 1),
            np.linspace(0., 1., divisions + 1),
            np.linspace(0., 1., divisions + 1),
        )
        mesh = MeshHex2.from_mesh(linear)
        python_time, python_volume = median_build(mesh, order, args.repeats, False)
        native_time, native_volume = median_build(mesh, order, args.repeats, True)
        np.testing.assert_allclose(native_volume, python_volume, rtol=2e-14, atol=2e-14)
        nq = Basis(mesh, ElementHex2(), intorder=order).X.shape[1]
        print(
            f"{mesh.nelements:5d}  {order:8d}  {nq:7d}  "
            f"{python_time * 1e3:15.3f}  {native_time * 1e3:14.3f}  "
            f"{python_time / native_time:7.3f}x  {native_volume:.12g}"
        )


if __name__ == "__main__":
    main()
