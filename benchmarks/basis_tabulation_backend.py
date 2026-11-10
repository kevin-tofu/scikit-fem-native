"""Compare Python and native Hex27 reference basis table generation."""

from __future__ import annotations

import argparse
import timeit

import numpy as np

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    print("qpoints  python_ms  native_ms  speedup")
    for count in (27, 125, 343, 729):
        points = np.random.default_rng(20261016).random((3, count))
        expected = python_hex27(points)
        actual = tabulate_hex27_shapes(points)
        np.testing.assert_allclose(actual[0], expected[0], rtol=2e-15, atol=2e-15)
        np.testing.assert_allclose(actual[1], expected[1], rtol=2e-15, atol=2e-15)
        python_ms = 1e3 * timeit.timeit(lambda: python_hex27(points), number=args.repeats) / args.repeats
        native_ms = 1e3 * timeit.timeit(lambda: tabulate_hex27_shapes(points), number=args.repeats) / args.repeats
        print(f"{count:7d}  {python_ms:9.3f}  {native_ms:9.3f}  {python_ms / native_ms:7.1f}x")


if __name__ == "__main__":
    main()
