"""Compare Python and native facet topology during Tri3 refinement."""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np

import skfemntv.basis as basis_module
from skfemntv import MeshTri
from skfemntv._skfn import build_tri_facet_topology


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    print("input_triangles  python_refine_ms  native_refine_ms  speedup")
    for divisions in (32, 64, 128):
        mesh = MeshTri.init_tensor(
            np.linspace(0., 1., divisions + 1),
            np.linspace(0., 1., divisions + 1),
        )
        inputs = {}
        elapsed = {}
        for backend, builder in (
            ("python", None), ("native", build_tri_facet_topology)
        ):
            basis_module._build_tri_facet_topology_native = builder
            samples = []
            refined = None
            for _ in range(args.repeats):
                fresh = MeshTri(mesh.p, mesh.t)
                start = time.perf_counter()
                refined = fresh.refined(1)
                samples.append(time.perf_counter() - start)
            elapsed[backend] = statistics.median(samples)
            inputs[backend] = refined

        np.testing.assert_array_equal(inputs["native"].p, inputs["python"].p)
        np.testing.assert_array_equal(inputs["native"].t, inputs["python"].t)
        print(
            f"{mesh.nelements:15d}  {elapsed['python'] * 1e3:16.3f}  "
            f"{elapsed['native'] * 1e3:16.3f}  "
            f"{elapsed['python'] / elapsed['native']:7.2f}x"
        )


if __name__ == "__main__":
    main()
