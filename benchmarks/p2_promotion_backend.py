"""Compare native and Python P1 triangle to P2 mesh promotion."""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np

import skfemntv.basis as basis_module
from skfemntv import Basis, ElementTriP2, MeshTri
from skfemntv._skfn import promote_tri_p2_mesh


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    print("triangles  python_promotion_ms  native_promotion_ms  speedup")
    for divisions in (8, 32, 64):
        mesh = MeshTri.init_tensor(
            np.linspace(0., 1., divisions + 1),
            np.linspace(0., 1., divisions + 1),
        )
        times = {}
        outputs = {}
        for backend, kernel in (
            ("python", None), ("native", promote_tri_p2_mesh)
        ):
            basis_module._promote_tri_p2_mesh_native = kernel
            elapsed = []
            result = None
            for _ in range(args.repeats):
                start = time.perf_counter()
                result = basis_module._promote_linear_p2_mesh(
                    mesh, basis_module.MeshTri2
                )
                elapsed.append(time.perf_counter() - start)
            times[backend] = statistics.median(elapsed)
            outputs[backend] = result

        np.testing.assert_array_equal(outputs["native"].p, outputs["python"].p)
        np.testing.assert_array_equal(outputs["native"].t, outputs["python"].t)
        print(
            f"{mesh.nelements:10d}  {times['python'] * 1e3:19.3f}  "
            f"{times['native'] * 1e3:19.3f}  "
            f"{times['python'] / times['native']:7.2f}x"
        )

    # Include full Basis setup at a moderate size to expose downstream costs.
    mesh = MeshTri.init_tensor(
        np.linspace(0., 1., 33), np.linspace(0., 1., 33)
    )
    elapsed = {}
    dofs = {}
    for backend, kernel in (("python", None), ("native", promote_tri_p2_mesh)):
        basis_module._promote_tri_p2_mesh_native = kernel
        samples = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            basis = Basis(mesh, ElementTriP2(), intorder=4)
            samples.append(time.perf_counter() - start)
        elapsed[backend] = statistics.median(samples)
        dofs[backend] = basis.element_dofs.copy()
    np.testing.assert_array_equal(dofs["native"], dofs["python"])
    print(
        f"full Basis ({mesh.nelements} triangles): "
        f"Python {elapsed['python'] * 1e3:.3f} ms, "
        f"C++ {elapsed['native'] * 1e3:.3f} ms, "
        f"{elapsed['python'] / elapsed['native']:.2f}x"
    )


if __name__ == "__main__":
    main()
