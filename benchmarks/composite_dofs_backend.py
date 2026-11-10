"""Compare Python and native composite H1 DOF assembly in full Basis setup."""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np

import skfemntv.basis as basis_module
from skfemntv import Basis, ElementTriP1, ElementTriP2, MeshTri, MeshTri2
from skfemntv._skfn import assemble_composite_dofs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    print("triangles  python_basis_ms  native_basis_ms  speedup")
    for divisions in (8, 32, 64):
        linear_mesh = MeshTri.init_tensor(
            np.linspace(0., 1., divisions + 1),
            np.linspace(0., 1., divisions + 1),
        )
        mesh = MeshTri2.from_mesh(linear_mesh)
        element = ElementTriP2() * ElementTriP1()
        samples = {}
        bases = {}
        for backend, kernel in (
            ("python", None), ("native", assemble_composite_dofs)
        ):
            basis_module._assemble_composite_dofs_native = kernel
            elapsed = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                basis = Basis(mesh, element, intorder=4)
                elapsed.append(time.perf_counter() - start)
            samples[backend] = statistics.median(elapsed)
            bases[backend] = basis

        python_basis = bases["python"]
        native_basis = bases["native"]
        assert python_basis.N == native_basis.N
        np.testing.assert_array_equal(python_basis.nodal_dofs, native_basis.nodal_dofs)
        np.testing.assert_array_equal(python_basis.element_dofs, native_basis.element_dofs)
        np.testing.assert_array_equal(python_basis.doflocs, native_basis.doflocs)
        for python_field, native_field in zip(
            python_basis.subbases, native_basis.subbases
        ):
            np.testing.assert_array_equal(
                python_field.element_dofs, native_field.element_dofs
            )
        print(
            f"{mesh.nelements:10d}  {samples['python'] * 1e3:15.3f}  "
            f"{samples['native'] * 1e3:14.3f}  "
            f"{samples['python'] / samples['native']:7.2f}x"
        )


if __name__ == "__main__":
    main()
