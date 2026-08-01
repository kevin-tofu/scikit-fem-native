"""Reproducible single-thread assembly benchmark.

The linear comparison performs equivalent work: tangent assembly plus one
matrix-vector product in scikit-fem, and residual+tangent assembly in skfemntv.
Initialization and sparsity construction are intentionally outside timings.
"""

import argparse
import statistics
import time

import numpy as np
from skfem import (
    Basis,
    ElementHex1,
    ElementTetP1,
    ElementVector,
    MeshHex,
    MeshTet,
)
from skfem.models.elasticity import linear_elasticity

from skfemntv import LinearElasticity, NativeAssembler, NeoHookean
from skfem_neo_hookean import forms as neo_hookean_forms


def median_seconds(function, repeats):
    for _ in range(2):
        function()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def make_basis(topology, points):
    axis = np.linspace(0.0, 1.0, points)
    if topology == "tet":
        return Basis(
            MeshTet.init_tensor(axis, axis, axis),
            ElementVector(ElementTetP1()),
            intorder=2,
        )
    return Basis(
        MeshHex.init_tensor(axis, axis, axis),
        ElementVector(ElementHex1()),
        intorder=2,
    )


def run(topology, points, repeats):
    basis = make_basis(topology, points)
    young, poisson = 100.0, 0.3
    lmbda = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
    mu = young / (2 * (1 + poisson))
    form = linear_elasticity(Lambda=lmbda, Mu=mu)
    linear = NativeAssembler.from_skfem(
        basis, LinearElasticity(young, poisson)
    )
    nonlinear = NativeAssembler.from_skfem(
        basis, NeoHookean(mu, lmbda)
    )
    reference_residual, reference_tangent = neo_hookean_forms(mu, lmbda)
    u = np.random.default_rng(0).normal(scale=1e-3, size=basis.N)

    def skfem_linear():
        tangent = form.assemble(basis)
        return tangent @ u

    skfem_time = median_seconds(skfem_linear, repeats)
    native_time = median_seconds(lambda: linear.evaluate(u), repeats)
    residual_time = median_seconds(
        lambda: nonlinear.evaluate(u, mode="residual"), repeats
    )
    nonlinear_time = median_seconds(lambda: nonlinear.evaluate(u), repeats)

    def skfem_nonlinear():
        field = basis.interpolate(u)
        residual = reference_residual.assemble(
            basis, displacement=field
        )
        tangent = reference_tangent.assemble(
            basis, displacement=field
        )
        return residual, tangent

    skfem_nonlinear_time = median_seconds(skfem_nonlinear, repeats)
    print(
        f"{topology:3} points={points:2} "
        f"elements={basis.mesh.nelements:7} dofs={basis.N:7} "
        f"nnz={linear.tangent.nnz:8} "
        f"skfem={1e3*skfem_time:9.3f}ms "
        f"skfemntv={1e3*native_time:8.3f}ms "
        f"speedup={skfem_time/native_time:7.1f}x "
        f"neo-R={1e3*residual_time:8.3f}ms "
        f"neo-RK={1e3*nonlinear_time:8.3f}ms "
        f"skfem-neo-RK={1e3*skfem_nonlinear_time:9.3f}ms "
        f"neo-speedup={skfem_nonlinear_time/nonlinear_time:7.1f}x"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--tet", type=int, nargs="*", default=[6, 10])
    parser.add_argument("--hex", type=int, nargs="*", default=[5, 8, 11])
    args = parser.parse_args()
    for value in args.tet:
        run("tet", value, args.repeats)
    for value in args.hex:
        run("hex", value, args.repeats)
