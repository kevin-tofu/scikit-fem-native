"""Compare cached CSR tangent scatter with the previous COO conversion path."""

from __future__ import annotations

import argparse
import csv
import io
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

import skfemntv
from skfemntv._skfn import (
    assemble_nonlinear_global,
    evaluate_j2_plasticity,
    evaluate_neo_hookean,
)


FAMILIES = ("tet4", "tet10", "hex8", "hex20", "hex27")
MATERIALS = ("neo_hookean", "j2")
FIELDS = (
    "material", "family", "grid", "elements", "dofs", "local_matrix_entries",
    "assembler_init_ms", "native_kernel_ms", "cpp_scatter_only_ms",
    "python_scatter_only_ms", "coo_scatter_only_ms", "cpp_total_ms",
    "python_cached_total_ms", "coo_total_ms", "coo_over_cpp",
    "python_over_cpp",
    "cached_graph_bytes", "coo_row_column_bytes_per_eval",
)


def make_basis(family: str, grid: int):
    coordinates = np.linspace(0.0, 1.0, grid + 1)
    if family.startswith("tet"):
        linear = skfemntv.MeshTet.init_tensor(coordinates, coordinates, coordinates)
        if family == "tet4":
            mesh, element = linear, skfemntv.ElementTetP1()
        else:
            mesh, element = skfemntv.MeshTet2.from_mesh(linear), skfemntv.ElementTetP2()
    else:
        linear = skfemntv.MeshHex.init_tensor(coordinates, coordinates, coordinates)
        if family == "hex8":
            mesh, element = linear, skfemntv.ElementHex1()
        elif family == "hex20":
            mesh, element = skfemntv.MeshHex20.from_mesh(linear), skfemntv.ElementHex20()
        else:
            mesh, element = skfemntv.MeshHex2.from_mesh(linear), skfemntv.ElementHex2()
    return skfemntv.Basis(
        mesh, skfemntv.ElementVector(element, dim=3), intorder=5
    )


def affine_displacement(basis, amount: float):
    displacement = np.zeros(basis.N)
    for component, dofs in enumerate(basis.split_indices()):
        displacement[dofs] = amount * basis.doflocs[component, dofs]
    return displacement


def make_assembler(basis, material: str):
    if material == "neo_hookean":
        return skfemntv.NeoHookeanAssembler(basis, mu=2.4, lambda_=3.1)
    return skfemntv.J2PlasticityAssembler(
        basis,
        young_modulus=1000.0,
        poisson_ratio=0.25,
        yield_stress=2.0,
        hardening_modulus=20.0,
    )


def legacy_coo_evaluate(model, material: str, displacement, committed_state):
    if material == "neo_hookean":
        local_residual, local_tangent = evaluate_neo_hookean(
            model._dofs,
            model._gradients,
            model._weights,
            displacement,
            model.mu,
            model.lambda_,
        )[:2]
    else:
        local_residual, local_tangent = evaluate_j2_plasticity(
            model._dofs,
            model._gradients,
            model._weights,
            displacement,
            committed_state,
            model.lambda_,
            model.mu,
            model.yield_stress,
            model.hardening_modulus,
        )[:2]

    dofs = model._dofs.reshape(model.basis.nelems, -1)
    residual = np.zeros(model.basis.N)
    np.add.at(residual, dofs.reshape(-1), local_residual.reshape(-1))
    rows = np.broadcast_to(dofs[:, :, None], local_tangent.shape).reshape(-1)
    columns = np.broadcast_to(dofs[:, None, :], local_tangent.shape).reshape(-1)
    tangent = coo_matrix(
        (local_tangent.reshape(-1), (rows, columns)),
        shape=(model.basis.N, model.basis.N),
    ).tocsr()
    return residual, tangent


def python_cached_evaluate(model, material: str, displacement, committed_state):
    """Reference the CSR-graph cache with Python residual/value aggregation."""
    if material == "neo_hookean":
        local_residual, local_tangent = evaluate_neo_hookean(
            model._dofs, model._gradients, model._weights,
            displacement, model.mu, model.lambda_,
        )[:2]
    else:
        local_residual, local_tangent = evaluate_j2_plasticity(
            model._dofs, model._gradients, model._weights, displacement,
            committed_state, model.lambda_, model.mu,
            model.yield_stress, model.hardening_modulus,
        )[:2]
    dofs = model._dofs.reshape(model.basis.nelems, -1)
    residual = np.zeros(model.basis.N)
    np.add.at(residual, dofs.reshape(-1), local_residual.reshape(-1))
    indptr, indices, scatter = model._tangent_graph
    data = np.bincount(
        scatter.reshape(-1), weights=local_tangent.reshape(-1), minlength=indices.size
    )
    tangent = csr_matrix(
        (data, indices.copy(), indptr.copy()), shape=(model.basis.N, model.basis.N)
    )
    return residual, tangent


def evaluate(model, material: str, displacement, committed_state):
    if material == "neo_hookean":
        result = model.evaluate(displacement)
    else:
        result = model.evaluate(displacement, committed_state)
    return result.residual, result.tangent


def native_local_kernel(model, material: str, displacement, committed_state):
    if material == "neo_hookean":
        return evaluate_neo_hookean(
            model._dofs, model._gradients, model._weights,
            displacement, model.mu, model.lambda_,
        )
    return evaluate_j2_plasticity(
        model._dofs, model._gradients, model._weights, displacement,
        committed_state, model.lambda_, model.mu,
        model.yield_stress, model.hardening_modulus,
    )


def cpp_scatter_only(model, local_residual, local_tangent):
    dofs = model._dofs.reshape(model.basis.nelems, -1)
    indptr, indices, scatter = model._tangent_graph
    residual, data = assemble_nonlinear_global(
        dofs, local_residual, local_tangent, scatter,
        model.basis.N, indices.size,
    )
    tangent = csr_matrix(
        (data, indices.copy(), indptr.copy()), shape=(model.basis.N, model.basis.N)
    )
    return residual, tangent


def python_scatter_only(model, local_residual, local_tangent):
    dofs = model._dofs.reshape(model.basis.nelems, -1)
    residual = np.zeros(model.basis.N)
    np.add.at(residual, dofs.reshape(-1), local_residual.reshape(-1))
    indptr, indices, scatter = model._tangent_graph
    data = np.bincount(
        scatter.reshape(-1), weights=local_tangent.reshape(-1), minlength=indices.size
    )
    tangent = csr_matrix(
        (data, indices.copy(), indptr.copy()), shape=(model.basis.N, model.basis.N)
    )
    return residual, tangent


def coo_scatter_only(model, local_residual, local_tangent):
    dofs = model._dofs.reshape(model.basis.nelems, -1)
    residual = np.zeros(model.basis.N)
    np.add.at(residual, dofs.reshape(-1), local_residual.reshape(-1))
    rows = np.broadcast_to(dofs[:, :, None], local_tangent.shape).reshape(-1)
    columns = np.broadcast_to(dofs[:, None, :], local_tangent.shape).reshape(-1)
    tangent = coo_matrix(
        (local_tangent.reshape(-1), (rows, columns)),
        shape=(model.basis.N, model.basis.N),
    ).tocsr()
    return residual, tangent


def median_ms(function, warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        function()
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples)


def run_case(material: str, family: str, grid: int, warmup: int, repeat: int):
    basis = make_basis(family, grid)
    start = time.perf_counter()
    model = make_assembler(basis, material)
    init_ms = (time.perf_counter() - start) * 1000.0
    displacement = affine_displacement(basis, 0.01)
    committed = model.initial_state() if material == "j2" else None
    cached = lambda: evaluate(model, material, displacement, committed)
    python_cached = lambda: python_cached_evaluate(
        model, material, displacement, committed
    )
    coo = lambda: legacy_coo_evaluate(model, material, displacement, committed)

    expected_residual, expected_tangent = coo()
    actual_residual, actual_tangent = cached()
    np.testing.assert_allclose(actual_residual, expected_residual, rtol=5e-12, atol=5e-12)
    np.testing.assert_allclose(
        actual_tangent.toarray(), expected_tangent.toarray(), rtol=5e-12, atol=5e-12
    )
    python_residual, python_tangent = python_cached()
    np.testing.assert_allclose(actual_residual, python_residual, rtol=5e-12, atol=5e-12)
    np.testing.assert_allclose(
        actual_tangent.toarray(), python_tangent.toarray(), rtol=5e-12, atol=5e-12
    )

    local_response = native_local_kernel(model, material, displacement, committed)
    local_residual, local_tangent = local_response[:2]
    cpp_scatter = lambda: cpp_scatter_only(model, local_residual, local_tangent)
    python_scatter = lambda: python_scatter_only(model, local_residual, local_tangent)
    coo_scatter = lambda: coo_scatter_only(model, local_residual, local_tangent)
    for scatter_response in (cpp_scatter(), python_scatter(), coo_scatter()):
        np.testing.assert_allclose(scatter_response[0], actual_residual, rtol=5e-12, atol=5e-12)
        np.testing.assert_allclose(
            scatter_response[1].toarray(), actual_tangent.toarray(), rtol=5e-12, atol=5e-12
        )

    kernel_ms = median_ms(
        lambda: native_local_kernel(model, material, displacement, committed), warmup, repeat
    )
    cpp_scatter_ms = median_ms(cpp_scatter, warmup, repeat)
    python_scatter_ms = median_ms(python_scatter, warmup, repeat)
    coo_scatter_ms = median_ms(coo_scatter, warmup, repeat)
    cpp_ms = median_ms(cached, warmup, repeat)
    python_ms = median_ms(python_cached, warmup, repeat)
    coo_ms = median_ms(coo, warmup, repeat)
    graph_bytes = sum(array.nbytes for array in model._tangent_graph)
    local_entries = model._dofs.shape[0] * (3 * model._dofs.shape[1]) ** 2
    return {
        "material": material,
        "family": family,
        "grid": grid,
        "elements": basis.nelems,
        "dofs": basis.N,
        "local_matrix_entries": local_entries,
        "assembler_init_ms": init_ms,
        "native_kernel_ms": kernel_ms,
        "cpp_scatter_only_ms": cpp_scatter_ms,
        "python_scatter_only_ms": python_scatter_ms,
        "coo_scatter_only_ms": coo_scatter_ms,
        "cpp_total_ms": cpp_ms,
        "python_cached_total_ms": python_ms,
        "coo_total_ms": coo_ms,
        "coo_over_cpp": coo_ms / cpp_ms,
        "python_over_cpp": python_ms / cpp_ms,
        "cached_graph_bytes": graph_bytes,
        "coo_row_column_bytes_per_eval": local_entries * 2 * np.dtype(np.int64).itemsize,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, default=2, help="cells per axis")
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=FAMILIES)
    parser.add_argument("--materials", nargs="+", choices=MATERIALS, default=MATERIALS)
    parser.add_argument("--output", type=Path, help="CSV output path; defaults to stdout")
    args = parser.parse_args()
    if args.grid < 1 or args.repeat < 1 or args.warmup < 0:
        parser.error("grid and repeat must be positive; warmup must be nonnegative")

    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=FIELDS)
    writer.writeheader()
    for material in args.materials:
        for family in args.families:
            writer.writerow(run_case(material, family, args.grid, args.warmup, args.repeat))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output.getvalue())
    else:
        sys.stdout.write(output.getvalue())


if __name__ == "__main__":
    main()
