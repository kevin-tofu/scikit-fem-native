"""Tet P1 benchmark matching scikit-fem's official performance.py."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict,dataclass
from pathlib import Path
import statistics
import time

import numpy as np
import skfem
from skfem.helpers import dot as reference_dot
from skfem.helpers import grad as reference_grad

import skfemntv
from skfemntv.helpers import ddot,dot,grad


@skfemntv.BilinearForm
def native_laplace(u,v,w):
    return ddot(grad(u),grad(v))


@skfemntv.LinearForm
def native_load(v,w):
    return dot(w.source,v)


@skfem.BilinearForm
def reference_laplace(u,v,w):
    return reference_dot(reference_grad(u),reference_grad(v))


@skfem.LinearForm
def reference_load(v,w):
    return v


@dataclass(frozen=True)
class Result:
    k: int
    subdivisions: int
    dofs: int
    elements: int
    skfn_cold_ms: float
    skfem_cold_ms: float
    skfn_basis_ms: float
    skfem_basis_ms: float
    skfn_warm_ms: float
    skfem_warm_ms: float


def elapsed(function):
    start=time.perf_counter()
    value=function()
    return value,(time.perf_counter()-start)*1e3


def median_time(function,repeat):
    return statistics.median(elapsed(function)[1] for _ in range(repeat))


def mesh_for_k(k):
    subdivisions=int(2**(k/3))
    axis=np.linspace(0.,1.,subdivisions)
    return subdivisions,skfemntv.MeshTet.init_tensor(axis,axis,axis)


def benchmark(k,repeat):
    subdivisions,mesh=mesh_for_k(k)
    reference_mesh=skfem.MeshTet(mesh.p,mesh.t)
    native_element=skfemntv.ElementVector(skfemntv.ElementTetP1(),dim=1)

    def native_cold():
        basis=skfemntv.Basis(mesh,native_element)
        return basis,skfemntv.asm(native_laplace,basis),skfemntv.asm(
            native_load,basis,source=np.array([1.])
        )

    def reference_cold():
        basis=skfem.Basis(reference_mesh,skfem.ElementTetP1())
        return basis,skfem.asm(reference_laplace,basis),skfem.asm(
            reference_load,basis
        )

    (basis,native_matrix,native_rhs),native_cold_ms=elapsed(native_cold)
    (reference_basis,reference_matrix,reference_rhs),reference_cold_ms=(
        elapsed(reference_cold)
    )
    difference=(native_matrix-reference_matrix).tocoo()
    error=max(
        float(np.max(np.abs(difference.data))) if difference.nnz else 0.,
        float(np.max(np.abs(native_rhs-reference_rhs))),
    )
    if error>3e-12:
        raise RuntimeError(f"implementation mismatch: {error:.3e}")

    _,native_basis_ms=elapsed(lambda:skfemntv.Basis(mesh,native_element))
    _,reference_basis_ms=elapsed(
        lambda:skfem.Basis(reference_mesh,skfem.ElementTetP1())
    )
    source=np.array([1.])
    native_warm_ms=median_time(lambda:(
        skfemntv.asm(native_laplace,basis),
        skfemntv.asm(native_load,basis,source=source),
    ),repeat)
    reference_warm_ms=median_time(lambda:(
        skfem.asm(reference_laplace,reference_basis),
        skfem.asm(reference_load,reference_basis),
    ),repeat)
    return Result(
        k,subdivisions,basis.N,mesh.nelements,
        native_cold_ms,reference_cold_ms,
        native_basis_ms,reference_basis_ms,
        native_warm_ms,reference_warm_ms,
    )


def markdown(results):
    lines=[
        "| DoFs | Elements | skfemntv cold [ms] | skfem cold [ms] | "
        "skfemntv cold speedup | skfemntv warm [ms] | skfem warm [ms] | "
        "skfemntv warm speedup |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result.dofs} | {result.elements} | "
            f"{result.skfn_cold_ms:.3f} | {result.skfem_cold_ms:.3f} | "
            f"{result.skfem_cold_ms/result.skfn_cold_ms:.2f}x | "
            f"{result.skfn_warm_ms:.3f} | {result.skfem_warm_ms:.3f} | "
            f"{result.skfem_warm_ms/result.skfn_warm_ms:.2f}x |"
        )
    return "\n".join(lines)


def write_csv(path,results):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(
            stream,fieldnames=Result.__dataclass_fields__,lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--k",nargs="+",type=int,default=list(range(6,21)))
    parser.add_argument("--repeat",type=int,default=1)
    parser.add_argument("--output",type=Path)
    arguments=parser.parse_args()
    if arguments.repeat<1:
        parser.error("repeat must be positive")
    results=[]
    for k in arguments.k:
        print(f"benchmarking k={k}...",flush=True)
        results.append(benchmark(k,arguments.repeat))
    print(markdown(results))
    if arguments.output:
        write_csv(arguments.output,results)


if __name__=="__main__":
    main()
