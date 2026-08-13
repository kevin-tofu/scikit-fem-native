"""Benchmark the experimental TetN1 vertical slice against scikit-fem."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics
import time

import numpy as np
import skfem
from skfem.helpers import curl,dot

import skfemntv


def _median(call,repeats,warmup):
    for _ in range(warmup):
        call()
    samples=[]
    for _ in range(repeats):
        started=time.perf_counter()
        call()
        samples.append(time.perf_counter()-started)
    return statistics.median(samples)


def _case(resolution,repeats,warmup):
    axis=np.linspace(0.,1.,resolution+1)
    mesh=skfemntv.MeshTet.init_tensor(axis,axis,axis)
    started=time.perf_counter()
    basis=skfemntv.AffineTetN1Basis(mesh,intorder=3)
    basis_seconds=time.perf_counter()-started
    started=time.perf_counter()
    assembler=skfemntv.TetN1Assembler(basis)
    setup_seconds=time.perf_counter()-started

    reference_mesh=skfem.MeshTet(mesh.p,mesh.t[:4])
    started=time.perf_counter()
    reference=skfem.Basis(
        reference_mesh,skfem.ElementTetN1(),intorder=3
    )
    reference_basis_seconds=time.perf_counter()-started

    @skfem.BilinearForm
    def maxwell(u,v,w):
        return dot(u,v)+.2*dot(curl(u),curl(v))

    assembly_seconds=_median(
        lambda:assembler.assemble_maxwell(
            mass_coefficient=1.,curl_coefficient=.2
        ),repeats,warmup,
    )
    reference_assembly_seconds=_median(
        lambda:skfem.asm(maxwell,reference),repeats,warmup
    )
    def integrate():
        elements=np.einsum(
            "ebiq,eciq,eq->ebc",
            basis._element_values,basis._element_values,basis.dx,
            optimize=True,
        )
        elements+=.2*np.einsum(
            "ebiq,eciq,eq->ebc",
            basis._element_curls,basis._element_curls,basis.dx,
            optimize=True,
        )
        return elements

    elements=integrate()
    integration_seconds=_median(integrate,repeats,warmup)
    scatter_seconds=_median(
        lambda:assembler._assemble_elements(elements),repeats,warmup
    )
    estimate=assembler.memory_estimate
    return {
        "resolution":resolution,
        "elements":mesh.nelements,
        "dofs":basis.N,
        "nnz":assembler._matrix.nnz,
        "basis_seconds":basis_seconds,
        "setup_seconds":setup_seconds,
        "assembly_seconds":assembly_seconds,
        "skfem_basis_seconds":reference_basis_seconds,
        "skfem_assembly_seconds":reference_assembly_seconds,
        "assembly_speedup":reference_assembly_seconds/assembly_seconds,
        "integration_seconds":integration_seconds,
        "scatter_seconds":scatter_seconds,
        "integration_fraction":integration_seconds/assembly_seconds,
        "scatter_fraction":scatter_seconds/assembly_seconds,
        "basis_bytes":estimate.basis_bytes,
        "peak_total_bytes_estimated":estimate.construction_peak_total_bytes_upper_bound,
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--resolutions",type=int,nargs="+",default=[4,8,12])
    parser.add_argument("--repeat",type=int,default=7)
    parser.add_argument("--warmup",type=int,default=2)
    parser.add_argument("--output",type=Path)
    arguments=parser.parse_args()
    rows=[
        _case(value,arguments.repeat,arguments.warmup)
        for value in arguments.resolutions
    ]
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True,exist_ok=True)
        with arguments.output.open("w",newline="",encoding="utf-8") as stream:
            writer=csv.DictWriter(stream,fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(
        "resolution dofs elements basis-ms setup-ms assembly-ms skfem-ms "
        "speedup integrate% scatter%"
    )
    for row in rows:
        print(
            f"{row['resolution']:10d} {row['dofs']:7d} {row['elements']:8d} "
            f"{1e3*row['basis_seconds']:8.2f} {1e3*row['setup_seconds']:8.2f} "
            f"{1e3*row['assembly_seconds']:11.3f} "
            f"{1e3*row['skfem_assembly_seconds']:9.3f} "
            f"{row['assembly_speedup']:7.2f} "
            f"{100*row['integration_fraction']:9.1f} "
            f"{100*row['scatter_fraction']:8.1f}"
        )


if __name__=="__main__":
    main()
