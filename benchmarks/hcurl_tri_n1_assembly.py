"""Benchmark the experimental TriN1 vertical slice against scikit-fem."""

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
        start=time.perf_counter()
        call()
        samples.append(time.perf_counter()-start)
    return statistics.median(samples)


def _case(resolution,repeats,warmup):
    coordinates=np.linspace(0.,1.,resolution+1)
    mesh=skfemntv.MeshTri.init_tensor(coordinates,coordinates)
    start=time.perf_counter()
    basis=skfemntv.AffineTriN1Basis(mesh,intorder=3)
    native_basis_seconds=time.perf_counter()-start
    start=time.perf_counter()
    assembler=skfemntv.TriN1Assembler(basis)
    native_setup_seconds=time.perf_counter()-start

    reference_mesh=skfem.MeshTri(mesh.p,mesh.t[:3])
    start=time.perf_counter()
    reference=skfem.Basis(reference_mesh,skfem.ElementTriN1(),intorder=3)
    skfem_basis_seconds=time.perf_counter()-start

    @skfem.BilinearForm
    def maxwell(u,v,w):
        return dot(u,v)+.2*curl(u)*curl(v)

    native_total=_median(
        lambda:assembler.assemble_maxwell(
            mass_coefficient=1.,curl_coefficient=.2
        ),repeats,warmup,
    )
    skfem_total=_median(
        lambda:skfem.asm(maxwell,reference),repeats,warmup
    )

    def integrate():
        mass=np.ones_like(basis.dx)
        curl_value=np.full_like(basis.dx,.2)
        elements=np.einsum(
            "ebiq,eciq,eq->ebc",
            basis._element_values,basis._element_values,basis.dx*mass,
            optimize=True,
        )
        elements+=np.einsum(
            "ebq,ecq,eq->ebc",
            basis._element_curls,basis._element_curls,basis.dx*curl_value,
            optimize=True,
        )
        return elements

    elements=integrate()
    integration=_median(integrate,repeats,warmup)
    scatter=_median(lambda:assembler._assemble_elements(elements),repeats,warmup)
    return {
        "resolution":resolution,
        "elements":mesh.nelements,
        "dofs":basis.N,
        "nnz":assembler._matrix.nnz,
        "native_basis_seconds":native_basis_seconds,
        "native_setup_seconds":native_setup_seconds,
        "native_assembly_seconds":native_total,
        "native_integration_seconds":integration,
        "native_scatter_seconds":scatter,
        "skfem_basis_seconds":skfem_basis_seconds,
        "skfem_assembly_seconds":skfem_total,
        "assembly_speedup":skfem_total/native_total,
        "integration_fraction":integration/native_total,
        "scatter_fraction":scatter/native_total,
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--resolutions",type=int,nargs="+",default=[16,32,64])
    parser.add_argument("--repeat",type=int,default=7)
    parser.add_argument("--warmup",type=int,default=2)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    rows=[_case(value,args.repeat,args.warmup) for value in args.resolutions]
    fields=tuple(rows[0])
    if args.output is not None:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=fields)
            writer.writeheader();writer.writerows(rows)
    print("resolution dofs elements native-ms skfem-ms speedup integrate% scatter%")
    for row in rows:
        print(
            f"{row['resolution']:10d} {row['dofs']:5d} {row['elements']:8d} "
            f"{1e3*row['native_assembly_seconds']:9.3f} "
            f"{1e3*row['skfem_assembly_seconds']:8.3f} "
            f"{row['assembly_speedup']:7.2f} "
            f"{100*row['integration_fraction']:9.1f} "
            f"{100*row['scatter_fraction']:8.1f}"
        )


if __name__=="__main__":
    main()
