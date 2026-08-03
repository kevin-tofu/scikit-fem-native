"""Thread scaling of fixed-CSR nonmatching-interface assembly."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from time import perf_counter

import numpy as np

import skfemntv


def surface(cells):
    x=np.linspace(0.,1.,cells+1)
    points=np.array([(i,j,0.) for j in x for i in x],dtype=float).T
    triangles=[]
    for j in range(cells):
        for i in range(cells):
            lower=j*(cells+1)+i
            triangles.extend((
                (lower,lower+1,lower+cells+2),
                (lower,lower+cells+2,lower+cells+1),
            ))
    return points,np.asarray(triangles,dtype=np.int64).T


def measure(function,repeat):
    samples=[]
    for _ in range(repeat):
        start=perf_counter();function();samples.append(perf_counter()-start)
    return min(samples),sum(samples)/len(samples)


def run(cells_values,threads,repeat,multiplier):
    rows=[]
    for cells in cells_values:
        points,triangles=surface(cells)
        start=perf_counter()
        integration=skfemntv.TriangleSupermesh(
            points,triangles,points,triangles
        )
        search_seconds=perf_counter()-start
        baseline=None
        for count in threads:
            if multiplier=="coupling":
                operation=lambda count=count:integration.assemble(
                    num_threads=count
                )
            else:
                operation=lambda count=count:integration.assemble_mortar(
                    multiplier,num_threads=count
                )
            minimum,average=measure(operation,repeat)
            if baseline is None:
                baseline=minimum
            rows.append({
                "cells":cells,
                "surface_triangles":triangles.shape[1],
                "overlap_cells":integration.diagnostics.integration_triangle_count,
                "mode":multiplier,
                "threads":count,
                "search_seconds":search_seconds,
                "assembly_min_seconds":minimum,
                "assembly_average_seconds":average,
                "speedup":baseline/minimum,
            })
    return rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--cells",default="32,64,128")
    parser.add_argument("--threads",default="1,2,4")
    parser.add_argument("--repeat",type=int,default=3)
    parser.add_argument(
        "--multiplier",default="coupling",
        choices=(
            "coupling","slave","master","overlap_p0",
            "slave_facet_p0","master_facet_p0","dual",
        ),
    )
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    cells=[int(value) for value in args.cells.split(",")]
    threads=[int(value) for value in args.threads.split(",")]
    rows=run(cells,threads,args.repeat,args.multiplier)
    columns=tuple(rows[0])
    print(" | ".join(columns))
    for row in rows:
        print(" | ".join(
            f"{row[column]:.6g}" if isinstance(row[column],float)
            else str(row[column]) for column in columns
        ))
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=columns)
            writer.writeheader();writer.writerows(rows)


if __name__=="__main__":
    main()
