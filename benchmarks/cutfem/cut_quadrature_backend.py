"""Compare Tri3 cut-quadrature generation in Python and C++."""

from __future__ import annotations

import argparse
import importlib
import statistics
import time

import numpy as np

import skfemntv


def median_seconds(function, repeats):
    function()
    samples=[]
    for _ in range(repeats):
        start=time.perf_counter()
        function()
        samples.append(time.perf_counter()-start)
    return statistics.median(samples)


def benchmark(resolution, repeats, intorder):
    levelset_module=importlib.import_module("skfemntv.levelset")
    native_kernel=levelset_module.cut_triangle_quadrature
    if native_kernel is None:
        raise RuntimeError("rebuild the C++ extension to run this benchmark")
    coordinates=np.linspace(0.,1.,resolution+1)
    mesh=skfemntv.MeshTri.init_tensor(coordinates,coordinates)
    level_set=skfemntv.LevelSet(
        lambda x:x[0]+.37*x[1]-.413,tolerance=0.
    )
    try:
        levelset_module.cut_triangle_quadrature=None
        python_time=median_seconds(
            lambda:level_set.cut_quadrature(mesh,intorder=intorder),repeats
        )
        levelset_module.cut_triangle_quadrature=native_kernel
        native_result=level_set.cut_quadrature(mesh,intorder=intorder)
        native_time=median_seconds(
            lambda:level_set.cut_quadrature(mesh,intorder=intorder),repeats
        )
    finally:
        levelset_module.cut_triangle_quadrature=native_kernel
    return {
        "resolution":resolution,
        "cells":mesh.nelements,
        "cut_cells":native_result.diagnostics.cut_cell_count,
        "quadrature_points":native_result.diagnostics.quadrature_point_count,
        "python_ms":python_time*1.e3,
        "cpp_ms":native_time*1.e3,
        "speedup":python_time/native_time,
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--resolutions",type=int,nargs="+",default=[32,64,128])
    parser.add_argument("--repeat",type=int,default=5)
    parser.add_argument("--intorder",type=int,default=2)
    args=parser.parse_args()
    if args.repeat<1 or args.intorder<1 or any(n<1 for n in args.resolutions):
        parser.error("resolutions, repeat, and intorder must be positive")
    print("resolution cells cut_cells qpoints Python-ms C++-ms speedup")
    for resolution in args.resolutions:
        row=benchmark(resolution,args.repeat,args.intorder)
        print(
            f"{row['resolution']:10d} {row['cells']:7d} "
            f"{row['cut_cells']:9d} {row['quadrature_points']:7d} "
            f"{row['python_ms']:9.3f} {row['cpp_ms']:8.3f} "
            f"{row['speedup']:7.1f}x"
        )


if __name__=="__main__":
    main()
