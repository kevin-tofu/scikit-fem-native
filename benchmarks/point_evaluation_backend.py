"""Compare point evaluation through the Python and native triangle kernels."""

from __future__ import annotations

import argparse
import importlib
import statistics
import time

import numpy as np

import skfemntv


def median_seconds(function,repeats):
    function()
    samples=[]
    for _ in range(repeats):
        start=time.perf_counter()
        function()
        samples.append(time.perf_counter()-start)
    return statistics.median(samples)


def benchmark(resolution,point_count,repeats,seed):
    evaluation=importlib.import_module("skfemntv.evaluation")
    native_kernel=evaluation._evaluate_triangles
    if native_kernel is None:
        raise RuntimeError("rebuild the C++ extension to run this benchmark")
    coordinates=np.linspace(0.,1.,resolution+1)
    linear_mesh=skfemntv.MeshTri.init_tensor(coordinates,coordinates)
    vertex_count=linear_mesh.p.shape[1]
    mesh=skfemntv.MeshTri2.from_mesh(linear_mesh)
    warped=mesh.p.copy()
    warped[1,vertex_count:] += (0.05/resolution) * np.sin(
        np.pi * warped[0,vertex_count:]
    )
    mesh=skfemntv.MeshTri2(p=warped,t=mesh.t.copy())
    basis=skfemntv.Basis(mesh,skfemntv.ElementTriP2())
    coefficients=np.sin(0.8*basis.doflocs[0])+np.cos(1.1*basis.doflocs[1])
    generator=np.random.default_rng(seed)
    points=generator.uniform(.1,.9,size=(2,point_count))

    try:
        evaluation._evaluate_triangles=None
        python_result=basis.evaluate_at(points,coefficients)
        python_time=median_seconds(
            lambda:basis.evaluate_at(points,coefficients),repeats
        )
        evaluation._evaluate_triangles=native_kernel
        native_result=basis.evaluate_at(points,coefficients)
        np.testing.assert_allclose(native_result.value,python_result.value,rtol=2e-11,atol=2e-11)
        np.testing.assert_allclose(native_result.grad,python_result.grad,rtol=2e-10,atol=2e-10)
        np.testing.assert_array_equal(native_result.cell_ids,python_result.cell_ids)
        np.testing.assert_array_equal(native_result.on_boundary,python_result.on_boundary)
        np.testing.assert_array_equal(native_result.ambiguous,python_result.ambiguous)
        np.testing.assert_array_equal(native_result.outside,python_result.outside)
        native_time=median_seconds(
            lambda:basis.evaluate_at(points,coefficients),repeats
        )
    finally:
        evaluation._evaluate_triangles=native_kernel
    return {
        "resolution":resolution,"cells":mesh.nelements,
        "points":point_count,"python_ms":python_time*1.e3,
        "cpp_ms":native_time*1.e3,"speedup":python_time/native_time,
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--resolutions",type=int,nargs="+",default=[16,32,64])
    parser.add_argument("--points",type=int,default=5000)
    parser.add_argument("--repeat",type=int,default=3)
    parser.add_argument("--seed",type=int,default=13579)
    args=parser.parse_args()
    if args.points<0 or args.repeat<1 or any(n<1 for n in args.resolutions):
        parser.error("resolutions and repeat must be positive; points nonnegative")
    print("resolution cells points Python-ms C++-ms speedup")
    for resolution in args.resolutions:
        row=benchmark(resolution,args.points,args.repeat,args.seed)
        print(
            f"{row['resolution']:10d} {row['cells']:7d} "
            f"{row['points']:6d} {row['python_ms']:9.3f} "
            f"{row['cpp_ms']:8.3f} {row['speedup']:7.2f}x"
        )


if __name__=="__main__":
    main()
