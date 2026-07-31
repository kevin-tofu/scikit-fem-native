"""Scaling benchmark for the native planar triangle-supermesh builder."""

from __future__ import annotations

import argparse
from time import perf_counter

import numpy as np

import skfn


def grid(cells: int,flip: bool):
    axis=np.linspace(0.,1.,cells+1)
    points=np.array([[x,y,0.] for y in axis for x in axis]).T
    node=lambda i,j:i+(cells+1)*j
    triangles=[]
    for j in range(cells):
        for i in range(cells):
            a=node(i,j);b=node(i+1,j)
            c=node(i,j+1);d=node(i+1,j+1)
            triangles.extend(
                ((a,b,c),(b,d,c))
                if not flip else ((a,b,d),(a,d,c))
            )
    return points,np.asarray(triangles,dtype=np.int64).T


def run(cells):
    master_points,master_triangles=grid(cells,False)
    slave_points,slave_triangles=grid(cells,True)
    start=perf_counter()
    supermesh=skfn.TriangleSupermesh(
        master_points,master_triangles,
        slave_points,slave_triangles,
    )
    seconds=perf_counter()-start
    arrays=(
        supermesh._row_dofs,supermesh._column_dofs,
        supermesh._row_shape,supermesh._column_shape,
        supermesh._weights,supermesh.global_coordinates,
        supermesh.master_normals,supermesh.slave_normals,
        supermesh.gap,
    )
    diagnostics=supermesh.diagnostics
    return {
        "cells_per_axis":cells,
        "triangles_per_side":master_triangles.shape[1],
        "all_pairs":diagnostics.total_pair_count,
        "aabb_candidates":diagnostics.candidate_pair_count,
        "integration_triangles":diagnostics.integration_triangle_count,
        "build_seconds":seconds,
        "output_megabytes":sum(array.nbytes for array in arrays)/1e6,
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument(
        "cells",nargs="*",type=int,default=[16,32,64,128]
    )
    args=parser.parse_args()
    for cells in args.cells:
        result=run(cells)
        print(" ".join(
            f"{key}={value:.6g}" if isinstance(value,float)
            else f"{key}={value}"
            for key,value in result.items()
        ))


if __name__=="__main__":
    main()
