"""Segmented versus flattened native cut-cell assembly benchmark."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict,dataclass
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import numpy as np

import skfemntv
from skfemntv.bilinear_form import NativeBilinearForm
from skfemntv.linear_form import NativeLinearForm


@dataclass(frozen=True)
class Result:
    resolution: int
    dofs: int
    cells: int
    cut_fraction: float
    intorder: int
    quadrature_points: int
    nonempty_cells: int
    threads: int
    segmented_linear_setup_ms: float
    flattened_linear_setup_ms: float
    segmented_bilinear_setup_ms: float
    flattened_bilinear_setup_ms: float
    segmented_linear_assembly_ms: float
    flattened_linear_assembly_ms: float
    segmented_bilinear_assembly_ms: float
    flattened_bilinear_assembly_ms: float


def median_time(function,repeat):
    function();samples=[]
    for _ in range(repeat):
        start=time.perf_counter();function()
        samples.append((time.perf_counter()-start)*1.e3)
    return statistics.median(samples)


def flattened_adapter(cut):
    """The former one-native-entity-per-real-point representation."""
    return SimpleNamespace(
        mesh=cut.mesh,elem=cut.elem,N=cut.N,dx=cut.dx,
        tabulated_shape=cut.tabulated_shape,
        tabulated_gradients=cut.tabulated_gradients,
        element_dofs=cut.element_dofs,
    )


def benchmark(resolution,fraction,intorder,threads,repeat,check):
    axis=np.linspace(0.,1.,resolution+1)
    mesh=skfemntv.MeshTri.init_tensor(axis,axis)
    basis=skfemntv.Basis(
        mesh,skfemntv.ElementVector(skfemntv.ElementTriP1(),dim=1)
    )
    level_set=skfemntv.LevelSet(lambda x:x[0]-fraction,tolerance=0.)
    quadrature=level_set.cut_quadrature(mesh,intorder=intorder)
    cut=skfemntv.CutCellBasis(basis,quadrature)
    flat=flattened_adapter(cut)

    segmented_linear=NativeLinearForm(cut)
    flattened_linear=NativeLinearForm(flat)
    segmented_bilinear=NativeBilinearForm(cut)
    flattened_bilinear=NativeBilinearForm(flat)
    force=np.array([1.],dtype=np.float64)
    value=np.ones(cut.dx.shape,dtype=np.float64)
    gradient=np.ones(cut.dx.shape,dtype=np.float64)
    if check:
        left,_=segmented_linear.assemble(value=force,num_threads=threads)
        right,_=flattened_linear.assemble(value=force,num_threads=threads)
        if len(right)<len(left):right=np.pad(right,(0,len(left)-len(right)))
        np.testing.assert_allclose(left,right,atol=2.e-14)
        left=segmented_bilinear.assemble(
            value=value,gradient=gradient,num_threads=threads
        )
        right=flattened_bilinear.assemble(
            value=value,gradient=gradient,num_threads=threads
        )
        np.testing.assert_allclose(left.toarray(),right.toarray(),atol=2.e-14)
    return Result(
        resolution=resolution,dofs=basis.N,cells=mesh.nelements,
        cut_fraction=fraction,intorder=intorder,
        quadrature_points=cut.npoints,nonempty_cells=cut.nelems,
        threads=threads,
        segmented_linear_setup_ms=median_time(
            lambda:NativeLinearForm(cut),repeat
        ),
        flattened_linear_setup_ms=median_time(
            lambda:NativeLinearForm(flat),repeat
        ),
        segmented_bilinear_setup_ms=median_time(
            lambda:NativeBilinearForm(cut),repeat
        ),
        flattened_bilinear_setup_ms=median_time(
            lambda:NativeBilinearForm(flat),repeat
        ),
        segmented_linear_assembly_ms=median_time(
            lambda:segmented_linear.assemble(value=force,num_threads=threads),repeat
        ),
        flattened_linear_assembly_ms=median_time(
            lambda:flattened_linear.assemble(value=force,num_threads=threads),repeat
        ),
        segmented_bilinear_assembly_ms=median_time(
            lambda:segmented_bilinear.assemble(
                value=value,gradient=gradient,num_threads=threads
            ),repeat
        ),
        flattened_bilinear_assembly_ms=median_time(
            lambda:flattened_bilinear.assemble(
                value=value,gradient=gradient,num_threads=threads
            ),repeat
        ),
    )


def write_csv(path,results):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=Result.__dataclass_fields__)
        writer.writeheader();writer.writerows(asdict(result) for result in results)


def write_plot(path,results,repeat):
    import matplotlib.pyplot as plt
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    figure,axes=plt.subplots(1,2,figsize=(12,4.8))
    for intorder in sorted({result.intorder for result in results}):
        selected=[result for result in results if result.intorder==intorder]
        x=[result.cut_fraction for result in selected]
        axes[0].plot(x,[result.segmented_bilinear_setup_ms for result in selected],
                     "o-",label=f"segmented, order {intorder}")
        axes[0].plot(x,[result.flattened_bilinear_setup_ms for result in selected],
                     "x--",label=f"flattened, order {intorder}")
        axes[1].plot(x,[result.segmented_bilinear_assembly_ms for result in selected],
                     "o-",label=f"segmented, order {intorder}")
        axes[1].plot(x,[result.flattened_bilinear_assembly_ms for result in selected],
                     "x--",label=f"flattened, order {intorder}")
    axes[0].set_title("Bilinear assembler setup")
    axes[1].set_title("Repeated bilinear assembly")
    for axis in axes:
        axis.set_xlabel("active volume fraction");axis.set_ylabel("median time [ms]")
        axis.grid(True,alpha=.3);axis.legend(fontsize=8)
    figure.suptitle(f"Segmented cut assembly (median of {repeat})")
    figure.tight_layout();figure.savefig(path,dpi=160);plt.close(figure)


def markdown(results):
    lines=[
        "| fraction | order | qpoints | threads | setup seg/flat ms | "
        "assembly seg/flat ms | assembly speedup |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        speedup=(result.flattened_bilinear_assembly_ms/
                 result.segmented_bilinear_assembly_ms)
        lines.append(
            f"| {result.cut_fraction:.2f} | {result.intorder} | "
            f"{result.quadrature_points} | {result.threads} | "
            f"{result.segmented_bilinear_setup_ms:.3f} / "
            f"{result.flattened_bilinear_setup_ms:.3f} | "
            f"{result.segmented_bilinear_assembly_ms:.3f} / "
            f"{result.flattened_bilinear_assembly_ms:.3f} | {speedup:.2f}x |"
        )
    return "\n".join(lines)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--resolution",type=int,default=64)
    parser.add_argument("--fractions",type=float,nargs="+",default=[.1,.5,.9])
    parser.add_argument("--intorders",type=int,nargs="+",default=[1,2,4])
    parser.add_argument("--threads",type=int,default=1)
    parser.add_argument("--repeat",type=int,default=3)
    parser.add_argument("--no-check",action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--plot-output")
    args=parser.parse_args()
    if args.resolution<1 or args.repeat<1:
        parser.error("resolution and repeat must be positive")
    effective=min(args.threads,skfemntv.available_num_threads())
    results=[
        benchmark(args.resolution,fraction,intorder,effective,args.repeat,
                  not args.no_check)
        for intorder in args.intorders for fraction in args.fractions
    ]
    if args.output:write_csv(args.output,results)
    if args.plot_output:write_plot(args.plot_output,results,args.repeat)
    print(markdown(results))


if __name__=="__main__":main()
