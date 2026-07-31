"""Compare skfn and scikit-fem Poisson assembly as the DoF count grows."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime,timezone
import os
import platform
from pathlib import Path
import statistics
import sys
import tempfile
import time

import numpy as np
import scipy
import skfem
from skfem.helpers import dot as skfem_dot
from skfem.helpers import grad as skfem_grad

import skfn
from skfn.helpers import ddot,dot,grad


@skfn.BilinearForm
def skfn_laplace(u,v,w):
    return ddot(grad(u),grad(v))


@skfn.LinearForm
def skfn_load(v,w):
    return dot(w.source,v)


@skfem.BilinearForm
def skfem_laplace(u,v,w):
    return skfem_dot(skfem_grad(u),skfem_grad(v))


@skfem.LinearForm
def skfem_load(v,w):
    return v


@dataclass(frozen=True)
class Result:
    resolution: int
    dofs: int
    elements: int
    skfn_basis_ms: float
    skfem_basis_ms: float
    skfn_matrix_ms: float
    skfem_matrix_ms: float
    skfn_rhs_ms: float
    skfem_rhs_ms: float

    @property
    def matrix_speedup(self) -> float:
        return self.skfem_matrix_ms/self.skfn_matrix_ms

    @property
    def rhs_speedup(self) -> float:
        return self.skfem_rhs_ms/self.skfn_rhs_ms

    @property
    def skfn_total_ms(self) -> float:
        return self.skfn_matrix_ms+self.skfn_rhs_ms

    @property
    def skfem_total_ms(self) -> float:
        return self.skfem_matrix_ms+self.skfem_rhs_ms

    @property
    def total_speedup(self) -> float:
        return self.skfem_total_ms/self.skfn_total_ms


def environment() -> str:
    return (
        f"Python {platform.python_version()}, NumPy {np.__version__}, "
        f"SciPy {scipy.__version__}, skfn {skfn.__version__}, "
        f"scikit-fem {skfem.__version__}, {platform.platform()}"
    )


def elapsed(function):
    start=time.perf_counter()
    value=function()
    return value,(time.perf_counter()-start)*1e3


def median_time(function,repeat: int) -> float:
    samples=[]
    for _ in range(repeat):
        _,duration=elapsed(function)
        samples.append(duration)
    return statistics.median(samples)


def compare_outputs(native_matrix,reference_matrix,native_rhs,reference_rhs):
    difference=(native_matrix-reference_matrix).tocoo()
    matrix_error=(
        float(np.max(np.abs(difference.data)))
        if difference.nnz else 0.
    )
    rhs_error=float(np.max(np.abs(native_rhs-reference_rhs)))
    if matrix_error>2e-12 or rhs_error>2e-12:
        raise RuntimeError(
            f"implementation mismatch: matrix={matrix_error:.3e}, "
            f"rhs={rhs_error:.3e}"
        )


def benchmark(resolution: int,repeat: int,warmup: int) -> Result:
    coordinates=np.linspace(0.,1.,resolution+1)
    mesh=skfn.MeshTri.init_tensor(coordinates,coordinates)
    reference_mesh=skfem.MeshTri(mesh.p,mesh.t)

    basis,skfn_basis_ms=elapsed(lambda:skfn.Basis(
        mesh,skfn.ElementVector(skfn.ElementTriP1(),dim=1),
        intorder=2,
    ))
    reference_basis,skfem_basis_ms=elapsed(lambda:skfem.Basis(
        reference_mesh,skfem.ElementTriP1(),intorder=2
    ))
    source=np.array([1.])
    native_matrix=skfn.asm(skfn_laplace,basis)
    native_rhs=skfn.asm(skfn_load,basis,source=source)
    reference_matrix=skfem.asm(skfem_laplace,reference_basis)
    reference_rhs=skfem.asm(skfem_load,reference_basis)
    compare_outputs(
        native_matrix,reference_matrix,native_rhs,reference_rhs
    )
    for _ in range(max(0,warmup-1)):
        skfn.asm(skfn_laplace,basis)
        skfn.asm(skfn_load,basis,source=source)
        skfem.asm(skfem_laplace,reference_basis)
        skfem.asm(skfem_load,reference_basis)

    return Result(
        resolution=resolution,dofs=basis.N,elements=mesh.nelements,
        skfn_basis_ms=skfn_basis_ms,skfem_basis_ms=skfem_basis_ms,
        skfn_matrix_ms=median_time(
            lambda:skfn.asm(skfn_laplace,basis),repeat
        ),
        skfem_matrix_ms=median_time(
            lambda:skfem.asm(skfem_laplace,reference_basis),repeat
        ),
        skfn_rhs_ms=median_time(
            lambda:skfn.asm(skfn_load,basis,source=source),repeat
        ),
        skfem_rhs_ms=median_time(
            lambda:skfem.asm(skfem_load,reference_basis),repeat
        ),
    )


def markdown(results: list[Result],include_metadata: bool=False) -> str:
    lines=[
        "# skfn vs. scikit-fem: Poisson assembly",
        "",
    ] if include_metadata else []
    if include_metadata:
        lines.extend([
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            "",
            f"Environment: `{environment()}`",
            "",
            "Warm-cache median timings; solve time is excluded.",
            "",
        ])
    lines.extend([
        "| DoFs | Elements | skfn K [ms] | skfem K [ms] | K speedup | "
        "skfn f [ms] | skfem f [ms] | f speedup | total speedup |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for result in results:
        lines.append(
            f"| {result.dofs} | {result.elements} | "
            f"{result.skfn_matrix_ms:.3f} | "
            f"{result.skfem_matrix_ms:.3f} | "
            f"{result.matrix_speedup:.2f}x | "
            f"{result.skfn_rhs_ms:.3f} | "
            f"{result.skfem_rhs_ms:.3f} | "
            f"{result.rhs_speedup:.2f}x | "
            f"{result.total_speedup:.2f}x |"
        )
    lines.extend([
        "",
        "Basis construction:",
        "",
        "| DoFs | skfn [ms] | skfem [ms] | speedup |",
        "|---:|---:|---:|---:|",
    ])
    for result in results:
        lines.append(
            f"| {result.dofs} | {result.skfn_basis_ms:.3f} | "
            f"{result.skfem_basis_ms:.3f} | "
            f"{result.skfem_basis_ms/result.skfn_basis_ms:.2f}x |"
        )
    return "\n".join(lines)


def write_csv(path: Path,results: list[Result]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=tuple(Result.__dataclass_fields__)+(
        "matrix_speedup","rhs_speedup","skfn_total_ms",
        "skfem_total_ms","total_speedup",
    )
    with path.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(
            stream,fieldnames=fields,lineterminator="\n"
        )
        writer.writeheader()
        for result in results:
            writer.writerow({field:getattr(result,field) for field in fields})


def write_plot(path: Path,results: list[Result]) -> None:
    if "MPLCONFIGDIR" not in os.environ:
        cache=Path(tempfile.gettempdir())/"skfn-matplotlib"
        cache.mkdir(exist_ok=True)
        os.environ["MPLCONFIGDIR"]=str(cache)
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit(
            "plotting requires: python -m pip install '.[benchmark]'"
        ) from error
    dofs=np.array([result.dofs for result in results])
    figure,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    comparisons=(
        (axes[0,0],"Poisson stiffness assembly","skfn_matrix_ms",
         "skfem_matrix_ms"),
        (axes[0,1],"Right-hand-side assembly","skfn_rhs_ms",
         "skfem_rhs_ms"),
        (axes[1,0],"Basis construction","skfn_basis_ms",
         "skfem_basis_ms"),
    )
    for axis,title,native,reference in comparisons:
        axis.loglog(
            dofs,[getattr(result,native) for result in results],
            "o-",label="skfn",linewidth=2,
        )
        axis.loglog(
            dofs,[getattr(result,reference) for result in results],
            "s-",label="scikit-fem",linewidth=2,
        )
        axis.set(title=title,xlabel="Degrees of freedom",ylabel="Time [ms]")
        axis.grid(True,which="both",alpha=.3)
        axis.legend()
    speedup=axes[1,1]
    speedup.semilogx(
        dofs,[result.matrix_speedup for result in results],
        "o-",label="stiffness K",linewidth=2,
    )
    speedup.semilogx(
        dofs,[result.rhs_speedup for result in results],
        "s-",label="right-hand side f",linewidth=2,
    )
    speedup.semilogx(
        dofs,[result.total_speedup for result in results],
        "^-",label="K + f",linewidth=2,
    )
    speedup.axhline(1.,color="black",linewidth=1,linestyle="--")
    speedup.set(
        title="scikit-fem time / skfn time",
        xlabel="Degrees of freedom",ylabel="Speedup [x]",
    )
    speedup.grid(True,which="both",alpha=.3)
    speedup.legend()
    figure.suptitle("Poisson P1 assembly scaling (warm-cache median)")
    path.parent.mkdir(parents=True,exist_ok=True)
    figure.savefig(path,dpi=180)
    plt.close(figure)


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument(
        "--sizes",nargs="+",type=int,default=[16,32,64,128,256],
        help="number of subdivisions along each coordinate axis",
    )
    parser.add_argument("--repeat",type=int,default=5)
    parser.add_argument("--warmup",type=int,default=2)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--markdown-output",type=Path)
    parser.add_argument("--plot-output",type=Path)
    arguments=parser.parse_args()
    if arguments.repeat<1 or any(size<1 for size in arguments.sizes):
        parser.error("sizes and repeat must be positive")

    print(environment(),file=sys.stderr)
    results=[]
    for size in arguments.sizes:
        print(f"benchmarking {size} x {size} cells...",file=sys.stderr)
        results.append(benchmark(size,arguments.repeat,arguments.warmup))
    print(markdown(results))
    if arguments.output is not None:
        write_csv(arguments.output,results)
        print(f"wrote {arguments.output}",file=sys.stderr)
    if arguments.markdown_output is not None:
        arguments.markdown_output.parent.mkdir(parents=True,exist_ok=True)
        arguments.markdown_output.write_text(
            markdown(results,include_metadata=True)+"\n",encoding="utf-8"
        )
        print(f"wrote {arguments.markdown_output}",file=sys.stderr)
    if arguments.plot_output is not None:
        write_plot(arguments.plot_output,results)
        print(f"wrote {arguments.plot_output}",file=sys.stderr)


if __name__=="__main__":
    main()
