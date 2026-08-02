"""Large native-only Neo-Hookean thread-scaling benchmark."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict,dataclass
from pathlib import Path
import resource
import statistics
import sys
import tempfile
import time

import numpy as np

import skfemntv


@dataclass(frozen=True)
class Result:
    topology: str
    points: int
    dofs: int
    elements: int
    quadrature_points: int
    color_count: int
    min_color_size: int
    max_color_size: int
    eligible_colors: int
    thread_threshold: int
    basis_ms: float
    assembler_ms: float
    rss_mb: float
    csr_mb: float
    r_t1_ms: float
    r_t2_ms: float
    r_t4_ms: float
    r_t8_ms: float
    rk_t1_ms: float
    rk_t2_ms: float
    rk_t4_ms: float
    rk_t8_ms: float
    r_speedup_t8: float
    rk_speedup_t8: float
    rk_efficiency_t8: float
    effective_t8: int


def current_rss_mb():
    statm=Path("/proc/self/statm")
    if statm.exists():
        resident=int(statm.read_text().split()[1])
        return resident*__import__("os").sysconf("SC_PAGE_SIZE")/1024.**2
    value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value/(1024. if sys.platform!="darwin" else 1024.**2)


def median_time(function,repeat):
    function()
    samples=[]
    for _ in range(repeat):
        start=time.perf_counter();function()
        samples.append((time.perf_counter()-start)*1e3)
    return statistics.median(samples)


def distort(points):
    result=points.copy();x,y,z=points
    result[0]=x+.06*y*z
    result[1]=y-.04*x*z
    result[2]=z+.05*x*y
    return result


def mesh_and_element(topology,points):
    axis=np.linspace(0.,1.,points)
    if topology.startswith("tet"):
        linear=skfemntv.MeshTet.init_tensor(axis,axis,axis)
        if topology=="tet10":
            return skfemntv.MeshTet2.from_mesh(linear),skfemntv.ElementTetP2()
        return linear,skfemntv.ElementTetP1()
    if topology.startswith("hex"):
        linear=skfemntv.MeshHex.init_tensor(axis,axis,axis)
        if topology=="hex27":
            return skfemntv.MeshHex2.from_mesh(linear),skfemntv.ElementHex2()
        return linear,skfemntv.ElementHex1()
    return (
        skfemntv.MeshWedge1.init_tensor(axis,axis,axis),
        skfemntv.ElementWedge1(),
    )


def csr_mb(matrix):
    return (
        matrix.data.nbytes+matrix.indices.nbytes+matrix.indptr.nbytes
    )/1024.**2


def benchmark(topology,points,intorder,repeat):
    mesh,element=mesh_and_element(topology,points)
    mesh=type(mesh)(distort(mesh.p),mesh.t)
    start=time.perf_counter()
    basis=skfemntv.Basis(
        mesh,skfemntv.ElementVector(element,dim=3),intorder=intorder
    )
    basis_ms=(time.perf_counter()-start)*1e3
    start=time.perf_counter()
    assembler=skfemntv.NativeAssembler.from_basis(
        basis,skfemntv.NeoHookean.from_young_poisson(100.,.3)
    )
    assembler_ms=(time.perf_counter()-start)*1e3
    diagnostics=assembler.parallel_diagnostics
    coordinates=basis.doflocs
    u=np.empty(basis.N,dtype=np.float64)
    u[0::3]=1e-3*coordinates[0,0::3]*coordinates[1,0::3]
    u[1::3]=-7e-4*coordinates[1,1::3]*coordinates[2,1::3]
    u[2::3]=5e-4*coordinates[2,2::3]*coordinates[0,2::3]
    available=skfemntv.available_num_threads()
    effective=(1,min(2,available),min(4,available),min(8,available))

    def timings(mode):
        return tuple(median_time(
            lambda:assembler.assemble(u,mode=mode,num_threads=threads),repeat
        ) for threads in effective)

    residual=timings("residual")
    tangent=timings("residual_tangent")
    effective8=effective[-1]
    return Result(
        topology=topology,points=points,dofs=basis.N,
        elements=mesh.nelements,quadrature_points=basis.X.shape[1],
        color_count=diagnostics["color_count"],
        min_color_size=diagnostics["min_color_size"],
        max_color_size=diagnostics["max_color_size"],
        eligible_colors=diagnostics["parallel_eligible_color_count"],
        thread_threshold=diagnostics["explicit_thread_threshold"],
        basis_ms=basis_ms,assembler_ms=assembler_ms,rss_mb=current_rss_mb(),
        csr_mb=csr_mb(assembler.tangent),
        r_t1_ms=residual[0],r_t2_ms=residual[1],
        r_t4_ms=residual[2],r_t8_ms=residual[3],
        rk_t1_ms=tangent[0],rk_t2_ms=tangent[1],
        rk_t4_ms=tangent[2],rk_t8_ms=tangent[3],
        r_speedup_t8=residual[0]/residual[3],
        rk_speedup_t8=tangent[0]/tangent[3],
        rk_efficiency_t8=tangent[0]/tangent[3]/effective8,
        effective_t8=effective8,
    )


def markdown(results):
    lines=[
        "| Mesh | DoFs | Elems | colors min/max/eligible | RSS/CSR [MiB] | "
        "R t1/t2/t4/t8 [ms] | R+K t1/t2/t4/t8 [ms] | R+K speedup/eff. |",
        "|:---|---:|---:|:---|:---|:---|:---|:---|",
    ]
    for r in results:
        lines.append(
            f"| {r.topology} | {r.dofs} | {r.elements} | "
            f"{r.color_count} {r.min_color_size}/{r.max_color_size}/"
            f"{r.eligible_colors} | {r.rss_mb:.1f}/{r.csr_mb:.1f} | "
            f"{r.r_t1_ms:.2f}/{r.r_t2_ms:.2f}/{r.r_t4_ms:.2f}/"
            f"{r.r_t8_ms:.2f} | "
            f"{r.rk_t1_ms:.2f}/{r.rk_t2_ms:.2f}/{r.rk_t4_ms:.2f}/"
            f"{r.rk_t8_ms:.2f} | {r.rk_speedup_t8:.2f}x/"
            f"{100.*r.rk_efficiency_t8:.1f}% |"
        )
    return "\n".join(lines)


def write_csv(path,results):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(
            stream,fieldnames=Result.__dataclass_fields__,lineterminator="\n"
        )
        writer.writeheader();writer.writerows(asdict(r) for r in results)


def write_plot(path,results,repeat):
    import os
    os.environ.setdefault(
        "MPLCONFIGDIR",str(Path(tempfile.gettempdir())/"skfemntv-matplotlib")
    )
    import matplotlib.pyplot as plt
    dofs=np.array([r.dofs for r in results])
    figure,axes=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
    for axis,prefix,title in (
        (axes[0],"r_","Residual only"),(axes[1],"rk_","Residual + tangent")
    ):
        for threads,style in ((1,"o-"),(2,"s-"),(4,"^-"),(8,"d-")):
            axis.loglog(
                dofs,[getattr(r,f"{prefix}t{threads}_ms") for r in results],
                style,label=f"t{threads}",linewidth=2,
            )
        axis.set(xlabel="Degrees of freedom",ylabel="Time [ms]",title=title)
        axis.grid(True,which="both",alpha=.3);axis.legend()
    axes[2].semilogx(
        dofs,[r.rk_speedup_t8 for r in results],"o-",label="R+K t1/t8",
        linewidth=2,
    )
    axes[2].semilogx(
        dofs,[r.r_speedup_t8 for r in results],"s-",label="R t1/t8",
        linewidth=2,
    )
    axes[2].axhline(1.,color="black",linestyle="--",linewidth=1)
    axes[2].set(xlabel="Degrees of freedom",ylabel="Speedup [x]",
                title="Native thread scaling")
    axes[2].grid(True,which="both",alpha=.3);axes[2].legend()
    first=results[0]
    figure.suptitle(
        f"{first.topology} native Neo-Hookean; median of {repeat} run(s); "
        f"parallel color threshold={first.thread_threshold}"
    )
    path.parent.mkdir(parents=True,exist_ok=True)
    figure.savefig(path,dpi=180);plt.close(figure)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument(
        "--topology",choices=("tet4","tet10","hex8","hex27","wedge6"),
        default="hex8",
    )
    parser.add_argument("--intorder",type=int,default=4)
    parser.add_argument("--points",nargs="+",type=int,default=[6,8,10,12])
    parser.add_argument("--repeat",type=int,default=1)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--plot-output",type=Path)
    args=parser.parse_args()
    if args.repeat<1 or args.intorder<1 or any(p<2 for p in args.points):
        parser.error("repeat/intorder must be positive and points must be >= 2")
    results=[]
    for points in args.points:
        print(f"benchmarking {args.topology}: {points} points/axis...",flush=True)
        results.append(benchmark(
            args.topology,points,args.intorder,args.repeat
        ))
    print(markdown(results))
    if args.output:write_csv(args.output,results)
    if args.plot_output:write_plot(args.plot_output,results,args.repeat)


if __name__=="__main__":
    main()
