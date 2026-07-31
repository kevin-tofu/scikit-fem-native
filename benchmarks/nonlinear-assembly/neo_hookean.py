"""Tet4/Hex8 Neo-Hookean residual/tangent assembly benchmark."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict,dataclass
from pathlib import Path
import statistics
import sys
import tempfile
import time

import numpy as np
import skfem

import skfn

sys.path.insert(0,str(Path(__file__).parents[1]))
from skfem_neo_hookean import forms as reference_forms


@dataclass(frozen=True)
class Result:
    topology: str
    intorder: int
    distorted: int
    points: int
    dofs: int
    elements: int
    skfn_residual_ms: float
    skfn_residual_parallel_ms: float
    skfn_tangent_ms: float
    skfn_tangent_parallel_ms: float
    skfem_tangent_ms: float
    native_threads: int


def median_time(function,repeat):
    function()
    samples=[]
    for _ in range(repeat):
        start=time.perf_counter()
        function()
        samples.append((time.perf_counter()-start)*1e3)
    return statistics.median(samples)


def benchmark(
    topology,points,intorder,distorted,repeat,native_threads,check
):
    axis=np.linspace(0.,1.,points)
    if topology=="tet":
        mesh=skfn.MeshTet.init_tensor(axis,axis,axis)
        element=skfn.ElementTetP1()
        reference_mesh_type=skfem.MeshTet
        reference_element=skfem.ElementTetP1()
    else:
        mesh=skfn.MeshHex.init_tensor(axis,axis,axis)
        element=skfn.ElementHex1()
        reference_mesh_type=skfem.MeshHex
        reference_element=skfem.ElementHex1()
    if distorted:
        physical=mesh.p.copy()
        x,y,z=physical
        physical[0]=x+.08*y*z
        physical[1]=y-.05*x*z
        physical[2]=z+.06*x*y
        mesh=type(mesh)(physical,mesh.t)
    quadrature=(
        (np.full((3,1),.25),np.array([1./6.]))
        if topology=="tet" and intorder==1 else None
    )
    basis=skfn.Basis(
        mesh,skfn.ElementVector(element,dim=3),
        intorder=intorder,quadrature=quadrature,
    )
    local_nodes=len(element.doflocs)
    element_dofs=basis.element_dofs.T.reshape(
        mesh.nelements,local_nodes,3
    )
    young,poisson=100.,.3
    lmbda=young*poisson/((1.+poisson)*(1.-2.*poisson))
    mu=young/(2.*(1.+poisson))
    kernel=skfn.NeoHookean(mu,lmbda)
    assembler=(
        skfn.NativeAssembler(
            mesh.p.T,mesh.t.T,element_dofs,kernel,
            quadrature=(basis.X.T,basis.W),
        )
        if topology=="tet" else
        skfn.NativeAssembler.from_basis(basis,kernel)
    )
    reference_basis=skfem.Basis(
        reference_mesh_type(mesh.p,mesh.t),
        skfem.ElementVector(reference_element),
        quadrature=(basis.X,basis.W),
    )
    residual_form,tangent_form=reference_forms(mu,lmbda)
    coordinates=basis.doflocs
    u=np.empty(basis.N,dtype=np.float64)
    u[0::3]=1e-3*coordinates[0,0::3]*coordinates[1,0::3]
    u[1::3]=-7e-4*coordinates[1,1::3]*coordinates[2,1::3]
    u[2::3]=5e-4*coordinates[2,2::3]*coordinates[0,2::3]

    def reference():
        field=reference_basis.interpolate(u)
        return (
            residual_form.assemble(reference_basis,displacement=field),
            tangent_form.assemble(reference_basis,displacement=field),
        )

    if check:
        native=assembler.assemble(u,num_threads=1)
        residual,tangent=reference()
        np.testing.assert_allclose(
            native.residual,residual,rtol=3e-11,atol=3e-11
        )
        np.testing.assert_allclose(
            native.tangent.toarray(),tangent.toarray(),
            rtol=3e-11,atol=3e-11,
        )
    effective=min(native_threads,skfn.available_num_threads())
    return Result(
        topology=topology,intorder=intorder,distorted=int(distorted),
        points=points,dofs=basis.N,elements=mesh.nelements,
        skfn_residual_ms=median_time(
            lambda:assembler.assemble(u,mode="residual",num_threads=1),repeat
        ),
        skfn_residual_parallel_ms=median_time(
            lambda:assembler.assemble(
                u,mode="residual",num_threads=effective
            ),repeat
        ),
        skfn_tangent_ms=median_time(
            lambda:assembler.assemble(u,num_threads=1),repeat
        ),
        skfn_tangent_parallel_ms=median_time(
            lambda:assembler.assemble(u,num_threads=effective),repeat
        ),
        skfem_tangent_ms=median_time(reference,repeat),
        native_threads=effective,
    )


def markdown(results):
    lines=[
        "| Mesh | Order | Distorted | DoFs | Elements | skfn R [ms] | "
        "skfn R parallel [ms] | "
        "skfn R+K [ms] | skfn R+K parallel [ms] | threads | "
        "skfem R+K [ms] | parallel speedup |",
        "|:---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result.topology} | {result.intorder} | "
            f"{'yes' if result.distorted else 'no'} | "
            f"{result.dofs} | {result.elements} | "
            f"{result.skfn_residual_ms:.3f} | "
            f"{result.skfn_residual_parallel_ms:.3f} | "
            f"{result.skfn_tangent_ms:.3f} | "
            f"{result.skfn_tangent_parallel_ms:.3f} | "
            f"{result.native_threads} | {result.skfem_tangent_ms:.3f} | "
            f"{result.skfem_tangent_ms/result.skfn_tangent_parallel_ms:.2f}x |"
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


def write_plot(path,results,repeat):
    if "MPLCONFIGDIR" not in __import__("os").environ:
        __import__("os").environ["MPLCONFIGDIR"]=str(
            Path(tempfile.gettempdir())/"skfn-matplotlib"
        )
    import matplotlib.pyplot as plt
    dofs=np.array([result.dofs for result in results])
    threads=results[0].native_threads
    figure,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    axis=axes[0]
    for values,label,marker in (
        ("skfn_tangent_ms","skfn R+K (1 thread)","o-"),
        ("skfn_tangent_parallel_ms",f"skfn R+K ({threads} threads)","^-"),
        ("skfem_tangent_ms","scikit-fem R+K","s-"),
    ):
        axis.loglog(
            dofs,[getattr(result,values) for result in results],
            marker,label=label,linewidth=2,
        )
    axis.set(xlabel="Degrees of freedom",ylabel="Time [ms]",
             title=(
                 f"{results[0].topology.title()} Neo-Hookean fused assembly "
                 f"(intorder={results[0].intorder})"
             ))
    axis.grid(True,which="both",alpha=.3);axis.legend()
    speedup=axes[1]
    speedup.semilogx(
        dofs,[result.skfem_tangent_ms/result.skfn_tangent_ms for result in results],
        "o-",label="skfn 1 thread",linewidth=2,
    )
    speedup.semilogx(
        dofs,[result.skfem_tangent_ms/result.skfn_tangent_parallel_ms for result in results],
        "^-",label=f"skfn {threads} threads",linewidth=2,
    )
    speedup.axhline(1.,color="black",linestyle="--",linewidth=1)
    speedup.set(xlabel="Degrees of freedom",ylabel="Speedup [x]",
                title="scikit-fem time / skfn time (>1 is faster)")
    speedup.grid(True,which="both",alpha=.3);speedup.legend()
    figure.suptitle(f"Median of {repeat} timed runs after one warm-up")
    path.parent.mkdir(parents=True,exist_ok=True)
    figure.savefig(path,dpi=180);plt.close(figure)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--topology",choices=("tet","hex"),default="tet")
    parser.add_argument("--intorder",type=int,default=1)
    parser.add_argument("--distorted",action="store_true")
    parser.add_argument("--points",nargs="+",type=int,default=[4,6,8,10,12,16])
    parser.add_argument("--repeat",type=int,default=1)
    parser.add_argument("--native-threads",type=int,default=4)
    parser.add_argument("--no-check",action="store_true")
    parser.add_argument("--output",type=Path)
    parser.add_argument("--plot-output",type=Path)
    arguments=parser.parse_args()
    if arguments.repeat<1 or arguments.native_threads<1:
        parser.error("repeat and native threads must be positive")
    results=[]
    for points in arguments.points:
        print(f"benchmarking {points} points/axis...",flush=True)
        results.append(benchmark(
            arguments.topology,points,arguments.intorder,
            arguments.distorted,arguments.repeat,arguments.native_threads,
            not arguments.no_check,
        ))
    print(markdown(results))
    if arguments.output:write_csv(arguments.output,results)
    if arguments.plot_output:write_plot(
        arguments.plot_output,results,arguments.repeat
    )


if __name__=="__main__":
    main()
