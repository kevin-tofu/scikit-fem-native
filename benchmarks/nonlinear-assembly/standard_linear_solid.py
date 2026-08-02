"""Standard Linear Solid load/hold/unload assembly benchmark."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict,dataclass
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time

import numpy as np
import skfem

import skfemntv

sys.path.insert(0,str(Path(__file__).parents[1]))
from skfem_j2 import forms as reference_forms


HISTORY=(.003,.003,.003,0.,0.)
TIME_STEPS=(.1,.25,.05,.4,.1)


@dataclass(frozen=True)
class Result:
    dofs: int
    elements: int
    integration_points: int
    steps: int
    skfn_1t_ms: float
    skfn_parallel_ms: float
    skfem_ms: float
    native_threads: int


def elastic_tangent(young,poisson):
    mu=young/(2.*(1.+poisson))
    lmbda=young*poisson/((1.+poisson)*(1.-2.*poisson))
    tangent=np.zeros((6,6));tangent[:3,:3]=lmbda
    tangent[0,0]+=2.*mu;tangent[1,1]+=2.*mu;tangent[2,2]+=2.*mu
    tangent[3,3]=tangent[4,4]=tangent[5,5]=2.*mu
    return tangent


def median_time(function,repeat):
    function();samples=[]
    for _ in range(repeat):
        start=time.perf_counter();function()
        samples.append((time.perf_counter()-start)*1e3)
    return statistics.median(samples)


def benchmark(points,repeat,native_threads,check):
    axis=np.linspace(0.,1.,points)
    mesh=skfemntv.MeshTet.init_tensor(axis,axis,axis)
    basis=skfemntv.Basis(
        mesh,skfemntv.ElementVector(skfemntv.ElementTetP1(),dim=3),intorder=2
    )
    reference_basis=skfem.Basis(
        skfem.MeshTet(mesh.p,mesh.t),skfem.ElementVector(skfem.ElementTetP1()),
        quadrature=(basis.X,basis.W),
    )
    material=skfemntv.StandardLinearSolid(100.,60.,.25,2.,.2)
    assembler=skfemntv.MaterialAssembler(basis,material)
    residual_form,tangent_form=reference_forms()
    coordinates=basis.doflocs
    displacements=[]
    for amplitude in HISTORY:
        u=np.zeros(basis.N)
        u[0::3]=amplitude*coordinates[0,0::3]
        u[1::3]=-.5*amplitude*coordinates[1,1::3]
        u[2::3]=-.5*amplitude*coordinates[2,2::3]
        displacements.append(u)
    equilibrium=elastic_tangent(
        material.equilibrium_modulus,material.poisson_ratio
    )
    branch=elastic_tangent(material.branch_modulus,material.poisson_ratio)
    entities,quadrature=basis.dx.shape

    def native(threads):
        state=assembler.initial_state();result=None
        for u,dt in zip(displacements,TIME_STEPS):
            result=assembler.assemble(
                u,state,num_threads=threads,time_step=dt
            )
            state=result.trial_state
        return result,state

    def reference():
        viscous=np.zeros((6,entities,quadrature));residual=tangent=None
        for u,dt in zip(displacements,TIME_STEPS):
            field=reference_basis.interpolate(u)
            strain=np.stack((
                field.grad[0,0],field.grad[1,1],field.grad[2,2],
                .5*(field.grad[0,1]+field.grad[1,0]),
                .5*(field.grad[1,2]+field.grad[2,1]),
                .5*(field.grad[0,2]+field.grad[2,0]),
            ))
            step_factor=1./(1.+dt/material.relaxation_time)
            viscous=step_factor*viscous+(1.-step_factor)*strain
            stress=(
                np.einsum("ij,j...->i...",equilibrium,strain)
                +np.einsum("ij,j...->i...",branch,strain-viscous)
            )
            step_algorithmic=equilibrium+step_factor*branch
            constitutive=np.broadcast_to(
                step_algorithmic[:,:,None,None],(6,6)+strain.shape[1:]
            )
            residual=residual_form.assemble(reference_basis,stress=stress)
            tangent=tangent_form.assemble(
                reference_basis,constitutive=constitutive
            )
        return residual,tangent,viscous

    effective=min(native_threads,skfemntv.available_num_threads())
    if check:
        result,state=native(effective)
        residual,tangent,viscous=reference()
        np.testing.assert_allclose(result.residual,residual,rtol=3e-11,atol=3e-11)
        difference=result.tangent-tangent;difference.eliminate_zeros()
        assert difference.nnz==0 or np.max(np.abs(difference.data))<3e-10
        np.testing.assert_allclose(
            state.storage,np.moveaxis(viscous,0,-1).reshape(-1,6),
            rtol=3e-11,atol=3e-11,
        )
    return Result(
        basis.N,mesh.nelements,assembler.state_count,len(HISTORY),
        median_time(lambda:native(1),repeat),
        median_time(lambda:native(effective),repeat),
        median_time(reference,repeat),effective,
    )


def markdown(results):
    lines=[
        "| DoFs | Elements | IPs | Steps | skfemntv 1t [ms] | skfemntv Nt [ms] | "
        "skfem [ms] | threads | skfem/skfemntv Nt |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| {row.dofs} | {row.elements} | {row.integration_points} | "
            f"{row.steps} | {row.skfn_1t_ms:.3f} | "
            f"{row.skfn_parallel_ms:.3f} | {row.skfem_ms:.3f} | "
            f"{row.native_threads} | {row.skfem_ms/row.skfn_parallel_ms:.2f}x |"
        )
    return "\n".join(lines)


def write_csv(path,results):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(
            stream,fieldnames=Result.__dataclass_fields__,lineterminator="\n"
        );writer.writeheader();writer.writerows(asdict(row) for row in results)


def write_plot(path,results,repeat):
    os.environ.setdefault(
        "MPLCONFIGDIR",str(Path(tempfile.gettempdir())/"skfemntv-matplotlib")
    )
    import matplotlib.pyplot as plt
    dofs=[row.dofs for row in results];threads=results[0].native_threads
    figure,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    for field,label,style in (
        ("skfn_1t_ms","skfemntv 1 thread","o-"),
        ("skfn_parallel_ms",f"skfemntv {threads} threads","^-"),
        ("skfem_ms","scikit-fem","s-"),
    ):
        axes[0].loglog(
            dofs,[getattr(row,field) for row in results],style,
            label=label,linewidth=2,
        )
    axes[0].set(
        xlabel="Degrees of freedom",ylabel="Five-step history time [ms]",
        title="Tet Standard Linear Solid load/hold/unload",
    );axes[0].grid(True,which="both",alpha=.3);axes[0].legend()
    axes[1].semilogx(
        dofs,[row.skfem_ms/row.skfn_1t_ms for row in results],
        "o-",label="skfem / skfemntv 1 thread",linewidth=2,
    )
    axes[1].semilogx(
        dofs,[row.skfem_ms/row.skfn_parallel_ms for row in results],
        "^-",label=f"skfem / skfemntv {threads} threads",linewidth=2,
    )
    axes[1].axhline(1.,color="black",linestyle="--",linewidth=1)
    axes[1].set(
        xlabel="Degrees of freedom",ylabel="Speedup [x]",
        title="History assembly speedup (>1 is faster)",
    );axes[1].grid(True,which="both",alpha=.3);axes[1].legend()
    figure.suptitle(
        f"No solve; median of {repeat} runs after warm-up; "
        f"amplitudes={HISTORY}; dt={TIME_STEPS}"
    )
    path.parent.mkdir(parents=True,exist_ok=True)
    figure.savefig(path,dpi=180);plt.close(figure)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--points",nargs="+",type=int,default=[4,6,8,10,12,16])
    parser.add_argument("--repeat",type=int,default=1)
    parser.add_argument("--native-threads",type=int,default=4)
    parser.add_argument("--no-check",action="store_true")
    parser.add_argument("--output",type=Path);parser.add_argument("--plot-output",type=Path)
    args=parser.parse_args()
    if args.repeat<1 or args.native_threads<1 or min(args.points)<2:
        parser.error("repeat, threads, and point counts must be positive")
    results=[]
    for points in args.points:
        print(f"benchmarking {points} points/axis...",flush=True)
        results.append(benchmark(
            points,args.repeat,args.native_threads,not args.no_check
        ))
    print(markdown(results))
    if args.output:write_csv(args.output,results)
    if args.plot_output:write_plot(args.plot_output,results,args.repeat)


if __name__=="__main__":main()
