"""J2 load/unload/reload history assembly benchmark without a solver."""

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
from skfem_j2 import forms as reference_forms,update as reference_update


HISTORY=(.0005,.004,.003,-.003,.0025)


@dataclass(frozen=True)
class Result:
    topology: str
    dofs: int
    elements: int
    integration_points: int
    steps: int
    skfn_1t_ms: float
    skfn_parallel_ms: float
    skfem_ms: float
    native_threads: int


def median_time(function,repeat):
    function()
    samples=[]
    for _ in range(repeat):
        start=time.perf_counter();function()
        samples.append((time.perf_counter()-start)*1e3)
    return statistics.median(samples)


def benchmark(topology,points,repeat,native_threads,check):
    axis=np.linspace(0.,1.,points)
    if topology=="tet":
        mesh=skfemntv.MeshTet.init_tensor(axis,axis,axis)
        element=skfemntv.ElementTetP1()
        reference_mesh=skfem.MeshTet(mesh.p,mesh.t)
        reference_element=skfem.ElementTetP1()
    else:
        mesh=skfemntv.MeshHex.init_tensor(axis,axis,axis)
        element=skfemntv.ElementHex1()
        reference_mesh=skfem.MeshHex(mesh.p,mesh.t)
        reference_element=skfem.ElementHex1()
    basis=skfemntv.Basis(
        mesh,skfemntv.ElementVector(element,dim=3),intorder=2
    )
    reference_basis=skfem.Basis(
        reference_mesh,skfem.ElementVector(reference_element),
        quadrature=(basis.X,basis.W),
    )
    material=skfemntv.J2Plasticity(210.,.3,.25,2.)
    assembler=skfemntv.J2Assembler(basis,material)
    residual_form,tangent_form=reference_forms()
    coordinates=basis.doflocs
    displacements=[]
    for amplitude in HISTORY:
        displacement=np.zeros(basis.N)
        displacement[0::3]=amplitude*coordinates[0,0::3]
        displacement[1::3]=-.5*amplitude*coordinates[1,1::3]
        displacement[2::3]=-.5*amplitude*coordinates[2,2::3]
        displacements.append(displacement)
    entities,quadrature=basis.dx.shape

    def native(threads):
        state=assembler.initial_state();result=None
        for displacement in displacements:
            result=assembler.assemble(
                displacement,state,num_threads=threads
            )
            state=result.trial_state
        return result,state

    def reference():
        plastic=np.zeros((6,entities,quadrature))
        alpha=np.zeros((entities,quadrature));residual=tangent=None
        for displacement in displacements:
            field=reference_basis.interpolate(displacement)
            strain=np.stack((
                field.grad[0,0],field.grad[1,1],field.grad[2,2],
                .5*(field.grad[0,1]+field.grad[1,0]),
                .5*(field.grad[1,2]+field.grad[2,1]),
                .5*(field.grad[0,2]+field.grad[2,0]),
            ))
            stress,constitutive,plastic,alpha=reference_update(
                strain,material.young_modulus,material.poisson_ratio,
                material.yield_stress,material.hardening_modulus,
                plastic,alpha,
            )
            residual=residual_form.assemble(reference_basis,stress=stress)
            tangent=tangent_form.assemble(
                reference_basis,constitutive=constitutive
            )
        return residual,tangent,plastic,alpha

    effective=min(native_threads,skfemntv.available_num_threads())
    if check:
        native_result,native_state=native(effective)
        residual,tangent,plastic,alpha=reference()
        np.testing.assert_allclose(
            native_result.residual,residual,rtol=4e-11,atol=4e-11
        )
        difference=native_result.tangent-tangent
        difference.eliminate_zeros()
        assert (difference.nnz==0
                or np.max(np.abs(difference.data))<4e-10)
        np.testing.assert_allclose(
            native_state.plastic_strain,
            np.moveaxis(plastic,0,-1).reshape(-1,6),
            rtol=4e-11,atol=4e-11,
        )
        np.testing.assert_allclose(
            native_state.equivalent_plastic_strain,alpha.reshape(-1),
            rtol=4e-11,atol=4e-11,
        )
    return Result(
        topology,basis.N,mesh.nelements,assembler.state_count,len(HISTORY),
        median_time(lambda:native(1),repeat),
        median_time(lambda:native(effective),repeat),
        median_time(reference,repeat),effective,
    )


def markdown(results):
    lines=[
        "| Mesh | DoFs | Elements | IPs | Steps | skfemntv 1t [ms] | "
        "skfemntv Nt [ms] | skfem [ms] | threads | skfem/skfemntv Nt |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result.topology} | {result.dofs} | {result.elements} | "
            f"{result.integration_points} | {result.steps} | "
            f"{result.skfn_1t_ms:.3f} | {result.skfn_parallel_ms:.3f} | "
            f"{result.skfem_ms:.3f} | {result.native_threads} | "
            f"{result.skfem_ms/result.skfn_parallel_ms:.2f}x |"
        )
    return "\n".join(lines)


def write_csv(path,results):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(
            stream,fieldnames=Result.__dataclass_fields__,lineterminator="\n"
        )
        writer.writeheader();writer.writerows(asdict(row) for row in results)


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
        title=f"{results[0].topology.title()} J2 load/unload/reload",
    )
    axes[0].grid(True,which="both",alpha=.3);axes[0].legend()
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
    )
    axes[1].grid(True,which="both",alpha=.3);axes[1].legend()
    figure.suptitle(
        f"No solve; median of {repeat} runs after one warm-up; "
        f"strain amplitudes={HISTORY}"
    )
    path.parent.mkdir(parents=True,exist_ok=True)
    figure.savefig(path,dpi=180);plt.close(figure)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--topology",choices=("tet","hex"),default="tet")
    parser.add_argument("--points",nargs="+",type=int,default=[4,6,8,10,12])
    parser.add_argument("--repeat",type=int,default=1)
    parser.add_argument("--native-threads",type=int,default=4)
    parser.add_argument("--no-check",action="store_true")
    parser.add_argument("--output",type=Path)
    parser.add_argument("--plot-output",type=Path)
    arguments=parser.parse_args()
    if arguments.repeat<1 or arguments.native_threads<1 or min(arguments.points)<2:
        parser.error("repeat, thread count, and point counts must be positive")
    results=[]
    for points in arguments.points:
        print(f"benchmarking {points} points/axis...",flush=True)
        results.append(benchmark(
            arguments.topology,points,arguments.repeat,
            arguments.native_threads,not arguments.no_check,
        ))
    print(markdown(results))
    if arguments.output:write_csv(arguments.output,results)
    if arguments.plot_output:write_plot(
        arguments.plot_output,results,arguments.repeat
    )


if __name__=="__main__":
    main()
