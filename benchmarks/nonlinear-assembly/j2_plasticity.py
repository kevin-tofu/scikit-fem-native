"""Small-strain J2 material and fused global assembly scaling benchmark."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict,dataclass
import os
from pathlib import Path
import statistics
import tempfile
import time

import numpy as np
import skfem

import skfn

import sys
sys.path.insert(0,str(Path(__file__).parents[1]))
from skfem_j2 import forms as reference_forms,update as reference_update


@dataclass(frozen=True)
class Result:
    topology: str
    intorder: int
    points: int
    dofs: int
    elements: int
    integration_points: int
    material_1t_ms: float
    material_parallel_ms: float
    residual_1t_ms: float
    residual_parallel_ms: float
    tangent_1t_ms: float
    tangent_parallel_ms: float
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


def benchmark(topology,points,intorder,repeat,native_threads,check):
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
    basis=skfn.Basis(
        mesh,skfn.ElementVector(element,dim=3),intorder=intorder
    )
    material=skfn.J2Plasticity(
        young_modulus=210.,poisson_ratio=.3,
        yield_stress=.25,hardening_modulus=2.,
    )
    assembler=skfn.J2Assembler(basis,material)
    reference_basis=skfem.Basis(
        reference_mesh_type(mesh.p,mesh.t),
        skfem.ElementVector(reference_element),quadrature=(basis.X,basis.W),
    )
    residual_form,tangent_form=reference_forms()
    state=assembler.initial_state()
    coordinates=basis.doflocs
    displacement=np.zeros(basis.N,dtype=np.float64)
    displacement[0::3]=.004*coordinates[0,0::3]
    displacement[1::3]=-.002*coordinates[1,1::3]
    displacement[2::3]=-.002*coordinates[2,2::3]
    strains=np.tile(
        np.array([.004,-.002,-.002,.0004,0.,0.]),
        (assembler.state_count,1),
    )
    effective=min(native_threads,skfn.available_num_threads())

    def reference():
        field=reference_basis.interpolate(displacement)
        strain=np.stack((
            field.grad[0,0],field.grad[1,1],field.grad[2,2],
            .5*(field.grad[0,1]+field.grad[1,0]),
            .5*(field.grad[1,2]+field.grad[2,1]),
            .5*(field.grad[0,2]+field.grad[2,0]),
        ))
        stress,constitutive,_,_=reference_update(
            strain,material.young_modulus,material.poisson_ratio,
            material.yield_stress,material.hardening_modulus,
        )
        return (
            residual_form.assemble(reference_basis,stress=stress),
            tangent_form.assemble(
                reference_basis,constitutive=constitutive
            ),
        )

    if check:
        serial=assembler.assemble(displacement,state,num_threads=1)
        residual=serial.residual.copy()
        tangent=serial.tangent.copy()
        trial_plastic=serial.trial_state.plastic_strain.copy()
        trial_alpha=serial.trial_state.equivalent_plastic_strain.copy()
        parallel=assembler.assemble(
            displacement,state,num_threads=effective
        )
        np.testing.assert_allclose(
            parallel.residual,residual,rtol=3e-13,atol=3e-13
        )
        np.testing.assert_array_equal(parallel.tangent.indptr,tangent.indptr)
        np.testing.assert_array_equal(parallel.tangent.indices,tangent.indices)
        np.testing.assert_allclose(
            parallel.tangent.data,tangent.data,rtol=3e-13,atol=3e-13
        )
        np.testing.assert_allclose(
            parallel.trial_state.plastic_strain,trial_plastic,
            rtol=3e-13,atol=3e-13,
        )
        np.testing.assert_allclose(
            parallel.trial_state.equivalent_plastic_strain,trial_alpha,
            rtol=3e-13,atol=3e-13,
        )
        assert np.max(trial_alpha)>0.
        reference_residual,reference_tangent=reference()
        np.testing.assert_allclose(
            residual,reference_residual,rtol=3e-11,atol=3e-11
        )
        difference=tangent-reference_tangent
        difference.eliminate_zeros()
        scale=max(1.,np.max(np.abs(reference_tangent.data)))
        assert (difference.nnz==0
                or np.max(np.abs(difference.data))<=3e-11*scale)

    return Result(
        topology=topology,intorder=intorder,points=points,dofs=basis.N,
        elements=mesh.nelements,integration_points=assembler.state_count,
        material_1t_ms=median_time(
            lambda:material.evaluate(strains,state,num_threads=1),repeat
        ),
        material_parallel_ms=median_time(
            lambda:material.evaluate(strains,state,num_threads=effective),repeat
        ),
        residual_1t_ms=median_time(
            lambda:assembler.assemble(
                displacement,state,mode="residual",num_threads=1
            ),repeat
        ),
        residual_parallel_ms=median_time(
            lambda:assembler.assemble(
                displacement,state,mode="residual",num_threads=effective
            ),repeat
        ),
        tangent_1t_ms=median_time(
            lambda:assembler.assemble(displacement,state,num_threads=1),repeat
        ),
        tangent_parallel_ms=median_time(
            lambda:assembler.assemble(
                displacement,state,num_threads=effective
            ),repeat
        ),
        skfem_tangent_ms=median_time(reference,repeat),
        native_threads=effective,
    )


def markdown(results):
    lines=[
        "| Mesh | Order | DoFs | Elements | IPs | Material 1t/Nt [ms] | "
        "R 1t/Nt [ms] | R+K 1t/Nt [ms] | skfem R+K [ms] | threads | "
        "skfem/skfn Nt |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result.topology} | {result.intorder} | {result.dofs} | "
            f"{result.elements} | {result.integration_points} | "
            f"{result.material_1t_ms:.3f}/{result.material_parallel_ms:.3f} | "
            f"{result.residual_1t_ms:.3f}/{result.residual_parallel_ms:.3f} | "
            f"{result.tangent_1t_ms:.3f}/{result.tangent_parallel_ms:.3f} | "
            f"{result.skfem_tangent_ms:.3f} | {result.native_threads} | "
            f"{result.skfem_tangent_ms/result.tangent_parallel_ms:.2f}x |"
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
    os.environ.setdefault(
        "MPLCONFIGDIR",str(Path(tempfile.gettempdir())/"skfn-matplotlib")
    )
    import matplotlib.pyplot as plt

    dofs=np.array([result.dofs for result in results])
    threads=results[0].native_threads
    figure,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    for field,label,style in (
        ("material_1t_ms","material update (1 thread)","o--"),
        ("material_parallel_ms",f"material update ({threads} threads)","^--"),
        ("tangent_1t_ms","fused R+K (1 thread)","o-"),
        ("tangent_parallel_ms",f"fused R+K ({threads} threads)","^-"),
        ("skfem_tangent_ms","scikit-fem R+K","s-"),
    ):
        axes[0].loglog(
            dofs,[getattr(result,field) for result in results],
            style,label=label,linewidth=2,
        )
    axes[0].set(
        xlabel="Degrees of freedom",ylabel="Time [ms]",
        title=(
            f"{results[0].topology.title()} J2 plasticity "
            f"(intorder={results[0].intorder})"
        ),
    )
    axes[0].grid(True,which="both",alpha=.3);axes[0].legend()
    for numerator,denominator,label,style in (
        ("tangent_1t_ms","tangent_parallel_ms","native parallel scaling","o--"),
        ("skfem_tangent_ms","tangent_1t_ms","skfem / skfn 1 thread","s-"),
        ("skfem_tangent_ms","tangent_parallel_ms",f"skfem / skfn {threads} threads","^-"),
    ):
        axes[1].semilogx(
            dofs,[getattr(result,numerator)/getattr(result,denominator)
                  for result in results],style,label=label,linewidth=2,
        )
    axes[1].axhline(1.,color="black",linestyle="--",linewidth=1)
    axes[1].set(
        xlabel="Degrees of freedom",ylabel="1-thread time / parallel time",
        title="Assembly speedup (>1 is faster)",
    )
    axes[1].grid(True,which="both",alpha=.3);axes[1].legend()
    figure.suptitle(
        f"Integration points shown in CSV; median of {repeat} timed runs "
        "after one warm-up"
    )
    path.parent.mkdir(parents=True,exist_ok=True)
    figure.savefig(path,dpi=180);plt.close(figure)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--topology",choices=("tet","hex"),default="tet")
    parser.add_argument("--intorder",type=int,default=2)
    parser.add_argument(
        "--points",nargs="+",type=int,default=[4,6,8,10,12,16]
    )
    parser.add_argument("--repeat",type=int,default=1)
    parser.add_argument("--native-threads",type=int,default=4)
    parser.add_argument("--no-check",action="store_true")
    parser.add_argument("--output",type=Path)
    parser.add_argument("--plot-output",type=Path)
    arguments=parser.parse_args()
    if (arguments.repeat<1 or arguments.native_threads<1
            or arguments.intorder<1 or min(arguments.points)<2):
        parser.error("repeat, thread count, intorder, and point counts must be positive")
    results=[]
    for points in arguments.points:
        print(f"benchmarking {points} points/axis...",flush=True)
        results.append(benchmark(
            arguments.topology,points,arguments.intorder,arguments.repeat,
            arguments.native_threads,not arguments.no_check,
        ))
    print(markdown(results))
    if arguments.output:write_csv(arguments.output,results)
    if arguments.plot_output:write_plot(
        arguments.plot_output,results,arguments.repeat
    )


if __name__=="__main__":
    main()
