"""Neo-Hookean assembly scaling across 3D mesh types and element orders."""

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

import skfemntv

sys.path.insert(0,str(Path(__file__).parents[1]))
from skfem_neo_hookean import forms as reference_forms


@dataclass(frozen=True)
class Result:
    topology: str
    intorder: int
    points: int
    dofs: int
    elements: int
    quadrature_points: int
    skfn_r_t1_ms: float
    skfn_r_t2_ms: float
    skfn_r_t4_ms: float
    skfn_rk_t1_ms: float
    skfn_rk_t2_ms: float
    skfn_rk_t4_ms: float
    skfem_rk_ms: float
    skfn_csr_mb: float
    skfem_csr_mb: float
    effective_t2: int
    effective_t4: int


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


def spaces(topology,points,intorder,distorted):
    axis=np.linspace(0.,1.,points)
    if topology.startswith("tet"):
        linear=skfemntv.MeshTet.init_tensor(axis,axis,axis)
        reference_linear=skfem.MeshTet(linear.p,linear.t)
        if topology=="tet10":
            mesh=skfemntv.MeshTet2.from_mesh(linear)
            element=skfemntv.ElementTetP2()
            reference_mesh=skfem.MeshTet2.from_mesh(reference_linear)
            reference_element=skfem.ElementTetP2()
        else:
            mesh=linear;element=skfemntv.ElementTetP1()
            reference_mesh=reference_linear
            reference_element=skfem.ElementTetP1()
    elif topology.startswith("hex"):
        linear=skfemntv.MeshHex.init_tensor(axis,axis,axis)
        reference_linear=skfem.MeshHex(linear.p,linear.t)
        if topology=="hex27":
            mesh=skfemntv.MeshHex2.from_mesh(linear)
            element=skfemntv.ElementHex2()
            reference_mesh=skfem.MeshHex2.from_mesh(reference_linear)
            reference_element=skfem.ElementHex2()
        else:
            mesh=linear;element=skfemntv.ElementHex1()
            reference_mesh=reference_linear
            reference_element=skfem.ElementHex1()
    else:
        mesh=skfemntv.MeshWedge1.init_tensor(axis,axis,axis)
        element=skfemntv.ElementWedge1()
        reference_mesh=skfem.MeshWedge1(mesh.p,mesh.t)
        reference_element=skfem.ElementWedge1()
    if distorted:
        mesh=type(mesh)(distort(mesh.p),mesh.t)
        reference_mesh=type(reference_mesh)(
            distort(reference_mesh.p),reference_mesh.t
        )
    native=skfemntv.Basis(
        mesh,skfemntv.ElementVector(element,dim=3),intorder=intorder
    )
    reference=skfem.Basis(
        reference_mesh,skfem.ElementVector(reference_element),
        quadrature=(native.X,native.W),
    )
    return native,reference


def permutation(native,reference):
    lookup={
        tuple(np.round(reference.doflocs[:,3*node],13)):node
        for node in range(reference.N//3)
    }
    result=[]
    for node in range(native.N//3):
        other=lookup[tuple(np.round(native.doflocs[:,3*node],13))]
        result.extend(3*other+component for component in range(3))
    return np.asarray(result)


def csr_mb(matrix):
    return (
        matrix.data.nbytes+matrix.indices.nbytes+matrix.indptr.nbytes
    )/1024.**2


def benchmark(topology,points,intorder,distorted,repeat,check):
    basis,reference=spaces(topology,points,intorder,distorted)
    order=permutation(basis,reference)
    young,poisson=100.,.3
    mu=young/(2.*(1.+poisson))
    lmbda=young*poisson/((1.+poisson)*(1.-2.*poisson))
    assembler=skfemntv.NativeAssembler.from_basis(
        basis,skfemntv.NeoHookean(mu,lmbda)
    )
    residual_form,tangent_form=reference_forms(mu,lmbda)
    coordinates=basis.doflocs
    u=np.empty(basis.N,dtype=np.float64)
    u[0::3]=1e-3*coordinates[0,0::3]*coordinates[1,0::3]
    u[1::3]=-7e-4*coordinates[1,1::3]*coordinates[2,1::3]
    u[2::3]=5e-4*coordinates[2,2::3]*coordinates[0,2::3]
    reference_u=np.empty(reference.N);reference_u[order]=u

    def reference_assembly():
        field=reference.interpolate(reference_u)
        return (
            residual_form.assemble(reference,displacement=field),
            tangent_form.assemble(reference,displacement=field),
        )

    native=assembler.assemble(u,num_threads=1)
    expected_residual,expected_tangent=reference_assembly()
    if check:
        np.testing.assert_allclose(
            native.residual,expected_residual[order],rtol=5e-10,atol=5e-10
        )
        np.testing.assert_allclose(
            native.tangent.toarray(),expected_tangent[order][:,order].toarray(),
            rtol=5e-10,atol=5e-10,
        )
    available=skfemntv.available_num_threads()
    t2=min(2,available);t4=min(4,available)
    timed=lambda mode,threads:median_time(
        lambda:assembler.assemble(u,mode=mode,num_threads=threads),repeat
    )
    return Result(
        topology=topology,intorder=intorder,points=points,dofs=basis.N,
        elements=basis.mesh.nelements,quadrature_points=basis.X.shape[1],
        skfn_r_t1_ms=timed("residual",1),
        skfn_r_t2_ms=timed("residual",t2),
        skfn_r_t4_ms=timed("residual",t4),
        skfn_rk_t1_ms=timed("residual_tangent",1),
        skfn_rk_t2_ms=timed("residual_tangent",t2),
        skfn_rk_t4_ms=timed("residual_tangent",t4),
        skfem_rk_ms=median_time(reference_assembly,repeat),
        skfn_csr_mb=csr_mb(native.tangent),
        skfem_csr_mb=csr_mb(expected_tangent),
        effective_t2=t2,effective_t4=t4,
    )


def markdown(results):
    lines=[
        "| Mesh | DoFs | Elems | QP/elem | skfemntv R t1/t2/t4 [ms] | "
        "skfemntv R+K t1/t2/t4 [ms] | skfem R+K [ms] | "
        "speedup t4 | CSR skfemntv/skfem [MiB] |",
        "|:---|---:|---:|---:|:---|:---|---:|---:|:---|",
    ]
    for r in results:
        lines.append(
            f"| {r.topology} | {r.dofs} | {r.elements} | "
            f"{r.quadrature_points} | "
            f"{r.skfn_r_t1_ms:.3f}/{r.skfn_r_t2_ms:.3f}/{r.skfn_r_t4_ms:.3f} | "
            f"{r.skfn_rk_t1_ms:.3f}/{r.skfn_rk_t2_ms:.3f}/{r.skfn_rk_t4_ms:.3f} | "
            f"{r.skfem_rk_ms:.3f} | {r.skfem_rk_ms/r.skfn_rk_t4_ms:.2f}x | "
            f"{r.skfn_csr_mb:.2f}/{r.skfem_csr_mb:.2f} |"
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
        (axes[0],"skfn_r_","Residual only"),
        (axes[1],"skfn_rk_","Residual + tangent"),
    ):
        for suffix,label,style in (
            ("t1_ms","skfemntv t1","o-"),("t2_ms","skfemntv t2","s-"),
            ("t4_ms","skfemntv t4","^-"),
        ):
            axis.loglog(
                dofs,[getattr(r,prefix+suffix) for r in results],
                style,label=label,linewidth=2,
            )
        if prefix=="skfn_rk_":
            axis.loglog(
                dofs,[r.skfem_rk_ms for r in results],"d-",
                label="scikit-fem",linewidth=2,
            )
        axis.set(xlabel="Degrees of freedom",ylabel="Time [ms]",title=title)
        axis.grid(True,which="both",alpha=.3);axis.legend()
    axes[2].semilogx(
        dofs,[r.skfem_rk_ms/r.skfn_rk_t1_ms for r in results],
        "o-",label="skfemntv t1",linewidth=2,
    )
    axes[2].semilogx(
        dofs,[r.skfem_rk_ms/r.skfn_rk_t4_ms for r in results],
        "^-",label="skfemntv t4",linewidth=2,
    )
    axes[2].axhline(1.,color="black",linestyle="--",linewidth=1)
    axes[2].set(
        xlabel="Degrees of freedom",ylabel="scikit-fem / skfemntv",
        title="R+K speedup (>1 is faster)",
    )
    axes[2].grid(True,which="both",alpha=.3);axes[2].legend()
    first=results[0]
    figure.suptitle(
        f"{first.topology} Neo-Hookean, intorder={first.intorder}; "
        f"median of {repeat} timed run(s) after warm-up"
    )
    path.parent.mkdir(parents=True,exist_ok=True)
    figure.savefig(path,dpi=180);plt.close(figure)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument(
        "--topology",choices=("tet4","tet10","hex8","hex27","wedge6"),
        default="tet4",
    )
    parser.add_argument("--intorder",type=int,default=4)
    parser.add_argument("--points",nargs="+",type=int,default=[2,3,4,5,6])
    parser.add_argument("--repeat",type=int,default=1)
    parser.add_argument("--distorted",action="store_true")
    parser.add_argument("--no-check",action="store_true")
    parser.add_argument("--output",type=Path)
    parser.add_argument("--plot-output",type=Path)
    args=parser.parse_args()
    if args.repeat<1 or args.intorder<1 or any(p<2 for p in args.points):
        parser.error("repeat/intorder must be positive and points must be >= 2")
    results=[]
    for points in args.points:
        print(f"benchmarking {args.topology}: {points} points/axis...",flush=True)
        results.append(benchmark(
            args.topology,points,args.intorder,args.distorted,args.repeat,
            not args.no_check,
        ))
    print(markdown(results))
    if args.output:write_csv(args.output,results)
    if args.plot_output:write_plot(args.plot_output,results,args.repeat)


if __name__=="__main__":
    main()
