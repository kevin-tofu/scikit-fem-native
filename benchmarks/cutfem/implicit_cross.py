"""Segmented implicit cross assembly with independent correctness checks."""
from __future__ import annotations
import argparse,csv,statistics,time
from dataclasses import asdict,dataclass
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.sparse import bmat,coo_matrix
import skfemntv
from skfemntv.bilinear_form import NativeCrossBilinearForm

@dataclass(frozen=True)
class Result:
    resolution:int;dofs_per_side:int;interface_cells:int;intorder:int
    quadrature_points:int;threads:int;contraction:str
    segmented_setup_ms:float;flattened_setup_ms:float
    segmented_assembly_ms:float;flattened_assembly_ms:float
    setup_speedup:float;assembly_speedup:float
    oracle_max_abs_error:float;parallel_max_abs_error:float
    null_mode_max_abs_error:float

def median_time(function,repeat):
    function();samples=[]
    for _ in range(repeat):
        start=time.perf_counter();function()
        samples.append((time.perf_counter()-start)*1.e3)
    return statistics.median(samples)

def flattened(basis):
    return SimpleNamespace(
        mesh=basis.mesh,elem=basis.elem,N=basis.N,dx=basis.dx,
        tabulated_shape=basis.tabulated_shape,
        tabulated_gradients=basis.tabulated_gradients,
        element_dofs=basis.element_dofs,
        global_coordinates=basis.global_coordinates,normals=basis.normals,
    )

def numpy_oracle(test,trial,kind):
    rows=[];columns=[];values=[]
    for cell in test.tind:
        selection=test.cell_slice(int(cell))
        if kind=="value":
            local=np.einsum("q,qa,qb->ab",test.weights[selection],
                            test.shape[selection],trial.shape[selection])
        else:
            local=np.einsum("q,qad,qbd->ab",test.weights[selection],
                            test.gradients[selection],trial.gradients[selection])
        row_dofs=test.cell_dofs[cell,:,0]
        column_dofs=trial.cell_dofs[cell,:,0]
        rows.extend(np.repeat(row_dofs,len(column_dofs)))
        columns.extend(np.tile(column_dofs,len(row_dofs)))
        values.extend(local.ravel())
    return coo_matrix(
        (values,(rows,columns)),shape=(test.N,trial.N)
    ).tocsr()

def sparse_error(left,right):
    difference=left-right
    return float(np.max(np.abs(difference.data),initial=0.))

def benchmark(resolution,intorder,threads,kind,repeat):
    axis=np.linspace(0.,1.,resolution+1)
    mesh=skfemntv.MeshTri.init_tensor(axis,axis)
    rule=skfemntv.LevelSet(lambda x:x[0]-.413,tolerance=0.).interface_quadrature(
        mesh,intorder=intorder)
    element=skfemntv.ElementVector(skfemntv.ElementTriP1(),dim=1)
    negative=skfemntv.ImplicitFacetBasis(skfemntv.Basis(mesh,element),rule,
                                        side="negative")
    positive=skfemntv.ImplicitFacetBasis(skfemntv.Basis(mesh,element),rule,
                                        side="positive")
    segmented=NativeCrossBilinearForm(negative,positive)
    old_negative=flattened(negative);old_positive=flattened(positive)
    old=NativeCrossBilinearForm(old_negative,old_positive)
    coefficient=np.ones(negative.dx.shape)
    assemble=lambda native,workers:native.assemble_kinds(
        kind,kind,coefficient,num_threads=workers)
    serial=assemble(segmented,1);parallel=assemble(segmented,threads)
    old_matrix=assemble(old,threads);oracle=numpy_oracle(negative,positive,kind)
    oracle_error=sparse_error(serial,oracle)
    parallel_error=sparse_error(parallel,serial)
    old_error=sparse_error(old_matrix,serial)
    assert oracle_error<3.e-13 and parallel_error<3.e-13 and old_error<3.e-13
    block=bmat([[oracle,-oracle],[-oracle,oracle]],format="csr")
    null_error=float(np.max(np.abs(block@np.ones(2*negative.N)),initial=0.))
    assert null_error<3.e-13
    segmented_setup=median_time(lambda:NativeCrossBilinearForm(
        negative,positive),repeat)
    flattened_setup=median_time(lambda:NativeCrossBilinearForm(
        old_negative,old_positive),repeat)
    segmented_time=median_time(lambda:assemble(segmented,threads),repeat)
    flattened_time=median_time(lambda:assemble(old,threads),repeat)
    return Result(resolution,negative.N,len(negative.tind),intorder,
        negative.npoints,threads,kind,segmented_setup,flattened_setup,
        segmented_time,flattened_time,flattened_setup/segmented_setup,
        flattened_time/segmented_time,oracle_error,parallel_error,null_error)

def write_csv(path,results):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=Result.__dataclass_fields__)
        writer.writeheader();writer.writerows(asdict(result) for result in results)

def write_plot(path,results,repeat):
    import matplotlib.pyplot as plt
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    figure,axis=plt.subplots(figsize=(8,5))
    for kind in sorted({result.contraction for result in results}):
        selected=[result for result in results if result.contraction==kind]
        axis.plot([r.intorder for r in selected],[r.assembly_speedup for r in selected],
                  "o-",label=kind)
    axis.axhline(1.,color="black",linewidth=.8)
    axis.set(xlabel="integration order",ylabel="flattened / segmented time",
             title=f"Implicit cross speedup (median of {repeat})")
    axis.grid(True,alpha=.3);axis.legend();figure.tight_layout()
    figure.savefig(path,dpi=160);plt.close(figure)

def report(results):
    lines=["| kind | order | qpoints | setup speedup | assembly speedup | oracle error | parallel error |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for r in results:lines.append(
        f"| {r.contraction} | {r.intorder} | {r.quadrature_points} | "
        f"{r.setup_speedup:.2f}x | {r.assembly_speedup:.2f}x | "
        f"{r.oracle_max_abs_error:.2e} | {r.parallel_max_abs_error:.2e} |")
    return "\n".join(lines)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--resolution",type=int,default=128)
    parser.add_argument("--intorders",type=int,nargs="+",default=[1,2,4,6])
    parser.add_argument("--contractions",nargs="+",choices=["value","gradient"],
                        default=["value","gradient"])
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--repeat",type=int,default=3)
    parser.add_argument("--output");parser.add_argument("--plot-output")
    args=parser.parse_args();workers=min(args.threads,skfemntv.available_num_threads())
    results=[benchmark(args.resolution,order,workers,kind,args.repeat)
             for kind in args.contractions for order in args.intorders]
    if args.output:write_csv(args.output,results)
    if args.plot_output:write_plot(args.plot_output,results,args.repeat)
    print(report(results))
if __name__=="__main__":main()
