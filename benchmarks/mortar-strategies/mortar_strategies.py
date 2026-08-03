"""Four tied-mortar strategies assembled through the public skfemntv API."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict,dataclass
from pathlib import Path
import statistics
import time

import numpy as np
from scipy.linalg import qr
from scipy.sparse import csr_matrix,vstack

import skfemntv


@dataclass(frozen=True)
class Result:
    cells: int
    master_facets: int
    slave_facets: int
    overlap_cells: int
    strategy: str
    multiplier_rows: int
    matrix_nnz: int
    matrix_mib: float
    search_ms: float
    assembly_ms: float
    reduction_ms: float
    maximum_local_rows: int
    constant_residual: float


def surface(cells,reverse=False):
    axis=np.linspace(0.,1.,cells+1)
    points=np.array(
        [(x,y,0.) for y in axis for x in axis],dtype=np.float64
    ).T
    triangles=[]
    for j in range(cells):
        for i in range(cells):
            lower=j*(cells+1)+i
            a=lower;b=lower+1;c=lower+cells+1;d=c+1
            triangles.extend(
                ((a,b,c),(b,d,c)) if reverse else ((a,b,d),(a,d,c))
            )
    return points,np.asarray(triangles,dtype=np.int64).T


def matrix_mib(matrix):
    return (
        matrix.data.nbytes+matrix.indices.nbytes+matrix.indptr.nbytes
    )/2.**20


def measured(function,repeat):
    samples=[];value=None
    for _ in range(repeat):
        start=time.perf_counter();value=function()
        samples.append((time.perf_counter()-start)*1.e3)
    return value,statistics.median(samples)


def reduce_local_rows(matrix,parent_facets,method,tolerance):
    """Reduce tied rows facet-locally; never materialize the global dense C."""
    blocks=[];maximum=0
    for facet in np.unique(parent_facets):
        rows=np.flatnonzero(parent_facets==facet)
        maximum=max(maximum,len(rows))
        block=matrix[rows]
        columns=np.unique(block.indices)
        dense=block[:,columns].toarray()
        if method=="algebraic-qr":
            _,factor,pivots=qr(dense.T,mode="economic",pivoting=True)
            diagonal=np.abs(np.diag(factor))
            scale=float(diagonal.max(initial=0.))
            rank=int(np.count_nonzero(diagonal>tolerance*scale))
            if rank:
                blocks.append(csr_matrix(block[pivots[:rank]]))
        elif method=="algebraic-svd":
            _,singular,right=np.linalg.svd(dense,full_matrices=False)
            scale=float(singular.max(initial=0.))
            rank=int(np.count_nonzero(singular>tolerance*scale))
            if rank:
                local=csr_matrix(
                    (singular[:rank,None]*right[:rank]),shape=(rank,len(columns))
                )
                row,column=local.nonzero()
                blocks.append(csr_matrix(
                    (local.data,(row,columns[column])),
                    shape=(rank,matrix.shape[1]),
                ))
        else:
            raise ValueError("method must be algebraic-qr or algebraic-svd")
    return (
        csr_matrix(vstack(blocks,format="csr"))
        if blocks else csr_matrix((0,matrix.shape[1]),dtype=np.float64)
    ),maximum


def benchmark(cells,repeat,tolerance,threads):
    master_points,master_triangles=surface(cells)
    slave_points,slave_triangles=surface(cells+1,reverse=True)
    start=time.perf_counter()
    integration=skfemntv.TriangleSupermesh(
        master_points,master_triangles,slave_points,slave_triangles
    )
    search_ms=(time.perf_counter()-start)*1.e3

    fine,fine_ms=measured(
        lambda:integration.assemble_mortar(
            "overlap_p0",num_threads=threads
        ),repeat,
    )
    coarse,coarse_ms=measured(
        lambda:integration.assemble_mortar(
            "slave_facet_p0",num_threads=threads
        ),repeat,
    )
    matrices={
        "fine":(fine.coupling_matrix,fine_ms,0.,0),
        "coarse-p0":(coarse.coupling_matrix,coarse_ms,0.,0),
    }
    parents=integration.slave_trace.parent_facets
    for method in ("algebraic-qr","algebraic-svd"):
        (matrix,maximum),elapsed=measured(
            lambda method=method:reduce_local_rows(
                fine.coupling_matrix,parents,method,tolerance
            ),repeat,
        )
        matrices[method]=(matrix,fine_ms,elapsed,maximum)

    ones=np.ones(fine.coupling_matrix.shape[1])
    return [
        Result(
            cells=cells,master_facets=master_triangles.shape[1],
            slave_facets=slave_triangles.shape[1],
            overlap_cells=integration.diagnostics.integration_triangle_count,
            strategy=strategy,multiplier_rows=matrix.shape[0],
            matrix_nnz=matrix.nnz,matrix_mib=matrix_mib(matrix),
            search_ms=search_ms,assembly_ms=assembly_ms,
            reduction_ms=reduction_ms,maximum_local_rows=maximum,
            constant_residual=float(np.linalg.norm(matrix@ones,ord=np.inf)),
        )
        for strategy,(matrix,assembly_ms,reduction_ms,maximum) in matrices.items()
    ]


def write_csv(path,results):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=Result.__dataclass_fields__)
        writer.writeheader();writer.writerows(asdict(result) for result in results)


def write_plot(path,results):
    import matplotlib.pyplot as plt
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    figure,axes=plt.subplots(1,2,figsize=(11,4.5))
    strategies=tuple(dict.fromkeys(result.strategy for result in results))
    for strategy in strategies:
        selected=[result for result in results if result.strategy==strategy]
        axes[0].plot(
            [result.overlap_cells for result in selected],
            [result.multiplier_rows for result in selected],"o-",label=strategy,
        )
        axes[1].plot(
            [result.overlap_cells for result in selected],
            [result.assembly_ms+result.reduction_ms for result in selected],
            "o-",label=strategy,
        )
    axes[0].set_ylabel("multiplier rows")
    axes[1].set_ylabel("assembly + reduction [ms]")
    for axis in axes:
        axis.set_xlabel("overlap cells");axis.set_xscale("log")
        axis.set_yscale("log");axis.grid(True,alpha=.3);axis.legend()
    figure.suptitle("Tied mortar strategies (QR/SVD outside skfemntv)")
    figure.tight_layout();figure.savefig(path,dpi=160);plt.close(figure)


def markdown(results):
    lines=[
        "| cells | overlaps | strategy | rows | nnz | assembly ms | "
        "reduction ms | max local rows | constant residual |",
        "|---:|---:|:---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result.cells} | {result.overlap_cells} | {result.strategy} | "
            f"{result.multiplier_rows} | {result.matrix_nnz} | "
            f"{result.assembly_ms:.3f} | {result.reduction_ms:.3f} | "
            f"{result.maximum_local_rows} | {result.constant_residual:.3e} |"
        )
    return "\n".join(lines)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--cells",default="4,8,16")
    parser.add_argument("--repeat",type=int,default=1)
    parser.add_argument("--rank-tolerance",type=float,default=1.e-10)
    parser.add_argument("--threads",type=int,default=1)
    parser.add_argument("--output")
    parser.add_argument("--plot-output")
    args=parser.parse_args()
    cells=[int(value) for value in args.cells.split(",")]
    if any(value<1 for value in cells) or args.repeat<1:
        parser.error("cells and repeat must be positive")
    if not 0.<args.rank_tolerance<1.:
        parser.error("rank-tolerance must be between zero and one")
    threads=min(args.threads,skfemntv.available_num_threads())
    results=[
        result for cells_value in cells
        for result in benchmark(
            cells_value,args.repeat,args.rank_tolerance,threads
        )
    ]
    if any(result.constant_residual>1.e-10 for result in results):
        raise RuntimeError("a strategy failed constant tied-field reproduction")
    if args.output:write_csv(args.output,results)
    if args.plot_output:write_plot(args.plot_output,results)
    print(markdown(results))


if __name__=="__main__":main()
