"""Profile native cross-basis contraction versus fixed-CSR scatter.

The synthetic workload uses overlapping four-node entities, three field
components, three physical gradient directions, and six quadrature points.
Initialization and sparsity construction are outside all timings.
"""

import argparse
import statistics

import numpy as np

from skfemntv._skfn import CrossBilinearAssembler


def median_native(call, repeats):
    for _ in range(2):
        call()
    samples=[call()[1] for _ in range(repeats)]
    return statistics.median(samples)


def assembler(entity_count):
    nodes=4
    components=3
    quadrature=6
    scalar_nodes=np.arange(entity_count)[:,None]+np.arange(nodes)[None,:]
    dofs=np.stack(
        [components*scalar_nodes+c for c in range(components)],axis=2
    )
    rng=np.random.default_rng(1)
    row_shape=rng.random((entity_count,quadrature,nodes))
    row_shape/=row_shape.sum(axis=2,keepdims=True)
    column_shape=rng.random((entity_count,quadrature,nodes))
    column_shape/=column_shape.sum(axis=2,keepdims=True)
    row_gradient=rng.normal(size=(entity_count,quadrature,nodes,3))
    row_gradient-=row_gradient.mean(axis=2,keepdims=True)
    column_gradient=rng.normal(size=(entity_count,quadrature,nodes,3))
    column_gradient-=column_gradient.mean(axis=2,keepdims=True)
    weights=np.full((entity_count,quadrature),1./quadrature)
    native=CrossBilinearAssembler(
        np.ascontiguousarray(dofs,dtype=np.int64),
        np.ascontiguousarray(dofs,dtype=np.int64),
        np.ascontiguousarray(row_shape),
        np.ascontiguousarray(column_shape),
        np.ascontiguousarray(weights),
        np.ascontiguousarray(row_gradient),
        np.ascontiguousarray(column_gradient),
    )
    scalar=np.ones((entity_count,quadrature))
    mixed=np.ones((entity_count,quadrature,3,3,3))
    return native,scalar,mixed


def run(entity_count,repeats):
    native,scalar,mixed=assembler(entity_count)
    cases=(
        ("value-value",scalar,"value","value"),
        ("value-gradient",mixed,"value","gradient"),
        ("gradient-gradient",scalar,"gradient","gradient"),
    )
    print(
        f"entities={entity_count} quadrature=6 nodes=4 components=3 "
        f"nnz={len(native.values)}"
    )
    for name,coefficient,row_kind,column_kind in cases:
        full=median_native(
            lambda:native.assemble(coefficient,row_kind,column_kind),repeats
        )
        contraction=median_native(
            lambda:native.contract_only(
                coefficient,row_kind,column_kind
            ),
            repeats,
        )
        remainder=max(0.,full-contraction)
        fraction=contraction/full if full else 0.
        print(
            f"{name:17} total={1e3*full:9.3f} ms "
            f"contract={1e3*contraction:9.3f} ms "
            f"scatter-est={1e3*remainder:9.3f} ms "
            f"contract/total={100*fraction:6.1f}%"
        )


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--entities",type=int,nargs="*",default=[1000,10000])
    parser.add_argument("--repeats",type=int,default=9)
    args=parser.parse_args()
    for count in args.entities:
        run(count,args.repeats)
