from __future__ import annotations

import numpy as np
from scipy.sparse import bmat

from .bilinear_form import NativeCrossBilinearForm
from .cut_basis import ImplicitFacetBasis
from .linear_form import NativeLinearForm


class ImplicitInterfacePair:
    """Two independent P1 traces sharing one reconstructed interface."""

    def __init__(self,negative: ImplicitFacetBasis,positive: ImplicitFacetBasis):
        if not isinstance(negative,ImplicitFacetBasis) or not isinstance(
            positive,ImplicitFacetBasis
        ):
            raise TypeError("implicit interface pair requires two ImplicitFacetBasis values")
        if negative.side!="negative" or positive.side!="positive":
            raise ValueError("implicit interface pair requires negative then positive sides")
        if not np.array_equal(negative.cell_offsets,positive.cell_offsets):
            raise ValueError("implicit traces have different cell offsets")
        if not np.allclose(negative.global_coordinates,positive.global_coordinates):
            raise ValueError("implicit traces have different physical points")
        if not np.allclose(negative.dx,positive.dx):
            raise ValueError("implicit traces have different weights")
        opposition=np.linalg.norm(
            negative.normal_vectors+positive.normal_vectors,axis=1
        )
        if np.max(opposition,initial=0.)>1.e-12:
            raise ValueError("implicit trace normals are not opposite")
        self.negative=negative;self.positive=positive
        self.master_size=negative.N;self.slave_size=positive.N
        self.global_coordinates=negative.global_coordinates
        self.master_normals=negative.normals
        self.slave_normals=positive.normals
        self._weights=negative.dx
        self._coefficient_shape=negative.dx.shape
        self.gap=np.zeros(self._coefficient_shape,dtype=np.float64)
        self.gap.flags.writeable=False
        self._cross_cache={};self._linear_cache={}

    def _basis(self,index):
        return self.negative if index==0 else self.positive

    def _cross(
        self,row,column,kind_row,kind_column,coefficient,num_threads,
    ):
        key=(row,column)
        assembler=self._cross_cache.get(key)
        if assembler is None:
            assembler=NativeCrossBilinearForm(
                self._basis(row),self._basis(column)
            )
            self._cross_cache[key]=assembler
        return assembler.assemble_kinds(
            kind_row,kind_column,coefficient,num_threads=num_threads or 0,
        )

    def assemble_traces(
        self,row_weights,column_weights,*,row_kind="value",
        column_kind="value",coefficient=1.,num_threads=None,
    ):
        valid={"value","gradient","normal_gradient"}
        if row_kind not in valid or column_kind not in valid:
            raise ValueError("trace kind must be value, gradient, or normal_gradient")
        row_weights=np.asarray(row_weights,dtype=np.float64)
        column_weights=np.asarray(column_weights,dtype=np.float64)
        if row_weights.shape!=(2,) or column_weights.shape!=(2,):
            raise ValueError("implicit trace weights must contain two values")
        blocks=[]
        for row in range(2):
            block_row=[]
            for column in range(2):
                block_row.append(
                    row_weights[row]*column_weights[column]*self._cross(
                        row,column,row_kind,column_kind,coefficient,num_threads
                    )
                )
            blocks.append(block_row)
        return bmat(blocks,format="csr")

    def _linear(self,index,kind,coefficient,num_threads):
        basis=self._basis(index)
        assembler=self._linear_cache.get(index)
        if assembler is None:
            assembler=NativeLinearForm(basis);self._linear_cache[index]=assembler
        components=basis.elem._dim;dimension=basis.mesh.dim()
        coefficient=np.asarray(coefficient,dtype=np.float64)
        value=None;gradient=None
        if kind=="value":
            if coefficient.ndim==0:
                value=np.full(components,float(coefficient))
            else:
                value=np.broadcast_to(
                    coefficient,basis.dx.shape+(components,)
                ).copy()
        elif kind=="normal_gradient":
            scalar=np.broadcast_to(coefficient,basis.dx.shape)
            gradient=(
                scalar[...,None,None]
                *basis.normals[...,None,:]
            )
            gradient=np.broadcast_to(
                gradient,basis.dx.shape+(components,dimension)
            ).copy()
        elif kind=="gradient":
            gradient=np.broadcast_to(
                coefficient,basis.dx.shape+(components,dimension)
            ).copy()
        else:
            raise ValueError("trace kind must be value, gradient, or normal_gradient")
        result,_=assembler.assemble(
            value=value,gradient=gradient,num_threads=num_threads or 0
        )
        if len(result)<basis.N:result=np.pad(result,(0,basis.N-len(result)))
        return result

    def assemble_linear_trace(
        self,trace_weights,*,trace_kind="value",coefficient=1.,
        num_threads=None,
    ):
        trace_weights=np.asarray(trace_weights,dtype=np.float64)
        if trace_weights.shape!=(2,):
            raise ValueError("implicit trace weights must contain two values")
        negative=self._linear(0,trace_kind,coefficient,num_threads)
        positive=self._linear(1,trace_kind,coefficient,num_threads)
        return np.concatenate((
            trace_weights[0]*negative,trace_weights[1]*positive
        ))
