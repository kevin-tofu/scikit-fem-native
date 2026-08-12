"""Reusable sparse assembly for the experimental affine TriN1 basis."""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix,csr_matrix

from .hcurl_basis import AffineTriN1Basis
from .preflight import AssemblyMemoryEstimate,enforce_memory_budget


def estimate_tri_n1_assembly_memory(basis):
    """Conservatively estimate the dedicated Python/CSR assembler storage."""
    if not isinstance(basis,AffineTriN1Basis):
        raise TypeError("basis must be AffineTriN1Basis")
    cells=basis.mesh.nelements
    local=3
    nnz_upper=min(basis.N*basis.N,cells*local*local)
    basis_bytes=sum(
        array.nbytes for array in (
            basis._element_values,basis._element_curls,
            basis.dx,basis.element_dofs,
        )
    )
    return AssemblyMemoryEstimate(
        kind="hcurl_tri_n1",rows=basis.N,columns=basis.N,
        entity_count=cells,quadrature_points_per_entity=len(basis.W),
        row_local_dofs=local,column_local_dofs=local,
        nnz_upper_bound=nnz_upper,basis_bytes=basis_bytes,
        native_tabulation_bytes=0,dof_map_bytes=0,
        csr_bytes_upper_bound=8*nnz_upper+8*nnz_upper+8*(basis.N+1),
        scatter_bytes=8*cells*local*local,
        coloring_bytes_upper_bound=0,
        pattern_temporary_bytes_upper_bound=24*cells*local*local,
        assumptions=(
            "one reusable CSR pattern",
            "int64 CSR indices and scatter positions",
            "COO-sized temporary pattern arrays",
        ),
    )


class TriN1Assembler:
    """Dedicated mass and curl-curl CSR assembler for ``AffineTriN1Basis``.

    Assembly methods overwrite and return one reusable CSR matrix.  Call
    ``.copy()`` when a result must survive the next assembly on this object.
    """

    def __init__(
        self,basis,*,memory_limit_bytes=None,memory_safety_factor=1.25,
    ):
        if not isinstance(basis,AffineTriN1Basis):
            raise TypeError("basis must be AffineTriN1Basis")
        self.basis=basis
        self.memory_estimate=estimate_tri_n1_assembly_memory(basis)
        enforce_memory_budget(
            self.memory_estimate,memory_limit_bytes,
            safety_factor=memory_safety_factor,
        )
        rows=np.repeat(basis.element_dofs.T,3,axis=1).ravel()
        columns=np.tile(basis.element_dofs.T,(1,3)).ravel()
        pattern=coo_matrix(
            (np.ones(len(rows)),(rows,columns)),shape=(basis.N,basis.N)
        ).tocsr()
        pattern.data.fill(0.)
        self._matrix=csr_matrix(
            (pattern.data,pattern.indices,pattern.indptr),
            shape=pattern.shape,copy=False,
        )
        positions={}
        for row in range(basis.N):
            for offset in range(pattern.indptr[row],pattern.indptr[row+1]):
                positions[(row,int(pattern.indices[offset]))]=offset
        self._scatter=np.asarray(
            [positions[(int(row),int(column))]
             for row,column in zip(rows,columns)],dtype=np.int64
        ).reshape(basis.mesh.nelements,3,3)

    def _coefficient(self,name,value):
        if value is None:
            return np.ones_like(self.basis.dx)
        coefficient=np.asarray(value,dtype=np.float64)
        try:
            return np.broadcast_to(coefficient,self.basis.dx.shape)
        except ValueError as error:
            raise ValueError(
                f"{name} coefficient must broadcast to {self.basis.dx.shape}"
            ) from error

    def _assemble_elements(self,elements):
        self._matrix.data.fill(0.)
        np.add.at(self._matrix.data,self._scatter.ravel(),elements.ravel())
        return self._matrix

    def assemble_mass(self,coefficient=None):
        coefficient=self._coefficient("mass",coefficient)
        elements=np.einsum(
            "ebiq,eciq,eq,eq->ebc",
            self.basis._element_values,self.basis._element_values,
            self.basis.dx,coefficient,
        )
        return self._assemble_elements(elements)

    def assemble_curl_curl(self,coefficient=None):
        coefficient=self._coefficient("curl",coefficient)
        elements=np.einsum(
            "ebq,ecq,eq,eq->ebc",
            self.basis._element_curls,self.basis._element_curls,
            self.basis.dx,coefficient,
        )
        return self._assemble_elements(elements)

    def assemble_maxwell(self,*,mass_coefficient=None,curl_coefficient=None):
        mass=self._coefficient("mass",mass_coefficient)
        curl_value=self._coefficient("curl",curl_coefficient)
        elements=np.einsum(
            "ebiq,eciq,eq,eq->ebc",
            self.basis._element_values,self.basis._element_values,
            self.basis.dx,mass,
        )
        elements+=np.einsum(
            "ebq,ecq,eq,eq->ebc",
            self.basis._element_curls,self.basis._element_curls,
            self.basis.dx,curl_value,
        )
        return self._assemble_elements(elements)


__all__=["TriN1Assembler","estimate_tri_n1_assembly_memory"]
