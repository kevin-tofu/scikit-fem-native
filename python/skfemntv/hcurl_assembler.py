"""Reusable sparse assembly for the experimental affine TriN1 basis."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from ._skfn import build_edge_csr_pattern
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
            "native row-adjacency construction bounded by COO-sized storage",
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
        # Pattern and element-to-CSR positions are constructed together in
        # native code; this avoids size-dependent Python tuple/dict objects.
        indptr,indices,self._scatter=build_edge_csr_pattern(
            np.ascontiguousarray(basis.element_dofs,dtype=np.int64),basis.N
        )
        self._matrix=csr_matrix(
            (np.zeros(len(indices)),indices,indptr),shape=(basis.N,basis.N),
            copy=False,
        )

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
            "ebiq,eciq,eq->ebc",
            self.basis._element_values,self.basis._element_values,
            self.basis.dx*coefficient,optimize=True,
        )
        return self._assemble_elements(elements)

    def assemble_curl_curl(self,coefficient=None):
        coefficient=self._coefficient("curl",coefficient)
        elements=np.einsum(
            "ebq,ecq,eq->ebc",
            self.basis._element_curls,self.basis._element_curls,
            self.basis.dx*coefficient,optimize=True,
        )
        return self._assemble_elements(elements)

    def assemble_maxwell(self,*,mass_coefficient=None,curl_coefficient=None):
        mass=self._coefficient("mass",mass_coefficient)
        curl_value=self._coefficient("curl",curl_coefficient)
        elements=np.einsum(
            "ebiq,eciq,eq->ebc",
            self.basis._element_values,self.basis._element_values,
            self.basis.dx*mass,optimize=True,
        )
        elements+=np.einsum(
            "ebq,ecq,eq->ebc",
            self.basis._element_curls,self.basis._element_curls,
            self.basis.dx*curl_value,optimize=True,
        )
        return self._assemble_elements(elements)


class TriN1LinearAssembler:
    """Reusable vector-load assembler for ``AffineTriN1Basis``.

    ``assemble_vector_load`` overwrites and returns one reusable vector.  Copy
    it when the result must survive another call on this assembler.
    """

    def __init__(self,basis):
        if not isinstance(basis,AffineTriN1Basis):
            raise TypeError("basis must be AffineTriN1Basis")
        self.basis=basis
        self._vector=np.zeros(basis.N,dtype=np.float64)

    def _values(self,field):
        values=(
            field(self.basis.global_coordinates)
            if callable(field) else field
        )
        values=np.asarray(values,dtype=np.float64)
        expected=(2,)+self.basis.dx.shape
        try:
            return np.broadcast_to(values,expected)
        except ValueError as error:
            raise ValueError(
                f"vector load must broadcast to {expected}"
            ) from error

    def assemble_vector_load(self,field):
        values=self._values(field)
        local=np.einsum(
            "ebiq,ieq,eq->eb",
            self.basis._element_values,values,self.basis.dx,
        )
        self._vector.fill(0.)
        np.add.at(self._vector,self.basis.element_dofs.T.ravel(),local.ravel())
        return self._vector


__all__=[
    "TriN1Assembler",
    "TriN1LinearAssembler",
    "estimate_tri_n1_assembly_memory",
]
