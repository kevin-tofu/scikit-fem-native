from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from ._skfn import BilinearFormAssembler,CrossBilinearAssembler


class NativeBilinearForm:
    """Native component-wise value and gradient bilinear assembly."""

    def __init__(self, basis):
        scalar = basis.elem.elem
        components = basis.elem._dim
        nodes = len(scalar.doflocs)
        entities, quadrature = basis.dx.shape
        dofs = basis.element_dofs.T.reshape(entities, nodes, components)
        self._native = BilinearFormAssembler(
            np.asarray(dofs, dtype=np.int64, order="C"),
            np.asarray(basis.tabulated_shape, dtype=np.float64, order="C"),
            np.asarray(
                basis.tabulated_gradients, dtype=np.float64, order="C"
            ),
            np.asarray(basis.dx, dtype=np.float64, order="C"),
        )
        self._coefficient_shape = (entities, quadrature)
        self._matrix = csr_matrix(
            (
                self._native.values,
                self._native.indices,
                self._native.indptr,
            ),
            shape=(self._native.ndofs, self._native.ndofs),
            copy=False,
        )
        self._matrix.resize((basis.N,basis.N))

    def assemble(self, *, value=None, gradient=None):
        value = self._coefficient("value", value)
        gradient = self._coefficient("gradient", gradient)
        self._native.assemble(value, gradient)
        return self._matrix

    def _coefficient(self, name, coefficient):
        if coefficient is None:
            return None
        coefficient = np.asarray(coefficient)
        if coefficient.dtype != np.float64:
            coefficient = coefficient.astype(np.float64)
        try:
            coefficient = np.broadcast_to(
                coefficient, self._coefficient_shape
            )
        except ValueError as error:
            raise ValueError(
                f"{name} coefficient must broadcast to "
                f"{self._coefficient_shape}"
            ) from error
        return np.ascontiguousarray(coefficient)


class NativeCrossBilinearForm:
    """Native assembly between aligned trial and test facet traces."""

    def __init__(self,test_basis,trial_basis):
        if test_basis.dx.shape!=trial_basis.dx.shape:
            raise ValueError("trial and test quadrature shapes must match")
        if not np.allclose(test_basis.dx,trial_basis.dx):
            raise ValueError("trial and test quadrature weights must match")
        test_scalar=test_basis.elem.elem
        trial_scalar=trial_basis.elem.elem
        test_components=test_basis.elem._dim
        trial_components=trial_basis.elem._dim
        if not np.allclose(
            test_basis.global_coordinates,
            trial_basis.global_coordinates,
        ):
            raise ValueError(
                "trial and test quadrature coordinates must match"
            )
        entities=test_basis.dx.shape[0]
        test_dofs=test_basis.element_dofs.T.reshape(
            entities,len(test_scalar.doflocs),test_components
        )
        trial_dofs=trial_basis.element_dofs.T.reshape(
            entities,len(trial_scalar.doflocs),trial_components
        )
        self._native=CrossBilinearAssembler(
            np.ascontiguousarray(test_dofs,dtype=np.int64),
            np.ascontiguousarray(trial_dofs,dtype=np.int64),
            np.ascontiguousarray(test_basis.tabulated_shape),
            np.ascontiguousarray(trial_basis.tabulated_shape),
            np.ascontiguousarray(test_basis.dx),
            np.ascontiguousarray(test_basis.tabulated_gradients),
            np.ascontiguousarray(trial_basis.tabulated_gradients),
        )
        self.shape=(test_basis.N,trial_basis.N)
        self.coefficient_shape=test_basis.dx.shape
        self.test_components=test_components
        self.trial_components=trial_components
        self.dimension=test_basis.mesh.dim()

    def assemble(self,kind,coefficient):
        coefficient=np.asarray(coefficient,dtype=np.float64)
        if kind in {"row_divergence","column_divergence"}:
            if kind=="row_divergence":
                if (
                    self.test_components!=self.dimension
                    or self.trial_components!=1
                ):
                    raise ValueError(
                        "row divergence requires vector test and scalar trial"
                    )
                tensor=np.zeros((
                    self.test_components,self.dimension,
                    self.trial_components,
                ))
                for component in range(self.test_components):
                    tensor[component,component,0]=1.
                row_kind,column_kind="gradient","value"
            else:
                if (
                    self.trial_components!=self.dimension
                    or self.test_components!=1
                ):
                    raise ValueError(
                        "column divergence requires scalar test and vector trial"
                    )
                tensor=np.zeros((
                    self.test_components,
                    self.trial_components,self.dimension,
                ))
                for component in range(self.trial_components):
                    tensor[0,component,component]=1.
                row_kind,column_kind="value","gradient"
            scalar=np.broadcast_to(
                coefficient,self.coefficient_shape
            )
            coefficient=scalar[(...,)+(None,)*tensor.ndim]*tensor
        else:
            if self.test_components!=self.trial_components:
                raise ValueError(
                    "value and gradient contractions require matching "
                    "component counts"
                )
            coefficient=np.broadcast_to(
                coefficient,self.coefficient_shape
            )
            row_kind=column_kind=(
                "gradient" if kind=="gradient" else "value"
            )
        self._native.assemble(
            np.ascontiguousarray(coefficient),row_kind,column_kind
        )
        matrix=csr_matrix(
            (
                self._native.values,self._native.indices,
                self._native.indptr,
            ),
            shape=(self._native.rows,self._native.columns),
            copy=False,
        )
        matrix.resize(self.shape)
        return matrix.copy()


class NativeCompositeBilinearForm:
    """Reusable rectangular native blocks for a composite H1 basis."""

    def __init__(self,basis):
        self.basis=basis
        self._assemblers={}

    def assemble(
        self,row_field,column_field,*,kind,coefficient,
    ):
        row=self.basis.subbases[row_field]
        column=self.basis.subbases[column_field]
        row_components=row.elem._dim
        column_components=column.elem._dim
        divergence=kind in {"row_divergence","column_divergence"}
        if row_components!=column_components and not divergence:
            raise ValueError(
                "component-mismatched composite coupling requires "
                "an explicit tensor contraction"
            )
        row_kind=(
            "gradient" if kind in {"gradient","row_divergence"}
            else "value"
        )
        column_kind=(
            "gradient" if kind in {"gradient","column_divergence"}
            else "value"
        )
        key=(row_field,column_field,row_kind,column_kind)
        native=self._assemblers.get(key)
        if native is None:
            row_nodes=len(row.elem.elem.doflocs)
            column_nodes=len(column.elem.elem.doflocs)
            entities,quadrature=self.basis.dx.shape
            row_dofs=row.element_dofs.T.reshape(
                entities,row_nodes,row_components
            )
            column_dofs=column.element_dofs.T.reshape(
                entities,column_nodes,column_components
            )
            native=CrossBilinearAssembler(
                np.ascontiguousarray(row_dofs,dtype=np.int64),
                np.ascontiguousarray(column_dofs,dtype=np.int64),
                np.ascontiguousarray(row.tabulated_shape),
                np.ascontiguousarray(column.tabulated_shape),
                np.ascontiguousarray(self.basis.dx),
                np.ascontiguousarray(row.tabulated_gradients),
                np.ascontiguousarray(column.tabulated_gradients),
            )
            self._assemblers[key]=native
        coefficient=np.asarray(coefficient,dtype=np.float64)
        if divergence:
            dimension=row.tabulated_gradients.shape[3]
            if kind=="row_divergence":
                if row_components!=dimension or column_components!=1:
                    raise ValueError(
                        "row divergence requires vector test and scalar trial"
                    )
                tensor=np.zeros(
                    (row_components,dimension,column_components)
                )
                for component in range(row_components):
                    tensor[component,component,0]=1.
            else:
                dimension=column.tabulated_gradients.shape[3]
                if column_components!=dimension or row_components!=1:
                    raise ValueError(
                        "column divergence requires scalar test and vector trial"
                    )
                tensor=np.zeros(
                    (row_components,column_components,dimension)
                )
                for component in range(column_components):
                    tensor[0,component,component]=1.
            scalar=np.broadcast_to(coefficient,self.basis.dx.shape)
            coefficient=scalar[(...,)+(None,)*tensor.ndim]*tensor
        else:
            coefficient=np.broadcast_to(coefficient,self.basis.dx.shape)
        coefficient=np.ascontiguousarray(coefficient)
        native.assemble(coefficient,row_kind,column_kind)
        matrix=csr_matrix(
            (native.values,native.indices,native.indptr),
            shape=(native.rows,native.columns),copy=False,
        )
        matrix.resize((self.basis.N,self.basis.N))
        return matrix.copy()
