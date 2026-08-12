from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from ._skfn import (
    BilinearFormAssembler,CrossBilinearAssembler,CutBilinearFormAssembler,
    CutCrossAssembler,
)
from .preflight import enforce_memory_budget, estimate_bilinear_memory


class NativeBilinearForm:
    """Native component-wise value and gradient bilinear assembly."""

    def __init__(
        self,
        basis,
        *,
        memory_limit_bytes: int | None = None,
        memory_safety_factor: float = 1.25,
    ):
        self.memory_estimate = estimate_bilinear_memory(basis)
        enforce_memory_budget(
            self.memory_estimate,memory_limit_bytes,
            safety_factor=memory_safety_factor,
        )
        scalar = basis.elem.elem
        components = basis.elem._dim
        nodes = len(scalar.doflocs)
        entities, quadrature = basis.dx.shape
        self._cut=hasattr(basis,"cell_offsets")
        if self._cut:
            self._native=CutBilinearFormAssembler(
                np.ascontiguousarray(basis.active_cell_dofs,dtype=np.int64),
                np.ascontiguousarray(basis.active_cell_offsets,dtype=np.int64),
                np.ascontiguousarray(basis.shape,dtype=np.float64),
                np.ascontiguousarray(basis.gradients,dtype=np.float64),
                np.ascontiguousarray(basis.weights,dtype=np.float64),
            )
        else:
            dofs = basis.element_dofs.T.reshape(entities, nodes, components)
            self._native = BilinearFormAssembler(
                np.asarray(dofs,dtype=np.int64,order="C"),
                np.asarray(basis.tabulated_shape,dtype=np.float64,order="C"),
                np.asarray(basis.tabulated_gradients,dtype=np.float64,order="C"),
                np.asarray(basis.dx,dtype=np.float64,order="C"),
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

    def assemble(self,*,value=None,gradient=None,symmetric_gradient=None,
                 divergence=None,num_threads=0):
        value = self._coefficient("value", value)
        gradient = self._coefficient("gradient", gradient)
        symmetric_gradient = self._coefficient(
            "symmetric_gradient", symmetric_gradient
        )
        divergence = self._coefficient("divergence", divergence)
        if self._cut:
            if symmetric_gradient is not None or divergence is not None:
                raise ValueError(
                    "cut-basis symmetric-gradient and divergence assembly "
                    "is not supported"
                )
            if value is not None:value=value.reshape(-1)
            if gradient is not None:gradient=gradient.reshape(-1)
            self._native.assemble(value,gradient,num_threads)
        else:
            self._native.assemble(
                value,gradient,symmetric_gradient,divergence,num_threads
            )
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

    def __init__(
        self,
        test_basis,
        trial_basis,
        *,
        memory_limit_bytes: int | None = None,
        memory_safety_factor: float = 1.25,
    ):
        if test_basis.dx.shape!=trial_basis.dx.shape:
            raise ValueError("trial and test quadrature shapes must match")
        if not np.allclose(test_basis.dx,trial_basis.dx):
            raise ValueError("trial and test quadrature weights must match")
        self.memory_estimate = estimate_bilinear_memory(
            test_basis, trial_basis
        )
        enforce_memory_budget(
            self.memory_estimate,memory_limit_bytes,
            safety_factor=memory_safety_factor,
        )
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
        self._cut=(
            hasattr(test_basis,"cell_offsets")
            and hasattr(trial_basis,"cell_offsets")
        )
        if self._cut:
            if not np.array_equal(
                test_basis.active_cell_offsets,
                trial_basis.active_cell_offsets,
            ):
                raise ValueError("cut cross bases have different cell offsets")
            self._native=CutCrossAssembler(
                np.ascontiguousarray(test_basis.active_cell_dofs,dtype=np.int64),
                np.ascontiguousarray(trial_basis.active_cell_dofs,dtype=np.int64),
                np.ascontiguousarray(test_basis.active_cell_offsets,dtype=np.int64),
                np.ascontiguousarray(test_basis.shape),
                np.ascontiguousarray(trial_basis.shape),
                np.ascontiguousarray(test_basis.weights),
                np.ascontiguousarray(test_basis.gradients),
                np.ascontiguousarray(trial_basis.gradients),
            )
        else:
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
        self.test_normals=getattr(test_basis,"normals",None)
        self.trial_normals=getattr(trial_basis,"normals",None)

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
        native_coefficient=np.ascontiguousarray(
            np.squeeze(coefficient,axis=1) if self._cut else coefficient
        )
        self._native.assemble(native_coefficient,row_kind,column_kind)
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

    def assemble_kinds(
        self,row_kind,column_kind,coefficient=1.,*,num_threads=0,
    ):
        """Assemble arbitrary value/gradient/normal-gradient contractions."""
        valid={"value","gradient","normal_gradient"}
        if row_kind not in valid or column_kind not in valid:
            raise ValueError("cross kind must be value, gradient, or normal_gradient")
        if self.test_components!=self.trial_components:
            raise ValueError("cross contractions require matching components")
        scalar=np.broadcast_to(
            np.asarray(coefficient,dtype=np.float64),self.coefficient_shape
        )
        row_gradient=row_kind!="value";column_gradient=column_kind!="value"
        tensor_shape=(self.test_components,)
        if row_gradient:tensor_shape+=(self.dimension,)
        tensor_shape+=(self.trial_components,)
        if column_gradient:tensor_shape+=(self.dimension,)
        tensor=np.zeros(self.coefficient_shape+tensor_shape,dtype=np.float64)
        row_normals=(
            None if row_kind!="normal_gradient" else
            np.asarray(self.test_normals,dtype=np.float64)
        )
        column_normals=(
            None if column_kind!="normal_gradient" else
            np.asarray(self.trial_normals,dtype=np.float64)
        )
        if row_kind=="normal_gradient" and row_normals is None:
            raise ValueError("row normal_gradient requires test-basis normals")
        if column_kind=="normal_gradient" and column_normals is None:
            raise ValueError("column normal_gradient requires trial-basis normals")
        for component in range(self.test_components):
            if not row_gradient and not column_gradient:
                tensor[...,component,component]=scalar
            elif row_gradient and not column_gradient:
                direction=(
                    row_normals if row_normals is not None else
                    np.ones(self.coefficient_shape+(self.dimension,))
                )
                for axis in range(self.dimension):
                    factor=(
                        direction[...,axis] if row_normals is not None
                        else (1. if self.dimension==1 else None)
                    )
                    if factor is None:
                        tensor[...,component,axis,component]=scalar
                    else:
                        tensor[...,component,axis,component]=scalar*factor
            elif not row_gradient and column_gradient:
                direction=(
                    column_normals if column_normals is not None else
                    np.ones(self.coefficient_shape+(self.dimension,))
                )
                for axis in range(self.dimension):
                    factor=(
                        direction[...,axis] if column_normals is not None
                        else (1. if self.dimension==1 else None)
                    )
                    if factor is None:
                        tensor[...,component,component,axis]=scalar
                    else:
                        tensor[...,component,component,axis]=scalar*factor
            else:
                for row_axis in range(self.dimension):
                    for column_axis in range(self.dimension):
                        if row_normals is None and column_normals is None:
                            factor=1. if row_axis==column_axis else 0.
                        else:
                            factor=1.
                            if row_normals is not None:
                                factor=factor*row_normals[...,row_axis]
                            elif row_axis!=column_axis:
                                factor=0.
                            if column_normals is not None:
                                factor=factor*column_normals[...,column_axis]
                        tensor[...,component,row_axis,component,column_axis]=(
                            scalar*factor
                        )
        native_tensor=np.ascontiguousarray(
            np.squeeze(tensor,axis=1) if self._cut else tensor
        )
        self._native.assemble(
            native_tensor,
            "gradient" if row_gradient else "value",
            "gradient" if column_gradient else "value",
            num_threads,
        )
        matrix=csr_matrix((
            self._native.values,self._native.indices,self._native.indptr,
        ),shape=(self._native.rows,self._native.columns),copy=False)
        matrix.resize(self.shape)
        return matrix.copy()

    def assemble_tensor(
        self,row_kind,column_kind,coefficient,*,num_threads=0,
    ):
        """Assemble a caller-supplied value/gradient contraction tensor."""
        valid={"value","gradient"}
        if row_kind not in valid or column_kind not in valid:
            raise ValueError("tensor kind must be value or gradient")
        row_axes=(self.test_components,)+(
            (self.dimension,) if row_kind=="gradient" else ()
        )
        column_axes=(self.trial_components,)+(
            (self.dimension,) if column_kind=="gradient" else ()
        )
        expected=self.coefficient_shape+row_axes+column_axes
        coefficient=np.asarray(coefficient,dtype=np.float64)
        try:
            coefficient=np.broadcast_to(coefficient,expected)
        except ValueError as error:
            raise ValueError(
                f"tensor coefficient must broadcast to {expected}"
            ) from error
        native_coefficient=np.ascontiguousarray(
            np.squeeze(coefficient,axis=1) if self._cut else coefficient
        )
        self._native.assemble(
            native_coefficient,row_kind,column_kind,num_threads
        )
        matrix=csr_matrix((
            self._native.values,self._native.indices,self._native.indptr,
        ),shape=(self._native.rows,self._native.columns),copy=False)
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
