from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.sparse import csr_matrix

from ._skfn import H1Assembler, TabulatedAssembler
from .evaluation import EvaluationDiagnostics, NativeEvaluation
from .kernels import LinearElasticity, NeoHookean


class NativeAssembler:
    """Reusable native Tet4 residual and tangent assembler."""

    def __init__(
        self,
        coordinates: np.ndarray,
        connectivity: np.ndarray,
        element_dofs: np.ndarray,
        kernel: LinearElasticity | NeoHookean,
        *,
        quadrature: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        if isinstance(kernel, LinearElasticity):
            kind = "linear_elastic"
            parameters = (kernel.young_modulus, kernel.poisson_ratio)
        elif isinstance(kernel, NeoHookean):
            kind = "neo_hookean"
            parameters = (kernel.mu, kernel.lmbda)
        else:
            raise TypeError("kernel must be LinearElasticity or NeoHookean")
        local_nodes = np.asarray(connectivity).shape[1]
        topology = {4: "tet4", 8: "hex8"}.get(local_nodes)
        if topology is None:
            raise ValueError("only Tet4 and Hex8 connectivity are currently supported")
        if quadrature is None:
            if topology == "tet4":
                points = np.array([[0.25, 0.25, 0.25]])
                weights = np.array([1.0 / 6.0])
            else:
                x = (1.0 + np.array([-1.0, 1.0]) / np.sqrt(3.0)) / 2.0
                points = np.array(
                    [[a, b, c] for c in x for b in x for a in x]
                )
                weights = np.full(8, 1.0 / 8.0)
        else:
            points, weights = quadrature
        self._native = H1Assembler(
            np.asarray(coordinates, dtype=np.float64, order="C"),
            np.asarray(connectivity, dtype=np.int64, order="C"),
            np.asarray(element_dofs, dtype=np.int64, order="C"),
            topology,
            kind,
            *parameters,
            np.asarray(points, dtype=np.float64, order="C"),
            np.asarray(weights, dtype=np.float64, order="C"),
        )
        self._initialize_matrix()

    def _initialize_matrix(self) -> None:
        self._tangent = csr_matrix(
            (self._native.values, self._native.indices, self._native.indptr),
            shape=(self._native.ndofs, self._native.ndofs),
            copy=False,
        )

    @classmethod
    def from_basis(
        cls, basis, kernel: LinearElasticity | NeoHookean
    ) -> NativeAssembler:
        mesh = basis.mesh
        if mesh.p.shape[0] != 3:
            raise ValueError("basis must describe a three-dimensional mesh")
        scalar_element = getattr(basis.elem, "elem", None)
        if scalar_element is None or getattr(basis.elem, "_dim", None) != 3:
            raise ValueError("basis must use ElementVector with three components")
        local_nodes = len(scalar_element.doflocs)
        if local_nodes > 27:
            raise ValueError("at most 27 scalar H1 basis functions are supported")
        expected_local_dofs = 3 * local_nodes
        if basis.element_dofs.shape[0] != expected_local_dofs:
            raise ValueError("only nodal vector H1 elements are currently supported")

        if hasattr(basis, "tabulated_gradients"):
            gradients = basis.tabulated_gradients
        else:
            # Compatibility bridge used only by reference tests.
            gradients = np.empty(
                (mesh.nelements, basis.X.shape[1], local_nodes, 3),
                dtype=np.float64,
            )
            for node in range(local_nodes):
                field = scalar_element.gbasis(
                    basis.mapping, basis.X, node
                )[0]
                gradients[:, :, node, :] = field.grad.transpose(1, 2, 0)

        if isinstance(kernel, LinearElasticity):
            kind = "linear_elastic"
            parameters = (kernel.young_modulus, kernel.poisson_ratio)
        elif isinstance(kernel, NeoHookean):
            kind = "neo_hookean"
            parameters = (kernel.mu, kernel.lmbda)
        else:
            raise TypeError("kernel must be LinearElasticity or NeoHookean")

        assembler = cls.__new__(cls)
        assembler._native = TabulatedAssembler(
            np.asarray(basis.element_dofs.T, dtype=np.int64, order="C"),
            np.asarray(gradients, dtype=np.float64, order="C"),
            np.asarray(basis.dx, dtype=np.float64, order="C"),
            kind,
            *parameters,
        )
        assembler._initialize_matrix()
        return assembler

    # Test/reference bridge; skfemntv itself never imports scikit-fem.
    from_skfem = from_basis

    @property
    def ndofs(self) -> int:
        return self._native.ndofs

    @property
    def tangent(self) -> csr_matrix:
        return self._tangent

    @property
    def parallel_diagnostics(self) -> dict[str,int]:
        """Read-only coloring sizes used by race-free parallel scatter."""
        return {
            "color_count": int(getattr(self._native,"color_count",0)),
            "min_color_size": int(getattr(self._native,"min_color_size",0)),
            "max_color_size": int(getattr(self._native,"max_color_size",0)),
            "explicit_thread_threshold": int(getattr(
                self._native,"explicit_thread_threshold",128
            )),
            "parallel_eligible_color_count": int(getattr(
                self._native,"parallel_eligible_color_count",0
            )),
        }

    def evaluate(
        self,
        u: np.ndarray,
        *,
        loads: np.ndarray | None = None,
        mode: Literal["residual_tangent", "residual"] = "residual_tangent",
        num_threads: int = 0,
    ) -> NativeEvaluation:
        if mode not in ("residual_tangent", "residual"):
            raise ValueError(f"unsupported evaluation mode: {mode!r}")
        num_threads=self._validated_num_threads(num_threads)
        u = np.asarray(u)
        if u.dtype != np.float64 or not u.flags.c_contiguous:
            raise TypeError("u must be a C-contiguous float64 array")
        if loads is not None:
            loads = np.asarray(loads)
            if loads.dtype != np.float64 or not loads.flags.c_contiguous:
                raise TypeError("loads must be a C-contiguous float64 array")
        residual, _, seconds = self._native.evaluate(
            u,loads,mode == "residual_tangent",num_threads
        )
        diagnostics = EvaluationDiagnostics(
            element_count=self._native.nelements,
            quadrature_evaluations=self._native.nelements,
            assembly_seconds=seconds,
        )
        return NativeEvaluation(
            residual=residual,
            tangent=self._tangent if mode == "residual_tangent" else None,
            trial_state=None,
            diagnostics=diagnostics,
        )

    def assemble(
        self,
        u: np.ndarray,
        state=None,
        *,
        loads: np.ndarray | None = None,
        mode: Literal["residual_tangent", "residual"] = "residual_tangent",
        num_threads: int = 0,
    ) -> NativeEvaluation:
        """Assemble the current residual and tangent.

        ``state`` reserves the stable call shape for stateful native kernels.
        Current elasticity and Neo-Hookean kernels are state-free.
        """
        if state is not None:
            raise ValueError("the selected kernel does not accept state")
        return self.evaluate(
            u,loads=loads,mode=mode,num_threads=num_threads
        )

    def evaluate_into(
        self,
        u: np.ndarray,
        residual: np.ndarray,
        *,
        tangent_values: np.ndarray | None = None,
        loads: np.ndarray | None = None,
        num_threads: int = 0,
    ) -> EvaluationDiagnostics:
        """Assemble directly into caller-owned contiguous arrays.

        ``tangent_values`` follows the assembler's fixed CSR ordering.  Passing
        ``assembler.tangent.data`` updates the matrix exposed by ``tangent``.
        """
        u = self._validated_vector("u", u, self.ndofs)
        num_threads=self._validated_num_threads(num_threads)
        residual = self._validated_vector("residual", residual, self.ndofs)
        if tangent_values is not None:
            tangent_values = self._validated_vector(
                "tangent_values", tangent_values, self._tangent.nnz
            )
        if loads is not None:
            loads = self._validated_vector("loads", loads, self.ndofs)
        seconds = self._native.evaluate_into(
            u,residual,tangent_values,loads,num_threads
        )
        return EvaluationDiagnostics(
            element_count=self._native.nelements,
            quadrature_evaluations=self._native.nelements,
            assembly_seconds=seconds,
        )

    @staticmethod
    def _validated_vector(name: str, value: np.ndarray, size: int) -> np.ndarray:
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{name} must be a NumPy array")
        if value.dtype != np.float64 or not value.flags.c_contiguous:
            raise TypeError(f"{name} must be a C-contiguous float64 array")
        if value.ndim != 1 or value.size != size:
            raise ValueError(f"{name} must have shape ({size},)")
        if not value.flags.writeable and name in ("residual", "tangent_values"):
            raise ValueError(f"{name} must be writeable")
        return value

    @staticmethod
    def _validated_num_threads(value: int) -> int:
        if isinstance(value,bool) or not isinstance(value,int) or value<0:
            raise ValueError("num_threads must be a nonnegative integer")
        if value==0:
            return 0
        from .runtime import available_num_threads
        return min(value,available_num_threads())
