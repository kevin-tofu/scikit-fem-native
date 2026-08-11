from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse.linalg import spsolve


@dataclass(frozen=True)
class GalleryComparison:
    example: str
    matrix_max_abs: float
    rhs_max_abs: float
    solution_max_abs: float
    tolerance: float = 5e-11

    def assert_matches(self) -> None:
        largest=max(
            self.matrix_max_abs,self.rhs_max_abs,self.solution_max_abs
        )
        if largest>self.tolerance:
            raise AssertionError(
                f"{self.example} differs: matrix={self.matrix_max_abs:.3e}, "
                f"rhs={self.rhs_max_abs:.3e}, "
                f"solution={self.solution_max_abs:.3e}, "
                f"tolerance={self.tolerance:.3e}"
            )

    def summary(self) -> str:
        return (
            f"{self.example}: matrix={self.matrix_max_abs:.3e}, "
            f"rhs={self.rhs_max_abs:.3e}, "
            f"solution={self.solution_max_abs:.3e}"
        )


def sparse_max_abs(left,right) -> float:
    difference=(left-right).tocoo()
    return (
        float(np.max(np.abs(difference.data)))
        if difference.nnz else 0.
    )


def coordinate_permutation(
    native_coordinates: np.ndarray,reference_coordinates: np.ndarray,
    tolerance: float=1e-12,
) -> np.ndarray:
    if native_coordinates.shape!=reference_coordinates.shape:
        raise ValueError(
            "backend DOF coordinate arrays must have identical shapes"
        )
    distances,native_to_reference=cKDTree(
        reference_coordinates.T
    ).query(native_coordinates.T)
    if (
        float(np.max(distances,initial=0.))>tolerance
        or np.unique(native_to_reference).size!=native_coordinates.shape[1]
    ):
        raise ValueError(
            "backend DOFs do not define the same coordinate set"
        )
    return native_to_reference


def reorder_matrix(matrix,native_to_reference: np.ndarray):
    reference_to_native=np.argsort(native_to_reference)
    return matrix[reference_to_native][:,reference_to_native]


def reorder_vector(
    values: np.ndarray,native_to_reference: np.ndarray
) -> np.ndarray:
    reordered=np.empty_like(values)
    reordered[native_to_reference]=values
    return reordered


def solve_dirichlet(matrix,rhs,boundary_dofs: np.ndarray) -> np.ndarray:
    free=np.setdiff1d(
        np.arange(matrix.shape[0],dtype=np.int64),boundary_dofs,
        assume_unique=False,
    )
    solution=np.zeros(matrix.shape[0],dtype=float)
    solution[free]=spsolve(matrix[free][:,free],rhs[free])
    return solution
