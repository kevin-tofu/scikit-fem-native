"""SciPy-backed utilities matching the common scikit-fem solver API."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy.sparse import issparse
from scipy.sparse.linalg import eigs, spsolve


def _flatten_dofs(dofs):
    if dofs is None:
        return None
    if isinstance(dofs, np.ndarray):
        return dofs.astype(np.intp, copy=False).ravel()
    if hasattr(dofs, "flatten") and callable(dofs.flatten):
        return np.asarray(dofs.flatten(), dtype=np.intp).ravel()
    if isinstance(dofs, Mapping):
        flattened = [_flatten_dofs(value) for value in dofs.values()]
        if not flattened:
            return np.empty(0, dtype=np.intp)
        return np.unique(np.concatenate(flattened))
    raise NotImplementedError("Unable to flatten the given set of DOFs.")


def _init_bc(A, b=None, x=None, I=None, D=None):
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")
    D = _flatten_dofs(D)
    I = _flatten_dofs(I)
    if I is None and D is None:
        raise Exception("Either I or D must be given!")
    if I is not None and D is not None:
        raise Exception("Give only I or only D!")
    all_dofs = np.arange(A.shape[0], dtype=np.intp)
    if D is None:
        I = np.unique(I)
        D = np.setdiff1d(all_dofs, I)
    else:
        D = np.unique(D)
        I = np.setdiff1d(all_dofs, D)
    if np.any(D < 0) or np.any(D >= A.shape[0]):
        raise IndexError("constrained DOF index out of range")
    if np.any(I < 0) or np.any(I >= A.shape[0]):
        raise IndexError("free DOF index out of range")
    if x is None:
        x = np.zeros(A.shape[0], dtype=A.dtype)
    elif b is None:
        b = np.zeros_like(x)
    return b, x, I, D


def condense(A, b=None, x=None, I=None, D=None, expand=True):
    """Eliminate constrained DOFs, following :func:`skfem.condense`."""
    b, x, I, D = _init_bc(A, b, x, I, D)
    result = [A[I][:, I]]
    if b is not None:
        if issparse(b):
            result.append(b[I][:, I])
        else:
            result.append(np.asarray(b)[I] - A[I][:, D] @ x[D])
    if expand:
        result.extend((x, I))
    return result[0] if len(result) == 1 else tuple(result)


def enforce(A, b=None, x=None, I=None, D=None, diag=1.0, overwrite=False):
    """Enforce DOFs by replacing constrained rows with diagonal entries."""
    b, x, I, D = _init_bc(A, b, x, I, D)
    Aout = A if overwrite else A.copy()
    Aout = Aout.tocsr(copy=False)
    for row in D:
        Aout.data[Aout.indptr[row]:Aout.indptr[row + 1]] = 0
    diagonal = Aout.diagonal()
    diagonal[D] = diag
    Aout.setdiag(diagonal)
    if b is None:
        return Aout
    if issparse(b):
        bout = enforce(b, D=D, diag=0.0, overwrite=overwrite)
    else:
        bout = b if overwrite else np.array(b, copy=True)
        bout[D] = x[D]
    return Aout, bout


def penalize(A, b=None, x=None, I=None, D=None, epsilon=None, overwrite=False):
    """Penalize constrained DOFs with a large diagonal contribution."""
    b, x, I, D = _init_bc(A, b, x, I, D)
    Aout = A if overwrite else A.copy()
    diagonal = Aout.diagonal()
    if epsilon is None:
        norm = np.linalg.norm(diagonal[D], np.inf)
        epsilon = 1e-10 / float(norm)
    diagonal[D] = 1.0 / epsilon
    Aout.setdiag(diagonal)
    if b is None:
        return Aout
    bout = b if overwrite else (b.copy() if issparse(b) else np.array(b, copy=True))
    if not issparse(b):
        bout[D] = x[D] / epsilon
    return Aout, bout


def solver_direct_scipy(**kwargs):
    """Return a direct sparse linear solver callable."""
    params = dict(kwargs)

    def solver(A, b, **solve_time_kwargs):
        return spsolve(A, b, **(params | solve_time_kwargs))

    return solver


def solver_eigen_scipy(**kwargs):
    """Return the default ARPACK generalized eigenvalue solver."""
    params = {"sigma": 10, "k": 5} | kwargs

    def solver(A, M, **solve_time_kwargs):
        return eigs(A, M=M, **(params | solve_time_kwargs))

    return solver


def solve(A, b, x=None, I=None, solver=None, **kwargs):
    """Solve a sparse linear system or generalized eigenvalue problem."""
    if issparse(b):
        solver = solver or solver_eigen_scipy(**kwargs)
        values, vectors = solver(A, b, **kwargs)
        if x is not None and I is not None:
            full = np.tile(np.asarray(x)[:, None], (1, vectors.shape[1]))
            if isinstance(I, tuple):
                np.add.at(full, I[0], np.asarray([I[1](v) for v in vectors.T]).T)
            else:
                full[np.asarray(I)] = vectors
            return values, full
        return values, vectors
    if not isinstance(b, np.ndarray):
        raise NotImplementedError("Provided argument types not supported")
    solver = solver or solver_direct_scipy(**kwargs)
    solution = solver(A, b, **kwargs)
    if x is not None and I is not None:
        full = np.array(x, copy=True)
        if isinstance(I, tuple):
            np.add.at(full, I[0], I[1](solution))
        else:
            full[np.asarray(I)] = solution
        return full
    return solution
