"""Solver and boundary-condition helpers with scikit-fem-compatible inputs.

The native package focuses on mesh and assembly kernels. These convenience
functions delegate linear algebra policy to scikit-fem when that optional
package is installed, while accepting native ``DofsView`` objects.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .basis import DofsView


def _normalize_dofs(value: Any) -> Any:
    if isinstance(value, DofsView):
        return value.flatten()
    if isinstance(value, dict):
        # scikit-fem's condense accepts dictionaries of named DofsView objects
        # and flattens them into one selection. Native DofsView is a separate
        # class, so normalize the mapping before passing it to scikit-fem.
        selections = [_normalize_dofs(item) for item in value.values()]
        if not selections:
            return np.empty(0, dtype=np.int64)
        return np.unique(np.concatenate([np.asarray(item).ravel() for item in selections]))
    if isinstance(value, tuple):
        return tuple(_normalize_dofs(item) for item in value)
    if isinstance(value, list):
        return [_normalize_dofs(item) for item in value]
    return value


def _call_skfem_utility(name: str, *args, **kwargs):
    try:
        from skfem import utils
    except ImportError as error:
        raise ImportError(
            f"skfemntv.{name} requires scikit-fem; install scikit-fem "
            "to use the compatibility utility."
        ) from error
    args = tuple(_normalize_dofs(value) for value in args)
    kwargs = {key: _normalize_dofs(value) for key, value in kwargs.items()}
    return getattr(utils, name)(*args, **kwargs)


def condense(*args, **kwargs):
    """Apply scikit-fem's boundary-condition condensation utility."""
    return _call_skfem_utility("condense", *args, **kwargs)


def enforce(*args, **kwargs):
    """Apply scikit-fem's boundary-condition enforcement utility."""
    return _call_skfem_utility("enforce", *args, **kwargs)


def penalize(*args, **kwargs):
    """Apply scikit-fem's penalty boundary-condition utility."""
    return _call_skfem_utility("penalize", *args, **kwargs)


def solve(*args, **kwargs):
    """Solve a linear system using scikit-fem's configured solver path."""
    return _call_skfem_utility("solve", *args, **kwargs)


__all__ = ["condense", "enforce", "penalize", "solve"]
