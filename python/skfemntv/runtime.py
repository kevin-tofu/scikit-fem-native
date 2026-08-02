"""Runtime controls and discoverable native-backend capabilities."""

from __future__ import annotations

from contextlib import contextmanager
import os

from ._skfn import get_num_threads as _get_num_threads
from ._skfn import set_num_threads as _set_num_threads


BACKEND = "skfemntv-native"
CAPABILITIES = frozenset({
    "native_backend",
    "native_threads",
    "parallel_basis_geometry",
    "parallel_linear_assembly",
    "parallel_bilinear_assembly",
    "parallel_fused_assembly",
})


def has_capability(name: str) -> bool:
    """Return whether this backend implements the named extension."""
    return name in CAPABILITIES


def available_num_threads() -> int:
    """Return CPUs available to this process, respecting affinity when possible."""
    affinity=getattr(os,"sched_getaffinity",None)
    if affinity is not None:
        try:
            return max(1,len(affinity(0)))
        except OSError:
            pass
    return max(1,os.cpu_count() or 1)


def get_num_threads() -> int:
    return _get_num_threads()


def set_num_threads(count: int) -> int:
    """Set the process-wide native thread limit and return the effective value."""
    if isinstance(count,bool) or not isinstance(count,int) or count<1:
        raise ValueError("thread count must be a positive integer")
    effective=min(count,available_num_threads())
    _set_num_threads(effective)
    return effective


@contextmanager
def thread_limit(count: int):
    """Temporarily constrain native work to at most ``count`` threads."""
    previous=get_num_threads()
    effective=set_num_threads(count)
    try:
        yield effective
    finally:
        _set_num_threads(previous)
