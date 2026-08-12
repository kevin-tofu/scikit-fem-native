"""Allocation-free memory estimates for native sparse assembly setup."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


_I64 = 8
_F64 = 8
_I32 = 4
_VECTOR_OVERHEAD = 24


def format_bytes(count: int) -> str:
    """Format an integer byte count using binary units."""
    value = float(count)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


@dataclass(frozen=True)
class AssemblyMemoryEstimate:
    """Conservative memory estimate made before constructing an assembler.

    ``basis_bytes`` are already retained by the supplied basis.  The
    incremental values describe memory added by native assembler construction.
    They intentionally exclude Python/scipy object overhead and allocator
    fragmentation.
    """

    kind: str
    rows: int
    columns: int
    entity_count: int
    quadrature_points_per_entity: int
    row_local_dofs: int
    column_local_dofs: int
    nnz_upper_bound: int
    basis_bytes: int
    native_tabulation_bytes: int
    dof_map_bytes: int
    csr_bytes_upper_bound: int
    scatter_bytes: int
    coloring_bytes_upper_bound: int
    pattern_temporary_bytes_upper_bound: int
    assumptions: tuple[str, ...]

    @property
    def persistent_incremental_bytes_upper_bound(self) -> int:
        return (
            self.native_tabulation_bytes
            + self.dof_map_bytes
            + self.csr_bytes_upper_bound
            + self.scatter_bytes
            + self.coloring_bytes_upper_bound
        )

    @property
    def construction_peak_incremental_bytes_upper_bound(self) -> int:
        return (
            self.persistent_incremental_bytes_upper_bound
            + self.pattern_temporary_bytes_upper_bound
        )

    @property
    def construction_peak_total_bytes_upper_bound(self) -> int:
        return self.basis_bytes + self.construction_peak_incremental_bytes_upper_bound

    @property
    def largest_incremental_allocation(self) -> tuple[str, int]:
        items = {
            "native tabulation": self.native_tabulation_bytes,
            "DOF map": self.dof_map_bytes,
            "CSR": self.csr_bytes_upper_bound,
            "scatter map": self.scatter_bytes,
            "coloring": self.coloring_bytes_upper_bound,
            "temporary CSR pattern": self.pattern_temporary_bytes_upper_bound,
        }
        return max(items.items(), key=lambda item: item[1])

    def fits_in(self, available_bytes: int, *, safety_factor: float = 1.25) -> bool:
        if available_bytes < 0:
            raise ValueError("available_bytes must be nonnegative")
        if safety_factor < 1.0:
            raise ValueError("safety_factor must be at least 1")
        return self.construction_peak_total_bytes_upper_bound * safety_factor <= available_bytes


class AssemblyMemoryBudgetError(MemoryError):
    """Raised before native allocation when an estimate exceeds its budget."""

    def __init__(
        self,
        estimate: AssemblyMemoryEstimate,
        memory_limit_bytes: int,
        safety_factor: float,
    ) -> None:
        self.estimate = estimate
        self.memory_limit_bytes = memory_limit_bytes
        self.safety_factor = safety_factor
        self.required_bytes = math.ceil(
            estimate.construction_peak_total_bytes_upper_bound * safety_factor
        )
        largest_name, largest_bytes = estimate.largest_incremental_allocation
        self.largest_allocation = largest_name
        self.largest_allocation_bytes = largest_bytes
        super().__init__(
            "native assembly memory budget exceeded: "
            f"required={format_bytes(self.required_bytes)}, "
            f"estimated_peak={format_bytes(estimate.construction_peak_total_bytes_upper_bound)}, "
            f"budget={format_bytes(memory_limit_bytes)}, "
            f"safety_factor={safety_factor:.3g}, "
            f"largest={largest_name} ({format_bytes(largest_bytes)}), "
            f"rows={estimate.rows}, columns={estimate.columns}, "
            f"entities={estimate.entity_count}, "
            f"local_dofs={estimate.row_local_dofs}x{estimate.column_local_dofs}"
        )


def enforce_memory_budget(
    estimate: AssemblyMemoryEstimate,
    memory_limit_bytes: int | None,
    *,
    safety_factor: float = 1.25,
) -> None:
    """Validate a memory budget and raise before native allocation if needed."""
    if safety_factor < 1.0:
        raise ValueError("memory_safety_factor must be at least 1")
    if memory_limit_bytes is None:
        return
    if isinstance(memory_limit_bytes, bool) or not isinstance(memory_limit_bytes, int):
        raise TypeError("memory_limit_bytes must be an integer or None")
    if memory_limit_bytes <= 0:
        raise ValueError("memory_limit_bytes must be positive")
    if not estimate.fits_in(memory_limit_bytes, safety_factor=safety_factor):
        raise AssemblyMemoryBudgetError(estimate, memory_limit_bytes, safety_factor)


@dataclass(frozen=True)
class CompositeAssemblyMemoryEstimate:
    """Sum of independently cached assemblers for selected field pairs."""

    estimates: tuple[AssemblyMemoryEstimate, ...]
    field_pairs: tuple[tuple[int, int], ...]
    basis_bytes: int
    assumptions: tuple[str, ...]

    @property
    def persistent_incremental_bytes_upper_bound(self) -> int:
        return sum(item.persistent_incremental_bytes_upper_bound for item in self.estimates)

    @property
    def construction_peak_incremental_bytes_upper_bound(self) -> int:
        # Assemblers persist, while only the largest temporary pattern is live
        # during sequential construction.
        return self.persistent_incremental_bytes_upper_bound + max(
            (item.pattern_temporary_bytes_upper_bound for item in self.estimates),
            default=0,
        )

    @property
    def construction_peak_total_bytes_upper_bound(self) -> int:
        return self.basis_bytes + self.construction_peak_incremental_bytes_upper_bound


def _array_bytes(basis) -> int:
    names = (
        "element_dofs",
        "tabulated_shape",
        "tabulated_gradients",
        "dx",
        "global_coordinates",
        "doflocs",
        "_geometry_determinants",
        "_geometry_tolerances",
        "_geometry_condition_numbers",
        "shape",
        "gradients",
        "weights",
        "active_cell_offsets",
        "active_cell_dofs",
        "quadrature_dofs",
    )
    seen: set[int] = set()
    total = 0
    for name in names:
        value = getattr(basis, name, None)
        if isinstance(value, np.ndarray) and id(value) not in seen:
            seen.add(id(value))
            total += int(value.nbytes)
    return total


def _validate_basis(basis, label: str) -> None:
    required = ("N", "element_dofs", "tabulated_shape", "tabulated_gradients", "dx")
    missing = [name for name in required if not hasattr(basis, name)]
    if missing:
        raise TypeError(f"{label} is not an independent assembled basis; missing {missing}")
    if basis.element_dofs.ndim != 2 or basis.dx.ndim != 2:
        raise ValueError(f"{label} has invalid DOF or quadrature arrays")


def estimate_bilinear_memory(test_basis, trial_basis=None) -> AssemblyMemoryEstimate:
    """Estimate persistent and constructor-peak memory for one assembler.

    Passing one basis models ``NativeBilinearForm``.  Passing two models a
    rectangular ``NativeCrossBilinearForm``.  The CSR estimate is a safe upper
    bound derived without building adjacency or allocating the sparse pattern.
    """
    trial_basis = test_basis if trial_basis is None else trial_basis
    _validate_basis(test_basis, "test_basis")
    _validate_basis(trial_basis, "trial_basis")
    if test_basis.dx.shape != trial_basis.dx.shape:
        raise ValueError("test and trial basis quadrature shapes must match")

    cut = hasattr(test_basis, "cell_offsets") or hasattr(trial_basis, "cell_offsets")
    if cut and not (
        hasattr(test_basis, "cell_offsets") and hasattr(trial_basis, "cell_offsets")
    ):
        raise ValueError("test and trial basis must both be cut bases")
    if cut and not np.array_equal(
        test_basis.active_cell_offsets, trial_basis.active_cell_offsets
    ):
        raise ValueError("cut test and trial bases have different active cell offsets")
    if cut:
        entities = int(test_basis.active_cell_dofs.shape[0])
        quadrature = int(len(test_basis.weights))
        row_local = int(np.prod(test_basis.active_cell_dofs.shape[1:]))
        column_local = int(np.prod(trial_basis.active_cell_dofs.shape[1:]))
    else:
        entities, quadrature = map(int, test_basis.dx.shape)
        row_local = int(test_basis.element_dofs.shape[0])
        column_local = int(trial_basis.element_dofs.shape[0])
    rows, columns = int(test_basis.N), int(trial_basis.N)
    local_entries = entities * row_local * column_local
    nnz_upper = min(rows * columns, local_entries)
    same = trial_basis is test_basis

    dof_bytes = entities * row_local * _I64
    if not same:
        dof_bytes += entities * column_local * _I64
    if cut:
        dof_bytes += (entities + 1) * _I64
        native_tabulation = int(
            test_basis.shape.nbytes
            + test_basis.gradients.nbytes
            + test_basis.weights.nbytes
        )
    else:
        native_tabulation = int(
            test_basis.tabulated_shape.nbytes
            + test_basis.tabulated_gradients.nbytes
            + test_basis.dx.nbytes
        )
    if not same:
        native_tabulation += int(
            trial_basis.shape.nbytes + trial_basis.gradients.nbytes
            if cut else
            trial_basis.tabulated_shape.nbytes
            + trial_basis.tabulated_gradients.nbytes
        )
    csr_bytes = (rows + 1) * _I64 + nnz_upper * (_I64 + _F64)
    scatter_bytes = local_entries * _I64
    # colors_ stores every entity once; dof_colors may transiently retain up
    # to one color entry per local incidence.  Count both conservatively.
    coloring_bytes = entities * _I32 + entities * row_local * _I32
    pattern_bytes = 2 * local_entries * _I64 + rows * _VECTOR_OVERHEAD
    basis_bytes = _array_bytes(test_basis)
    if not same:
        basis_bytes += _array_bytes(trial_basis)

    return AssemblyMemoryEstimate(
        kind=("cut-" if cut else "") + (
            "bilinear" if same else "cross-bilinear"
        ),
        rows=rows,
        columns=columns,
        entity_count=entities,
        quadrature_points_per_entity=quadrature,
        row_local_dofs=row_local,
        column_local_dofs=column_local,
        nnz_upper_bound=nnz_upper,
        basis_bytes=basis_bytes,
        native_tabulation_bytes=native_tabulation,
        dof_map_bytes=dof_bytes,
        csr_bytes_upper_bound=csr_bytes,
        scatter_bytes=scatter_bytes,
        coloring_bytes_upper_bound=coloring_bytes,
        pattern_temporary_bytes_upper_bound=pattern_bytes,
        assumptions=(
            "int64 CSR indices, DOF maps, and scatter entries",
            "float64 matrix values and native tabulation",
            "CSR nnz is an adjacency-free upper bound, not an exact count",
            "Python objects, allocator capacity, and fragmentation are excluded",
            "temporary row-vector capacity is modeled at twice its logical entries",
            "native assemblers are constructed sequentially",
        ),
    )


def estimate_composite_bilinear_memory(
    basis,
    field_pairs: Iterable[tuple[int, int]] | None = None,
) -> CompositeAssemblyMemoryEstimate:
    """Estimate independently cached assemblers for a composite basis.

    Each pair represents one cached contraction-kind block.  By default one
    block for every ordered test/trial field pair is included.  Pass duplicate
    pairs when one pair caches multiple value/gradient contraction kinds.
    """
    subbases = getattr(basis, "subbases", None)
    if subbases is None:
        raise TypeError("basis must use ElementComposite")
    count = len(subbases)
    pairs = tuple(
        (row, column) for row in range(count) for column in range(count)
    ) if field_pairs is None else tuple((int(row), int(column)) for row, column in field_pairs)
    for row, column in pairs:
        if not (0 <= row < count and 0 <= column < count):
            raise IndexError(f"composite field pair {(row, column)} is out of range")
    estimates = tuple(
        estimate_bilinear_memory(subbases[row], subbases[column])
        for row, column in pairs
    )
    return CompositeAssemblyMemoryEstimate(
        estimates=estimates,
        field_pairs=pairs,
        basis_bytes=_array_bytes(basis) + sum(_array_bytes(item) for item in subbases),
        assumptions=(
            "all selected field-pair assemblers remain cached",
            "only one temporary CSR pattern is live during sequential construction",
            "shared NumPy buffers may make the retained-basis total conservative",
        ),
    )


__all__ = [
    "AssemblyMemoryBudgetError",
    "AssemblyMemoryEstimate",
    "CompositeAssemblyMemoryEstimate",
    "estimate_bilinear_memory",
    "estimate_composite_bilinear_memory",
    "enforce_memory_budget",
    "format_bytes",
]
