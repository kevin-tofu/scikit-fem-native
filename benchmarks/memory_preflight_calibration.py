"""Calibrate assembly preflight estimates against isolated-process RSS peaks.

Each case runs in a fresh child process so its peak resident set is not
contaminated by earlier assemblers.  Large cases are skipped before native
assembler creation when their estimate exceeds the configured budget.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

try:
    import resource
except ImportError:  # ``resource`` is not provided by CPython on Windows.
    resource = None

import numpy as np

import skfemntv


FAMILIES = ("tet4", "tet10", "hex8", "hex20")
CSV_COLUMNS = (
    "family", "components", "target_dofs", "actual_dofs", "elements",
    "local_dofs", "quadrature_points", "status", "skip_reason",
    "basis_seconds", "assembler_seconds", "assembly_seconds", "actual_nnz",
    "nnz_upper_bound", "nnz_upper_ratio", "basis_bytes_estimated",
    "persistent_bytes_estimated", "peak_total_bytes_estimated",
    "rss_before_basis", "rss_after_basis", "rss_peak",
    "rss_basis_delta", "rss_assembler_peak_delta", "peak_estimate_to_rss_ratio",
)


def _windows_working_set_bytes() -> tuple[int, int]:
    """Return current and peak process working sets on Windows, in bytes."""
    import ctypes
    from ctypes import wintypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCounters),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(
        kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(counters.WorkingSetSize), int(counters.PeakWorkingSetSize)


def _rss_peak_bytes() -> int:
    if os.name == "nt":
        return _windows_working_set_bytes()[1]
    assert resource is not None
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; the other supported Unix platforms report KiB.
    return value if sys.platform == "darwin" else value * 1024


def _rss_current_bytes() -> int:
    if os.name == "nt":
        return _windows_working_set_bytes()[0]
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return _rss_peak_bytes()


def _axis_count(family: str, target_dofs: int, components: int) -> int:
    desired_nodes = max(1, math.ceil(target_dofs / components))
    # Quadratic meshes add shared edge nodes.  Half-resolution is a useful
    # structured-grid starting point; actual DOFs are always reported.
    scale = (
        0.5 if family == "tet10"
        else 4.0 ** (-1.0 / 3.0) if family == "hex20"
        else 1.0
    )
    return max(2, math.ceil(desired_nodes ** (1.0 / 3.0) * scale))


def _basis(family: str, target_dofs: int, components: int):
    count = _axis_count(family, target_dofs, components)
    axis = np.linspace(0.0, 1.0, count)
    if family.startswith("tet"):
        mesh = skfemntv.MeshTet.init_tensor(axis, axis, axis)
        if family == "tet10":
            mesh = skfemntv.MeshTet2.from_mesh(mesh)
        scalar = skfemntv.ElementTetP2() if family == "tet10" else skfemntv.ElementTetP1()
    else:
        mesh = skfemntv.MeshHex.init_tensor(axis, axis, axis)
        if family == "hex20":
            mesh = skfemntv.MeshHex20.from_mesh(mesh)
        scalar = skfemntv.ElementHex20() if family == "hex20" else skfemntv.ElementHex1()
    return skfemntv.Basis(mesh, skfemntv.ElementVector(scalar, dim=components))


def run_case(family: str, target_dofs: int, components: int, budget_bytes: int) -> dict:
    rss_before = _rss_current_bytes()
    started = time.perf_counter()
    basis = _basis(family, target_dofs, components)
    basis_seconds = time.perf_counter() - started
    rss_after_basis = _rss_current_bytes()
    estimate = skfemntv.estimate_bilinear_memory(basis)
    common = {
        "family": family,
        "components": components,
        "target_dofs": target_dofs,
        "actual_dofs": basis.N,
        "elements": basis.nelems,
        "local_dofs": estimate.row_local_dofs,
        "quadrature_points": estimate.quadrature_points_per_entity,
        "basis_seconds": basis_seconds,
        "nnz_upper_bound": estimate.nnz_upper_bound,
        "basis_bytes_estimated": estimate.basis_bytes,
        "persistent_bytes_estimated": estimate.persistent_incremental_bytes_upper_bound,
        "peak_total_bytes_estimated": estimate.construction_peak_total_bytes_upper_bound,
        "rss_before_basis": rss_before,
        "rss_after_basis": rss_after_basis,
        "rss_basis_delta": max(0, rss_after_basis - rss_before),
    }
    if budget_bytes and estimate.construction_peak_total_bytes_upper_bound > budget_bytes:
        return {
            **common,
            "status": "skipped-budget",
            "skip_reason": (
                f"estimated {skfemntv.format_bytes(estimate.construction_peak_total_bytes_upper_bound)} "
                f"> budget {skfemntv.format_bytes(budget_bytes)}"
            ),
            "assembler_seconds": "",
            "assembly_seconds": "",
            "actual_nnz": "",
            "nnz_upper_ratio": "",
            "rss_peak": _rss_peak_bytes(),
            "rss_assembler_peak_delta": "",
            "peak_estimate_to_rss_ratio": "",
        }

    started = time.perf_counter()
    assembler = skfemntv.NativeBilinearForm(basis)
    assembler_seconds = time.perf_counter() - started
    started = time.perf_counter()
    matrix = assembler.assemble(gradient=1.0)
    assembly_seconds = time.perf_counter() - started
    rss_peak = _rss_peak_bytes()
    rss_peak_delta = max(0, rss_peak - rss_after_basis)
    measured_total = max(1, rss_peak - rss_before)
    return {
        **common,
        "status": "measured",
        "skip_reason": "",
        "assembler_seconds": assembler_seconds,
        "assembly_seconds": assembly_seconds,
        "actual_nnz": matrix.nnz,
        "nnz_upper_ratio": estimate.nnz_upper_bound / max(1, matrix.nnz),
        "rss_peak": rss_peak,
        "rss_assembler_peak_delta": rss_peak_delta,
        "peak_estimate_to_rss_ratio": estimate.construction_peak_total_bytes_upper_bound / measured_total,
    }


def _worker(arguments: argparse.Namespace) -> int:
    result = run_case(
        arguments.family,
        arguments.target_dofs,
        arguments.components,
        arguments.memory_budget_bytes,
    )
    print(json.dumps(result, separators=(",", ":")))
    return 0


def _run_isolated(script: Path, family: str, target: int, components: int, budget: int) -> dict:
    command = [
        sys.executable,
        str(script),
        "--worker",
        "--family", family,
        "--target-dofs", str(target),
        "--components", str(components),
        "--memory-budget-bytes", str(budget),
    ]
    environment = dict(os.environ)
    process = subprocess.run(command, check=False, capture_output=True, text=True, env=environment)
    if process.returncode != 0:
        return {
            "family": family,
            "components": components,
            "target_dofs": target,
            "status": "failed",
            "skip_reason": process.stderr.strip() or f"worker exit {process.returncode}",
        }
    return json.loads(process.stdout)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({column: row.get(column, "") for column in CSV_COLUMNS} for row in rows)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", default=",".join(FAMILIES))
    parser.add_argument("--targets", default="10000,100000,1000000")
    parser.add_argument("--component-counts", default="1,3")
    parser.add_argument("--memory-budget-gib", type=float, default=8.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--family", choices=FAMILIES, help=argparse.SUPPRESS)
    parser.add_argument("--target-dofs", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--components", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--memory-budget-bytes", type=int, default=0, help=argparse.SUPPRESS)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    if arguments.worker:
        return _worker(arguments)
    families = tuple(item.strip() for item in arguments.families.split(",") if item.strip())
    unknown = set(families) - set(FAMILIES)
    if unknown:
        raise ValueError(f"unknown families: {sorted(unknown)}")
    targets = tuple(int(item) for item in arguments.targets.split(","))
    components = tuple(int(item) for item in arguments.component_counts.split(","))
    if any(value <= 0 for value in (*targets, *components)):
        raise ValueError("targets and component counts must be positive")
    budget = int(arguments.memory_budget_gib * 1024**3)
    script = Path(__file__).resolve()
    rows = []
    for target in targets:
        for family in families:
            for count in components:
                print(f"calibrating family={family} components={count} target_dofs={target}", flush=True)
                row = _run_isolated(script, family, target, count, budget)
                rows.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
    if arguments.output:
        _write_csv(arguments.output, rows)
    return int(any(row["status"] == "failed" for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
