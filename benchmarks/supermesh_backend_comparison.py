"""Compare scikit-fem/Shapely and skfemntv Mortar assembly backends."""

from __future__ import annotations

import argparse
import csv
import json
import resource
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from skfem.supermeshing import intersect

import skfemntv


@dataclass(frozen=True)
class Surface:
    points: np.ndarray
    triangles: np.ndarray


@dataclass(frozen=True)
class BenchmarkResult:
    case: str
    backend: str
    multiplier_space: str
    master_facets: int
    slave_facets: int
    overlap_elements: int
    multiplier_rows: int
    matrix_nnz: int
    matrix_mib: float
    search_ms: float
    assembly_ms: float
    overlap_area: float
    constant_residual: float
    peak_rss_mib: float


def grid(cells: int, *, reverse: bool = False) -> Surface:
    axis = np.linspace(0.0, 1.0, cells + 1)
    points = np.asarray(
        [(x, y, 0.0) for y in axis for x in axis],
        dtype=np.float64,
    ).T
    triangles = []
    for j in range(cells):
        for i in range(cells):
            a = j * (cells + 1) + i
            b = a + 1
            c = a + cells + 1
            d = c + 1
            triangles.extend(
                ((a, b, c), (b, d, c))
                if not reverse
                else ((a, b, d), (a, d, c))
            )
    return Surface(points, np.asarray(triangles, dtype=np.int64).T)


def _area(points: np.ndarray) -> float:
    edge_a = points[:, 1] - points[:, 0]
    edge_b = points[:, 2] - points[:, 0]
    return 0.5 * abs(float(np.linalg.det(np.column_stack((edge_a, edge_b)))))


def _integrated_shape(
    triangle_points: np.ndarray,
    overlap_points: np.ndarray,
    area: float,
) -> np.ndarray:
    transform = np.vstack((triangle_points, np.ones(3)))
    coordinates = np.vstack((overlap_points, np.ones(3)))
    barycentric = np.linalg.solve(transform, coordinates)
    return area * barycentric.mean(axis=1)


def _matrix_mib(matrix: csr_matrix) -> float:
    return (
        matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
    ) / 2.0**20


def _peak_rss_mib() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / 1024.0


def assemble_skfem_shapely(
    master: Surface,
    slave: Surface,
    multiplier_space: str,
) -> tuple[csr_matrix, int, float, float, float]:
    started = time.perf_counter()
    supermesh, master_parents, slave_parents = intersect(
        (master.points[:2], master.triangles),
        (slave.points[:2], slave.triangles),
    )
    search_ms = (time.perf_counter() - started) * 1.0e3
    started = time.perf_counter()
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    slave_rows: dict[int, int] = {}
    overlap_area = 0.0
    retained = 0
    master_offset = master.points.shape[1]
    for element, (master_parent, slave_parent) in enumerate(
        zip(master_parents, slave_parents, strict=True)
    ):
        overlap_points = supermesh.p[:, supermesh.t[:, element]]
        area = _area(overlap_points)
        if area <= 0.0:
            continue
        if multiplier_space == "slave_facet_p0":
            slave_parent = int(slave_parent)
            if slave_parent not in slave_rows:
                slave_rows[slave_parent] = len(slave_rows)
            row = slave_rows[slave_parent]
        else:
            row = retained
        master_nodes = master.triangles[:, int(master_parent)]
        slave_nodes = slave.triangles[:, int(slave_parent)]
        master_weights = _integrated_shape(
            master.points[:2, master_nodes], overlap_points, area
        )
        slave_weights = _integrated_shape(
            slave.points[:2, slave_nodes], overlap_points, area
        )
        rows.extend([row] * 6)
        columns.extend(master_nodes.tolist())
        columns.extend((master_offset + slave_nodes).tolist())
        values.extend(master_weights.tolist())
        values.extend((-slave_weights).tolist())
        overlap_area += area
        retained += 1
    row_count = len(slave_rows) if multiplier_space == "slave_facet_p0" else retained
    matrix = csr_matrix(
        (values, (rows, columns)),
        shape=(row_count, master_offset + slave.points.shape[1]),
    )
    assembly_ms = (time.perf_counter() - started) * 1.0e3
    return matrix, retained, overlap_area, search_ms, assembly_ms


def assemble_native(
    master: Surface,
    slave: Surface,
    multiplier_space: str,
) -> tuple[csr_matrix, int, float, float, float]:
    started = time.perf_counter()
    supermesh = skfemntv.TriangleSupermesh(
        master.points,
        master.triangles,
        slave.points,
        slave.triangles,
    )
    search_ms = (time.perf_counter() - started) * 1.0e3
    started = time.perf_counter()
    result = supermesh.assemble_mortar(multiplier_space)
    assembly_ms = (time.perf_counter() - started) * 1.0e3
    return (
        result.coupling_matrix.tocsr(),
        int(supermesh.diagnostics.integration_triangle_count),
        float(result.overlap_area),
        search_ms,
        assembly_ms,
    )


def measured(function, repeat: int):
    values = []
    result = None
    for _ in range(repeat):
        started = time.perf_counter()
        result = function()
        values.append((time.perf_counter() - started) * 1.0e3)
    return result, statistics.median(values)


def run_case(
    case: str,
    master: Surface,
    slave: Surface,
    repeat: int,
) -> list[BenchmarkResult]:
    results = []
    for backend, assembler in (
        ("skfem", assemble_skfem_shapely),
        ("skfemntv", assemble_native),
    ):
        for multiplier_space in ("overlap_p0", "slave_facet_p0"):
            assembled, _total_ms = measured(
                lambda assembler=assembler, multiplier_space=multiplier_space: assembler(
                    master, slave, multiplier_space
                ),
                repeat,
            )
            matrix, overlaps, area, search_ms, assembly_ms = assembled
            residual = float(np.linalg.norm(
                matrix @ np.ones(matrix.shape[1]),
                ord=np.inf,
            ))
            results.append(BenchmarkResult(
                case=case,
                backend=backend,
                multiplier_space=multiplier_space,
                master_facets=master.triangles.shape[1],
                slave_facets=slave.triangles.shape[1],
                overlap_elements=overlaps,
                multiplier_rows=matrix.shape[0],
                matrix_nnz=matrix.nnz,
                matrix_mib=_matrix_mib(matrix),
                search_ms=search_ms,
                assembly_ms=assembly_ms,
                overlap_area=area,
                constant_residual=residual,
                peak_rss_mib=_peak_rss_mib(),
            ))
    return results


def write_outputs(output: Path, results: list[BenchmarkResult]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    records = [asdict(result) for result in results]
    (output / "results.json").write_text(
        json.dumps(records, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=BenchmarkResult.__dataclass_fields__)
        writer.writeheader()
        writer.writerows(records)

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    labels = [
        f"{item.case}\n{item.backend}\n{item.multiplier_space}"
        for item in results
    ]
    axes[0, 0].bar(labels, [item.search_ms for item in results])
    axes[0, 0].set_title("Supermesh search")
    axes[0, 0].set_ylabel("ms")
    axes[0, 1].bar(labels, [item.assembly_ms for item in results])
    axes[0, 1].set_title("Mortar assembly")
    axes[0, 1].set_ylabel("ms")
    axes[1, 0].bar(labels, [item.multiplier_rows for item in results])
    axes[1, 0].set_title("Multiplier rows")
    axes[1, 1].bar(labels, [item.matrix_mib for item in results])
    axes[1, 1].set_title("Coupling matrix storage")
    axes[1, 1].set_ylabel("MiB")
    for axis in axes.ravel():
        axis.tick_params(axis="x", labelrotation=75, labelsize=7)
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "comparison.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, default=12)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-results/supermesh-backends"),
    )
    args = parser.parse_args()
    cells = args.cells
    cases = (
        ("matching", grid(cells), grid(cells)),
        ("nonmatching", grid(cells), grid(cells + 1, reverse=True)),
        ("imbalanced", grid(4 * cells), grid(max(2, cells // 3), reverse=True)),
    )
    results = [
        result
        for case, master, slave in cases
        for result in run_case(case, master, slave, args.repeat)
    ]
    write_outputs(args.output, results)
    for result in results:
        print(" ".join(
            f"{key}={value:.6g}" if isinstance(value, float)
            else f"{key}={value}"
            for key, value in asdict(result).items()
        ))


if __name__ == "__main__":
    main()
