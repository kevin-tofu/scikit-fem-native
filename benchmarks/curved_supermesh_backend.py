"""Compare Python and native curved tetrahedral surface evaluation."""

from __future__ import annotations

import statistics
import time

import numpy as np
import skfemntv
import skfemntv.supermesh as supermesh_module


def build_case():
    mesh = skfemntv.MeshTet2()
    points = mesh.p.copy()
    points[:, 4] += np.array([0.0, 0.0, 0.25])
    curved = skfemntv.MeshTet2(points, mesh.t)
    facets = skfemntv.FacetBasis(
        curved,
        skfemntv.ElementVector(skfemntv.ElementTetP2()),
        intorder=4,
    )
    return facets


def run(evaluator, facets, repeats=3):
    supermesh_module._evaluate_tet_surface_native = evaluator
    results = []
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        coupling = skfemntv.TriangleSupermesh.from_facets(
            facets, facets, geometry_tolerance=0.01, max_subdivision_level=4
        )
        matrix = coupling.assemble()
        timings.append(time.perf_counter() - start)
        results.append((coupling, matrix.copy()))
    return statistics.median(timings), results[-1]


def main():
    native = supermesh_module._evaluate_tet_surface_native
    if native is None:
        raise RuntimeError("native extension is unavailable")
    facets = build_case()
    python_time, (python_coupling, python_matrix) = run(None, facets)
    native_time, (native_coupling, native_matrix) = run(native, facets)

    np.testing.assert_array_equal(
        native_coupling._master_parent_facets,
        python_coupling._master_parent_facets,
    )
    np.testing.assert_array_equal(
        native_coupling._slave_parent_facets,
        python_coupling._slave_parent_facets,
    )
    np.testing.assert_allclose(
        native_coupling.global_coordinates,
        python_coupling.global_coordinates,
        rtol=2e-11,
        atol=2e-11,
    )
    np.testing.assert_allclose(
        native_coupling._weights, python_coupling._weights,
        rtol=2e-11, atol=2e-11,
    )
    np.testing.assert_allclose(
        native_matrix.toarray(), python_matrix.toarray(),
        rtol=3e-10, atol=3e-12,
    )
    print(f"Python curved supermesh: {python_time * 1e3:.3f} ms")
    print(f"C++ curved supermesh:    {native_time * 1e3:.3f} ms")
    print(f"Speedup:                 {python_time / native_time:.2f}x")
    print(f"Search triangles:         {native_coupling.diagnostics.master_search_triangle_count}")
    print(f"Integration triangles:    {native_coupling.diagnostics.integration_triangle_count}")


if __name__ == "__main__":
    main()
