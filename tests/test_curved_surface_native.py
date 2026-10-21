import numpy as np
import pytest
import skfem

import skfemntv
import skfemntv.supermesh as supermesh_module


def test_native_tet10_surface_evaluation_matches_python(monkeypatch):
    native_evaluator = supermesh_module._evaluate_tet_surface_native
    if native_evaluator is None:
        pytest.skip("native curved-surface evaluator is unavailable")
    mesh = skfemntv.MeshTet2()
    points = mesh.p.copy()
    points[:, 4] += np.array([0., 0., .25])
    curved = skfemntv.MeshTet2(points, mesh.t)
    facets = skfemntv.FacetBasis(
        curved,
        skfemntv.ElementVector(skfemntv.ElementTetP2()),
        intorder=4,
    )
    surface = supermesh_module.BoundarySurface(
        facets, geometry_tolerance=.02, max_subdivision_level=3
    )
    search_triangle = 0
    parent = surface.parents[search_triangle]
    coordinates = curved.p[:, curved.t[:, parent]]
    reference_points = np.array([
        [.1, .2, .2], [.2, .1, .15], [.25, .1, .1],
    ])
    shape, _ = facets.volume_basis._evaluate_reference(reference_points.T)
    physical_points = shape @ coordinates.T

    actual = surface.evaluate(search_triangle, physical_points)
    monkeypatch.setattr(supermesh_module, "_evaluate_tet_surface_native", None)
    expected = surface.evaluate(search_triangle, physical_points)

    np.testing.assert_allclose(actual[0], expected[0], rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(actual[1], expected[1], rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(actual[2], expected[2], rtol=2e-12, atol=2e-12)


def test_native_tet10_surface_evaluation_matches_skfem_element():
    if supermesh_module._evaluate_tet_surface_native is None:
        pytest.skip("native curved-surface evaluator is unavailable")
    mesh = skfemntv.MeshTet2()
    points = mesh.p.copy()
    points[:, 4] += np.array([0., 0., .25])
    curved = skfemntv.MeshTet2(points, mesh.t)
    facets = skfemntv.FacetBasis(
        curved,
        skfemntv.ElementVector(skfemntv.ElementTetP2()),
        intorder=4,
    )
    surface = supermesh_module.BoundarySurface(
        facets, geometry_tolerance=.02, max_subdivision_level=3
    )
    search_triangle = 0
    parent = surface.parents[search_triangle]
    coordinates = curved.p[:, curved.t[:, parent]]
    reference_points = np.array([
        [.1, .2, .2], [.2, .1, .15], [.25, .1, .1],
    ])

    reference_element = skfem.ElementTetP2()
    reference_values = np.stack([
        reference_element.lbasis(reference_points.T, i)[0]
        for i in range(10)
    ], axis=1)
    reference_gradients = np.stack([
        reference_element.lbasis(reference_points.T, i)[1].T
        for i in range(10)
    ], axis=1)
    physical_points = reference_values @ coordinates.T

    actual_values, actual_gradients, actual_normals = surface.evaluate(
        search_triangle, physical_points
    )
    tangents = surface.reference_tangents[
        surface.search_faces[search_triangle]
    ]
    centroid = coordinates.mean(axis=1)
    expected_gradients = []
    expected_normals = []
    for values, gradients, point in zip(
        reference_values, reference_gradients, physical_points
    ):
        jacobian = coordinates @ gradients
        expected_gradients.append(gradients @ np.linalg.inv(jacobian))
        normal = np.cross(*(jacobian @ tangents).T)
        normal /= np.linalg.norm(normal)
        if normal @ (point - centroid) < 0.:
            normal = -normal
        expected_normals.append(normal)

    np.testing.assert_allclose(actual_values, reference_values, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(
        actual_gradients, expected_gradients, rtol=2e-11, atol=2e-11
    )
    np.testing.assert_allclose(
        actual_normals, expected_normals, rtol=2e-12, atol=2e-12
    )
