import numpy as np
import pytest
import skfem
import skfemntv


@pytest.mark.parametrize("dimension", [2, 3])
def test_composite_evaluate_probes_and_interpolator(dimension):
    if dimension == 2:
        mesh = skfemntv.MeshTri.init_tensor(
            np.array((0.0, 0.5, 1.0)), np.array((0.0, 0.5, 1.0))
        )
        reference_mesh = skfem.MeshTri(mesh.p, mesh.t)
        native_fields = (
            skfemntv.ElementTriP2(),
            skfemntv.ElementVector(skfemntv.ElementTriP1(), dim=2),
        )
        reference_fields = (
            skfem.ElementTriP2(),
            skfem.ElementVector(skfem.ElementTriP1(), dim=2),
        )
        points = np.array(((0.2, 0.7, 0.4), (0.3, 0.2, 0.8)))

        def scalar_field(x):
            return x[0] ** 2 + x[0] * x[1] + 2.0 * x[1]

        def vector_field(x):
            return np.asarray((x[0] + 2.0 * x[1], 2.0 * x[0] - x[1]))

        def vector_gradient():
            return np.asarray(((1.0, 2.0), (2.0, -1.0)))
    else:
        mesh = skfemntv.MeshTet.init_tensor(
            np.array((0.0, 1.0)), np.array((0.0, 1.0)), np.array((0.0, 1.0))
        )
        reference_mesh = skfem.MeshTet(mesh.p, mesh.t)
        native_fields = (
            skfemntv.ElementTetP2(),
            skfemntv.ElementVector(skfemntv.ElementTetP1(), dim=3),
        )
        reference_fields = (
            skfem.ElementTetP2(),
            skfem.ElementVector(skfem.ElementTetP1(), dim=3),
        )
        points = np.array(((0.1, 0.2), (0.2, 0.1), (0.3, 0.2)))

        def scalar_field(x):
            return x[0] ** 2 + x[1] * x[2] + 2.0 * x[2]

        def vector_field(x):
            return np.asarray((
                x[0] + 2.0 * x[2],
                2.0 * x[0] - x[1],
                x[1] + 3.0 * x[2],
            ))

        def vector_gradient():
            return np.asarray(((1.0, 0.0, 2.0), (2.0, -1.0, 0.0), (0.0, 1.0, 3.0)))

    native_basis = skfemntv.Basis(
        mesh, skfemntv.ElementComposite(*native_fields), intorder=5
    )
    reference_bases = tuple(
        skfem.Basis(reference_mesh, field, intorder=5)
        for field in reference_fields
    )
    coefficients = np.zeros(native_basis.N)
    indices = native_basis.split_indices()
    scalar_dofs = indices[0]
    scalar_coordinates = native_basis.doflocs[:, scalar_dofs]
    coefficients[scalar_dofs] = scalar_field(scalar_coordinates)
    vector_dofs = indices[1]
    vector_coordinates = mesh.p[:, native_basis.subbases[1].active_nodes]
    coefficients[vector_dofs] = vector_field(vector_coordinates).reshape(
        -1, order="F"
    )

    evaluations = native_basis.evaluate_at(points, coefficients)
    assert isinstance(evaluations, tuple) and len(evaluations) == 2
    expected_scalar = reference_bases[0].probes(points) @ coefficients[scalar_dofs]
    expected_vector = (
        reference_bases[1].probes(points) @ coefficients[vector_dofs]
    ).reshape(dimension, points.shape[1])
    np.testing.assert_allclose(evaluations[0].value, expected_scalar, atol=2e-12)
    np.testing.assert_allclose(evaluations[1].value, expected_vector, atol=2e-12)
    np.testing.assert_allclose(
        evaluations[1].grad,
        np.broadcast_to(vector_gradient()[:, :, None], (dimension, dimension, points.shape[1])),
        atol=2e-12,
    )
    assert all(np.all(item.cell_ids >= 0) for item in evaluations)

    probes = native_basis.probes(points)
    assert isinstance(probes, tuple) and len(probes) == 2
    np.testing.assert_allclose(probes[0] @ coefficients, evaluations[0].value)
    np.testing.assert_allclose(
        (probes[1] @ coefficients).reshape(dimension, -1),
        evaluations[1].value,
    )

    interpolator = native_basis.interpolator(coefficients)
    interpolated = interpolator(points)
    assert isinstance(interpolated, tuple) and len(interpolated) == 2
    np.testing.assert_allclose(interpolated[0], evaluations[0].value)
    np.testing.assert_allclose(interpolated[1], evaluations[1].value)
    grid = points[:, :2, None]
    grid_values = interpolator(grid)
    assert grid_values[0].shape == (2, 1)
    assert grid_values[1].shape == (dimension, 2, 1)


def test_composite_point_evaluation_reports_outside_points_for_each_field():
    mesh = skfemntv.MeshTri.init_tensor([0.0, 1.0], [0.0, 1.0])
    basis = skfemntv.Basis(
        mesh,
        skfemntv.ElementComposite(
            skfemntv.ElementTriP1(), skfemntv.ElementTriP2()
        ),
        intorder=5,
    )
    points = np.array(((0.2, 2.0), (0.3, 2.0)))
    result = basis.evaluate_at(points, np.zeros(basis.N), outside="nan")
    assert len(result) == 2
    for field_result in result:
        assert field_result.outside.tolist() == [False, True]
        assert np.isnan(field_result.value[-1])
