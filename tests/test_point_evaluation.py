import numpy as np
import pytest

import skfemntv


@pytest.mark.parametrize(
    "mesh,element,points,field,expected_value,expected_gradient",
    [
        (
            skfemntv.MeshTri2(),
            skfemntv.ElementTriP2(),
            np.array(((0.2, 0.6), (0.3, 0.2))),
            lambda x, y, z: x**2 + 2.0 * x * y + 3.0 * y**2,
            np.array((0.43, 0.72)),
            np.array(((1.0, 1.6), (2.2, 2.4))),
        ),
        (
            skfemntv.MeshTet2(),
            skfemntv.ElementTetP2(),
            np.array(((0.1, 0.2), (0.2, 0.1), (0.3, 0.2))),
            lambda x, y, z: x**2 + 2.0 * x * y + 3.0 * y * z,
            np.array((0.23, 0.14)),
            np.array(((0.6, 0.6), (1.1, 1.0), (0.6, 0.3))),
        ),
    ],
)
def test_point_evaluation_reproduces_quadratic_polynomials(
    mesh, element, points, field, expected_value, expected_gradient
):
    basis = skfemntv.Basis(mesh, element)
    x, y = basis.doflocs[:2]
    z = basis.doflocs[2] if basis.doflocs.shape[0] == 3 else np.zeros_like(x)
    coefficients = field(x, y, z)

    result = basis.evaluate_at(points, coefficients)

    np.testing.assert_allclose(result.value, expected_value, atol=1e-13)
    np.testing.assert_allclose(result.grad, expected_gradient, atol=1e-13)
    assert np.all(result.cell_ids >= 0)
    assert not np.any(result.outside)


def test_point_evaluation_vector_field_shapes_and_values():
    mesh = skfemntv.MeshTet()
    basis = skfemntv.Basis(
        mesh, skfemntv.ElementVector(skfemntv.ElementTetP1())
    )
    coefficients = np.empty(basis.N)
    x, y, z = mesh.p
    coefficients[basis.nodal_dofs[0]] = x + 2.0 * y
    coefficients[basis.nodal_dofs[1]] = y - z
    coefficients[basis.nodal_dofs[2]] = 3.0 * z

    result = basis.evaluate_at(np.array(((0.1,), (0.2,), (0.3,))), coefficients)

    assert result.value.shape == (3, 1)
    assert result.grad.shape == (3, 3, 1)
    np.testing.assert_allclose(result.value[:, 0], (0.5, -0.1, 0.9))
    np.testing.assert_allclose(
        result.grad[:, :, 0], ((1.0, 2.0, 0.0), (0.0, 1.0, -1.0), (0.0, 0.0, 3.0))
    )


def test_point_evaluation_reports_shared_facets_and_outside_points():
    mesh = skfemntv.MeshTri.init_tensor(
        np.array((0.0, 1.0)), np.array((0.0, 1.0))
    )
    basis = skfemntv.Basis(mesh, skfemntv.ElementTriP1())
    coefficients = basis.doflocs[0] + basis.doflocs[1]
    points = np.array(((0.5, 2.0), (0.5, 2.0)))

    result = basis.evaluate_at(points, coefficients, outside="nan")

    assert result.ambiguous[0]
    assert result.on_boundary[0]
    assert result.outside[1]
    assert result.cell_ids[1] == -1
    assert np.isnan(result.value[1])
    with pytest.raises(ValueError, match="lies outside the basis mesh"):
        basis.evaluate_at(points, coefficients)


def test_point_evaluation_preserves_global_cell_ids_for_restricted_basis():
    mesh = skfemntv.MeshTri.init_tensor(
        np.array((0.0, 1.0, 2.0)), np.array((0.0, 1.0))
    )
    basis = skfemntv.Basis(mesh, skfemntv.ElementTriP1()).with_elements((2,))
    coefficients = basis.doflocs[0] + basis.doflocs[1]

    result = basis.evaluate_at(np.array(((1.7,), (0.2,))), coefficients)

    assert result.cell_ids[0] == 2
