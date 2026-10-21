import numpy as np

from skfemntv._skfn import tabulate_hex27_shapes


def _python_hex27(points):
    shape = np.empty((points.shape[1], 27))
    gradient = np.empty((points.shape[1], 27, 3))
    for q, (x, y, z) in enumerate(points.T):
        values = tuple(
            np.array([2. * (v - .5) * (v - 1.), 4. * v * (1. - v),
                2. * v * (v - .5)]) for v in (x, y, z)
        )
        derivatives = tuple(
            np.array([4. * v - 3., 4. - 8. * v, 4. * v - 1.])
            for v in (x, y, z)
        )
        node = 0
        for k in range(3):
            for j in range(3):
                for i in range(3):
                    shape[q, node] = values[0][i] * values[1][j] * values[2][k]
                    gradient[q, node] = (
                        derivatives[0][i] * values[1][j] * values[2][k],
                        values[0][i] * derivatives[1][j] * values[2][k],
                        values[0][i] * values[1][j] * derivatives[2][k],
                    )
                    node += 1
    return shape, gradient


def test_native_hex27_tabulation_matches_python_reference():
    rng = np.random.default_rng(20261016)
    points = rng.random((3, 83))
    actual = tabulate_hex27_shapes(points)
    expected = _python_hex27(points)
    np.testing.assert_allclose(actual[0], expected[0], rtol=2e-15, atol=2e-15)
    np.testing.assert_allclose(actual[1], expected[1], rtol=2e-15, atol=2e-15)


def test_native_hex27_tabulation_rejects_wrong_point_shape():
    import pytest

    with pytest.raises(ValueError, match=r"shape \(3, nq\)"):
        tabulate_hex27_shapes(np.zeros((2, 4)))
