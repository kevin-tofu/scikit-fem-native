import numpy as np

import skfemntv
from skfemntv.helpers import ddot, div, sym_grad


def _crack_tip_bases(resolution=8):
    axis = np.linspace(-1.0, 1.0, resolution + 1)
    mesh = skfemntv.MeshTri.init_tensor(axis, axis)
    mesh.p[1] += 0.037 * (1.0 - np.abs(mesh.p[0]))
    parent = skfemntv.Basis(
        mesh,
        skfemntv.ElementVector(skfemntv.ElementTriP1(), dim=1),
        intorder=2,
    )
    level_set = skfemntv.LevelSet(
        lambda x: x[1] - 0.037, tolerance=1.0e-12
    )
    classification = level_set.classify(mesh)
    assert len(classification.cut) > 0
    inside = skfemntv.ModeICrackTipCutBasis(
        parent,
        level_set,
        side="inside",
        tip=np.array([0.0, 0.037]),
        tip_radius=0.6,
        intorder=3,
        classification=classification,
    )
    outside = skfemntv.ModeICrackTipCutBasis(
        parent,
        level_set,
        side="outside",
        tip=np.array([0.0, 0.037]),
        tip_radius=0.6,
        intorder=3,
        classification=classification,
    )
    assert len(inside.enriched_nodes) > 0
    return mesh, parent, inside, outside


def test_mode_i_tip_branch_functions_have_expected_crack_face_jump():
    from skfemntv.xfem import _mode_i_tip_functions

    epsilon = 1.0e-10
    points = np.array([[-0.4, epsilon], [-0.4, -epsilon]])
    values, gradients = _mode_i_tip_functions(points, np.zeros(2))
    np.testing.assert_allclose(values[0, 0], -values[1, 0], rtol=1.0e-8)
    np.testing.assert_allclose(values[:, 1:], 0.0, atol=1.0e-8)
    assert np.isfinite(gradients).all()


def test_vector_mode_i_cut_basis_interpolates_affine_displacement():
    mesh, _, inside, _ = _crack_tip_bases()
    vector_basis = skfemntv.ShiftedHeavisideVectorCutBasis(inside, components=2)
    coefficients = np.zeros(vector_basis.N)
    scalar_count = inside.N
    scalar_coefficients = np.zeros((scalar_count, 2))
    scalar_coefficients[: mesh.p.shape[1], 0] = 1.0 + 2.0 * mesh.p[0] - mesh.p[1]
    scalar_coefficients[: mesh.p.shape[1], 1] = -0.5 + mesh.p[0] + 3.0 * mesh.p[1]
    coefficients[:] = scalar_coefficients.ravel()

    field = vector_basis.interpolate(coefficients)
    expected = np.vstack((
        1.0 + 2.0 * inside.points[:, 0] - inside.points[:, 1],
        -0.5 + inside.points[:, 0] + 3.0 * inside.points[:, 1],
    ))
    np.testing.assert_allclose(field.value, expected, atol=2.0e-13)
    np.testing.assert_allclose(field.grad[0], np.array([[2.0, -1.0]] * inside.npoints).T)
    np.testing.assert_allclose(field.grad[1], np.array([[1.0, 3.0]] * inside.npoints).T)


def test_mode_i_cut_basis_assembles_native_symmetric_elasticity():
    mesh, parent, inside, outside = _crack_tip_bases()
    inside_vector = skfemntv.ShiftedHeavisideVectorCutBasis(
        inside, components=2
    )
    outside_vector = skfemntv.ShiftedHeavisideVectorCutBasis(
        outside, components=2
    )
    lame_lambda = 1.0
    lame_mu = 1.0

    @skfemntv.BilinearForm
    def elasticity(u, v, w):
        return (
            2.0 * lame_mu * ddot(sym_grad(u), sym_grad(v))
            + lame_lambda * div(u) * div(v)
        )

    matrix = skfemntv.asm(elasticity, inside_vector)
    matrix += skfemntv.asm(elasticity, outside_vector)
    assert matrix.shape == (inside_vector.N, inside_vector.N)
    assert matrix.nnz > 0
    np.testing.assert_allclose(
        (matrix - matrix.T).data, 0.0, atol=2.0e-12
    )
    regular_parent = skfemntv.Basis(
        mesh,
        skfemntv.ElementVector(skfemntv.ElementTriP1(), dim=2),
        intorder=4,
    )
    regular_matrix = skfemntv.asm(elasticity, regular_parent)
    base_indices = np.arange(2 * parent.N)
    np.testing.assert_allclose(
        matrix[base_indices][:, base_indices].toarray(),
        regular_matrix.toarray(),
        atol=3.0e-12,
    )
