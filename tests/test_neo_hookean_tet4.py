import numpy as np
from scipy.sparse.linalg import spsolve

from skfemntv import NativeAssembler, NeoHookeanTet4


def one_tet():
    coordinates = np.array(
        [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
    )
    connectivity = np.array([[0, 1, 2, 3]], dtype=np.int64)
    dofs = np.arange(12, dtype=np.int64).reshape(1, 4, 3)
    return coordinates, connectivity, dofs


def test_undeformed_configuration_has_zero_residual():
    assembler = NativeAssembler(*one_tet(), NeoHookeanTet4(10., 20.))
    out = assembler.evaluate(np.zeros(12))
    np.testing.assert_allclose(out.residual, 0., atol=1e-14)
    np.testing.assert_allclose(out.tangent.toarray(), out.tangent.toarray().T)


def test_consistent_tangent_matches_directional_difference():
    assembler = NativeAssembler(*one_tet(), NeoHookeanTet4(13., 17.))
    rng = np.random.default_rng(42)
    u = rng.normal(scale=.025, size=12)
    direction = rng.normal(size=12)
    out = assembler.evaluate(u)
    tangent_action = out.tangent @ direction
    eps = 2e-7
    plus = assembler.evaluate(u + eps*direction, mode="residual").residual.copy()
    minus = assembler.evaluate(u - eps*direction, mode="residual").residual.copy()
    np.testing.assert_allclose(
        tangent_action, (plus-minus)/(2*eps), rtol=2e-8, atol=2e-8
    )


def test_rigid_rotation_has_zero_internal_force():
    assembler = NativeAssembler(*one_tet(), NeoHookeanTet4(7., 11.))
    angle = .31
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.],
                         [np.sin(angle), np.cos(angle), 0.],
                         [0., 0., 1.]])
    coordinates = one_tet()[0]
    u = ((rotation @ coordinates.T).T - coordinates).reshape(-1)
    np.testing.assert_allclose(assembler.evaluate(u).residual, 0., atol=2e-14)


def test_newton_converges_for_loaded_tet():
    assembler = NativeAssembler(
        *one_tet(), NeoHookeanTet4.from_young_poisson(100., .3)
    )
    fixed = np.array([0, 1, 2, 4, 5, 8])
    free = np.setdiff1d(np.arange(12), fixed)
    loads = np.zeros(12)
    loads[11] = 1.
    u = np.zeros(12)
    norms = []
    for _ in range(10):
        out = assembler.evaluate(u, loads=loads)
        norms.append(np.linalg.norm(out.residual[free]))
        if norms[-1] < 1e-11:
            break
        u[free] += spsolve(
            out.tangent[free][:, free], -out.residual[free]
        )
    assert norms[-1] < 1e-11
    assert len(norms) <= 6
    assert u[11] > 0.


def test_inverted_deformation_is_rejected():
    assembler = NativeAssembler(*one_tet(), NeoHookeanTet4(10., 20.))
    u = np.zeros(12)
    u[3] = -2.
    with np.testing.assert_raises_regex(ValueError, "non-positive"):
        assembler.evaluate(u)
