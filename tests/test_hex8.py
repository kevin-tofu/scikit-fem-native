import numpy as np
from skfem import Basis, ElementHex1, ElementVector, MeshHex
from skfem.models.elasticity import linear_elasticity

from skfemntv import LinearElasticHex8, NativeAssembler, NeoHookeanHex8


def hex_basis():
    mesh = MeshHex.init_tensor(
        np.linspace(0., 1., 3),
        np.linspace(0., 1., 2),
        np.linspace(0., 1., 2),
    )
    return Basis(mesh, ElementVector(ElementHex1()))


def test_hex8_linear_elasticity_matches_skfem():
    basis = hex_basis()
    young, nu = 71., .29
    assembler = NativeAssembler.from_skfem(
        basis, LinearElasticHex8(young, nu)
    )
    lmbda = young*nu/((1.+nu)*(1.-2.*nu))
    mu = young/(2.*(1.+nu))
    expected = linear_elasticity(Lambda=lmbda, Mu=mu).assemble(basis)
    actual = assembler.evaluate(np.zeros(basis.N)).tangent
    np.testing.assert_allclose(
        actual.toarray(), expected.toarray(), rtol=2e-12, atol=2e-12
    )


def test_distorted_hex8_matches_skfem_at_multiple_quadrature_points():
    original = MeshHex.init_tensor(
        np.array([0., 1.]), np.array([0., 1.]), np.array([0., 1.])
    )
    points = original.p.copy()
    points[:, 7] += np.array([.2, -.1, .15])
    basis = Basis(MeshHex(points, original.t), ElementVector(ElementHex1()))
    young, nu = 43., .21
    assembler = NativeAssembler.from_skfem(
        basis, LinearElasticHex8(young, nu)
    )
    lmbda = young*nu/((1.+nu)*(1.-2.*nu))
    mu = young/(2.*(1.+nu))
    expected = linear_elasticity(Lambda=lmbda, Mu=mu).assemble(basis)
    actual = assembler.evaluate(np.zeros(basis.N)).tangent
    np.testing.assert_allclose(
        actual.toarray(), expected.toarray(), rtol=3e-12, atol=3e-12
    )


def test_hex8_neo_hookean_consistent_tangent():
    basis = hex_basis()
    assembler = NativeAssembler.from_skfem(
        basis, NeoHookeanHex8(mu=13., lmbda=17.)
    )
    rng = np.random.default_rng(9)
    u = rng.normal(scale=.01, size=basis.N)
    direction = rng.normal(size=basis.N)
    tangent_action = assembler.evaluate(u).tangent @ direction
    eps = 2e-7
    plus = assembler.evaluate(
        u+eps*direction, mode="residual"
    ).residual.copy()
    minus = assembler.evaluate(
        u-eps*direction, mode="residual"
    ).residual.copy()
    np.testing.assert_allclose(
        tangent_action, (plus-minus)/(2.*eps), rtol=5e-8, atol=5e-8
    )


def test_hex8_rigid_translation():
    basis = hex_basis()
    assembler = NativeAssembler.from_skfem(
        basis, NeoHookeanHex8(mu=3., lmbda=5.)
    )
    u = np.tile(np.array([.2, -.1, .3]), basis.mesh.p.shape[1])
    np.testing.assert_allclose(
        assembler.evaluate(u).residual, 0., atol=2e-14
    )
