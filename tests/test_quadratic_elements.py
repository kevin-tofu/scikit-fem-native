import numpy as np
import pytest
from skfem import (
    Basis,
    ElementHex2,
    ElementTetP2,
    ElementVector,
    MeshHex2,
    MeshTet2,
)
from skfem.models.elasticity import linear_elasticity

from skfn import LinearElasticity, NativeAssembler, NeoHookean


@pytest.mark.parametrize(
    ("mesh", "element"),
    [(MeshTet2(), ElementTetP2()), (MeshHex2(), ElementHex2())],
)
def test_quadratic_linear_elasticity_matches_skfem(mesh, element):
    basis = Basis(mesh, ElementVector(element), intorder=4)
    young, nu = 57.0, 0.24
    lmbda = young * nu / ((1 + nu) * (1 - 2 * nu))
    mu = young / (2 * (1 + nu))
    expected = linear_elasticity(Lambda=lmbda, Mu=mu).assemble(basis)
    assembler = NativeAssembler.from_skfem(
        basis, LinearElasticity(young, nu)
    )
    actual = assembler.evaluate(np.zeros(basis.N)).tangent
    np.testing.assert_allclose(
        actual.toarray(), expected.toarray(), rtol=3e-12, atol=3e-12
    )


@pytest.mark.parametrize(
    ("mesh", "element"),
    [(MeshTet2(), ElementTetP2()), (MeshHex2(), ElementHex2())],
)
def test_quadratic_neo_hookean_tangent_is_consistent(mesh, element):
    basis = Basis(mesh, ElementVector(element), intorder=4)
    assembler = NativeAssembler.from_skfem(
        basis, NeoHookean(mu=8.0, lmbda=12.0)
    )
    rng = np.random.default_rng(81)
    u = rng.normal(scale=0.003, size=basis.N)
    direction = rng.normal(size=basis.N)
    tangent_action = assembler.evaluate(u).tangent @ direction
    epsilon = 1e-7
    plus = assembler.evaluate(
        u + epsilon * direction, mode="residual"
    ).residual.copy()
    minus = assembler.evaluate(
        u - epsilon * direction, mode="residual"
    ).residual.copy()
    np.testing.assert_allclose(
        tangent_action,
        (plus - minus) / (2 * epsilon),
        rtol=2e-7,
        atol=2e-7,
    )
