import numpy as np
import pytest

skfem = pytest.importorskip("skfem")
from skfem import Basis, ElementTetP1, ElementVector, MeshTet
from skfem.helpers import ddot, sym_grad, trace
from skfem.models.elasticity import linear_elasticity

from skfemntv import LinearElasticTet4, NativeAssembler


def test_matches_scikit_fem():
    mesh = MeshTet.init_tensor(
        np.linspace(0, 1, 3), np.linspace(0, 1, 2), np.linspace(0, 1, 2)
    )
    basis = Basis(mesh, ElementVector(ElementTetP1()))
    young, nu = 71.0, 0.29
    assembler = NativeAssembler.from_skfem(
        basis, LinearElasticTet4(young, nu)
    )
    lam = young * nu / ((1 + nu) * (1 - 2 * nu))
    mu = young / (2 * (1 + nu))
    reference = linear_elasticity(Lambda=lam, Mu=mu).assemble(basis)
    native = assembler.evaluate(np.zeros(basis.N)).tangent
    np.testing.assert_allclose(native.toarray(), reference.toarray(), rtol=1e-12, atol=1e-12)
