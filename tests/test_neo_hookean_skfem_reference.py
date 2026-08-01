import sys
from pathlib import Path

import numpy as np
import pytest
from skfem import (
    Basis,
    ElementHex1,
    ElementTetP1,
    ElementVector,
    MeshHex,
    MeshTet,
)

sys.path.insert(0, str(Path(__file__).parents[1] / "benchmarks"))
from skfem_neo_hookean import forms

from skfemntv import NativeAssembler, NeoHookean


@pytest.mark.parametrize("topology", ["tet", "hex"])
def test_native_neo_hookean_matches_equivalent_skfem_forms(topology):
    if topology == "tet":
        mesh, element = MeshTet(), ElementTetP1()
    else:
        mesh, element = MeshHex(), ElementHex1()
    basis = Basis(mesh, ElementVector(element), intorder=2)
    mu, lmbda = 3.0, 5.0
    u = np.random.default_rng(1).normal(scale=0.01, size=basis.N)
    field = basis.interpolate(u)
    residual_form, tangent_form = forms(mu, lmbda)
    expected_residual = residual_form.assemble(
        basis, displacement=field
    )
    expected_tangent = tangent_form.assemble(
        basis, displacement=field
    )
    actual = NativeAssembler.from_skfem(
        basis, NeoHookean(mu, lmbda)
    ).evaluate(u)
    np.testing.assert_allclose(
        actual.residual, expected_residual, rtol=2e-12, atol=2e-12
    )
    np.testing.assert_allclose(
        actual.tangent.toarray(),
        expected_tangent.toarray(),
        rtol=2e-12,
        atol=2e-12,
    )
