import subprocess
import sys

import numpy as np
import pytest
import skfem
from skfem.models.elasticity import linear_elasticity

import skfn


@pytest.mark.parametrize("topology", ["tet", "hex"])
def test_independent_basis_matches_skfem_reference(topology):
    if topology == "tet":
        mesh = skfn.MeshTet.init_tensor(
            np.linspace(0., 1., 3),
            np.linspace(0., 1., 2),
            np.linspace(0., 1., 2),
        )
        element = skfn.ElementTetP1()
        reference_mesh = skfem.MeshTet(mesh.p, mesh.t)
        reference_element = skfem.ElementTetP1()
    else:
        mesh = skfn.MeshHex.init_tensor(
            np.linspace(0., 1., 3),
            np.linspace(0., 1., 2),
            np.linspace(0., 1., 2),
        )
        element = skfn.ElementHex1()
        reference_mesh = skfem.MeshHex(mesh.p, mesh.t)
        reference_element = skfem.ElementHex1()
    basis = skfn.Basis(mesh, skfn.ElementVector(element), intorder=2)
    reference_basis = skfem.Basis(
        reference_mesh,
        skfem.ElementVector(reference_element),
        intorder=2,
    )
    young, poisson = 31., .23
    lmbda = young*poisson/((1+poisson)*(1-2*poisson))
    mu = young/(2*(1+poisson))
    expected = linear_elasticity(Lambda=lmbda, Mu=mu).assemble(
        reference_basis
    )
    actual = skfn.NativeAssembler.from_basis(
        basis, skfn.LinearElasticity(young, poisson)
    ).evaluate(np.zeros(basis.N)).tangent
    np.testing.assert_allclose(
        actual.toarray(), expected.toarray(), rtol=3e-12, atol=3e-12
    )


def test_runtime_import_does_not_import_scikit_fem():
    script = r"""
import sys
class BlockSkfem:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "skfem" or fullname.startswith("skfem."):
            raise RuntimeError("runtime attempted to import scikit-fem")
sys.meta_path.insert(0, BlockSkfem())
import skfn
import numpy as np
mesh = skfn.MeshTet()
basis = skfn.Basis(mesh, skfn.ElementVector(skfn.ElementTetP1()))
assembler = skfn.NativeAssembler.from_basis(
    basis, skfn.LinearElasticity(10.0, 0.2)
)
assert assembler.evaluate(np.zeros(basis.N)).tangent.shape == (12, 12)
"""
    subprocess.run([sys.executable, "-c", script], check=True)
