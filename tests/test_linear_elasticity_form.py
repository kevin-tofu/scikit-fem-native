import numpy as np
from skfem import Basis as SkfemBasis
from skfem import asm as skfem_asm
from skfem import ElementTetP1 as SkfemElementTetP1
from skfem import ElementVector as SkfemElementVector
from skfem import MeshTet as SkfemMeshTet
from skfem.models.elasticity import linear_elasticity

import skfemntv


def test_isotropic_linear_elasticity_form_matches_scikit_fem():
    material = skfemntv.IsotropicLinearElasticity(210.0, 0.29)
    native_mesh = skfemntv.MeshTet()
    native_basis = skfemntv.Basis(
        native_mesh,
        skfemntv.ElementVector(skfemntv.ElementTetP1(), dim=3),
    )
    native_stiffness = skfemntv.asm(
        skfemntv.isotropic_linear_elasticity_form(material), native_basis
    ).toarray()

    skfem_basis = SkfemBasis(
        SkfemMeshTet(native_mesh.p, native_mesh.t),
        SkfemElementVector(SkfemElementTetP1(), dim=3),
    )
    reference_stiffness = skfem_asm(
        linear_elasticity(material.lame_lambda, material.lame_mu), skfem_basis
    ).toarray()

    np.testing.assert_allclose(
        np.linalg.eigvalsh(native_stiffness),
        np.linalg.eigvalsh(reference_stiffness),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
