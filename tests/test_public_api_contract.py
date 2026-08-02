import numpy as np
import pytest
import skfem as reference
from skfem.helpers import dot as reference_dot

import skfemntv
import skfemntv as skfem
from skfemntv.helpers import dot


def test_public_exports_are_unique_and_resolvable():
    assert len(skfemntv.__all__)==len(set(skfemntv.__all__))
    for name in skfemntv.__all__:
        assert hasattr(skfemntv,name),name


def test_import_alias_documented_form_matches_skfem():
    mesh=skfem.MeshTet.init_tensor(
        np.linspace(0.,1.,3),
        np.linspace(0.,1.,2),
        np.linspace(0.,1.,2),
    )
    basis=skfem.Basis(
        mesh,skfem.ElementVector(skfem.ElementTetP1())
    )

    @skfem.BilinearForm
    def mass(u,v,w):
        return (1.+w.x[0])*dot(u,v)

    actual=skfem.asm(mass,basis)

    reference_basis=reference.Basis(
        reference.MeshTet(mesh.p,mesh.t),
        reference.ElementVector(reference.ElementTetP1()),
    )

    @reference.BilinearForm
    def reference_mass(u,v,w):
        return (1.+w.x[0])*reference_dot(u,v)

    expected=reference.asm(reference_mass,reference_basis)
    np.testing.assert_allclose(
        actual.toarray(),expected.toarray(),rtol=5e-13,atol=5e-13
    )


def test_unsupported_form_raises_without_fallback():
    basis=skfem.Basis(
        skfem.MeshTet(),
        skfem.ElementVector(skfem.ElementTetP1()),
    )

    @skfem.BilinearForm
    def unsupported(u,v,w):
        return u

    with pytest.raises(skfem.UnsupportedNativeForm):
        skfem.asm(unsupported,basis)


def test_solver_policy_is_not_exported():
    assert not hasattr(skfemntv,"solve")
    assert not hasattr(skfemntv,"condense")
