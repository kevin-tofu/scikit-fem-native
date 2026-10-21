import numpy as np
import pytest
import skfem
from skfem.helpers import div as reference_div

import skfemntv
from skfemntv.helpers import div


@pytest.mark.parametrize("family", ["tri", "tet"])
def test_rt0_dg0_divergence_blocks_match_scikit_fem(family):
    if family == "tri":
        mesh = skfemntv.MeshTri.init_tensor([0., 0.5, 1.], [0., 0.5, 1.])
        reference_mesh = skfem.MeshTri(mesh.p, mesh.t)
        rt_element = skfemntv.ElementTriRT0()
        reference_rt_element = skfem.ElementTriRT0()
        p0_element = skfemntv.ElementTriP0()
        reference_p0_element = skfem.ElementTriP0()
    else:
        mesh = skfemntv.MeshTet.init_tensor(
            [0., 1.], [0., 1.], [0., 1.]
        )
        reference_mesh = skfem.MeshTet(mesh.p, mesh.t)
        rt_element = skfemntv.ElementTetRT0()
        reference_rt_element = skfem.ElementTetRT0()
        p0_element = skfemntv.ElementTetP0()
        reference_p0_element = skfem.ElementTetP0()

    rt0 = skfemntv.Basis(mesh, rt_element, intorder=5)
    dg0 = skfemntv.Basis(
        mesh, skfemntv.ElementDG(p0_element), quadrature=rt0.quadrature
    )
    reference_rt0 = skfem.Basis(
        reference_mesh, reference_rt_element, quadrature=rt0.quadrature
    )
    reference_dg0 = skfem.Basis(
        reference_mesh, skfem.ElementDG(reference_p0_element),
        quadrature=rt0.quadrature,
    )
    assert dg0.N == reference_dg0.N
    assert rt0.N == reference_rt0.N

    @skfemntv.BilinearForm
    def rt0_trial_pressure_test(q, p, w):
        return (0.7 + 0.3 * w.x[-1]) * div(q) * p

    @skfem.BilinearForm
    def reference_rt0_trial_pressure_test(q, p, w):
        return (0.7 + 0.3 * w.x[-1]) * reference_div(q) * p

    actual = skfemntv.asm(rt0_trial_pressure_test, rt0, dg0)
    expected = skfem.asm(
        reference_rt0_trial_pressure_test, reference_rt0, reference_dg0
    )
    assert actual.shape == (dg0.N, rt0.N)
    np.testing.assert_allclose(
        actual.toarray(), expected.toarray(), rtol=3e-13, atol=3e-13
    )

    @skfemntv.BilinearForm
    def pressure_trial_rt0_test(p, q, w):
        return p * div(q) * (0.7 + 0.3 * w.x[-1])

    @skfem.BilinearForm
    def reference_pressure_trial_rt0_test(p, q, w):
        return p * reference_div(q) * (0.7 + 0.3 * w.x[-1])

    actual_transpose = skfemntv.asm(pressure_trial_rt0_test, dg0, rt0)
    expected_transpose = skfem.asm(
        reference_pressure_trial_rt0_test, reference_dg0, reference_rt0
    )
    assert actual_transpose.shape == (rt0.N, dg0.N)
    np.testing.assert_allclose(
        actual_transpose.toarray(), expected_transpose.toarray(),
        rtol=3e-13, atol=3e-13,
    )
