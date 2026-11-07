import numpy as np
import pytest
import skfem
from skfem.helpers import dot

import skfemntv


@pytest.mark.parametrize(
    ("native_mesh", "native_element", "reference_element"),
    [
        (skfemntv.MeshTet(), skfemntv.ElementTetP1(), skfem.ElementTetP1()),
        (skfemntv.MeshTet2(), skfemntv.ElementTetP2(), skfem.ElementTetP2()),
        (skfemntv.MeshHex(), skfemntv.ElementHex1(), skfem.ElementHex1()),
        (skfemntv.MeshHex2(), skfemntv.ElementHex2(), skfem.ElementHex2()),
    ],
)
def test_independent_surface_traction_matches_skfem(
    native_mesh, native_element, reference_element
):
    native = skfemntv.FacetBasis(
        native_mesh, skfemntv.ElementVector(native_element), intorder=4
    )
    if isinstance(native_mesh, skfemntv.MeshTet2):
        reference_mesh = skfem.MeshTet2.from_mesh(
            skfem.MeshTet(native_mesh.p[:, :4], native_mesh.t[:4])
        )
    elif isinstance(native_mesh, skfemntv.MeshHex2):
        corner_local = np.array([0, 2, 6, 18, 8, 20, 24, 26])
        reference_mesh = skfem.MeshHex2.from_mesh(
            skfem.MeshHex(
                native_mesh.p[:, native_mesh.t[corner_local, 0]],
                np.arange(8)[:, None],
            )
        )
    elif isinstance(native_mesh, skfemntv.MeshTet):
        reference_mesh = skfem.MeshTet(native_mesh.p, native_mesh.t)
    else:
        reference_mesh = skfem.MeshHex(native_mesh.p, native_mesh.t)
    reference = skfem.FacetBasis(
        reference_mesh,
        skfem.ElementVector(reference_element),
        facets=reference_mesh.boundary_facets(),
        intorder=4,
    )
    traction = np.array([.3, -1.2, 2.4])
    actual, _ = skfemntv.NativeLinearForm(native).assemble(value=traction)

    @skfem.LinearForm
    def load(v, w):
        return dot(w.traction, v)

    expected = load.assemble(
        reference, traction=traction[:, None, None]
    )
    lookup = {
        tuple(np.round(reference.doflocs[:, 3*node], 14)): node
        for node in range(reference.N // 3)
    }
    order = []
    for node in range(native.N // 3):
        other = lookup[tuple(np.round(native.mesh.p[:, node], 14))]
        order.extend(3*other + component for component in range(3))
    np.testing.assert_allclose(
        actual, expected[np.asarray(order)], rtol=8e-12, atol=8e-12
    )


def test_curved_quadratic_facet_uses_quadrature_point_jacobian():
    mesh = skfemntv.MeshTet2()
    flat_area = skfemntv.FacetBasis(
        mesh, skfemntv.ElementVector(skfemntv.ElementTetP2()), intorder=4
    ).dx.sum()
    curved_points = mesh.p.copy()
    curved_points[:, 4] += np.array([0., 0., .2])
    curved = skfemntv.MeshTet2(curved_points, mesh.t)
    curved_area = skfemntv.FacetBasis(
        curved, skfemntv.ElementVector(skfemntv.ElementTetP2()), intorder=4
    ).dx.sum()
    assert not np.isclose(curved_area, flat_area)
