import numpy as np
import pytest
import skfem

import skfemntv
import skfemntv.basis as basis_module


def _python_promote_tri_mesh(mesh):
    facets = mesh.facets
    edge_ids = {
        tuple(sorted(map(int, facets[:, facet]))): mesh.p.shape[1] + facet
        for facet in range(facets.shape[1])
    }
    edge_pairs = ((0, 1), (1, 2), (0, 2))
    edge_rows = [[] for _ in edge_pairs]
    for cell in mesh.t.T:
        for row, (first, second) in zip(edge_rows, edge_pairs):
            row.append(edge_ids[tuple(sorted((int(cell[first]), int(cell[second]))))])
    midpoint_coordinates = np.column_stack([
        mesh.p[:, facets[:, facet]].mean(axis=1)
        for facet in range(facets.shape[1])
    ])
    return (
        np.column_stack((mesh.p, midpoint_coordinates)),
        np.vstack((mesh.t, *edge_rows)),
    )


def test_linear_triangle_p2_promotion_preserves_python_numbering():
    mesh = skfemntv.MeshTri.init_tensor(
        np.linspace(0., 1., 5), np.linspace(0., 1., 4)
    )
    expected_points, expected_cells = _python_promote_tri_mesh(mesh)

    basis = skfemntv.Basis(mesh, skfemntv.ElementTriP2(), intorder=4)
    actual = basis._mesh

    np.testing.assert_array_equal(actual.p, expected_points)
    np.testing.assert_array_equal(actual.t, expected_cells)


def test_linear_triangle_p2_promotion_preserves_named_regions():
    mesh = skfemntv.MeshTri.init_tensor(
        np.linspace(0., 1., 4), np.linspace(0., 1., 3)
    ).with_boundaries({
        "bottom": lambda x: np.isclose(x[1], 0.),
    }).with_subdomains({
        "left": lambda x: x[0] < .5,
    })
    basis = skfemntv.Basis(mesh, skfemntv.ElementTriP2(), intorder=4)
    promoted = basis._mesh

    assert set(promoted.boundaries) == {"bottom"}
    assert set(promoted.subdomains) == {"left"}
    boundary = promoted.boundaries["bottom"]
    assert len(boundary) == 3
    boundary_points = promoted.p[:, promoted.facets[:, np.asarray(boundary)]]
    np.testing.assert_allclose(boundary_points[1], 0.)
    assert len(promoted.subdomains["left"]) == len(mesh.subdomains["left"])


def test_native_p2_promotion_builds_skfem_compatible_basis(monkeypatch):
    native_promote = basis_module._promote_tri_p2_mesh_native
    if native_promote is None:
        pytest.skip("native triangle P2 promotion kernel is unavailable")
    calls = 0

    def tracked_promotion(*args):
        nonlocal calls
        calls += 1
        return native_promote(*args)

    monkeypatch.setattr(
        basis_module, "_promote_tri_p2_mesh_native", tracked_promotion
    )
    linear = skfemntv.MeshTri.init_tensor([0., .5, 1.], [0., .5, 1.])
    native = skfemntv.Basis(linear, skfemntv.ElementTriP2(), intorder=4)
    reference_mesh = skfem.MeshTri(linear.p, linear.t)
    reference = skfem.Basis(
        reference_mesh, skfem.ElementTriP2(), intorder=4
    )

    assert calls == 1
    np.testing.assert_array_equal(native.doflocs, reference.doflocs)
    np.testing.assert_array_equal(native.element_dofs, reference.element_dofs)
