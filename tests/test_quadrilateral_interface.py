import numpy as np
import pytest

import skfemntv


def _surface(mesh, element, height):
    facets = mesh.facets_satisfying(
        lambda x: np.isclose(x[2], height), boundaries_only=True
    )
    return skfemntv.FacetBasis(
        mesh, skfemntv.ElementVector(element), facets=facets, intorder=4
    )


def _affine_coefficients(basis, gradient, offset):
    components = np.arange(basis.N) % 3
    return (
        np.einsum("ij,ji->i", gradient[components], basis.doflocs)
        + offset[components]
    )


def _nonmatching_hex8_interface():
    master_mesh = skfemntv.MeshHex.init_tensor([0., 1.], [0., 1.], [0., 1.])
    slave_mesh = skfemntv.MeshHex.init_tensor(
        [0., .5, 1.], [0., .5, 1.], [1., 2.]
    )
    master = _surface(master_mesh, skfemntv.ElementHex1(), 1.)
    slave = _surface(slave_mesh, skfemntv.ElementHex1(), 1.)
    return master, slave


def test_interface_supermesh_formalizes_nonmatching_quad_facets():
    master, slave = _nonmatching_hex8_interface()
    integration = skfemntv.InterfaceSupermesh.from_facets(master, slave)
    assert isinstance(integration, skfemntv.InterfaceSupermesh)
    assert integration.diagnostics.master_search_triangle_count == 2
    assert integration.diagnostics.slave_search_triangle_count == 8
    assert integration.diagnostics.overlap_area == pytest.approx(1.)
    np.testing.assert_allclose(
        integration.assemble() @ np.ones(slave.N),
        skfemntv.NativeLinearForm(master).assemble(value=np.ones(3))[0],
        rtol=4e-13,
        atol=4e-13,
    )


def test_nonmatching_hex8_symmetric_nitsche_preserves_affine_quad_trace():
    master, slave = _nonmatching_hex8_interface()
    integration = skfemntv.InterfaceSupermesh.from_facets(master, slave)
    result = skfemntv.assemble_symmetric_nitsche(
        master,
        slave,
        integration=integration,
        master_lame=(3., 1.1),
        slave_lame=(1.7, .6),
    )
    gradient = np.array([
        [.2, -.04, .07],
        [.03, .1, -.02],
        [-.05, .08, .13],
    ])
    offset = np.array([.4, -.7, 1.2])
    affine = np.concatenate((
        _affine_coefficients(master, gradient, offset),
        _affine_coefficients(slave, gradient, offset),
    ))
    np.testing.assert_allclose(result.penalty @ affine, 0., atol=5e-13)
    np.testing.assert_allclose(
        result.adjoint_consistency @ affine, 0., atol=5e-13
    )
    np.testing.assert_allclose(
        result.matrix.toarray(), result.matrix.toarray().T,
        rtol=4e-14, atol=4e-14,
    )
    diagnostics = result.stabilization
    assert diagnostics is not None
    assert diagnostics.master_characteristic_length.shape == (
        integration.diagnostics.integration_triangle_count,
    )
    assert diagnostics.slave_characteristic_length.shape == (
        integration.diagnostics.integration_triangle_count,
    )


@pytest.mark.parametrize(
    "mesh_factory, element_factory",
    [
        (
            lambda: skfemntv.MeshHex20.from_mesh(skfemntv.MeshHex()),
            skfemntv.ElementHex20,
        ),
        (
            lambda: skfemntv.MeshHex2.from_mesh(skfemntv.MeshHex()),
            skfemntv.ElementHex2,
        ),
    ],
)
def test_high_order_quad_interface_nitsche_preserves_rigid_translation(
    mesh_factory, element_factory,
):
    mesh = mesh_factory()
    basis = _surface(mesh, element_factory(), 0.)
    integration = skfemntv.InterfaceSupermesh.from_facets(basis, basis)
    result = skfemntv.assemble_symmetric_nitsche(
        basis, basis, integration=integration, master_lame=(2.4, .9)
    )
    translation = np.tile(np.array([.3, -1.1, .8]), basis.N // 3)
    common = np.concatenate((translation, translation))
    np.testing.assert_allclose(result.matrix @ common, 0., atol=2e-12)
    assert integration.diagnostics.master_search_triangle_count >= 2


def test_nonmatching_quad_nitsche_threaded_and_serial_match():
    master, slave = _nonmatching_hex8_interface()
    integration = skfemntv.InterfaceSupermesh.from_facets(master, slave)
    kwargs = dict(
        integration=integration,
        master_lame=(2.8, 1.),
        slave_lame=(1.5, .55),
    )
    serial = skfemntv.assemble_symmetric_nitsche(
        master, slave, num_threads=1, **kwargs
    )
    threaded = skfemntv.assemble_symmetric_nitsche(
        master, slave, num_threads=4, **kwargs
    )
    np.testing.assert_allclose(
        threaded.matrix.toarray(), serial.matrix.toarray(),
        rtol=4e-14, atol=4e-14,
    )


def test_nonmatching_quad_master_slave_exchange_preserves_operator():
    master, slave = _nonmatching_hex8_interface()
    forward = skfemntv.assemble_symmetric_nitsche(
        master, slave,
        master_lame=(2.8, 1.), slave_lame=(1.5, .55),
    )
    reverse = skfemntv.assemble_symmetric_nitsche(
        slave, master,
        master_lame=(1.5, .55), slave_lame=(2.8, 1.),
    )
    reverse_order = np.concatenate((
        np.arange(master.N, master.N + slave.N),
        np.arange(master.N),
    ))
    expected = forward.matrix.toarray()[np.ix_(reverse_order, reverse_order)]
    np.testing.assert_allclose(
        reverse.matrix.toarray(), expected, rtol=5e-13, atol=5e-13
    )
