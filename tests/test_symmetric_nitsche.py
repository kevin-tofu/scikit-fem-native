import numpy as np
import pytest

import skfemntv
from skfemntv.helpers import isotropic_traction_tensor


def interface():
    master_mesh = skfemntv.MeshTet2.from_mesh(skfemntv.MeshTet())
    slave_mesh = skfemntv.MeshTet()
    master = skfemntv.FacetBasis(
        master_mesh,
        skfemntv.ElementVector(skfemntv.ElementTetP2()),
        intorder=4,
    )
    slave = skfemntv.FacetBasis(
        slave_mesh,
        skfemntv.ElementVector(skfemntv.ElementTetP1()),
        intorder=4,
    )
    integration = skfemntv.TriangleSupermesh.from_facets(master, slave)
    return master, slave, integration


def test_symmetric_nitsche_exposes_each_term_and_is_symmetric():
    master, slave, integration = interface()
    result = skfemntv.assemble_symmetric_nitsche(
        master,
        slave,
        integration=integration,
        penalty=18.,
        master_lame=(2.3, .8),
        slave_lame=(1.7, .6),
        average_weights=(.4, .6),
    )
    np.testing.assert_allclose(
        result.adjoint_consistency.toarray(),
        result.consistency.toarray().T,
        rtol=3e-14,
        atol=3e-14,
    )
    np.testing.assert_allclose(
        result.matrix.toarray(),
        result.matrix.toarray().T,
        rtol=3e-14,
        atol=3e-14,
    )
    np.testing.assert_allclose(
        result.matrix.toarray(),
        (
            result.penalty
            - result.consistency
            - result.adjoint_consistency
        ).toarray(),
    )


def test_symmetric_nitsche_matches_explicit_two_material_trace_assembly():
    master, slave, integration = interface()
    penalty = 11.
    master_lame = (3.1, 1.2)
    slave_lame = (1.4, .55)
    weights = (.25, .75)
    result = skfemntv.assemble_symmetric_nitsche(
        master,
        slave,
        integration=integration,
        penalty=penalty,
        master_lame=master_lame,
        slave_lame=slave_lame,
        average_weights=weights,
    )
    normal = np.moveaxis(integration.master_normals, -1, 0)
    expected_penalty = integration.assemble_traces(
        (1., -1.), (1., -1.), coefficient=penalty
    )
    expected_consistency = (
        integration.assemble_traces(
            (1., -1.), (weights[0], 0.),
            row_kind="value", column_kind="gradient",
            coefficient=isotropic_traction_tensor(normal, *master_lame),
        )
        + integration.assemble_traces(
            (1., -1.), (0., weights[1]),
            row_kind="value", column_kind="gradient",
            coefficient=isotropic_traction_tensor(normal, *slave_lame),
        )
    )
    expected = expected_penalty - expected_consistency - expected_consistency.T
    np.testing.assert_allclose(
        result.matrix.toarray(), expected.toarray(), rtol=3e-14, atol=3e-14
    )


def test_symmetric_nitsche_preserves_common_rigid_translation():
    master, slave, integration = interface()
    result = skfemntv.assemble_symmetric_nitsche(
        master,
        slave,
        integration=integration,
        penalty=20.,
        master_lame=(2.3, .8),
    )
    translation = np.tile(np.array([.3, -1.1, 2.4]), (master.N + slave.N) // 3)
    np.testing.assert_allclose(result.matrix @ translation, 0., atol=3e-13)


def test_symmetric_nitsche_affine_patch_has_only_consistency_flux():
    master, slave, integration = interface()
    result = skfemntv.assemble_symmetric_nitsche(
        master, slave, integration=integration, master_lame=(2.3, .8)
    )
    gradient = np.array([
        [.2, -.1, .05],
        [.03, .15, -.04],
        [-.02, .07, .11],
    ])
    offset = np.array([.4, -1.2, .8])
    master_components = np.arange(master.N) % 3
    slave_components = np.arange(slave.N) % 3
    master_values = np.einsum(
        "ij,ji->i", gradient[master_components], master.doflocs
    ) + offset[master_components]
    slave_values = np.einsum(
        "ij,ji->i", gradient[slave_components], slave.doflocs
    ) + offset[slave_components]
    affine = np.concatenate((master_values, slave_values))
    np.testing.assert_allclose(result.penalty @ affine, 0., atol=3e-13)
    np.testing.assert_allclose(
        result.adjoint_consistency @ affine, 0., atol=3e-13
    )
    np.testing.assert_allclose(
        result.matrix @ affine,
        -(result.consistency @ affine),
        atol=4e-13,
    )


def test_symmetric_nitsche_threaded_and_serial_assembly_match():
    master, slave, integration = interface()
    kwargs = dict(
        integration=integration,
        penalty=16.,
        master_lame=(2.3, .8),
        slave_lame=(1.9, .7),
    )
    serial = skfemntv.assemble_symmetric_nitsche(
        master, slave, num_threads=1, **kwargs
    )
    threaded = skfemntv.assemble_symmetric_nitsche(
        master, slave, num_threads=4, **kwargs
    )
    np.testing.assert_allclose(
        threaded.matrix.toarray(), serial.matrix.toarray(),
        rtol=3e-14, atol=3e-14,
    )


def test_automatic_stabilization_uses_facet_scale_and_material_weights():
    master, slave, integration = interface()
    factor = 12.
    master_lame = (3., 1.)
    slave_lame = (1., .5)
    result = skfemntv.assemble_symmetric_nitsche(
        master,
        slave,
        integration=integration,
        master_lame=master_lame,
        slave_lame=slave_lame,
        stabilization_factor=factor,
    )
    diagnostics = result.stabilization
    assert diagnostics is not None
    master_parent = integration.master_trace.parent_facets
    slave_parent = integration.slave_trace.parent_facets
    master_h = np.sqrt(master.dx.sum(axis=1)[master_parent])
    slave_h = np.sqrt(slave.dx.sum(axis=1)[slave_parent])
    master_scale = (master_lame[0] + 2. * master_lame[1]) / master_h
    slave_scale = (slave_lame[0] + 2. * slave_lame[1]) / slave_h
    expected_master_weight = slave_scale / (master_scale + slave_scale)
    expected_penalty = (
        factor * 2. * master_scale * slave_scale
        / (master_scale + slave_scale)
    )
    np.testing.assert_allclose(
        diagnostics.master_characteristic_length, master_h
    )
    np.testing.assert_allclose(
        diagnostics.master_average_weight[:, 0], expected_master_weight
    )
    np.testing.assert_allclose(diagnostics.penalty[:, 0], expected_penalty)
    assert diagnostics.minimum_penalty == pytest.approx(expected_penalty.min())
    assert diagnostics.maximum_penalty == pytest.approx(expected_penalty.max())
    with pytest.raises(ValueError):
        diagnostics.penalty[0, 0] = 0.


def test_automatic_penalty_scales_linearly_with_stabilization_factor():
    master, slave, integration = interface()
    low = skfemntv.assemble_symmetric_nitsche(
        master, slave, integration=integration,
        master_lame=(2.3, .8), stabilization_factor=4.,
    )
    high = skfemntv.assemble_symmetric_nitsche(
        master, slave, integration=integration,
        master_lame=(2.3, .8), stabilization_factor=20.,
    )
    np.testing.assert_allclose(
        high.stabilization.penalty, 5. * low.stabilization.penalty
    )
    np.testing.assert_allclose(
        high.penalty.toarray(), 5. * low.penalty.toarray(), atol=5e-15
    )


def test_explicit_average_weights_override_automatic_material_weights():
    master, slave, integration = interface()
    result = skfemntv.assemble_symmetric_nitsche(
        master, slave, integration=integration,
        master_lame=(4., 1.5), slave_lame=(1., .4),
        average_weights=(.3, .7),
    )
    np.testing.assert_allclose(result.stabilization.master_average_weight, .3)
    np.testing.assert_allclose(result.stabilization.slave_average_weight, .7)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"penalty": 0.}, "penalty"),
        ({"master_lame": (1., 0.)}, "master_lame"),
        ({"average_weights": (.2, .2)}, "sum to one"),
        ({"penalty": None, "stabilization_factor": 0.}, "stabilization_factor"),
    ],
)
def test_symmetric_nitsche_rejects_invalid_parameters(kwargs, message):
    master, slave, integration = interface()
    options = dict(
        integration=integration,
        penalty=10.,
        master_lame=(2., .7),
    )
    options.update(kwargs)
    with pytest.raises(ValueError, match=message):
        skfemntv.assemble_symmetric_nitsche(master, slave, **options)
