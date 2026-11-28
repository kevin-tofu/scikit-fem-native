from pathlib import Path
import runpy

import numpy as np

import skfemntv
from skfemntv.helpers import dot, grad, jump
from skfemntv.xfem import ShiftedHeavisideCutBasis


def test_shifted_heaviside_xfem_represents_piecewise_constant_jump():
    axis = np.linspace(0.0, 1.0, 9)
    mesh = skfemntv.MeshTri.init_tensor(axis, axis)
    parent = skfemntv.Basis(
        mesh,
        skfemntv.ElementVector(skfemntv.ElementTriP1(), dim=1),
        intorder=2,
    )
    level_set = skfemntv.LevelSet(
        lambda x: (x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2 - 0.24**2
    )
    classification = level_set.classify(mesh)
    assert len(classification.cut) > 0
    inside = ShiftedHeavisideCutBasis(
        parent, level_set, side="inside", classification=classification
    )
    outside = ShiftedHeavisideCutBasis(
        parent, level_set, side="outside", classification=classification
    )

    nodal_sign = np.where(level_set.values(mesh) < 0.0, -1.0, 1.0)
    coefficients = np.concatenate((
        1.5 + 0.5 * nodal_sign,
        np.full(len(inside.enriched_nodes), 0.5),
    ))
    inside_values = inside.interpolate(coefficients)
    outside_values = outside.interpolate(coefficients)

    np.testing.assert_allclose(inside_values.value, 1.0, atol=2e-14)
    np.testing.assert_allclose(outside_values.value, 2.0, atol=2e-14)
    np.testing.assert_allclose(inside_values.grad, 0.0, atol=2e-14)
    np.testing.assert_allclose(outside_values.grad, 0.0, atol=2e-14)


def test_xfem_cut_basis_assembles_native_reaction_diffusion_blocks():
    axis = np.linspace(0.0, 1.0, 7)
    mesh = skfemntv.MeshTri.init_tensor(axis, axis)
    parent = skfemntv.Basis(
        mesh,
        skfemntv.ElementVector(skfemntv.ElementTriP1(), dim=1),
        intorder=2,
    )
    level_set = skfemntv.LevelSet(lambda x: x[0] - 0.47)
    classification = level_set.classify(mesh)
    inside = ShiftedHeavisideCutBasis(
        parent, level_set, side="inside", classification=classification
    )
    outside = ShiftedHeavisideCutBasis(
        parent, level_set, side="outside", classification=classification
    )

    @skfemntv.BilinearForm
    def reaction_diffusion(u, v, w):
        return dot(grad(u), grad(v)) + u * v

    matrix = skfemntv.asm(reaction_diffusion, inside)
    matrix += skfemntv.asm(reaction_diffusion, outside)
    assert matrix.shape == (inside.N, inside.N)
    assert matrix.nnz > inside.N
    np.testing.assert_allclose(matrix.toarray(), matrix.toarray().T, atol=2e-13)


def test_shifted_heaviside_xfem_l2_projection_converges_under_refinement():
    example = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "xfem_shifted_heaviside.py"
    )
    projection_error = runpy.run_path(str(example))["l2_projection_error"]
    coarse_error = projection_error(8)
    fine_error = projection_error(16)
    assert fine_error < coarse_error / 2.5


def test_shifted_heaviside_interface_trace_preserves_side_jump_and_assembles():
    axis = np.linspace(0.0, 1.0, 9)
    mesh = skfemntv.MeshTri.init_tensor(axis, axis)
    parent = skfemntv.Basis(
        mesh,
        skfemntv.ElementVector(skfemntv.ElementTriP1(), dim=1),
        intorder=2,
    )
    level_set = skfemntv.LevelSet(
        lambda x: (x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2 - 0.24**2
    )
    classification = level_set.classify(mesh)
    inside = ShiftedHeavisideCutBasis(
        parent, level_set, side="inside", classification=classification
    )
    outside = ShiftedHeavisideCutBasis(
        parent, level_set, side="outside", classification=classification
    )
    interface_rule = level_set.interface_quadrature(mesh, intorder=3)
    inside_trace = inside.interface_trace(interface_rule)
    outside_trace = outside.interface_trace(interface_rule)
    interface = skfemntv.ShiftedHeavisideInterfacePair(
        inside_trace, outside_trace
    )

    nodal_sign = np.where(level_set.values(mesh) < 0.0, -1.0, 1.0)
    coefficients = np.concatenate((
        1.5 + 0.5 * nodal_sign,
        np.full(len(inside.enriched_nodes), 0.5),
    ))
    np.testing.assert_allclose(
        inside_trace.interpolate(coefficients).value, 1.0, atol=2e-14
    )
    np.testing.assert_allclose(
        outside_trace.interpolate(coefficients).value, 2.0, atol=2e-14
    )

    @skfemntv.BilinearForm
    def jump_penalty(u, v, w):
        return 2.0 * dot(jump(u), jump(v))

    matrix = skfemntv.asm(
        jump_penalty,
        inside_trace,
        outside_trace,
        integration=interface,
    )
    assert matrix.shape == (inside.N, inside.N)
    np.testing.assert_allclose(matrix.toarray(), matrix.toarray().T, atol=2e-13)
    continuous_coefficients = np.concatenate((
        np.ones(parent.N), np.zeros(len(inside.enriched_nodes))
    ))
    np.testing.assert_allclose(
        matrix @ continuous_coefficients, 0.0, atol=2e-13
    )
    assert coefficients @ matrix @ coefficients > 0.0
