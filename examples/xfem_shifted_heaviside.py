"""Solve a reaction-diffusion problem with a shifted-Heaviside XFEM field.

Run from the repository root with::

    PYTHONPATH=packages/skfem-native/python python \
        packages/skfem-native/examples/xfem_shifted_heaviside.py

The circular interface carries a unit jump: ``u=1`` inside and ``u=2``
outside.  For ``-Delta(u) + u = f`` with homogeneous natural boundary data,
``f`` equals the piecewise constant solution.  This is a compact check of
enrichment, cut-volume assembly, and solve; it is not a crack-tip XFEM model.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

import skfemntv
from skfemntv.helpers import dot, grad
from skfemntv.linear_form import NativeLinearForm
from skfemntv.xfem import ShiftedHeavisideCutBasis


def l2_projection_error(resolution: int) -> float:
    """L2 projection error for a piecewise-quadratic field with a jump."""
    axis = np.linspace(0.0, 1.0, resolution + 1)
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
        parent, level_set, side="inside", intorder=4,
        classification=classification,
    )
    outside = ShiftedHeavisideCutBasis(
        parent, level_set, side="outside", intorder=4,
        classification=classification,
    )

    @skfemntv.BilinearForm
    def mass(u, v, w):
        return u * v

    matrix = skfemntv.asm(mass, inside) + skfemntv.asm(mass, outside)
    inside_exact = (
        1.0 + inside.points[:, 0] ** 2 + 0.5 * inside.points[:, 1] ** 2
    )
    outside_exact = 2.0 * (
        1.0 + outside.points[:, 0] ** 2 + 0.5 * outside.points[:, 1] ** 2
    )
    inside_load, _ = NativeLinearForm(inside).assemble(
        value=np.asarray(inside_exact[:, None, None], dtype=np.float64)
    )
    outside_load, _ = NativeLinearForm(outside).assemble(
        value=np.asarray(outside_exact[:, None, None], dtype=np.float64)
    )
    coefficients = spsolve(matrix.tocsr(), inside_load + outside_load)
    inside_error = inside.interpolate(coefficients).value - inside_exact
    outside_error = outside.interpolate(coefficients).value - outside_exact
    return float(np.sqrt(
        inside.integrate(inside_error**2)
        + outside.integrate(outside_error**2)
    ))


def main() -> None:
    resolution = 24
    axis = np.linspace(0.0, 1.0, resolution + 1)
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

    @skfemntv.BilinearForm
    def reaction_diffusion(u, v, w):
        return dot(grad(u), grad(v)) + u * v

    matrix = csr_matrix(skfemntv.asm(reaction_diffusion, inside))
    matrix += skfemntv.asm(reaction_diffusion, outside)
    inside_load, _ = NativeLinearForm(inside).assemble(value=np.array([1.0]))
    outside_load, _ = NativeLinearForm(outside).assemble(value=np.array([2.0]))
    solution = spsolve(matrix.tocsr(), inside_load + outside_load)

    inside_field = inside.interpolate(solution)
    outside_field = outside.interpolate(solution)
    error_squared = (
        inside.integrate((inside_field.value - 1.0) ** 2)
        + outside.integrate((outside_field.value - 2.0) ** 2)
    )
    gradient_squared = (
        inside.integrate(np.sum(inside_field.grad**2, axis=0))
        + outside.integrate(np.sum(outside_field.grad**2, axis=0))
    )
    print("Shifted-Heaviside XFEM / reaction-diffusion jump benchmark")
    print(f"background cells: {mesh.nelements}")
    print(f"cut cells: {len(classification.cut)}")
    print(f"enriched nodes: {len(inside.enriched_nodes)}")
    print(f"global DOFs: {inside.N}")
    print(f"piecewise-constant solution L2 error: {np.sqrt(error_squared):.6e}")
    print(f"piecewise-constant solution gradient L2 norm: {np.sqrt(gradient_squared):.6e}")
    print("piecewise-quadratic jump-field L2 projection error:")
    for projection_resolution in (8, 16, 32):
        error = l2_projection_error(projection_resolution)
        print(f"  resolution={projection_resolution:2d}: {error:.6e}")


if __name__ == "__main__":
    main()
