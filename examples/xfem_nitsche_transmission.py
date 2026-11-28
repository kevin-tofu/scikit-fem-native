"""Solve a two-material transmission problem with shifted-Heaviside XFEM.

Run from the repository root with::

    PYTHONPATH=packages/skfem-native/python python \
        packages/skfem-native/examples/xfem_nitsche_transmission.py

The exact solution has a prescribed nonzero jump and continuous normal flux
across a circular material interface.  The two one-sided XFEM traces share
one enriched coefficient vector and are coupled by symmetric Nitsche terms.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import spsolve

import skfemntv
from skfemntv.helpers import dot, grad
from skfemntv.linear_form import NativeLinearForm


def solve(resolution: int) -> tuple[float, float, float]:
    """Return piecewise L2 error, interface-jump error, and asymmetry."""
    axis = np.linspace(0.0, 1.0, resolution + 1)
    mesh = skfemntv.MeshTri.init_tensor(axis, axis)
    parent = skfemntv.Basis(
        mesh,
        skfemntv.ElementVector(skfemntv.ElementTriP1(), dim=1),
        intorder=2,
    )
    radius = 0.31
    level_set = skfemntv.LevelSet(
        lambda x: (x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2 - radius**2
    )
    classification = level_set.classify(mesh)
    inside = skfemntv.ShiftedHeavisideCutBasis(
        parent, level_set, side="inside", classification=classification
    )
    outside = skfemntv.ShiftedHeavisideCutBasis(
        parent, level_set, side="outside", classification=classification
    )

    conductivity_inside = 1.0
    conductivity_outside = 4.0
    radial_inside = 1.0
    radial_outside = (
        conductivity_inside * radial_inside / conductivity_outside
    )
    prescribed_jump = 0.08
    offset_inside = (
        radial_outside - radial_inside
    ) * radius**2 + prescribed_jump
    source = -4.0 * conductivity_inside * radial_inside

    @skfemntv.BilinearForm
    def diffusion(u, v, w):
        return w.conductivity * dot(grad(u), grad(v))

    matrix = skfemntv.asm(
        diffusion, inside, conductivity=conductivity_inside
    )
    matrix += skfemntv.asm(
        diffusion, outside, conductivity=conductivity_outside
    )
    inside_load, _ = NativeLinearForm(inside).assemble(
        value=np.array([source], dtype=np.float64)
    )
    outside_load, _ = NativeLinearForm(outside).assemble(
        value=np.array([source], dtype=np.float64)
    )
    rhs = np.zeros(inside.N)
    rhs[: len(inside_load)] += inside_load
    rhs[: len(outside_load)] += outside_load

    rule = level_set.interface_quadrature(mesh, intorder=4)
    inside_trace = inside.interface_trace(rule)
    outside_trace = outside.interface_trace(rule)
    interface = skfemntv.ShiftedHeavisideInterfacePair(
        inside_trace, outside_trace
    )
    flux_weights = (
        0.5 * conductivity_inside,
        -0.5 * conductivity_outside,
    )
    penalty = 30.0 * conductivity_outside * resolution
    matrix += interface.assemble_traces(
        (1.0, -1.0),
        flux_weights,
        row_kind="value",
        column_kind="normal_gradient",
        coefficient=-1.0,
    )
    matrix += interface.assemble_traces(
        flux_weights,
        (1.0, -1.0),
        row_kind="normal_gradient",
        column_kind="value",
        coefficient=-1.0,
    )
    matrix += interface.assemble_traces(
        (1.0, -1.0),
        (1.0, -1.0),
        coefficient=penalty,
    )
    rhs += interface.assemble_linear_trace(
        flux_weights,
        trace_kind="normal_gradient",
        coefficient=-prescribed_jump,
    )
    rhs += interface.assemble_linear_trace(
        (1.0, -1.0),
        coefficient=penalty * prescribed_jump,
    )
    asymmetry = float(np.max(np.abs((matrix - matrix.T).data), initial=0.0))

    boundary_nodes = mesh.boundary_nodes()
    fixed = parent.get_dofs(nodes=boundary_nodes).all()
    boundary_coordinates = mesh.p[:, boundary_nodes]
    boundary_values = radial_outside * np.sum(
        (boundary_coordinates - 0.5) ** 2, axis=0
    )
    fixed_values = np.zeros(inside.N)
    fixed_values[fixed] = boundary_values
    active = np.flatnonzero(matrix.diagonal() > 0.0)
    free = np.setdiff1d(active, fixed)
    solution = fixed_values.copy()
    reduced_rhs = rhs[free] - matrix[free][:, fixed] @ solution[fixed]
    solution[free] = spsolve(matrix[free][:, free], reduced_rhs)

    inside_radius_squared = np.sum((inside.points - 0.5) ** 2, axis=1)
    outside_radius_squared = np.sum((outside.points - 0.5) ** 2, axis=1)
    inside_exact = radial_inside * inside_radius_squared + offset_inside
    outside_exact = radial_outside * outside_radius_squared
    inside_error = inside.interpolate(solution).value - inside_exact
    outside_error = outside.interpolate(solution).value - outside_exact
    error = np.sqrt(
        inside.integrate(inside_error**2)
        + outside.integrate(outside_error**2)
    )
    trace_jump = (
        inside_trace.interpolate(solution).value
        - outside_trace.interpolate(solution).value
    )
    jump_error = float(np.sqrt(
        rule.weights @ (trace_jump - prescribed_jump) ** 2
    ))
    return float(error), jump_error, asymmetry


def main() -> None:
    print("Shifted-Heaviside XFEM / two-material Nitsche transmission")
    print("resolution  piecewise_L2_error  interface_jump_L2_error  matrix_asymmetry")
    for resolution in (8, 16, 32):
        error, jump_error, asymmetry = solve(resolution)
        print(
            f"{resolution:10d}  {error:18.6e}  {jump_error:23.6e}"
            f"  {asymmetry:16.6e}"
        )


if __name__ == "__main__":
    main()
