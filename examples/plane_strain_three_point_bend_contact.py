"""Displacement-controlled sheet bending with rigid frictionless tools.

This is a small-strain plane-strain proof case: a rigid flat punch moves a
sheet onto two rigid supports, then unloads. Contact is node-to-rigid and
normal-only; ``--curved-tools`` uses circular tool cross-sections whose gaps
and normals are updated during Newton iterations. KKTKit handles unilateral
rows while scikit-fem-native evaluates the J2 internal force and tangent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import skfemntv
from scikit_fvm_native.kktkit_bridge import solve_kktkit_contact_step


def run(nx: int = 32, ny: int = 4, thickness: float = 0.1,
        maximum_stroke: float = 0.02,
        maximum_increment: float = 5.0e-4,
        order: int = 1, curved_tools: bool = False,
        numisheet_material: bool = False) -> dict:
    if not np.isfinite(maximum_stroke) or maximum_stroke < 0.0:
        raise ValueError("maximum_stroke must be finite and nonnegative")
    if not np.isfinite(maximum_increment) or maximum_increment <= 0.0:
        raise ValueError("maximum_increment must be finite and positive")
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")
    x_nodes = np.linspace(-0.5, 0.5, nx + 1)
    y_nodes = np.linspace(0.0, thickness, ny + 1)
    mesh = skfemntv.MeshTri.init_tensor(x_nodes, y_nodes)
    scalar_element = skfemntv.ElementTriP1() if order == 1 else skfemntv.ElementTriP2()
    basis = skfemntv.Basis(mesh, skfemntv.ElementVector(scalar_element, dim=2), intorder=4)
    if numisheet_material:
        reference_path = Path(__file__).with_name("numisheet93_draw_bending_reference.json")
        reference = json.loads(reference_path.read_text())
        material = reference["material"]
        ratios = material["hill48_coefficients_R11_R22_R33_R12_R13_R23"]
        hardening_curve = material[
            "hardening_curve_equivalent_plastic_strain_yield_stress_MPa"
        ]
        assembler = skfemntv.PlaneStrainHill48Assembler(
            basis,
            young_modulus=material["young_modulus"],
            poisson_ratio=material["poisson_ratio"],
            yield_stress=material["initial_yield_stress"],
            stress_ratios=ratios,
            hardening_curve=hardening_curve,
        )
        reported_material = {
            "source": "NUMISHEET '93 reference Hill48 and 51-point hardening curve",
            "young_modulus_MPa": material["young_modulus"],
            "poisson_ratio": material["poisson_ratio"],
            "initial_yield_stress_MPa": material["initial_yield_stress"],
            "stress_ratios_R11_R22_R33_R12_R13_R23": ratios,
            "hardening_points": len(hardening_curve),
        }
    else:
        reported_material = {
            "source": "illustrative isotropic J2 parameters",
            "young_modulus": 1000.0,
            "poisson_ratio": 0.3,
            "yield_stress": 2.0,
            "hardening_modulus": 20.0,
        }
        assembler = skfemntv.PlaneStrainJ2Assembler(
            basis,
            young_modulus=1000.0,
            poisson_ratio=0.3,
            yield_stress=2.0,
            hardening_modulus=20.0,
        )
    # Vector H1 DOFs are node-interleaved; using DOF coordinates includes
    # quadratic edge nodes for TriP2 tool and support contact rows.
    node_count = basis.N // 2
    dof_x, dof_y = 2 * np.arange(node_count), 2 * np.arange(node_count) + 1
    node_x = basis.doflocs[0, dof_x]
    node_y = basis.doflocs[1, dof_y]

    top_boundary = np.flatnonzero(np.isclose(node_y, thickness))
    lower_boundary = np.flatnonzero(np.isclose(node_y, 0.0))
    top_nodes = (top_boundary if curved_tools else top_boundary[
        np.abs(node_x[top_boundary]) <= 0.125 + 1e-12
    ])
    support_nodes = (lower_boundary if curved_tools else lower_boundary[
        ((node_x[lower_boundary] >= -0.375 - 1e-12)
         & (node_x[lower_boundary] <= -0.25 + 1e-12))
        | ((node_x[lower_boundary] >= 0.25 - 1e-12)
           & (node_x[lower_boundary] <= 0.375 + 1e-12))
    ])
    pin_node = int(np.flatnonzero(
        np.isclose(node_y, 0.0) & np.isclose(node_x, -0.3125)
    )[0])
    horizontal_anchor_node = int(np.flatnonzero(
        np.isclose(node_y, 0.0) & np.isclose(node_x, 0.0)
    )[0])
    support_nodes = support_nodes[support_nodes != pin_node]

    def circle_profile(center_x: float, center_y: float, radius: float):
        rows = [["START", center_x + radius, center_y]]
        for piece in range(1, 4):
            angle = -2. * np.pi * piece / 3.
            rows.append([
                "CIRCL", center_x + radius * np.cos(angle),
                center_y + radius * np.sin(angle), center_x, center_y,
            ])
        return skfemntv.AnalyticalRigidProfile2D.from_keyword_profile(rows)

    tool_radius = 0.125
    punch_profile = circle_profile(0., thickness + tool_radius, tool_radius)
    support_profiles = {
        -1: circle_profile(-0.3125, -tool_radius, tool_radius),
        1: circle_profile(0.3125, -tool_radius, tool_radius),
    }

    # Two equality rows remove rigid translation. The vertical pin lies on a
    # lower die support; its equality row replaces that node's unilateral row.
    contact_count = top_nodes.size + support_nodes.size
    constraints = np.zeros((2 + contact_count, basis.N))
    constraints[0, dof_x[horizontal_anchor_node]] = 1.0
    constraints[1, dof_y[pin_node]] = 1.0
    row = 2
    punch_rows = np.arange(row, row + top_nodes.size)
    constraints[punch_rows, dof_y[top_nodes]] = -1.0
    row += top_nodes.size
    support_rows = np.arange(row, row + support_nodes.size)
    constraints[support_rows, dof_y[support_nodes]] = 1.0
    unilateral = np.r_[False, False, np.ones(contact_count, dtype=bool)]
    fixed_dofs = np.array([dof_x[horizontal_anchor_node], dof_y[pin_node]])

    displacement = np.zeros(basis.N)
    committed = assembler.initial_state()
    # Small stroke increments keep the contact set and plastic return map
    # inside the local convergence basin.  The increment scales with the
    # requested stroke, so user-selected stroke sizes remain equally resolved.
    increments_per_leg = max(1, int(np.ceil(maximum_stroke / maximum_increment)))
    strokes = np.r_[
        np.linspace(maximum_stroke / increments_per_leg, maximum_stroke,
                    increments_per_leg),
        np.linspace(maximum_stroke * (increments_per_leg - 1)
                    / increments_per_leg, 0.0, increments_per_leg),
    ]
    step_records = []

    def current_gaps(u: np.ndarray, punch_y: float) -> np.ndarray:
        if curved_tools:
            x = node_x + u[dof_x]
            y = node_y + u[dof_y]
            punch = punch_profile.translated([0., float(punch_y - thickness)])
            punch_gap = punch.project(np.column_stack((x[top_nodes], y[top_nodes]))).gaps
            support_gap = np.empty(support_nodes.size)
            for side in (-1, 1):
                mask = node_x[support_nodes] < 0. if side < 0 else node_x[support_nodes] >= 0.
                selected = np.flatnonzero(mask)
                profile = support_profiles[side]
                support_gap[selected] = profile.project(np.column_stack((
                    x[support_nodes[selected]], y[support_nodes[selected]],
                ))).gaps
            return np.r_[punch_gap, support_gap]
        top_gap = punch_y - (node_y[top_nodes] + u[dof_y[top_nodes]])
        support_gap = node_y[support_nodes] + u[dof_y[support_nodes]]
        return np.r_[top_gap, support_gap]

    def current_constraints(u: np.ndarray, punch_y: float) -> np.ndarray:
        if not curved_tools:
            return constraints
        result = constraints.copy()
        x = node_x + u[dof_x]
        y = node_y + u[dof_y]
        punch = punch_profile.translated([0., float(punch_y - thickness)])
        punch_normals = punch.project(np.column_stack((x[top_nodes], y[top_nodes]))).normals
        result[punch_rows, dof_x[top_nodes]] = punch_normals[:, 0]
        result[punch_rows, dof_y[top_nodes]] = punch_normals[:, 1]
        support_normals = np.empty((support_nodes.size, 2))
        for side in (-1, 1):
            mask = node_x[support_nodes] < 0. if side < 0 else node_x[support_nodes] >= 0.
            selected = np.flatnonzero(mask)
            support_normals[selected] = support_profiles[side].project(np.column_stack((
                x[support_nodes[selected]], y[support_nodes[selected]],
            ))).normals
        result[support_rows, dof_x[support_nodes]] = support_normals[:, 0]
        result[support_rows, dof_y[support_nodes]] = support_normals[:, 1]
        return result

    accepted_stroke = 0.0

    def advance_to(target_stroke: float, subdivision_depth: int = 0) -> None:
        nonlocal displacement, committed, accepted_stroke
        punch_y = thickness - target_stroke
        trial_displacement = displacement.copy()
        evaluation = None
        contact_result = None
        try:
            for iteration in range(1, 31):
                evaluation = assembler.evaluate(trial_displacement, committed)
                gaps = current_gaps(trial_displacement, punch_y)
                constraints = current_constraints(trial_displacement, punch_y)
                constraint_rhs = np.r_[-trial_displacement[fixed_dofs], -gaps]
                contact_result = solve_kktkit_contact_step(
                    evaluation.tangent, evaluation.residual, constraints,
                    constraint_rhs, unilateral, tolerance=1e-9,
                    max_iterations=100,
                )
                if not contact_result.converged:
                    raise RuntimeError(
                        f"contact active set failed: {contact_result.message}"
                    )
                candidate = trial_displacement + contact_result.primal
                candidate[fixed_dofs] = 0.0
                candidate_evaluation = assembler.evaluate(candidate, committed)
                candidate_gaps = current_gaps(candidate, punch_y)
                candidate_constraints = current_constraints(candidate, punch_y)
                stationarity = candidate_evaluation.residual + (
                    candidate_constraints.T @ contact_result.multipliers
                )
                stationarity[fixed_dofs] = 0.0
                equilibrium_error = float(np.linalg.norm(stationarity, ord=np.inf))
                complementarity = float(np.max(np.abs(
                    candidate_gaps * contact_result.multipliers[2:]
                ), initial=0.0))
                trial_displacement = candidate
                if (equilibrium_error <= 2e-7
                        and np.min(candidate_gaps, initial=0.0) >= -2e-8
                        and np.max(contact_result.multipliers[2:], initial=0.0) <= 2e-8
                        and complementarity <= 2e-7):
                    evaluation = candidate_evaluation
                    break
            else:
                raise RuntimeError(
                    f"Newton contact equilibrium failed after 30 iterations; "
                    f"equilibrium={equilibrium_error:.3e}, "
                    f"minimum_gap={float(np.min(candidate_gaps)):.3e}, "
                    f"complementarity={complementarity:.3e}"
                )
            if evaluation is None or contact_result is None:
                raise RuntimeError("Newton iteration did not produce a trial state")
        except RuntimeError as error:
            if subdivision_depth >= 8:
                raise RuntimeError(
                    f"contact increment to stroke {target_stroke:.8g} failed "
                    f"after 8 subdivisions: {error}"
                ) from error
            midpoint = 0.5 * (accepted_stroke + target_stroke)
            advance_to(midpoint, subdivision_depth + 1)
            advance_to(target_stroke, subdivision_depth + 1)
            return

        # Only accepted equilibrium states advance either displacement or
        # plastic history. A failed attempt leaves both committed values intact.
        displacement = trial_displacement
        committed = assembler.commit(evaluation)
        accepted_stroke = target_stroke
        if curved_tools:
            punch_force = float(np.sum(
                contact_result.multipliers[punch_rows]
                * current_constraints(displacement, punch_y)[punch_rows, dof_y[top_nodes]]
            ))
        else:
            punch_force = float(-np.sum(contact_result.multipliers[punch_rows]))
        active = contact_result.active_rows
        step_records.append({
            "step": len(step_records) + 1,
            "stroke": float(target_stroke),
            "punch_force": punch_force,
            "newton_iterations": iteration,
            "active_punch_nodes": int(np.count_nonzero(active[punch_rows])),
            "active_support_nodes": int(np.count_nonzero(active[support_rows])),
            "minimum_contact_gap": float(np.min(current_gaps(displacement, punch_y))),
            "equilibrium_residual_inf": equilibrium_error,
            "maximum_equivalent_plastic_strain": float(np.max(committed[..., 6])),
        })

    for stroke in strokes:
        advance_to(float(stroke))

    center_top = int(np.flatnonzero(
        np.isclose(node_y, thickness) & np.isclose(node_x, 0.0)
    )[0])
    center_bottom = int(np.flatnonzero(
        np.isclose(node_y, 0.0) & np.isclose(node_x, 0.0)
    )[0])
    initial_thickness = thickness
    final_thickness = (
        node_y[center_top] + displacement[dof_y[center_top]]
        - node_y[center_bottom] - displacement[dof_y[center_bottom]]
    )
    return {
        "problem": (
            "small-strain plane-strain Hill48 bending with circular rigid frictionless tools"
            if curved_tools and numisheet_material else
            "small-strain plane-strain J2 bending with circular rigid frictionless tools"
            if curved_tools else
            "small-strain plane-strain Hill48 bending with flat frictionless tools"
            if numisheet_material else
            "small-strain plane-strain J2 three-point bending with rigid frictionless flat tools"
        ),
        "units": ("millimetres, newtons, and MPa for the reference material; illustrative geometry"
                  if numisheet_material else
                  "consistent illustrative units; not calibrated SI material data"),
        "geometry": {
            "span_length": 1.0,
            "initial_thickness": float(thickness),
            "punch_contact_x_interval": [-0.125, 0.125],
            "support_contact_x_intervals": [[-0.375, -0.25], [0.25, 0.375]],
            "tool_profile": "circular punch and cylindrical die rolls" if curved_tools else "flat punch and supports",
            "tool_radius": tool_radius if curved_tools else None,
        },
        "mesh": {"nx": nx, "ny": ny, "order": order,
                 "triangles": int(mesh.nelements), "geometry_vertices": int(mesh.p.shape[1]),
                 "field_nodes": int(node_count), "displacement_dofs": int(basis.N)},
        "material": reported_material,
        "maximum_stroke": maximum_stroke,
        "requested_maximum_increment": maximum_increment,
        "accepted_increment_count": len(step_records),
        "steps": step_records,
        "punch_force_by_stroke": [
            [record["stroke"], record["punch_force"]] for record in step_records
        ],
        "final_stroke": float(strokes[-1]),
        "final_punch_force": step_records[-1]["punch_force"],
        "maximum_punch_force": max(record["punch_force"] for record in step_records),
        "centerline_thickness_initial": float(initial_thickness),
        "centerline_thickness_after_unload": float(final_thickness),
        "centerline_thickness_change": float(final_thickness - initial_thickness),
        "centerline_springback_vertical_displacement": float(
            displacement[dof_y[center_top]]
        ),
        "maximum_equilibrium_residual_inf": max(
            record["equilibrium_residual_inf"] for record in step_records
        ),
        "max_equivalent_plastic_strain": float(np.max(committed[..., 6])),
        "material_source": reported_material["source"],
        "scope": ("node-to-analytic-circle contact; small strain; reference material on illustrative geometry"
                  if curved_tools and numisheet_material else
                  "node-to-analytic-circle contact; small strain; illustrative outputs"
                  if curved_tools else
                  "node-to-flat-tool contact; small strain; reference material on illustrative geometry"
                  if numisheet_material else
                  "node-to-flat-tool contact; small strain; illustrative force/thickness/springback outputs"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=32)
    parser.add_argument("--ny", type=int, default=4)
    parser.add_argument("--maximum-stroke", type=float, default=0.02)
    parser.add_argument("--maximum-increment", type=float, default=5.0e-4)
    parser.add_argument("--order", type=int, choices=(1, 2), default=1)
    parser.add_argument("--curved-tools", action="store_true",
                        help="use a circular punch and two cylindrical die rolls")
    parser.add_argument("--numisheet-material", action="store_true",
                        help="use reference Hill48 ratios and the tabular hardening curve")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = run(
        args.nx, args.ny, maximum_stroke=args.maximum_stroke,
        maximum_increment=args.maximum_increment, order=args.order,
        curved_tools=args.curved_tools, numisheet_material=args.numisheet_material,
    )
    rendered = json.dumps(result, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as stream:
            stream.write(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
