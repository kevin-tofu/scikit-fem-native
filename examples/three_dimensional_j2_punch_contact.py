"""Small 3D Tet4 J2/Hill48 sheet with frictionless rigid-plane punch contact.

This is a 3D contact-kernel proof case, not a calibrated forming benchmark.
Pass ``--hill48`` to use the six-direction anisotropic yield surface.
Pass ``--numisheet-material`` to read its ratios and tabular hardening from the
checked-in NUMISHEET '93 reference data.
The punch and two supports are represented by nodal normal constraints; the
example runs displacement-controlled loading and unloading and commits plastic
history only after each converged equilibrium state.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import skfemntv
from scikit_fvm_native.kktkit_bridge import solve_kktkit_contact_step


def run(nx: int = 8, ny: int = 4, nz: int = 1,
        maximum_stroke: float = 0.001, hill48: bool = False,
        numisheet_material: bool = False,
        facet_contact: bool = False) -> dict:
    x = np.linspace(-0.5, 0.5, nx + 1)
    y = np.linspace(-0.1, 0.1, ny + 1)
    z = np.linspace(0.0, 0.1, nz + 1)
    mesh = skfemntv.MeshTet.init_tensor(x, y, z)
    basis = skfemntv.Basis(
        mesh, skfemntv.ElementVector(skfemntv.ElementTetP1(), dim=3), intorder=2
    )
    material = {
        "young_modulus": 1000.0, "poisson_ratio": 0.3,
        "yield_stress": 2.0, "hardening_modulus": 20.0,
    }
    if numisheet_material:
        reference_path = Path(__file__).with_name("numisheet93_draw_bending_reference.json")
        reference = json.loads(reference_path.read_text())
        material = {
            "young_modulus": reference["material"]["young_modulus"],
            "poisson_ratio": reference["material"]["poisson_ratio"],
            "yield_stress": reference["material"]["initial_yield_stress"],
            "stress_ratios": tuple(
                reference["material"]["hill48_coefficients_R11_R22_R33_R12_R13_R23"]
            ),
            "hardening_curve": reference["material"][
                "hardening_curve_equivalent_plastic_strain_yield_stress_MPa"
            ],
        }
        hill48 = True
    elif hill48:
        material["stress_ratios"] = (1.0, 1.0402, 1.24897, 1.07895, 1.0, 1.0)
    if hill48:
        assembler = skfemntv.Hill48PlasticityAssembler(basis, **material)
    else:
        assembler = skfemntv.J2PlasticityAssembler(basis, **material)

    node_count = mesh.p.shape[1]
    dof = 3 * np.arange(node_count)[:, None] + np.arange(3)[None, :]
    px, py, pz = mesh.p
    top = np.flatnonzero(
        np.isclose(pz, 0.1) & (np.abs(px) <= 0.125 + 1e-12)
        & (np.abs(py) <= 0.05 + 1e-12)
    )
    bottom = np.isclose(pz, 0.0)
    supports = np.flatnonzero(
        bottom & (np.isclose(px, -0.25) | np.isclose(px, 0.25))
    )
    if facet_contact:
        boundary_facets = mesh.boundary_facets()
        facet_nodes = mesh.facets[:3, boundary_facets]
        facet_centers = mesh.p[:, facet_nodes].mean(axis=1)
        on_top = np.all(np.isclose(pz[facet_nodes], 0.1), axis=0)
        on_bottom = np.all(np.isclose(pz[facet_nodes], 0.0), axis=0)
        top_facets = facet_nodes[:, on_top & (np.abs(facet_centers[0]) <= 0.125 + 1e-12)
                                 & (np.abs(facet_centers[1]) <= 0.05 + 1e-12)]
        # Approximate each support cylinder by a narrow band of bottom facets.
        support_band = np.minimum(np.abs(facet_centers[0] - 0.25),
                                  np.abs(facet_centers[0] + 0.25)) <= 0.08
        support_facets = facet_nodes[:, on_bottom & support_band]
        if top_facets.shape[1] == 0 or support_facets.shape[1] == 0:
            raise ValueError("mesh is too coarse to represent punch and support facets")
    else:
        top_facets = support_facets = None
    anchor_x = int(np.flatnonzero(bottom & np.isclose(px, -0.5)
                                  & np.isclose(py, -0.1))[0])
    anchor_y = int(np.flatnonzero(bottom & np.isclose(px, -0.25)
                                  & np.isclose(py, -0.1))[0])
    anchor_z = int(np.flatnonzero(bottom & np.isclose(px, -0.25)
                                  & np.isclose(py, 0.0))[0])
    supports = supports[supports != anchor_z]
    if top.size == 0 or supports.size == 0:
        raise ValueError("mesh is too coarse to represent punch and support patches")

    fixed = np.array([dof[anchor_x, 0], dof[anchor_y, 1], dof[anchor_z, 2]])
    punch_count = top.size if not facet_contact else top_facets.shape[1]
    support_count = supports.size if not facet_contact else support_facets.shape[1]
    constraints = np.zeros((3 + punch_count + support_count, basis.N))
    constraints[0, fixed[0]] = 1.0
    constraints[1, fixed[1]] = 1.0
    constraints[2, fixed[2]] = 1.0
    first_punch = 3
    punch_rows = np.arange(first_punch, first_punch + punch_count)
    first_support = first_punch + punch_count
    support_rows = np.arange(first_support, first_support + support_count)
    if facet_contact:
        for local_node in range(3):
            constraints[punch_rows, dof[top_facets[local_node], 2]] = -1. / 3.
            constraints[support_rows, dof[support_facets[local_node], 2]] = 1. / 3.
    else:
        constraints[punch_rows, dof[top, 2]] = -1.0
        constraints[support_rows, dof[supports, 2]] = 1.0
    unilateral = np.r_[np.zeros(3, dtype=bool),
                       np.ones(punch_count + support_count, dtype=bool)]

    displacement = np.zeros(basis.N)
    committed = assembler.initial_state()
    nleg = max(2, int(np.ceil(maximum_stroke / 5e-4)))
    strokes = np.r_[np.linspace(maximum_stroke / nleg, maximum_stroke, nleg),
                    np.linspace(maximum_stroke * (nleg - 1) / nleg, 0.0, nleg)]
    records = []

    def gaps(u: np.ndarray, tool_z: float) -> np.ndarray:
        if facet_contact:
            top_z = pz[top_facets] + u[dof[top_facets, 2]]
            support_z = pz[support_facets] + u[dof[support_facets, 2]]
            return np.r_[tool_z - top_z.mean(axis=0), support_z.mean(axis=0)]
        return np.r_[tool_z - (pz[top] + u[dof[top, 2]]),
                     pz[supports] + u[dof[supports, 2]]]

    for step, stroke in enumerate(strokes, start=1):
        tool_z = 0.1 - float(stroke)
        converged = False
        contact = None
        evaluation = None
        for iteration in range(1, 31):
            evaluation = assembler.evaluate(displacement, committed)
            current_gap = gaps(displacement, tool_z)
            rhs = np.r_[-displacement[fixed], -current_gap]
            contact = solve_kktkit_contact_step(
                evaluation.tangent, evaluation.residual, constraints, rhs,
                unilateral, tolerance=1e-9, max_iterations=100,
            )
            if not contact.converged:
                raise RuntimeError(f"3D contact active set failed at step {step}: {contact.message}")
            candidate = displacement + contact.primal
            candidate[fixed] = 0.0
            candidate_evaluation = assembler.evaluate(candidate, committed)
            candidate_gap = gaps(candidate, tool_z)
            stationarity = candidate_evaluation.residual + constraints.T @ contact.multipliers
            stationarity[fixed] = 0.0
            equilibrium = float(np.linalg.norm(stationarity, ord=np.inf))
            comp = float(np.max(np.abs(candidate_gap * contact.multipliers[3:]), initial=0.0))
            if (equilibrium <= 2e-7
                    and np.min(candidate_gap, initial=0.0) >= -2e-8
                    and np.max(contact.multipliers[3:], initial=0.0) <= 2e-8
                    and comp <= 2e-7):
                displacement = candidate
                evaluation = candidate_evaluation
                converged = True
                break
            displacement = candidate
        if not converged or evaluation is None or contact is None:
            raise RuntimeError(
                f"3D Newton contact equilibrium failed at step {step}: "
                f"equilibrium={equilibrium:.3e}, gap={np.min(candidate_gap):.3e}"
            )
        committed = assembler.commit(evaluation)
        force = float(-np.sum(contact.multipliers[punch_rows]))
        records.append({
            "step": step, "stroke": float(stroke), "punch_force": force,
            "newton_iterations": iteration,
            "active_punch_nodes" if not facet_contact else "active_punch_facets":
                int(np.count_nonzero(contact.active_rows[punch_rows])),
            "active_support_nodes" if not facet_contact else "active_support_facets":
                int(np.count_nonzero(contact.active_rows[support_rows])),
            "minimum_gap": float(np.min(gaps(displacement, tool_z))),
            "equilibrium_residual_inf": equilibrium,
            "max_equivalent_plastic_strain": float(np.max(committed[..., 6])),
        })

    center_top = int(np.flatnonzero(np.isclose(px, 0.0) & np.isclose(py, 0.0)
                                    & np.isclose(pz, 0.1))[0])
    center_bottom = int(np.flatnonzero(np.isclose(px, 0.0) & np.isclose(py, 0.0)
                                       & np.isclose(pz, 0.0))[0])
    thickness_after = 0.1 + displacement[dof[center_top, 2]] - displacement[dof[center_bottom, 2]]
    return {
        "problem": ("3D small-strain Tet4 Hill48 sheet with facet-centroid rigid-plane contact"
                    if hill48 and facet_contact else
                    "3D small-strain Tet4 J2 sheet with facet-centroid rigid-plane contact"
                    if facet_contact else
                    "3D small-strain Tet4 Hill48 sheet with nodal rigid-plane contact"
                    if hill48 else
                    "3D small-strain Tet4 J2 sheet with nodal rigid-plane contact"),
        "mesh": {"nx": nx, "ny": ny, "nz": nz, "tetrahedra": int(mesh.nelements),
                 "nodes": int(node_count), "displacement_dofs": int(basis.N)},
        "maximum_stroke": maximum_stroke,
        "material_source": ("NUMISHEET '93 Hill48 ratios and tabular hardening"
                            if numisheet_material else
                            "illustrative isotropic Hill48/J2 parameters"),
        "steps": records,
        "maximum_punch_force": max(r["punch_force"] for r in records),
        "center_thickness_change": float(thickness_after - 0.1),
        "center_springback_z": float(displacement[dof[center_top, 2]]),
        "maximum_equivalent_plastic_strain": float(np.max(committed[..., 6])),
        "contact_discretization": ("facet-centroid collocation with linear face interpolation"
                                   if facet_contact else "nodal normal constraints"),
        "scope": "3D kernel proof; flat-tool contact; not a calibrated forming result",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=8)
    parser.add_argument("--ny", type=int, default=4)
    parser.add_argument("--nz", type=int, default=1)
    parser.add_argument("--maximum-stroke", type=float, default=0.001)
    parser.add_argument("--hill48", action="store_true",
                        help="use anisotropic Hill48 plasticity in the 3D contact solve")
    parser.add_argument("--numisheet-material", action="store_true",
                        help="use the reference Hill48 ratios and tabular hardening curve")
    parser.add_argument("--facet-contact", action="store_true",
                        help="collocate rigid-plane gaps at boundary-facet centroids")
    args = parser.parse_args()
    print(json.dumps(run(args.nx, args.ny, args.nz, args.maximum_stroke,
                         args.hill48, args.numisheet_material,
                         args.facet_contact), indent=2))


if __name__ == "__main__":
    main()
