"""Check Hill48 return mapping against the published NUMISHEET hardening table."""

import json
from pathlib import Path

import numpy as np

import skfemntv


def main():
    reference_path = Path(__file__).with_name("numisheet93_draw_bending_reference.json")
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    material_data = reference["material"]
    curve = np.asarray(
        material_data["hardening_curve_equivalent_plastic_strain_yield_stress_MPa"],
        dtype=float,
    )
    material = skfemntv.Hill48PlasticityMaterial(
        material_data["young_modulus"],
        material_data["poisson_ratio"],
        yield_stress=material_data["initial_yield_stress"],
        stress_ratios=material_data[
            "hill48_coefficients_R11_R22_R33_R12_R13_R23"
        ],
        hardening_curve=curve,
    )

    knot_error = float(np.max(np.abs(
        material.yield_stress_at(curve[:, 0]) - curve[:, 1]
    )))
    alpha = 0.5 * (curve[17, 0] + curve[18, 0])
    expected_midpoint = 0.5 * (curve[17, 1] + curve[18, 1])
    midpoint_error = abs(material.yield_stress_at(alpha) - expected_midpoint)

    strain = np.diag([0.005, 0.0, 0.0])
    trial = material.evaluate(strain, material.initial_state())
    return_error = abs(
        float(trial.equivalent_stress)
        - material.yield_stress_at(float(trial.trial_state[6]))
    )
    mesh = skfemntv.MeshTri.init_tensor(
        np.linspace(0.0, 1.0, 3), np.linspace(0.0, 0.2, 2)
    )
    basis = skfemntv.Basis(
        mesh, skfemntv.ElementVector(skfemntv.ElementTriP1(), dim=2), intorder=2
    )
    assembler = skfemntv.PlaneStrainHill48Assembler(
        basis,
        material_data["young_modulus"],
        material_data["poisson_ratio"],
        yield_stress=material_data["initial_yield_stress"],
        stress_ratios=material_data[
            "hill48_coefficients_R11_R22_R33_R12_R13_R23"
        ],
        hardening_curve=curve,
    )
    displacement = np.zeros(basis.N)
    x_dofs = np.arange(0, basis.N, 2)
    displacement[x_dofs] = 0.005 * basis.doflocs[0, x_dofs]
    committed = assembler.initial_state()
    assembled = assembler.evaluate(displacement, committed)
    direction = np.random.default_rng(42).standard_normal(basis.N)
    direction /= np.linalg.norm(direction)
    step = 1e-8
    plus = assembler.evaluate(displacement + step * direction, committed).residual
    minus = assembler.evaluate(displacement - step * direction, committed).residual
    finite_difference = (plus - minus) / (2.0 * step)
    tangent_error = float(np.linalg.norm(
        assembled.tangent @ direction - finite_difference
    ) / max(np.linalg.norm(finite_difference), 1e-30))
    print(f"hardening table points: {curve.shape[0]}")
    print(f"maximum table-knot error [MPa]: {knot_error:.3e}")
    print(f"piecewise-linear midpoint error [MPa]: {midpoint_error:.3e}")
    print(f"plastic return consistency error [MPa]: {return_error:.3e}")
    print(f"equivalent plastic strain: {trial.trial_state[6]:.8e}")
    print(f"assembled tabular-hardening tangent relative error: {tangent_error:.3e}")
    if (knot_error > 1e-10 or midpoint_error > 1e-10 or return_error > 1e-7
            or tangent_error > 1e-5):
        raise AssertionError("NUMISHEET Hill48 material point check failed")


if __name__ == "__main__":
    main()
