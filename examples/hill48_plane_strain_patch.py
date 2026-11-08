"""Check Hill48 ratios, the isotropic J2 limit, and TriP1 element tangent."""

import numpy as np

import skfemntv


def main():
    ratios = (1.0, 1.0402, 1.24897, 1.07895, 1.0, 1.0)
    material = skfemntv.Hill48PlasticityMaterial(
        1000.0, 0.3, 2.0, 20.0, stress_ratios=ratios
    )

    # The documented Abaqus R ratios are normalized so each uniaxial and
    # shear-yield stress maps back to the reference yield stress.
    equivalent_checks = []
    for component, ratio in enumerate(ratios):
        stress = np.zeros((3, 3))
        if component < 3:
            stress[component, component] = 2.0 * ratio
        else:
            i, j = ((0, 1), (0, 2), (1, 2))[component - 3]
            stress[i, j] = stress[j, i] = 2.0 * ratio / np.sqrt(3.0)
        equivalent_checks.append(float(material.equivalent_stress(stress)))

    strain = np.array([
        [0.01, 0.004, 0.0],
        [0.004, -0.003, 0.0],
        [0.0, 0.0, 0.0],
    ])
    state = np.zeros(7)
    isotropic = skfemntv.Hill48PlasticityMaterial(1000.0, 0.3, 2.0, 20.0)
    hill_point = isotropic.evaluate(strain, state)
    j2_point = skfemntv.J2PlasticityMaterial(1000.0, 0.3, 2.0, 20.0).evaluate(
        strain, state
    )
    isotropic_limit_error = float(np.max(np.abs(hill_point.stress - j2_point.stress)))
    state_limit_error = float(np.max(np.abs(hill_point.trial_state - j2_point.trial_state)))

    mesh = skfemntv.MeshTri.init_tensor(
        np.linspace(0.0, 1.0, 3), np.linspace(0.0, 0.2, 2)
    )
    basis = skfemntv.Basis(
        mesh, skfemntv.ElementVector(skfemntv.ElementTriP1(), dim=2), intorder=2
    )
    assembler = skfemntv.PlaneStrainHill48Assembler(
        basis, 1000.0, 0.3, 2.0, 20.0, stress_ratios=(1., 1., 1., 1., 1., 1.)
    )
    displacement = np.zeros(basis.N)
    x_dofs = np.arange(0, basis.N, 2)
    displacement[x_dofs] = 0.01 * basis.doflocs[0, x_dofs]
    committed = assembler.initial_state()
    trial = assembler.evaluate(displacement, committed)
    rng = np.random.default_rng(2)
    direction = rng.standard_normal(basis.N)
    direction /= np.linalg.norm(direction)
    step = 1e-7
    plus = assembler.evaluate(displacement + step * direction, committed).residual
    minus = assembler.evaluate(displacement - step * direction, committed).residual
    finite_difference = (plus - minus) / (2.0 * step)
    tangent_error = float(np.linalg.norm(
        trial.tangent @ direction - finite_difference
    ) / max(np.linalg.norm(finite_difference), 1e-30))

    print(f"Hill48 directional yield checks: {equivalent_checks}")
    print(f"unit-ratio J2 stress difference: {isotropic_limit_error:.3e}")
    print(f"unit-ratio J2 history difference: {state_limit_error:.3e}")
    print(f"TriP1 plastic points: {np.count_nonzero(trial.plastic_points)}")
    print(f"assembled tangent directional relative error: {tangent_error:.3e}")

    if not np.allclose(equivalent_checks, 2.0, rtol=1e-12, atol=1e-12):
        raise AssertionError("Hill48 R ratios do not recover directional yield stresses")
    if isotropic_limit_error > 1e-10 or state_limit_error > 1e-10:
        raise AssertionError("unit-ratio Hill48 must reduce to J2")
    if tangent_error > 2e-6:
        raise AssertionError("Hill48 assembled tangent failed its directional check")


if __name__ == "__main__":
    main()
