"""Solve a small 3D elasticity problem using only scikit-fem-native."""

from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import spsolve

import skfemntv


@skfemntv.LinearForm
def downward_body_force(v, _w):
    """Apply a unit body force in the negative z direction."""
    return -1.0 * v[2]


def solve_cube(divisions: int = 2):
    """Assemble and solve a unit cube clamped at its bottom face."""
    if divisions < 1:
        raise ValueError("divisions must be positive")

    axis = np.linspace(0.0, 1.0, divisions + 1)
    mesh = skfemntv.MeshTet.init_tensor(axis, axis, axis)
    basis = skfemntv.Basis(
        mesh,
        skfemntv.ElementVector(skfemntv.ElementTetP1(), dim=3),
    )

    material = skfemntv.IsotropicLinearElasticity(
        young_modulus=210.0,
        poisson_ratio=0.3,
    )
    stiffness_form = skfemntv.isotropic_linear_elasticity_form(material)
    stiffness = skfemntv.asm(stiffness_form, basis)
    load = skfemntv.asm(downward_body_force, basis)

    fixed = basis.get_dofs(
        nodes=lambda x: np.isclose(x[2], 0.0),
    ).all()
    free = np.setdiff1d(np.arange(basis.N), fixed)

    displacement = np.zeros(basis.N)
    displacement[free] = spsolve(
        stiffness[free][:, free],
        load[free] - stiffness[free][:, fixed] @ displacement[fixed],
    )
    return mesh, basis, displacement


def main() -> None:
    mesh, basis, displacement = solve_cube()
    nodal_displacement = displacement.reshape(3, -1)
    maximum_displacement = np.linalg.norm(nodal_displacement, axis=0).max()
    print(f"elements={mesh.nelements}")
    print(f"degrees_of_freedom={basis.N}")
    print(f"maximum_displacement={maximum_displacement:.6e}")


if __name__ == "__main__":
    main()
