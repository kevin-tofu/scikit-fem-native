"""Solve and plot a two-dimensional electrostatic capacitor problem."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from scipy.sparse.linalg import spsolve

import skfemntv
from skfemntv.helpers import ddot, grad


@skfemntv.BilinearForm
def electrostatic_laplace(u, v, w):
    """Unit-permittivity Laplace operator for the electric potential."""
    return ddot(grad(u), grad(v))


def solve_capacitor(axis: np.ndarray) -> tuple[skfemntv.MeshTri, object, np.ndarray]:
    mesh = skfemntv.MeshTri.init_tensor(axis, axis)
    basis = skfemntv.Basis(
        mesh,
        skfemntv.ElementVector(skfemntv.ElementTriP1(), dim=1),
        intorder=3,
    )
    stiffness = skfemntv.asm(electrostatic_laplace, basis)
    left = basis.get_dofs(nodes=lambda x: np.isclose(x[0], 0.0)).all()
    right = basis.get_dofs(nodes=lambda x: np.isclose(x[0], 1.0)).all()
    boundary = np.unique(np.concatenate((left, right)))
    free = np.setdiff1d(np.arange(basis.N), boundary)
    boundary_values = np.zeros(basis.N)
    boundary_values[left] = 1.0
    rhs = np.zeros(basis.N)
    rhs[free] = -stiffness[free][:, boundary] @ boundary_values[boundary]
    potential = boundary_values.copy()
    potential[free] = spsolve(stiffness[free][:, free], rhs[free])
    return mesh, basis, potential


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("electrostatic_potential.png"),
    )
    args = parser.parse_args()

    axis = np.linspace(0.0, 1.0, 31)
    mesh, basis, potential = solve_capacitor(axis)
    field = basis.interpolate(potential).grad
    element_centers = mesh.p[:, mesh.t].mean(axis=1)
    electric_field = -field.mean(axis=2)

    triangulation = mtri.Triangulation(mesh.p[0], mesh.p[1], mesh.t.T)
    figure, axes = plt.subplots(figsize=(7.0, 5.4), constrained_layout=True)
    plot = axes.tripcolor(
        triangulation,
        potential,
        shading="gouraud",
        cmap="viridis",
    )
    axes.quiver(
        element_centers[0],
        element_centers[1],
        electric_field[0],
        electric_field[1],
        color="white",
        alpha=0.8,
        pivot="mid",
        scale=16.0,
        width=0.0025,
    )
    axes.set(
        aspect="equal",
        xlabel="x",
        ylabel="y",
        title="Electrostatic potential and electric field",
    )
    figure.colorbar(plot, ax=axes, label="potential $\\phi$")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
