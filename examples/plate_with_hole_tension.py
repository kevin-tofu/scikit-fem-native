"""Linear plane-stress tension of a rectangular plate with a circular hole.

Run with ``python examples/plate_with_hole_tension.py``.  The example writes
``examples/plate_with_hole_tension.png``.
"""

from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.sparse.linalg import spsolve

import skfemntv
from skfemntv.helpers import ddot, div, sym_grad


LENGTH = 4.0
HEIGHT = 1.0
RADIUS = 0.18
YOUNG = 210.0e9
POISSON = 0.3
STRAIN = 1.0e-3

hole_center = np.array((LENGTH / 2.0, HEIGHT / 2.0))
angles = np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False)
corner_angle = np.arctan2(HEIGHT / 2.0, LENGTH / 2.0)
angles = np.sort(np.concatenate((
    angles,
    [corner_angle, np.pi - corner_angle, np.pi + corner_angle,
     2.0 * np.pi - corner_angle],
)))
radial = np.linspace(0.0, 1.0, 81) ** 1.2
cos_angle, sin_angle = np.cos(angles), np.sin(angles)
reach_x = np.divide(
    LENGTH / 2.0, np.abs(cos_angle),
    out=np.full_like(cos_angle, np.inf), where=np.abs(cos_angle) > 1.0e-14,
)
reach_y = np.divide(
    HEIGHT / 2.0, np.abs(sin_angle),
    out=np.full_like(sin_angle, np.inf), where=np.abs(sin_angle) > 1.0e-14,
)
outer_radius = np.minimum(reach_x, reach_y)
radius = RADIUS + radial[:, None] * (outer_radius[None, :] - RADIUS)
points = np.stack((
    hole_center[0] + radius * cos_angle[None, :],
    hole_center[1] + radius * sin_angle[None, :],
))
nradial, ntheta = len(radial), len(angles)
points = points.reshape(2, -1)
triangles = []
for ir in range(nradial - 1):
    inner = ir * ntheta
    outer = (ir + 1) * ntheta
    for it in range(ntheta):
        next_it = (it + 1) % ntheta
        a, b = inner + it, outer + it
        c, d = outer + next_it, inner + next_it
        triangles.extend(((a, b, c), (a, c, d)))
mesh = skfemntv.MeshTri(points, np.asarray(triangles, dtype=np.int32).T)
basis = skfemntv.Basis(
    mesh, skfemntv.ElementVector(skfemntv.ElementTriP1()), intorder=3
)

# Plane stress: sigma = 2 mu eps + lambda tr(eps) I.
mu = YOUNG / (2.0 * (1.0 + POISSON))
lame_lambda = YOUNG * POISSON / (1.0 - POISSON**2)


@skfemntv.BilinearForm
def elasticity(u, v, w):
    return 2.0 * mu * ddot(sym_grad(u), sym_grad(v)) + (
        lame_lambda * div(u) * div(v)
    )


matrix = skfemntv.asm(elasticity, basis)
left_nodes = np.flatnonzero(np.isclose(mesh.p[0], 0.0))
right_nodes = np.flatnonzero(np.isclose(mesh.p[0], LENGTH))
left_x = basis.nodal_dofs[0, left_nodes]
right_x = basis.nodal_dofs[0, right_nodes]
# Pin one transverse DOF to remove rigid vertical translation.  All other
# transverse motion, including that of the loaded edge, remains free.
pin_y = basis.nodal_dofs[1, np.flatnonzero(
    np.isclose(mesh.p[0], 0.0) & np.isclose(mesh.p[1], 0.0)
)]
prescribed = np.zeros(basis.N)
prescribed[right_x] = STRAIN * LENGTH
constrained = np.union1d(np.union1d(left_x, right_x), pin_y)
free = np.setdiff1d(np.arange(basis.N), constrained)
displacement = prescribed.copy()
rhs = -(matrix @ prescribed)[free]
displacement[free] = spsolve(
    matrix[free][:, free], rhs
)
relative_residual = np.linalg.norm((matrix @ displacement)[free]) / max(
    np.linalg.norm(rhs), 1.0
)

# Recover von Mises stress at quadrature points, then show its cell average.
field = basis.interpolate(displacement)
strain = sym_grad(field)
stress = 2.0 * mu * strain + lame_lambda * div(field)[None, None] * np.eye(2)[:, :, None, None]
sigma_xx, sigma_yy, sigma_xy = stress[0, 0], stress[1, 1], stress[0, 1]
von_mises = np.sqrt(
    sigma_xx**2 - sigma_xx * sigma_yy + sigma_yy**2 + 3.0 * sigma_xy**2
)
cell_stress = von_mises.mean(axis=1) / (YOUNG * STRAIN)

cells = np.column_stack((np.full(mesh.t.shape[1], 3), mesh.t.T)).ravel()
cell_types = np.full(mesh.t.shape[1], pv.CellType.TRIANGLE, dtype=np.uint8)
grid = pv.UnstructuredGrid(cells, cell_types, np.column_stack((mesh.p.T, np.zeros(mesh.p.shape[1]))))
grid.cell_data["von Mises / nominal stress"] = cell_stress
plotter = pv.Plotter(off_screen=True, window_size=(1500, 420))
plotter.set_background("white")
plotter.add_mesh(
    grid,
    scalars="von Mises / nominal stress",
    cmap="inferno",
    clim=(0.0, 4.0),
    show_edges=False,
    scalar_bar_args={
        "title": r"$\sigma_{vm}/(E\epsilon_{nom})$",
        "vertical": False,
        "position_x": 0.34,
        "position_y": 0.03,
        "width": 0.32,
        "height": 0.08,
    },
)
plotter.add_text("Linear plane-stress tension: von Mises stress", font_size=16)
plotter.view_xy()
plotter.camera.parallel_projection = True
plotter.camera.zoom(1.15)
output = Path(__file__).with_name("plate_with_hole_tension.png")
plotter.screenshot(str(output))
plotter.close()
print(
    f"Saved {output}; peak cell-average stress ratio={cell_stress.max():.2f}; "
    f"relative free-DOF residual={relative_residual:.2e}"
)
