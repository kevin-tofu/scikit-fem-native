"""Run a level-set wheel compliance optimization using skfemntv state solves."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import gc
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from time import perf_counter

import meshio
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

# A global editable-install redirector can point at another checkout; this
# tutorial is intended to use the source tree selected by PYTHONPATH.
sys.meta_path = [
    finder
    for finder in sys.meta_path
    if type(finder).__name__ != "ScikitBuildRedirectingFinder"
]

import skfemntv


@dataclass(frozen=True)
class WheelOptimizationConfig:
    mesh_path: Path
    output_directory: Path
    iterations: int = 30
    target_volume_fraction: float = 0.5
    evolution_gamma: float = 0.1
    evolution_steps: int = 10
    evolution_ghost_penalty: float = 0.01
    penalty_growth: float = 1.1
    penalty_update_interval: int = 5
    constraint_update_tolerance: float = 0.01
    quadrature_order: int = 4
    solver: str = "scipy"
    threads: int = 1


@dataclass(frozen=True)
class DesignEvaluation:
    compliance: float
    volume_fraction: float
    volume_residual: float
    state_path: Path | None = None
    adjoint_path: Path | None = None


@dataclass(frozen=True)
class IterationRecord:
    iteration: int
    compliance: float
    volume_fraction: float
    volume_residual: float
    augmented_lagrangian: float
    lagrange_multiplier: float
    penalty: float
    evolution_gamma: float
    max_velocity: float
    elapsed_seconds: float


def _initial_level_set(points: np.ndarray) -> np.ndarray:
    x, y, z = points.T
    trigonometric = (
        -np.cos(4.0 * np.pi * x)
        * np.cos(4.0 * np.pi * y)
        * np.cos(4.0 * np.pi * z)
        / 4.0
        - 0.1 / 4.0
    )
    radial = -np.sqrt(x * x + y * y) + 0.9
    return np.minimum(trigonometric, radial)


def _mesh_volume(points: np.ndarray, tetrahedra: np.ndarray) -> float:
    coordinates = points[tetrahedra]
    first = coordinates[:, 1] - coordinates[:, 0]
    second = coordinates[:, 2] - coordinates[:, 0]
    third = coordinates[:, 3] - coordinates[:, 0]
    return float(
        np.sum(np.abs(np.einsum("ij,ij->i", np.cross(first, second), third)))
        / 6.0
    )


def _tetrahedron_quadrature(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Duffy-product integration rule on the reference tetrahedron."""
    count = max(1, (int(order) + 4) // 2)
    points, weights = np.polynomial.legendre.leggauss(count)
    points = (points + 1.0) / 2.0
    weights = weights / 2.0
    coordinates: list[tuple[float, float, float]] = []
    quadrature_weights: list[float] = []
    for first, x in enumerate(points):
        for second, y_unit in enumerate(points):
            for third, z_unit in enumerate(points):
                coordinates.append(
                    (x, (1.0 - x) * y_unit, (1.0 - x) * (1.0 - y_unit) * z_unit)
                )
                quadrature_weights.append(
                    (1.0 - x) ** 2
                    * (1.0 - y_unit)
                    * weights[first]
                    * weights[second]
                    * weights[third]
                )
    return np.asarray(coordinates), np.asarray(quadrature_weights)


class WheelLevelSetOptimizer:
    """CutFEM wheel level-set optimization following the GridapTopOpt example."""

    def __init__(self, config: WheelOptimizationConfig) -> None:
        self.config = config
        source = meshio.read(config.mesh_path)
        if "tetra" not in source.cells_dict:
            raise ValueError("the wheel background mesh must contain Tet4 cells")
        self.points = np.asarray(source.points, dtype=float)
        self.tetrahedra = np.asarray(source.cells_dict["tetra"], dtype=np.int64)
        self.mesh = skfemntv.MeshTet(self.points.T, self.tetrahedra.T)
        self.background_volume = _mesh_volume(self.points, self.tetrahedra)
        self.mesh_scale = self._median_cell_diameter()
        self._cell_diameters = self._all_cell_diameters()
        self.cell_gradients, self.cell_volumes = self._tetra_geometry()
        radial_coordinate = np.linalg.norm(self.points[:, :2], axis=1)
        self.velocity_dirichlet_nodes = np.flatnonzero(
            np.abs(radial_coordinate - 0.1) <= 1.0e-6
        )
        if not len(self.velocity_dirichlet_nodes):
            raise RuntimeError(
                "could not locate the inner-shell Gamma_N nodes for velocity "
                "extension Dirichlet conditions"
            )
        self.mass_matrix = self._assemble_mass_matrix()
        self.velocity_matrix = self._assemble_velocity_extension_matrix()
        self.state_solver = (
            Path(__file__).resolve().parents[2]
            / "examples"
            / "wheel_torsion_cutfem_assembly.py"
        )
        if not self.state_solver.is_file():
            raise FileNotFoundError(self.state_solver)
        self.config.output_directory.mkdir(parents=True, exist_ok=True)
        self.log_directory = self.config.output_directory / "solver-logs"
        self.log_directory.mkdir(exist_ok=True)

    def _median_cell_diameter(self) -> float:
        coordinates = self.points[self.tetrahedra]
        lengths = []
        for first in range(4):
            for second in range(first):
                lengths.append(
                    np.linalg.norm(
                        coordinates[:, first] - coordinates[:, second], axis=1
                    )
                )
        return float(np.median(np.max(np.vstack(lengths), axis=0)))

    def _all_cell_diameters(self) -> np.ndarray:
        coordinates = self.points[self.tetrahedra]
        diameters = np.zeros(len(self.tetrahedra), dtype=float)
        for first in range(4):
            for second in range(first):
                diameters = np.maximum(
                    diameters,
                    np.linalg.norm(
                        coordinates[:, first] - coordinates[:, second], axis=1
                    ),
                )
        return diameters

    def _tetra_geometry(self) -> tuple[np.ndarray, np.ndarray]:
        coordinates = self.points[self.tetrahedra]
        gradients = np.empty((len(self.tetrahedra), 4, 3), dtype=float)
        volumes = np.empty(len(self.tetrahedra), dtype=float)
        for cell, nodes in enumerate(coordinates):
            jacobian = (nodes[1:] - nodes[0]).T
            inverse = np.linalg.inv(jacobian)
            gradients[cell, 1:] = inverse
            gradients[cell, 0] = -np.sum(inverse, axis=0)
            volumes[cell] = abs(np.linalg.det(jacobian)) / 6.0
        return gradients, volumes

    def _assemble_cell_matrix(self, local_matrices: np.ndarray) -> csr_matrix:
        rows = np.repeat(self.tetrahedra, 4, axis=1).reshape(-1)
        columns = np.tile(self.tetrahedra, (1, 4)).reshape(-1)
        matrix = coo_matrix(
            (local_matrices.reshape(-1), (rows, columns)),
            shape=(len(self.points), len(self.points)),
        ).tocsr()
        matrix.sum_duplicates()
        return matrix

    def _assemble_mass_matrix(self) -> csr_matrix:
        local_mass = self.cell_volumes[:, None, None] / 20.0 * (
            np.ones((4, 4)) + np.eye(4)
        )
        return self._assemble_cell_matrix(local_mass)

    def _assemble_velocity_extension_matrix(self) -> csr_matrix:
        alpha_coefficient = (
            self.config.evolution_gamma * self.config.evolution_steps
        )
        alpha = (alpha_coefficient * self._cell_diameters) ** 2 + 1.0e-6
        local_stiffness = self.cell_volumes[:, None, None] * np.einsum(
            "cik,cjk->cij", self.cell_gradients, self.cell_gradients
        )
        local_mass = self.cell_volumes[:, None, None] / 20.0 * (
            np.ones((4, 4)) + np.eye(4)
        )
        matrix = self._assemble_cell_matrix(
            local_mass + alpha[:, None, None] * local_stiffness
        ).tolil()
        fixed = self.velocity_dirichlet_nodes
        matrix[fixed, :] = 0.0
        matrix[:, fixed] = 0.0
        matrix[fixed, fixed] = 1.0
        return matrix.tocsr()

    def _write_candidate_mesh(self, path: Path, level_set: np.ndarray) -> None:
        meshio.write(
            path,
            meshio.Mesh(
                points=self.points,
                cells=[("tetra", self.tetrahedra)],
                point_data={"level_set": np.asarray(level_set, dtype=float)},
            ),
            file_format="vtu",
        )

    def _evaluate(
        self,
        level_set: np.ndarray,
        label: str,
        *,
        save_state: bool = False,
    ) -> DesignEvaluation:
        volume_fraction = self._volume_fraction(level_set)
        volume_residual = volume_fraction - self.config.target_volume_fraction
        with tempfile.TemporaryDirectory(
            prefix="wheel-levelset-eval-", dir=self.config.output_directory
        ) as temporary_directory:
            temporary = Path(temporary_directory)
            candidate_mesh = temporary / "candidate.vtu"
            state_path = (
                self.config.output_directory / f"state-{label}.vtu"
                if save_state
                else temporary / "state.vtu"
            )
            adjoint_path = (
                self.config.output_directory / f"adjoint-{label}.npz"
                if save_state
                else temporary / "adjoint.npz"
            )
            self._write_candidate_mesh(candidate_mesh, level_set)
            import_wrapper = (
                "import runpy,sys; "
                "sys.meta_path=[finder for finder in sys.meta_path "
                "if type(finder).__name__ != 'ScikitBuildRedirectingFinder']; "
                "sys.argv=sys.argv[1:]; "
                "runpy.run_path(sys.argv[0],run_name='__main__')"
            )
            command = [
                sys.executable,
                "-c",
                import_wrapper,
                str(self.state_solver),
                str(candidate_mesh),
                "--level-set-field",
                "level_set",
                "--quadrature-order",
                str(self.config.quadrature_order),
                "--threads",
                str(self.config.threads),
                "--solver",
                self.config.solver,
                "--output",
                str(state_path),
                "--adjoint-output",
                str(adjoint_path),
            ]
            package_python = str(Path(__file__).resolve().parents[2] / "python")
            python_paths = [package_python]
            for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep):
                if entry:
                    absolute_entry = str(
                        (Path.cwd() / entry).resolve()
                        if not Path(entry).is_absolute()
                        else Path(entry)
                    )
                    if absolute_entry not in python_paths:
                        python_paths.append(absolute_entry)
            child_environment = os.environ.copy()
            child_environment["PYTHONPATH"] = os.pathsep.join(python_paths)
            completed = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parents[2],
                env=child_environment,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            log_path = self.log_directory / f"{label}.log"
            log_path.write_text(completed.stdout, encoding="utf-8")
            if completed.returncode != 0:
                raise RuntimeError(
                    f"wheel state solve failed for {label}; see {log_path}\n"
                    f"{completed.stdout[-2000:]}"
                )
            match = re.search(
                r"linear_solve_compliance=([-+0-9.eE]+)", completed.stdout
            )
            if match is None:
                raise RuntimeError(
                    f"state solver did not report compliance; see {log_path}"
                )
            compliance = float(match.group(1))
        print(
            f"[wheel-levelset-optimization] evaluation={label} "
            f"compliance={compliance:.10g} "
            f"volume_fraction={volume_fraction:.10g} "
            f"volume_residual={volume_residual:.4e}",
            flush=True,
        )
        return DesignEvaluation(
            compliance,
            volume_fraction,
            volume_residual,
            state_path if save_state else None,
            adjoint_path if save_state else None,
        )

    def _volume_fraction(self, level_set: np.ndarray) -> float:
        cut = skfemntv.LevelSet(level_set, tolerance=0.0).cut_quadrature(
            self.mesh, side="inside", intorder=self.config.quadrature_order
        )
        fraction = cut.diagnostics.total_measure / self.background_volume
        del cut
        gc.collect()
        return float(fraction)

    def _shape_derivatives(
        self, level_set: np.ndarray, state_path: Path
    ) -> tuple[np.ndarray, np.ndarray]:
        """Assemble adjoint-based compliance and analytic volume gradients."""
        state = meshio.read(state_path)
        displacement = np.asarray(state.point_data["displacement"], dtype=float)
        adjoint_stem = state_path.stem.removeprefix("state-")
        adjoint_path = state_path.with_name(f"adjoint-{adjoint_stem}.npz")
        if not adjoint_path.is_file():
            raise FileNotFoundError(
                f"compliance adjoint field is missing: {adjoint_path}"
            )
        with np.load(adjoint_path) as adjoint_data:
            adjoint_displacement = np.asarray(
                adjoint_data["adjoint_displacement"], dtype=float
            )
        interface = skfemntv.LevelSet(level_set, tolerance=0.0).interface_quadrature(
            self.mesh, intorder=self.config.quadrature_order
        )
        cell_ids = interface.cells
        local_displacement = displacement[self.tetrahedra]
        local_adjoint_displacement = adjoint_displacement[self.tetrahedra]
        displacement_gradient = np.einsum(
            "cai,caj->cij", local_displacement, self.cell_gradients
        )
        strain = 0.5 * (
            displacement_gradient + displacement_gradient.transpose(0, 2, 1)
        )
        poisson_ratio = 0.3
        lame_lambda = poisson_ratio / (
            (1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio)
        )
        shear_modulus = 1.0 / (2.0 * (1.0 + poisson_ratio))
        stress = (
            lame_lambda
            * np.trace(strain, axis1=1, axis2=2)[:, None, None]
            * np.eye(3)
            + 2.0 * shear_modulus * strain
        )
        energy_density = np.einsum("cij,cij->c", strain, stress)
        adjoint_gradient = np.einsum(
            "cai,caj->cij", local_adjoint_displacement, self.cell_gradients
        )
        adjoint_strain = 0.5 * (
            adjoint_gradient + adjoint_gradient.transpose(0, 2, 1)
        )
        cross_energy_density = np.einsum(
            "cij,cij->c", adjoint_strain, stress
        )
        level_set_gradient = np.einsum(
            "ca,caj->cj", level_set[self.tetrahedra], self.cell_gradients
        )
        normal_gradient = np.abs(
            np.einsum(
                "qi,qi->q", interface.normals, level_set_gradient[cell_ids]
            )
        )
        if np.any(normal_gradient <= 1.0e-12):
            raise FloatingPointError(
                "level-set gradient is singular on the reconstructed interface"
            )
        reference = interface.reference_points
        shape_values = np.column_stack(
            (1.0 - np.sum(reference, axis=1), reference)
        )
        weighted_measure = interface.weights / normal_gradient
        local_compliance_gradient = (
            weighted_measure[:, None]
            * (cross_energy_density[cell_ids] - energy_density[cell_ids])[:, None]
            * shape_values
        )
        local_volume_gradient = (
            -weighted_measure[:, None]
            * shape_values
            / self.background_volume
        )
        compliance_gradient = np.zeros(len(self.points), dtype=float)
        volume_gradient = np.zeros(len(self.points), dtype=float)
        np.add.at(
            compliance_gradient,
            self.tetrahedra[cell_ids],
            local_compliance_gradient,
        )
        np.add.at(
            volume_gradient,
            self.tetrahedra[cell_ids],
            local_volume_gradient,
        )
        return compliance_gradient, volume_gradient

    def _extend_velocity(self, shape_gradient: np.ndarray) -> np.ndarray:
        rhs = np.array(shape_gradient, dtype=float, copy=True)
        rhs[self.velocity_dirichlet_nodes] = 0.0
        velocity = np.asarray(spsolve(self.velocity_matrix, rhs), dtype=float)
        velocity[self.velocity_dirichlet_nodes] = 0.0
        if not np.all(np.isfinite(velocity)):
            raise FloatingPointError("Hilbertian velocity extension is non-finite")
        return velocity

    def _evolve_level_set(
        self,
        level_set: np.ndarray,
        velocity: np.ndarray,
        evolution_gamma: float | None = None,
    ) -> np.ndarray:
        """Solve the stabilized P1 transport equation for ten CN steps."""
        velocity_norm = float(np.max(np.abs(velocity), initial=0.0))
        if velocity_norm <= 1.0e-30:
            return level_set.copy()
        level_set_gradient = np.einsum(
            "ca,caj->cj", level_set[self.tetrahedra], self.cell_gradients
        )
        gradient_norm = np.linalg.norm(level_set_gradient, axis=1)
        beta_at_nodes = (
            velocity[self.tetrahedra][:, :, None]
            / (1.0e-20 + velocity_norm)
            * level_set_gradient[:, None, :]
            / (1.0e-20 + gradient_norm[:, None, None])
        )
        # Integrate beta_h(x) * test_i(x) exactly for P1 beta and P1 tests;
        # beta is not replaced by its cell average as it is in a midpoint rule.
        integrated_beta_test = self.cell_volumes[:, None, None] / 20.0 * (
            np.sum(beta_at_nodes, axis=1)[:, None, :]
            + beta_at_nodes
        )
        local_advection = np.einsum(
            "cik,cjk->cij", integrated_beta_test, self.cell_gradients
        )
        advection = self._assemble_cell_matrix(local_advection)

        facet_rows: list[int] = []
        facet_columns: list[int] = []
        facet_values: list[float] = []
        parents = self.mesh.f2t
        facets = self.mesh.facets
        # GridapTopOpt's CutFEMEvolver uses SkeletonTriangulation(model), which
        # includes every interior facet. Match that stabilization domain here;
        # restricting it to cut-neighbor facets changes the level-set trajectory.
        stabilization_facets = np.flatnonzero(parents[1] >= 0)
        # Degree-four triangle quadrature matches Gridap's background
        # skeleton measure. Weights are normalized to sum to one here.
        barycentric_points = np.array(
            [
                [0.445948490915965, 0.445948490915965, 0.108103018168070],
                [0.445948490915965, 0.108103018168070, 0.445948490915965],
                [0.108103018168070, 0.445948490915965, 0.445948490915965],
                [0.091576213509771, 0.091576213509771, 0.816847572980459],
                [0.091576213509771, 0.816847572980459, 0.091576213509771],
                [0.816847572980459, 0.091576213509771, 0.091576213509771],
            ]
        )
        quadrature_weights = np.array(
            [
                0.223381589678010,
                0.223381589678010,
                0.223381589678010,
                0.109951743655322,
                0.109951743655322,
                0.109951743655322,
            ]
        )
        for facet_index in stabilization_facets:
            first_cell, second_cell = parents[:, facet_index]
            # Gridap's skeleton chooses the first incident cell in global cell
            # order as plus. Preserve that convention independently of the
            # local facet orientation returned by MeshTet.
            if second_cell < first_cell:
                first_cell, second_cell = second_cell, first_cell
            face_points = self.points[facets[:, facet_index]]
            cross = np.cross(
                face_points[1] - face_points[0],
                face_points[2] - face_points[0],
            )
            double_area = float(np.linalg.norm(cross))
            if double_area <= 1.0e-30:
                continue
            normal = cross / double_area
            face_area = 0.5 * double_area
            facet_velocity = barycentric_points @ velocity[
                facets[:, facet_index]
            ]
            beta_normal = float(
                np.dot(
                    quadrature_weights,
                    np.abs(
                        facet_velocity
                        / (1.0e-20 + velocity_norm)
                        * float(level_set_gradient[first_cell] @ normal)
                        / (1.0e-20 + gradient_norm[first_cell])
                    ),
                )
            )
            if beta_normal <= 1.0e-30:
                continue
            first_nodes = self.tetrahedra[first_cell]
            second_nodes = self.tetrahedra[second_cell]
            jump_gradient = np.concatenate(
                (
                    self.cell_gradients[first_cell] @ normal,
                    -self.cell_gradients[second_cell] @ normal,
                )
            )
            local_nodes = np.concatenate((first_nodes, second_nodes))
            mean_h_squared = 0.5 * (
                self._cell_diameters[first_cell] ** 2
                + self._cell_diameters[second_cell] ** 2
            )
            coefficient = (
                self.config.evolution_ghost_penalty
                * mean_h_squared
                * beta_normal
                * face_area
            )
            local_matrix = coefficient * np.outer(jump_gradient, jump_gradient)
            facet_rows.extend(np.repeat(local_nodes, len(local_nodes)))
            facet_columns.extend(np.tile(local_nodes, len(local_nodes)))
            facet_values.extend(local_matrix.reshape(-1))
        stabilization = coo_matrix(
            (facet_values, (facet_rows, facet_columns)),
            shape=self.mass_matrix.shape,
        ).tocsr()
        operator = advection + stabilization
        selected_gamma = (
            self.config.evolution_gamma
            if evolution_gamma is None
            else evolution_gamma
        )
        time_step = selected_gamma * float(
            np.min(self._cell_diameters)
        )
        lhs = self.mass_matrix / time_step + 0.5 * operator
        rhs_operator = self.mass_matrix / time_step - 0.5 * operator
        from scipy.sparse.linalg import splu

        factor = splu(lhs.tocsc())
        evolved = level_set.copy()
        for _ in range(self.config.evolution_steps):
            evolved = factor.solve(rhs_operator @ evolved)
        return self._correct_level_set(np.asarray(evolved, dtype=float))

    @staticmethod
    def _correct_level_set(level_set: np.ndarray, tolerance: float = 1.0e-10) -> np.ndarray:
        """Match GridapTopOpt.correct_ls! for zero and near-zero nodal values."""
        corrected = np.array(level_set, dtype=float, copy=True)
        near_zero = np.abs(corrected) < tolerance
        corrected[near_zero] = tolerance * np.sign(corrected[near_zero])
        corrected[corrected == 0.0] = tolerance
        return corrected

    def _reinitialize(self, level_set: np.ndarray) -> np.ndarray:
        """Solve GridapTopOpt's artificial-viscosity SDF weak form."""
        interface = skfemntv.LevelSet(level_set, tolerance=0.0).interface_quadrature(
            self.mesh, intorder=self.config.quadrature_order
        )
        if not len(interface.points):
            raise RuntimeError("level-set reinitialization found no interface")
        reference_points, reference_weights = _tetrahedron_quadrature(
            self.config.quadrature_order
        )
        reference_shape = np.column_stack(
            (1.0 - np.sum(reference_points, axis=1), reference_points)
        )
        initial_local = level_set[self.tetrahedra]
        initial_gradient = np.einsum(
            "ca,caj->cj", initial_local, self.cell_gradients
        )
        initial_gradient_norm_sq = np.einsum(
            "ci,ci->c", initial_gradient, initial_gradient
        )
        initial_at_quadrature = initial_local @ reference_shape.T
        cell_sizes = self._cell_diameters
        sign_at_quadrature = initial_at_quadrature / np.sqrt(
            initial_at_quadrature**2
            + cell_sizes[:, None] ** 2 * initial_gradient_norm_sq[:, None]
        )
        # The Duffy rule integrates over the unit reference tetrahedron, whose
        # volume is 1/6. ``cell_volumes`` is already the physical tetrahedron
        # volume (|det J| / 6), so scale it by six to obtain |det J| before
        # applying the reference weights.
        volume_weights = (
            6.0 * self.cell_volumes[:, None] * reference_weights[None, :]
        )

        interface_shape = np.column_stack(
            (
                1.0 - np.sum(interface.reference_points, axis=1),
                interface.reference_points,
            )
        )
        interface_mass = np.zeros(
            (len(self.tetrahedra), 4, 4), dtype=float
        )
        np.add.at(
            interface_mass,
            interface.cells,
            interface.weights[:, None, None]
            * interface_shape[:, :, None]
            * interface_shape[:, None, :],
        )
        penalty = 20.0 / cell_sizes
        viscosity = 0.5
        epsilon = 1.0e-20

        def residual(local_values: np.ndarray) -> np.ndarray:
            gradient = np.einsum(
                "ca,caj->cj", local_values, self.cell_gradients
            )
            gradient_magnitude = np.linalg.norm(gradient, axis=1)
            normalized_gradient = gradient / (epsilon + gradient_magnitude[:, None])
            transport_factor = np.einsum(
                "ci,ci->c", normalized_gradient, gradient
            )[:, None] - 1.0
            transport_residual = np.einsum(
                "cq,qi,cq->ci",
                sign_at_quadrature * transport_factor,
                reference_shape,
                volume_weights,
            )
            viscosity_coefficient = (
                viscosity
                * cell_sizes
                * np.linalg.norm(normalized_gradient, axis=1)
            )
            viscosity_residual = (
                self.cell_volumes[:, None]
                * viscosity_coefficient[:, None]
                * np.einsum("cj,cij->ci", gradient, self.cell_gradients)
            )
            boundary_residual = penalty[:, None] * np.einsum(
                "cij,cj->ci", interface_mass, local_values
            )
            return transport_residual + viscosity_residual + boundary_residual

        def gridap_jacobian(local_values: np.ndarray) -> np.ndarray:
            """Apply the reference a(u, du, v) tangent on each tetrahedron."""
            gradient = np.einsum(
                "ca,caj->cj", local_values, self.cell_gradients
            )
            normalized_gradient = gradient / (
                epsilon + np.linalg.norm(gradient, axis=1)[:, None]
            )
            viscosity_coefficient = (
                viscosity
                * cell_sizes
                * np.linalg.norm(normalized_gradient, axis=1)
            )
            jacobian = np.zeros((len(self.tetrahedra), 4, 4), dtype=float)
            for column in range(4):
                trial_gradient = self.cell_gradients[:, column, :]
                first_term = np.einsum(
                    "cq,qi,cq->ci",
                    sign_at_quadrature,
                    reference_shape,
                    volume_weights,
                ) * np.einsum("cj,cj->c", normalized_gradient, trial_gradient)[:, None]
                second_term = (
                    self.cell_volumes[:, None]
                    * viscosity_coefficient[:, None]
                    * np.einsum(
                        "cj,cij->ci",
                        trial_gradient,
                        self.cell_gradients,
                    )
                )
                jacobian[:, :, column] = (
                    first_term + second_term + penalty[:, None] * interface_mass[:, :, column]
                )
            return jacobian

        nodal_values = np.array(level_set, dtype=float, copy=True)
        values = nodal_values[self.tetrahedra]
        initial_residual = np.zeros(len(self.points), dtype=float)
        np.add.at(initial_residual, self.tetrahedra, residual(values))
        initial_residual_norm = float(np.linalg.norm(initial_residual))
        residual_scale = max(initial_residual_norm, 1.0e-30)
        residual_tolerance = max(1.0e-12, 1.0e-14 * initial_residual_norm)
        relative_residual = 1.0
        for iteration in range(20):
            local_residual = residual(values)
            global_residual = np.zeros(len(self.points), dtype=float)
            np.add.at(global_residual, self.tetrahedra, local_residual)
            residual_norm = float(np.linalg.norm(global_residual))
            relative_residual = residual_norm / residual_scale
            if residual_norm <= residual_tolerance:
                break

            local_jacobian = gridap_jacobian(values)
            jacobian = self._assemble_cell_matrix(local_jacobian)
            update = np.asarray(spsolve(jacobian, -global_residual), dtype=float)
            if not np.all(np.isfinite(update)):
                raise FloatingPointError(
                    "stabilized level-set reinitialization Newton step is non-finite"
                )
            nodal_values += update
            values = nodal_values[self.tetrahedra]
        else:
            raise RuntimeError(
                "stabilized level-set reinitialization did not converge: "
                f"relative_residual={relative_residual:.3e}"
            )

        print(
            f"[wheel-levelset-optimization] reinitialization="
            f"newton_iterations={iteration} relative_residual={relative_residual:.3e}",
            flush=True,
        )
        return self._correct_level_set(nodal_values)

    def _save_level_set(self, iteration: int, level_set: np.ndarray) -> Path:
        classification = skfemntv.LevelSet(
            level_set, tolerance=0.0
        ).classify(self.mesh)
        output = self.config.output_directory / f"levelset-{iteration:03d}.vtu"
        meshio.write(
            output,
            meshio.Mesh(
                points=self.points,
                cells=[("tetra", self.tetrahedra)],
                point_data={"level_set": level_set},
                cell_data={
                    "classification": [classification.labels.astype(np.int8)],
                    "active": [
                        np.isin(
                            np.arange(self.mesh.nelements), classification.active
                        ).astype(np.int8)
                    ],
                },
            ),
            file_format="vtu",
        )
        return output

    @staticmethod
    def _augmented_lagrangian(
        evaluation: DesignEvaluation, lagrange_multiplier: float, penalty: float
    ) -> float:
        return (
            evaluation.compliance
            - lagrange_multiplier * evaluation.volume_residual
            + 0.5 * penalty * evaluation.volume_residual**2
        )

    @staticmethod
    def _has_objective_oscillations(values: list[float]) -> bool:
        if len(values) < 26:
            return False
        # Gridap's history slice contains 26 samples (25 intervals).
        window = np.asarray(values[-26:], dtype=float)
        mean_value = float(np.mean(window))
        down_crossings = np.flatnonzero(
            (window[:-1] > mean_value) & (window[1:] <= mean_value)
        )
        return len(down_crossings) >= 2

    def _converged(self, records: list[IterationRecord]) -> bool:
        if (
            len(records) < 10
            or records[-1].iteration < 10
            or abs(records[-1].volume_residual) >= 0.01
        ):
            return False
        latest = records[-1].augmented_lagrangian
        if abs(latest) <= 1.0e-30:
            return False
        tolerance = 0.01 * float(np.min(self._cell_diameters))
        return all(
            abs(latest - previous.augmented_lagrangian) / abs(latest) < tolerance
            for previous in records[-6:-1]
        )

    def run(self) -> tuple[IterationRecord, ...]:
        level_set = self._correct_level_set(_initial_level_set(self.points))
        # GridapTopOpt reinitialises the corrected initial level set before
        # its first state and functional evaluation.
        level_set = self._reinitialize(level_set)
        self._save_level_set(0, level_set)
        lagrange_multiplier = 0.0
        penalty = 0.0
        evolution_gamma = self.config.evolution_gamma
        oscillation_iteration = -1
        objective_history: list[float] = []
        records: list[IterationRecord] = []
        csv_path = self.config.output_directory / "optimization-history.csv"
        columns = (
            "iteration",
            "compliance",
            "volume_fraction",
            "volume_residual",
            "augmented_lagrangian",
            "lagrange_multiplier",
            "penalty",
            "evolution_gamma",
            "max_velocity",
            "elapsed_seconds",
        )
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(columns)
            current = self._evaluate(
                level_set, "iteration-000-base", save_state=True
            )
            penalty = min(
                1.0e10,
                0.1
                * abs(current.compliance)
                / max(abs(current.volume_residual), 1.0e-12) ** 1.5,
            )
            current_lagrangian = self._augmented_lagrangian(
                current, lagrange_multiplier, penalty
            )
            objective_history.append(current_lagrangian)
            initial_record = IterationRecord(
                iteration=0,
                compliance=current.compliance,
                volume_fraction=current.volume_fraction,
                volume_residual=current.volume_residual,
                augmented_lagrangian=current_lagrangian,
                lagrange_multiplier=lagrange_multiplier,
                penalty=penalty,
                evolution_gamma=evolution_gamma,
                max_velocity=0.0,
                elapsed_seconds=0.0,
            )
            records.append(initial_record)
            writer.writerow(tuple(getattr(initial_record, name) for name in columns))
            stream.flush()

            for iteration in range(1, self.config.iterations + 1):
                started = perf_counter()
                if (
                    iteration >= 51
                    and iteration >= oscillation_iteration + 27
                    and evolution_gamma > 0.001
                    and self._has_objective_oscillations(objective_history)
                ):
                    oscillation_iteration = iteration + 1
                    evolution_gamma *= 0.75
                    print(
                        "[wheel-levelset-optimization] objective oscillation "
                        f"detected; evolution_gamma={evolution_gamma:.6g}",
                        flush=True,
                    )

                if current.state_path is None:
                    raise RuntimeError("baseline state solve did not save its VTU")
                compliance_gradient, volume_gradient = self._shape_derivatives(
                    level_set, current.state_path
                )
                augmented_gradient = compliance_gradient + (
                    -lagrange_multiplier + penalty * current.volume_residual
                ) * volume_gradient
                velocity = self._extend_velocity(augmented_gradient)
                max_velocity = float(np.max(np.abs(velocity), initial=0.0))
                level_set = self._reinitialize(
                    self._evolve_level_set(level_set, velocity, evolution_gamma)
                )
                self._save_level_set(iteration, level_set)

                # The reference updates lambda from C on the evolved geometry,
                # not from the geometry that generated the velocity.
                current = self._evaluate(
                    level_set,
                    f"iteration-{iteration:03d}-base",
                    save_state=True,
                )
                current_lagrangian = self._augmented_lagrangian(
                    current, lagrange_multiplier, penalty
                )
                objective_history.append(current_lagrangian)
                lagrange_multiplier -= penalty * current.volume_residual
                if (
                    iteration % self.config.penalty_update_interval == 0
                    and abs(current.volume_residual)
                    > self.config.constraint_update_tolerance
                ):
                    penalty = min(1.0e10, penalty * self.config.penalty_growth)

                record = IterationRecord(
                    iteration=iteration,
                    compliance=current.compliance,
                    volume_fraction=current.volume_fraction,
                    volume_residual=current.volume_residual,
                    augmented_lagrangian=current_lagrangian,
                    lagrange_multiplier=lagrange_multiplier,
                    penalty=penalty,
                    evolution_gamma=evolution_gamma,
                    max_velocity=max_velocity,
                    elapsed_seconds=perf_counter() - started,
                )
                records.append(record)
                writer.writerow(tuple(getattr(record, name) for name in columns))
                stream.flush()
                print(
                    f"[wheel-levelset-optimization] iteration={iteration}/"
                    f"{self.config.iterations} compliance={current.compliance:.8g} "
                    f"volume_fraction={current.volume_fraction:.8g} "
                    f"max_velocity={max_velocity:.5e} "
                    f"lambda={lagrange_multiplier:.5e} penalty={penalty:.5e} "
                    f"elapsed={record.elapsed_seconds:.1f}s",
                    flush=True,
                )
                if self._converged(records):
                    print(
                        f"[wheel-levelset-optimization] converged at "
                        f"iteration={iteration}",
                        flush=True,
                    )
                    break

        if current.state_path is None:
            raise RuntimeError("final state solve did not save its VTU")
        import shutil

        shutil.copyfile(current.state_path, self.config.output_directory / "state-final.vtu")
        print(
            f"[wheel-levelset-optimization] final_compliance="
            f"{current.compliance:.10g} final_volume_fraction="
            f"{current.volume_fraction:.10g} "
            f"levelset_vtu={self.config.output_directory / f'levelset-{records[-1].iteration:03d}.vtu'}",
            flush=True,
        )
        return tuple(records)


def main() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mesh",
        type=Path,
        default=repository_root
        / "tutorials/wheel-levelset/data/Wheel_3d-background.vtu",
        help="canonical Tet4 background mesh; its existing level-set field is ignored",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repository_root / "tutorial-output/wheel-levelset-optimization",
    )
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--target-volume-fraction", type=float, default=0.5)
    parser.add_argument("--evolution-gamma", type=float, default=0.1)
    parser.add_argument("--solver", choices=("scipy", "petsc-mumps"), default="scipy")
    parser.add_argument("--quadrature-order", type=int, default=4)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    if not args.mesh.is_file():
        parser.error(f"mesh does not exist: {args.mesh}")
    config = WheelOptimizationConfig(
        mesh_path=args.mesh,
        output_directory=args.output_dir,
        iterations=args.iterations,
        target_volume_fraction=args.target_volume_fraction,
        evolution_gamma=args.evolution_gamma,
        quadrature_order=args.quadrature_order,
        solver=args.solver,
        threads=args.threads,
    )
    WheelLevelSetOptimizer(config).run()


if __name__ == "__main__":
    main()
