"""Assemble a 3-D elastic-wheel CutFEM state from GridapTopOpt.

By default this solves the initial state on the reference geometry and level
set. ``--level-set-field`` instead uses a nodal level set already present in
the input mesh, so an optimized Gridap geometry can be evaluated with the
native CutFEM assembly. This is still a state solve, not a reproduction of the
optimization trajectory: shape derivatives, level-set evolution, and the
optimizer are not implemented here. For the P1
initial-state forms, volume and ghost-facet integrands are constant on each
affine simplex, while the linear traction needs a degree-two surface rule. A
first-order volume rule keeps the reference-scale mesh within memory limits.

Generate the background mesh from the pinned reference geometry, for example::

    gmsh papers/levelset/Wheel_3d.geo \
      -3 -format msh2 -o /tmp/Wheel_3d.msh

Then run from the repository root::

    PYTHONPATH=python python \
      examples/wheel_torsion_cutfem_assembly.py \
      /tmp/Wheel_3d.msh --quadrature-order 1

Reference source: GridapTopOpt.jl v0.2.0,
``scripts/Examples/Unfitted/Wheel3D_CutFEM.jl``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import resource
import sys
from time import perf_counter

import meshio
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

import skfemntv
from skfemntv.helpers import dot, jump, normal_grad
from skfemntv.linear_elasticity import IsotropicLinearElasticity
from skfemntv.mesh_io import from_meshio
from skfemntv.preflight import estimate_bilinear_memory, format_bytes
from skfemntv.weakforms import isotropic_linear_elasticity_weak_form


@dataclass(frozen=True)
class WheelAssemblyConfig:
    mesh_path: Path
    output_path: Path | None = None
    level_set_field: str | None = None
    reference_state_vtu: Path | None = None
    quadrature_order: int = 4
    surface_quadrature_order: int = 2
    threads: int = 1
    solver: str = "petsc-mumps"
    inner_radius: float = 0.1
    radius_tolerance: float = 1.0e-6
    traction_scale: float = 100.0
    ghost_penalty_scale: float = 1.0e-7
    isolated_volume_penalty: float = 1.0
    stop_after_cut_basis: bool = False
    stop_after_assembly: bool = False
    preflight_only: bool = False
    assembly_memory_limit_bytes: int | None = None
    adjoint_output_path: Path | None = None


@dataclass
class StageMemoryRecorder:
    """Report process peak RSS after each expensive wheel-analysis stage."""

    started_at: float
    previous_peak_bytes: int = 0

    @staticmethod
    def current_rss_bytes() -> int | None:
        try:
            resident_pages=int(Path("/proc/self/statm").read_text().split()[1])
        except (OSError,ValueError,IndexError):
            return None
        return resident_pages*int(os.sysconf("SC_PAGE_SIZE"))

    @classmethod
    def start(cls) -> StageMemoryRecorder:
        return cls(started_at=perf_counter())

    def checkpoint(self, stage: str) -> None:
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_bytes=int(peak if sys.platform=="darwin" else peak*1024)
        new_peak=max(0,peak_bytes-self.previous_peak_bytes)
        self.previous_peak_bytes=max(self.previous_peak_bytes,peak_bytes)
        current_bytes=self.current_rss_bytes()
        current_text=(
            "unavailable" if current_bytes is None
            else f"{current_bytes/1024**2:.1f}"
        )
        print(
            f"[wheel-memory] stage={stage} "
            f"peak_rss_mib={peak_bytes/1024**2:.1f} "
            f"current_rss_mib={current_text} "
            f"new_peak_mib={new_peak/1024**2:.1f} "
            f"elapsed_s={perf_counter()-self.started_at:.1f}",
            flush=True,
        )


def wheel_level_set(coordinates: np.ndarray) -> np.ndarray:
    """Initial level set copied from the GridapTopOpt reference driver."""
    x, y, z = coordinates
    trigonometric = (
        -np.cos(4.0 * np.pi * x)
        * np.cos(4.0 * np.pi * y)
        * np.cos(4.0 * np.pi * z)
        / 4.0
        - 0.1 / 4.0
    )
    radial = -np.sqrt(x * x + y * y) + 0.9
    return np.minimum(trigonometric, radial)


def _reference_dirichlet_nodes(mesh: skfemntv.MeshTet) -> np.ndarray:
    """Apply the five support predicates exactly as written in Julia."""
    x, y = mesh.p[:2]
    upper_circle = np.sqrt(np.maximum(0.0, 1.0 - x * x))
    lower_circle = -upper_circle
    upper = np.abs(y - upper_circle) < 1.0e-4
    lower = np.abs(y - lower_circle) < 1.0e-4
    fixed = (
        ((np.cos(np.deg2rad(30.0)) <= x)
         & (x <= np.cos(np.deg2rad(15.0))) & upper)
        | ((np.cos(np.deg2rad(97.5)) <= x)
           & (x <= np.cos(np.deg2rad(82.5))) & upper)
        | ((np.cos(np.deg2rad(165.0)) <= x)
           & (x <= np.cos(np.deg2rad(150.0))) & upper)
        | ((np.cos(np.deg2rad(142.5)) <= x)
           & (x <= np.cos(np.deg2rad(127.5))) & lower)
        | ((np.cos(np.deg2rad(52.5)) <= x)
           & (x <= np.cos(np.deg2rad(37.5))) & lower)
    )
    return np.flatnonzero(fixed)


def _cell_diameters(mesh: skfemntv.MeshTet) -> np.ndarray:
    coordinates = mesh.p[:, mesh.t]
    diameters = np.zeros(mesh.nelements)
    for first in range(mesh.t.shape[0]):
        for second in range(first):
            edge_length = np.linalg.norm(
                coordinates[:, first] - coordinates[:, second], axis=0
            )
            diameters = np.maximum(diameters, edge_length)
    return diameters


def _isolated_active_cells(
    mesh: skfemntv.MeshTet,
    active_cells: np.ndarray,
    fixed_nodes: np.ndarray,
) -> np.ndarray:
    """Mark active connected components with no fixed-boundary node."""
    local_id = np.full(mesh.nelements, -1, dtype=np.int64)
    local_id[active_cells] = np.arange(len(active_cells))
    parents = mesh.f2t
    interior = (parents[1] >= 0)
    first = parents[0, interior]
    second = parents[1, interior]
    connected = (local_id[first] >= 0) & (local_id[second] >= 0)
    first_local = local_id[first[connected]]
    second_local = local_id[second[connected]]
    edge_rows = np.concatenate((first_local, second_local))
    edge_columns = np.concatenate((second_local, first_local))
    adjacency = coo_matrix(
        (np.ones(len(edge_rows)), (edge_rows, edge_columns)),
        shape=(len(active_cells), len(active_cells)),
    ).tocsr()
    _, labels = connected_components(adjacency, directed=False)
    fixed_mask = np.zeros(mesh.p.shape[1], dtype=bool)
    fixed_mask[fixed_nodes] = True
    touches_fixed = np.any(fixed_mask[mesh.t[:, active_cells]], axis=0)
    anchored_components = np.unique(labels[touches_fixed])
    return active_cells[~np.isin(labels, anchored_components)]


def _elasticity_near_nullspace(
    mesh: skfemntv.MeshTet,
    nodal_dofs: np.ndarray,
    free_dofs: np.ndarray,
) -> np.ndarray:
    """Build the six rigid-body modes restricted to the free P1 DOFs."""
    coordinates = mesh.p.T
    centered = coordinates - coordinates.mean(axis=0)
    modes = np.zeros((len(free_dofs), 6), dtype=float)
    full_modes = np.zeros((nodal_dofs.max() + 1, 6), dtype=float)
    for component in range(3):
        full_modes[nodal_dofs[component], component] = 1.0
    rotations = (
        np.column_stack((np.zeros(len(coordinates)), -centered[:, 2], centered[:, 1])),
        np.column_stack((centered[:, 2], np.zeros(len(coordinates)), -centered[:, 0])),
        np.column_stack((-centered[:, 1], centered[:, 0], np.zeros(len(coordinates)))),
    )
    for axis, rotation in enumerate(rotations, start=3):
        for component in range(3):
            full_modes[nodal_dofs[component], axis] = rotation[:, component]
    modes[:] = full_modes[free_dofs]
    modes, triangular = np.linalg.qr(modes, mode="reduced")
    independent = np.abs(np.diag(triangular)) > 1.0e-12
    return modes[:, independent]


def _reference_state_vector(
    reference_path: Path,
    mesh: skfemntv.MeshTet,
    active_cells: np.ndarray,
    basis,
) -> np.ndarray:
    """Map Gridap's element-local VTK displacement onto native active DOFs."""
    reference = meshio.read(reference_path)
    if "uh" not in reference.point_data or "tetra" not in reference.cells_dict:
        raise ValueError("reference VTU must contain tetrahedra and point field 'uh'")
    reference_points = np.asarray(reference.points, dtype=float)
    coordinate_distances, point_to_node = cKDTree(mesh.p.T).query(reference_points)
    if np.max(coordinate_distances, initial=0.0) > 1.0e-12:
        raise ValueError(
            "reference points do not match solve mesh within 1e-12: "
            f"maximum distance={np.max(coordinate_distances):.3e}"
        )
    reference_cells = point_to_node[reference.cells_dict["tetra"]]
    solve_cells = mesh.t.T
    reference_keys = np.sort(reference_cells, axis=1)
    solve_keys = np.sort(solve_cells, axis=1)
    all_keys = np.vstack((solve_keys, reference_keys))
    _, key_ids = np.unique(all_keys, axis=0, return_inverse=True)
    solve_key_ids = key_ids[:len(solve_keys)]
    reference_key_ids = key_ids[len(solve_keys):]
    raw_cell_for_key = np.full(key_ids.max() + 1, -1, dtype=np.int64)
    raw_cell_for_key[reference_key_ids] = np.arange(len(reference_cells))
    raw_cell_ids = raw_cell_for_key[solve_key_ids]
    if np.any(raw_cell_ids < 0):
        raise ValueError("reference and solve mesh tetrahedra do not match")

    raw_displacement = np.asarray(reference.point_data["uh"], dtype=float).reshape(-1, 3)
    node_sum = np.zeros((mesh.p.shape[1], 3), dtype=float)
    node_min = np.full_like(node_sum, np.inf)
    node_max = np.full_like(node_sum, -np.inf)
    node_count = np.zeros(mesh.p.shape[1], dtype=np.int64)
    selected_cells = np.asarray(active_cells, dtype=np.int64)
    selected_raw_cells = raw_cell_ids[selected_cells]
    raw_connectivity = np.asarray(reference.cells_dict["tetra"], dtype=np.int64)
    for local_node in range(4):
        raw_points = raw_connectivity[selected_raw_cells, local_node]
        global_nodes = point_to_node[raw_points]
        values = raw_displacement[raw_points]
        np.add.at(node_sum, global_nodes, values)
        np.minimum.at(node_min, global_nodes, values)
        np.maximum.at(node_max, global_nodes, values)
        np.add.at(node_count, global_nodes, 1)
    used_nodes = node_count > 0
    spread = float(np.max(node_max[used_nodes] - node_min[used_nodes], initial=0.0))
    if spread > 1.0e-8:
        raise ValueError(
            "Gridap displacement is not single-valued on active mesh nodes: "
            f"max spread={spread:.3e}"
        )
    nodal_displacement = np.zeros_like(node_sum)
    nodal_displacement[used_nodes] = (
        node_sum[used_nodes] / node_count[used_nodes, None]
    )
    vector = np.zeros(basis.N, dtype=float)
    for component in range(3):
        vector[basis.nodal_dofs[component]] = nodal_displacement[:, component]
    return vector


def _solve_petsc(
    matrix,
    right_hand_side: np.ndarray,
    solver: str,
    near_nullspace: np.ndarray | None = None,
):
    """Solve the reduced SPD system through PETSc KSP."""
    try:
        from petsc4py import PETSc
    except ImportError as error:
        raise RuntimeError(
            "PETSc solve requested; activate the PETSc-enabled Python runtime"
        ) from error

    matrix = matrix.tocsr()
    petsc_csr = (
        np.asarray(matrix.indptr, dtype=PETSc.IntType),
        np.asarray(matrix.indices, dtype=PETSc.IntType),
        np.asarray(matrix.data, dtype=PETSc.ScalarType),
    )
    operator = PETSc.Mat().createAIJ(
        size=matrix.shape, csr=petsc_csr, comm=PETSc.COMM_SELF
    )
    operator.assemble()
    near_nullspace_object = None
    if near_nullspace is not None and near_nullspace.shape[1]:
        vectors = []
        vector_arrays = []
        for column in range(near_nullspace.shape[1]):
            vector = operator.createVecRight()
            vector_array = np.asarray(
                near_nullspace[:, column], dtype=PETSc.ScalarType
            ).copy()
            vector.setArray(vector_array)
            vectors.append(vector)
            vector_arrays.append(vector_array)
        near_nullspace_object = PETSc.NullSpace().create(
            constant=False,
            vectors=vectors,
            comm=PETSc.COMM_SELF,
        )
        operator.setNearNullSpace(near_nullspace_object)
    rhs_array = np.asarray(right_hand_side, dtype=PETSc.ScalarType)
    rhs = PETSc.Vec().createWithArray(rhs_array, comm=PETSc.COMM_SELF)
    solution = rhs.duplicate()

    ksp = PETSc.KSP().create(comm=PETSc.COMM_SELF)
    ksp.setOperators(operator)
    if solver == "petsc-gamg":
        # GAMG's default coarse-grid cycle may be indefinite.  GMRES does not
        # require the preconditioner to be positive definite, unlike CG.
        ksp.setType("gmres")
        ksp.getPC().setType("gamg")
        ksp.setTolerances(rtol=1.0e-9, atol=1.0e-12, max_it=2000)
    elif solver in {"petsc-lu", "petsc-mumps"}:
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        if solver == "petsc-mumps":
            ksp.getPC().setFactorSolverType("mumps")
    else:
        raise ValueError(f"unsupported PETSc solver {solver!r}")
    ksp.setFromOptions()
    ksp.solve(rhs, solution)
    residual = rhs.duplicate()
    operator.mult(solution, residual)
    residual.axpy(-1.0, rhs)
    result = solution.getArray(readonly=True).copy()
    diagnostics = (
        int(ksp.getIterationNumber()),
        float(residual.norm()) / max(float(rhs.norm()), 1.0e-30),
        int(ksp.getConvergedReason()),
    )
    ksp.destroy()
    if near_nullspace_object is not None:
        near_nullspace_object.destroy()
    for vector in vectors if near_nullspace is not None and near_nullspace.shape[1] else ():
        vector.destroy()
    residual.destroy()
    solution.destroy()
    rhs.destroy()
    operator.destroy()
    return result, diagnostics


def assemble(config: WheelAssemblyConfig) -> None:
    memory=StageMemoryRecorder.start()
    memory.checkpoint("start")
    if not config.mesh_path.is_file():
        raise FileNotFoundError(config.mesh_path)
    imported = meshio.read(config.mesh_path)
    imported_level_set = None
    if config.level_set_field is not None:
        if config.level_set_field not in imported.point_data:
            raise ValueError(
                f"mesh has no point field {config.level_set_field!r}; "
                f"available={tuple(imported.point_data)}"
            )
        imported_level_set = np.asarray(
            imported.point_data[config.level_set_field], dtype=float
        ).reshape(-1).copy()
        if len(imported_level_set) != len(imported.points):
            raise ValueError("level-set point field does not match mesh points")
    mesh = from_meshio(imported, force_meshio_type="tetra")
    del imported
    memory.checkpoint("mesh-loaded")
    level_set = skfemntv.LevelSet(
        wheel_level_set if imported_level_set is None else imported_level_set,
        tolerance=0.0,
    )
    classification = level_set.classify(mesh)
    memory.checkpoint("level-set-classified")
    cut_quadrature = level_set.cut_quadrature(
        mesh, side="inside", intorder=config.quadrature_order
    )
    active_cells = np.unique(cut_quadrature.cells)
    memory.checkpoint("cut-quadrature-built")
    print(
        "[wheel-memory] cut_cells="
        f"{cut_quadrature.diagnostics.cut_cell_count} "
        f"active_cells={len(active_cells)} "
        f"quadrature_points={len(cut_quadrature.weights)} "
        f"inside_volume={cut_quadrature.diagnostics.total_measure:.12g}",
        flush=True,
    )

    displacement_element = skfemntv.ElementVector(
        skfemntv.ElementTetP1(), dim=3
    )
    full_basis = skfemntv.Basis(
        mesh, displacement_element, intorder=config.quadrature_order
    )
    cut_basis = skfemntv.CutCellBasis(full_basis, cut_quadrature)
    memory.checkpoint("cut-basis-built")
    removed_dof_map_bytes=(
        mesh.nelements*mesh.t.shape[0]*displacement_element._dim*8
        +cut_basis.npoints*full_basis.element_dofs.shape[0]*8
    )
    print(
        "[wheel-memory] cut_basis_avoided_redundant_dof_maps_mib="
        f"{removed_dof_map_bytes/1024**2:.1f} "
        f"full_cell_map_mib={mesh.nelements*mesh.t.shape[0]*displacement_element._dim*8/1024**2:.1f} "
        f"quadrature_dof_map_mib={cut_basis.npoints*full_basis.element_dofs.shape[0]*8/1024**2:.1f}",
        flush=True,
    )
    if config.preflight_only:
        stiffness_estimate=estimate_bilinear_memory(cut_basis)
        print(
            "[wheel-memory-estimate] stage=elasticity "
            f"basis={format_bytes(stiffness_estimate.basis_bytes)} "
            f"assembly_peak={format_bytes(stiffness_estimate.construction_peak_incremental_bytes_upper_bound)} "
            f"total_peak={format_bytes(stiffness_estimate.construction_peak_total_bytes_upper_bound)} "
            f"nnz_upper={stiffness_estimate.nnz_upper_bound} "
            f"entities={stiffness_estimate.entity_count} "
            f"local_dofs={stiffness_estimate.row_local_dofs}",
            flush=True,
        )
        ghost_facets=classification.ghost_facets(mesh)
        ghost_bases=tuple(
            skfemntv.InteriorFacetBasis(
                mesh,
                displacement_element,
                facets=ghost_facets,
                intorder=config.quadrature_order,
                side=side,
            )
            for side in (0,1)
        )
        ghost_estimate=estimate_bilinear_memory(*ghost_bases)
        print(
            "[wheel-memory-estimate] stage=ghost-penalty-one-side-pair "
            f"basis={format_bytes(ghost_estimate.basis_bytes)} "
            f"assembly_peak={format_bytes(ghost_estimate.construction_peak_incremental_bytes_upper_bound)} "
            f"total_peak={format_bytes(ghost_estimate.construction_peak_total_bytes_upper_bound)} "
            f"nnz_upper={ghost_estimate.nnz_upper_bound} "
            f"entities={ghost_estimate.entity_count} "
            f"local_dofs={ghost_estimate.row_local_dofs}",
            flush=True,
        )
        print(
            "preflight stopped before sparse matrix assembly and PETSc solve; "
            "ghost side-pair assemblers are streamed one at a time",
            flush=True,
        )
        return
    if config.stop_after_cut_basis:
        print(
            "profile stopped after CutCellBasis construction; "
            "stiffness, ghost-penalty, and solve stages were skipped",
            flush=True,
        )
        return

    material = IsotropicLinearElasticity(young_modulus=1.0, poisson_ratio=0.3)
    weak_form = isotropic_linear_elasticity_weak_form(material)

    @skfemntv.BilinearForm
    def elasticity(trial, test, _parameters):
        return weak_form(trial, test, None)

    stiffness = skfemntv.asm(
        elasticity,
        cut_basis,
        num_threads=config.threads,
        memory_limit_bytes=config.assembly_memory_limit_bytes,
    )
    memory.checkpoint("elasticity-assembled")

    ghost_facets = classification.ghost_facets(mesh)
    ghost_bases = tuple(
        skfemntv.InteriorFacetBasis(
            mesh,
            displacement_element,
            facets=ghost_facets,
            intorder=config.quadrature_order,
            side=side,
        )
        for side in (0,1)
    )
    memory.checkpoint("ghost-facet-bases-built")
    element_sizes = _cell_diameters(mesh)
    facet_cells = mesh.f2t[:, np.asarray(ghost_facets)]
    mean_cubed_facet_size = 0.5 * (
        element_sizes[facet_cells[0]] ** 3
        + element_sizes[facet_cells[1]] ** 3
    )
    ghost_gamma = (
        config.ghost_penalty_scale
        * (material.lame_lambda + material.lame_mu)
        * mean_cubed_facet_size
    )[:, None]

    @skfemntv.BilinearForm
    def ghost_penalty(trial, test, parameters):
        trial_jump = jump(parameters, normal_grad(parameters, trial))
        test_jump = jump(parameters, normal_grad(parameters, test))
        return parameters.gamma * trial_jump * test_jump

    ghost_matrix = skfemntv.asm(
        ghost_penalty,
        ghost_bases,
        ghost_bases,
        gamma=ghost_gamma,
        cache_native_assemblers=False,
        memory_limit_bytes=config.assembly_memory_limit_bytes,
    )
    del ghost_bases, element_sizes, facet_cells, mean_cubed_facet_size, ghost_gamma
    memory.checkpoint("ghost-penalty-assembled")

    boundary_facets = mesh.boundary_facets()
    facet_nodes = mesh.facets[:, boundary_facets]
    radii = np.sqrt(
        mesh.p[0, facet_nodes] ** 2 + mesh.p[1, facet_nodes] ** 2
    )
    inner_facets = boundary_facets[
        np.all(
            np.abs(radii - config.inner_radius) <= config.radius_tolerance,
            axis=0,
        )
    ]
    if inner_facets.size == 0:
        raise ValueError(
            "no inner cylindrical facets found; check mesh radius and tolerance"
        )
    inner_basis = skfemntv.FacetBasis(
        mesh,
        displacement_element,
        facets=inner_facets,
        intorder=config.surface_quadrature_order,
    )

    coordinates = inner_basis.global_coordinates
    traction = np.zeros_like(coordinates)
    traction[..., 0] = -config.traction_scale * coordinates[..., 1]
    traction[..., 1] = config.traction_scale * coordinates[..., 0]

    @skfemntv.LinearForm
    def torsional_traction(test, parameters):
        return dot(parameters.traction, test)

    load = skfemntv.asm(
        torsional_traction,
        inner_basis,
        traction=traction,
        num_threads=config.threads,
    )
    del coordinates, traction
    memory.checkpoint("traction-assembled")

    fixed_nodes = _reference_dirichlet_nodes(mesh)
    if fixed_nodes.size == 0:
        raise ValueError("the mesh has no nodes on the five reference supports")
    isolated_cells = _isolated_active_cells(mesh, active_cells, fixed_nodes)
    quadrature_isolated = np.isin(cut_basis.cells, isolated_cells)[:, None]

    @skfemntv.BilinearForm
    def isolated_volume_term(trial, test, parameters):
        return parameters.isolated * dot(trial, test)

    isolated_matrix = (
        skfemntv.asm(
            isolated_volume_term,
            cut_basis,
            isolated=config.isolated_volume_penalty * quadrature_isolated,
            num_threads=config.threads,
        )
        if isolated_cells.size
        else csr_matrix((full_basis.N, full_basis.N))
    )
    del quadrature_isolated
    memory.checkpoint("isolated-cell-penalty-assembled")

    active_dofs = np.unique(full_basis.element_dofs[:, active_cells])
    fixed_dofs = np.intersect1d(
        full_basis.get_dofs(nodes=fixed_nodes).all(), active_dofs
    )
    free_dofs = np.setdiff1d(active_dofs, fixed_dofs, assume_unique=True)
    system = stiffness + ghost_matrix + isolated_matrix
    reduced_system = system[free_dofs][:, free_dofs]
    memory.checkpoint("system-reduced")
    if config.reference_state_vtu is not None:
        reference_solution = _reference_state_vector(
            config.reference_state_vtu, mesh, active_cells, full_basis
        )
        reference_free = reference_solution[free_dofs]
        reference_residual = (
            reduced_system @ reference_free - load[free_dofs]
        )
        full_reference_residual = np.zeros(full_basis.N, dtype=float)
        full_reference_residual[free_dofs] = reference_residual
        residual_component_norms = tuple(
            float(np.linalg.norm(full_reference_residual[full_basis.nodal_dofs[axis]]))
            for axis in range(3)
        )
        residual_by_node = np.linalg.norm(
            full_reference_residual[full_basis.nodal_dofs], axis=0
        )
        largest_residual_nodes = np.argsort(residual_by_node)[-5:][::-1]
        largest_residual_text = ";".join(
            f"{mesh.p[:, node].tolist()}:{residual_by_node[node]:.3e}"
            for node in largest_residual_nodes
        )
        reference_residual_relative = float(
            np.linalg.norm(reference_residual)
            / max(np.linalg.norm(load[free_dofs]), 1.0e-30)
        )
        reference_energy = float(
            reference_solution @ (stiffness @ reference_solution)
        )
        reference_ghost_action = ghost_matrix @ reference_solution
        reference_isolated_action = isolated_matrix @ reference_solution
        reference_volume_residual = (
            stiffness @ reference_solution - load
        )
        load_norm = max(float(np.linalg.norm(load[free_dofs])), 1.0e-30)
        reference_work = float(load @ reference_solution)
        print(
            "[wheel-reference-state] "
            f"native_equilibrium_relative_residual={reference_residual_relative:.6e} "
            f"residual_xyz_l2={residual_component_norms[0]:.6e},"
            f"{residual_component_norms[1]:.6e},"
            f"{residual_component_norms[2]:.6e} "
            f"largest_residual_nodes_xyz_norm={largest_residual_text} "
            f"volume_residual_over_load={np.linalg.norm(reference_volume_residual[free_dofs])/load_norm:.6e} "
            f"ghost_action_over_load={np.linalg.norm(reference_ghost_action[free_dofs])/load_norm:.6e} "
            f"isolated_action_over_load={np.linalg.norm(reference_isolated_action[free_dofs])/load_norm:.6e} "
            f"ghost_energy={float(reference_solution @ reference_ghost_action):.12g} "
            f"isolated_energy={float(reference_solution @ reference_isolated_action):.12g} "
            f"native_energy_on_gridap_state={reference_energy:.12g} "
            f"native_load_work_on_gridap_state={reference_work:.12g} "
            f"active_gridap_u_max={np.linalg.norm(reference_solution):.12g}",
            flush=True,
        )
    if config.stop_after_assembly:
        print(
            "[wheel-profile] stopped before solve "
            f"reduced_shape={reduced_system.shape} "
            f"reduced_nnz={reduced_system.nnz} "
            f"stiffness_nnz={stiffness.nnz} "
            f"ghost_nnz={ghost_matrix.nnz}",
            flush=True,
        )
        return
    if config.solver == "scipy":
        solution_free = spsolve(reduced_system.tocsc(), load[free_dofs])
        solver_diagnostics = (0, float("nan"), 1)
    else:
        solution_free, solver_diagnostics = _solve_petsc(
            reduced_system,
            load[free_dofs],
            config.solver,
            near_nullspace=(
                _elasticity_near_nullspace(
                    mesh,
                    full_basis.nodal_dofs,
                    free_dofs,
                )
                if config.solver == "petsc-gamg"
                else None
            ),
        )
        if solver_diagnostics[2] <= 0:
            raise RuntimeError(
                "PETSc failed to converge: "
                f"solver={config.solver} iterations={solver_diagnostics[0]} "
                f"relative_residual={solver_diagnostics[1]:.3e} "
                f"converged_reason={solver_diagnostics[2]}"
            )
    memory.checkpoint("linear-solve-complete")
    solution = np.zeros(full_basis.N)
    solution[free_dofs] = solution_free
    if not np.all(np.isfinite(solution[free_dofs])):
        raise FloatingPointError("wheel solve produced non-finite displacements")
    if config.adjoint_output_path is not None:
        # GridapTopOpt differentiates J(u, phi) through the state map.  For
        # J = u.T K_physical u, its state derivative is 2 K_physical u; the
        # adjoint uses the full stabilized state operator.
        adjoint_rhs = 2.0 * (stiffness @ solution)
        if config.solver == "scipy":
            adjoint_free = spsolve(
                reduced_system.T.tocsc(), adjoint_rhs[free_dofs]
            )
            adjoint_diagnostics = (0, float("nan"), 1)
        else:
            adjoint_free, adjoint_diagnostics = _solve_petsc(
                reduced_system.T.tocsr(),
                adjoint_rhs[free_dofs],
                config.solver,
                near_nullspace=(
                    _elasticity_near_nullspace(
                        mesh,
                        full_basis.nodal_dofs,
                        free_dofs,
                    )
                    if config.solver == "petsc-gamg"
                    else None
                ),
            )
            if adjoint_diagnostics[2] <= 0:
                raise RuntimeError(
                    "PETSc adjoint failed to converge: "
                    f"solver={config.solver} "
                    f"iterations={adjoint_diagnostics[0]} "
                    f"relative_residual={adjoint_diagnostics[1]:.3e} "
                    f"converged_reason={adjoint_diagnostics[2]}"
                )
        adjoint_solution = np.zeros(full_basis.N)
        adjoint_solution[free_dofs] = adjoint_free
        config.adjoint_output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            config.adjoint_output_path,
            adjoint_displacement=adjoint_solution[
                full_basis.nodal_dofs
            ].T,
        )
        print(
            "[wheel-adjoint] "
            f"output={config.adjoint_output_path} "
            f"solver={config.solver} iterations={adjoint_diagnostics[0]} "
            f"relative_residual={adjoint_diagnostics[1]:.3e} "
            f"converged_reason={adjoint_diagnostics[2]}",
            flush=True,
        )
    if config.reference_state_vtu is not None:
        active_nodes = np.unique(mesh.t[:, active_cells])
        native_active_displacement = solution[
            full_basis.nodal_dofs[:, active_nodes]
        ]
        reference_active_displacement = reference_solution[
            full_basis.nodal_dofs[:, active_nodes]
        ]
        displacement_relative_l2 = float(
            np.linalg.norm(native_active_displacement-reference_active_displacement)
            / max(np.linalg.norm(reference_active_displacement), 1.0e-30)
        )
        free_displacement_relative_l2 = float(
            np.linalg.norm(solution[free_dofs]-reference_solution[free_dofs])
            / max(np.linalg.norm(reference_solution[free_dofs]), 1.0e-30)
        )
        free_difference = solution[free_dofs]-reference_solution[free_dofs]
        reference_energy = float(
            reference_solution[free_dofs]
            @ (reduced_system @ reference_solution[free_dofs])
        )
        difference_energy = float(
            free_difference @ (reduced_system @ free_difference)
        )
        energy_norm_relative_error = float(
            np.sqrt(max(difference_energy, 0.0) / max(reference_energy, 1.0e-30))
        )
        fixed_displacement_max = float(
            np.max(np.abs(reference_solution[fixed_dofs]), initial=0.0)
        )
        active_difference = np.linalg.norm(
            native_active_displacement-reference_active_displacement, axis=0
        )
        largest_difference_node = int(np.argmax(active_difference))
        native_max_displacement = float(
            np.max(np.linalg.norm(native_active_displacement, axis=0), initial=0.0)
        )
        reference_max_displacement = float(
            np.max(np.linalg.norm(reference_active_displacement, axis=0), initial=0.0)
        )
        max_displacement_relative_error = abs(
            native_max_displacement-reference_max_displacement
        ) / max(reference_max_displacement, 1.0e-30)
        print(
            "[wheel-reference-state] "
            f"active_displacement_relative_l2={displacement_relative_l2:.6e} "
            f"free_displacement_relative_l2={free_displacement_relative_l2:.6e} "
            f"stiffness_energy_norm_relative_error={energy_norm_relative_error:.6e} "
            f"reference_fixed_displacement_max={fixed_displacement_max:.6e} "
            f"largest_displacement_difference="
            f"{active_difference[largest_difference_node]:.6e}@"
            f"{mesh.p[:, active_nodes[largest_difference_node]].tolist()} "
            f"native_max_displacement={native_max_displacement:.12g} "
            f"gridap_max_displacement={reference_max_displacement:.12g} "
            f"max_displacement_relative_error={max_displacement_relative_error:.6e}",
            flush=True,
        )
    compliance = float(solution @ (stiffness @ solution))
    external_work = float(load @ solution)
    output_path = config.output_path or config.mesh_path.with_name(
        f"{config.mesh_path.stem}-initial-state.vtu"
    )
    point_displacement = solution[full_basis.nodal_dofs].T
    point_level_set = level_set.values(mesh)
    active_cell_mask = np.isin(np.arange(mesh.nelements), active_cells)
    meshio.write(
        output_path,
        meshio.Mesh(
            points=mesh.p.T,
            cells=[("tetra", mesh.t.T)],
            point_data={
                "level_set": point_level_set,
                "displacement": point_displacement,
            },
            cell_data={
                "classification": [classification.labels.astype(np.int32)],
                "active_design": [active_cell_mask.astype(np.int8)],
            },
        ),
        file_format="vtu",
    )
    memory.checkpoint("vtu-written")
    surface_x = inner_basis.global_coordinates[..., 0]
    surface_y = inner_basis.global_coordinates[..., 1]
    torque = config.traction_scale * float(
        np.sum(
            (surface_x**2 + surface_y**2) * inner_basis.dx
        )
    )
    exact_torque = (
        config.traction_scale
        * 2.0
        * np.pi
        * config.inner_radius**3
        * float(np.ptp(mesh.p[2]))
    )
    torque_relative_error = abs(torque - exact_torque) / exact_torque

    symmetry_error = float(
        np.max(np.abs((stiffness - stiffness.T).data), initial=0.0)
    )
    if not np.all(np.isfinite(stiffness.data)) or not np.all(np.isfinite(load)):
        raise FloatingPointError("wheel assembly produced non-finite values")
    print("Elastic-wheel CutFEM state solve")
    print(f"mesh: {config.mesh_path}")
    print(
        "level_set_source="
        + ("analytic-initial-wheel" if config.level_set_field is None
           else f"point-field:{config.level_set_field}")
    )
    print(
        f"tetrahedra={mesh.nelements} nodes={mesh.p.shape[1]} "
        f"displacement_dofs={full_basis.N}"
    )
    print(
        f"cut_inside_volume={cut_quadrature.weights.sum():.12g} "
        f"cut_cells={cut_basis.nelems} quadrature_points={cut_basis.npoints}"
    )
    print(
        f"material: E={material.young_modulus:g} nu={material.poisson_ratio:g} "
        f"lambda={material.lame_lambda:.12g} mu={material.lame_mu:.12g}"
    )
    print(
        f"stiffness_shape={stiffness.shape} stiffness_nnz={stiffness.nnz} "
        f"symmetry_error={symmetry_error:.3e}"
    )
    print(
        f"ghost_facets={len(ghost_facets)} ghost_penalty_nnz={ghost_matrix.nnz} "
        f"isolated_cells={len(isolated_cells)} "
        f"isolated_penalty_nnz={isolated_matrix.nnz}"
    )
    print(
        f"active_cells={len(active_cells)} active_dofs={len(active_dofs)} "
        f"fixed_support_nodes={len(fixed_nodes)} fixed_dofs={len(fixed_dofs)} "
        f"free_dofs={len(free_dofs)}"
    )
    print(
        f"inner_facets={inner_facets.size} traction_scale={config.traction_scale:g} "
        f"load_l2={np.linalg.norm(load):.12g} torque={torque:.12g} "
        f"continuum_torque={exact_torque:.12g} "
        f"torque_relative_error={torque_relative_error:.3%}"
    )
    print(
        f"linear_solve_compliance={compliance:.12g} "
        f"external_work={external_work:.12g} "
        f"work_energy_relative_gap={abs(compliance-external_work)/max(abs(external_work),1.e-30):.3e}"
    )
    print(
        f"solver={config.solver} iterations={solver_diagnostics[0]} "
        f"relative_residual={solver_diagnostics[1]:.3e} "
        f"converged_reason={solver_diagnostics[2]}"
    )
    print(f"vtu={output_path}")
    if config.adjoint_output_path is None:
        print(
            "state solve only: no shape derivative, level-set evolution, or "
            "optimization is performed"
        )
    else:
        print(
            "state and compliance-adjoint solve complete; level-set "
            "sensitivity assembly and optimization are performed by the caller"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mesh", type=Path, help="Gmsh Wheel_3d.msh tetrahedral mesh"
    )
    parser.add_argument("--quadrature-order", type=int, default=4)
    parser.add_argument("--surface-quadrature-order", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--stop-after-cut-basis",
        action="store_true",
        help="profile reference-scale geometry and basis stages without matrix assembly",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="estimate stiffness and ghost-pair assembly memory, then stop before assembly",
    )
    parser.add_argument(
        "--stop-after-assembly",
        action="store_true",
        help="assemble and reduce the system, report its size, then skip the linear solve",
    )
    parser.add_argument(
        "--assembly-memory-limit-gib",
        type=float,
        help="preflight budget for each native assembler, in GiB",
    )
    parser.add_argument(
        "--solver",
        choices=("petsc-gamg", "petsc-lu", "petsc-mumps", "scipy"),
        default="petsc-mumps",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--adjoint-output",
        type=Path,
        help="save the displacement adjoint for the compliance shape derivative",
    )
    parser.add_argument(
        "--level-set-field",
        help="use this nodal level-set field from the input mesh instead of the initial wheel field",
    )
    parser.add_argument(
        "--reference-state-vtu",
        type=Path,
        help="compare the assembled native equilibrium equation against Gridap point field 'uh'",
    )
    arguments = parser.parse_args()
    if arguments.quadrature_order < 1:
        parser.error("--quadrature-order must be positive")
    if arguments.surface_quadrature_order < 1:
        parser.error("--surface-quadrature-order must be positive")
    if arguments.threads < 1:
        parser.error("--threads must be positive")
    if (
        arguments.assembly_memory_limit_gib is not None
        and arguments.assembly_memory_limit_gib <= 0.
    ):
        parser.error("--assembly-memory-limit-gib must be positive")
    assembly_memory_limit_bytes=(
        None if arguments.assembly_memory_limit_gib is None else
        int(arguments.assembly_memory_limit_gib*1024**3)
    )
    assemble(
        WheelAssemblyConfig(
            mesh_path=arguments.mesh,
            output_path=arguments.output,
            level_set_field=arguments.level_set_field,
            reference_state_vtu=arguments.reference_state_vtu,
            quadrature_order=arguments.quadrature_order,
            surface_quadrature_order=arguments.surface_quadrature_order,
            threads=arguments.threads,
            solver=arguments.solver,
            stop_after_cut_basis=arguments.stop_after_cut_basis,
            stop_after_assembly=arguments.stop_after_assembly,
            preflight_only=arguments.preflight_only,
            assembly_memory_limit_bytes=assembly_memory_limit_bytes,
            adjoint_output_path=arguments.adjoint_output,
        )
    )


if __name__ == "__main__":
    main()
