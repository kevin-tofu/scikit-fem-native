import numpy as np
import pytest

import skfemntv
from skfemntv.helpers import ddot, grad


def _tet_basis(*, components=1, elements=None):
    mesh = skfemntv.MeshTet.init_tensor(
        np.linspace(0.0, 1.0, 4),
        np.linspace(0.0, 1.0, 3),
        np.linspace(0.0, 1.0, 2),
    )
    return skfemntv.Basis(
        mesh,
        skfemntv.ElementVector(skfemntv.ElementTetP1(), dim=components),
        elements=elements,
    )


def test_scalar_bilinear_preflight_reports_formula_based_upper_bounds():
    basis = _tet_basis()
    estimate = skfemntv.estimate_bilinear_memory(basis)
    entities = basis.dx.shape[0]
    local = basis.element_dofs.shape[0]

    assert estimate.kind == "bilinear"
    assert estimate.rows == estimate.columns == basis.N
    assert estimate.nnz_upper_bound == min(basis.N**2, entities * local**2)
    assert estimate.scatter_bytes == entities * local**2 * 8
    assert estimate.dof_map_bytes == entities * local * 8
    assert estimate.construction_peak_total_bytes_upper_bound > estimate.basis_bytes
    assert "MiB" in skfemntv.format_bytes(2**20)


def test_vector_space_increases_local_scatter_quadratically():
    scalar = skfemntv.estimate_bilinear_memory(_tet_basis(components=1))
    vector = skfemntv.estimate_bilinear_memory(_tet_basis(components=3))

    assert vector.row_local_dofs == 3 * scalar.row_local_dofs
    assert vector.scatter_bytes == 9 * scalar.scatter_bytes
    assert vector.construction_peak_incremental_bytes_upper_bound > scalar.construction_peak_incremental_bytes_upper_bound


def test_restricted_basis_estimate_uses_only_selected_entities():
    full_basis = _tet_basis()
    selected = np.arange(0, full_basis.mesh.nelements, 2)
    restricted = _tet_basis(elements=selected)
    full = skfemntv.estimate_bilinear_memory(full_basis)
    partial = skfemntv.estimate_bilinear_memory(restricted)

    assert partial.entity_count == len(selected)
    assert partial.scatter_bytes * 2 == full.scatter_bytes
    assert partial.basis_bytes < full.basis_bytes


def test_cross_and_composite_estimates_declare_cached_blocks():
    mesh = skfemntv.MeshTet()
    composite = skfemntv.Basis(
        mesh,
        skfemntv.ElementVector(skfemntv.ElementTetP1(), dim=3)
        * skfemntv.ElementTetP1(),
    )
    velocity, pressure = composite.subbases
    cross = skfemntv.estimate_bilinear_memory(velocity, pressure)
    complete = skfemntv.estimate_composite_bilinear_memory(composite)
    diagonal = skfemntv.estimate_composite_bilinear_memory(
        composite, field_pairs=((0, 0), (1, 1))
    )

    assert cross.kind == "cross-bilinear"
    assert cross.row_local_dofs == 12
    assert cross.column_local_dofs == 4
    assert complete.field_pairs == ((0, 0), (0, 1), (1, 0), (1, 1))
    assert complete.persistent_incremental_bytes_upper_bound > diagonal.persistent_incremental_bytes_upper_bound


def test_fits_in_applies_explicit_safety_factor():
    estimate = skfemntv.estimate_bilinear_memory(_tet_basis())
    peak = estimate.construction_peak_total_bytes_upper_bound
    assert estimate.fits_in(peak, safety_factor=1.0)
    assert not estimate.fits_in(peak, safety_factor=1.01)
    with pytest.raises(ValueError, match="at least 1"):
        estimate.fits_in(peak, safety_factor=0.9)


def test_native_bilinear_budget_rejects_before_native_construction():
    basis = _tet_basis(components=3)
    estimate = skfemntv.estimate_bilinear_memory(basis)
    with pytest.raises(skfemntv.AssemblyMemoryBudgetError) as captured:
        skfemntv.NativeBilinearForm(
            basis,
            memory_limit_bytes=estimate.construction_peak_total_bytes_upper_bound,
        )
    error = captured.value
    assert error.estimate == estimate
    assert error.required_bytes > error.memory_limit_bytes
    assert error.largest_allocation in str(error)
    assert "local_dofs=" in str(error)


def test_native_bilinear_budget_accepts_exact_safety_adjusted_requirement():
    basis = _tet_basis()
    estimate = skfemntv.estimate_bilinear_memory(basis)
    required = int(np.ceil(estimate.construction_peak_total_bytes_upper_bound * 1.25))
    assembler = skfemntv.NativeBilinearForm(
        basis,
        memory_limit_bytes=required,
    )
    assert assembler.memory_estimate == estimate


def test_cross_bilinear_budget_and_argument_validation():
    test_basis = _tet_basis(components=3)
    trial_basis = _tet_basis(components=1)
    estimate = skfemntv.estimate_bilinear_memory(test_basis, trial_basis)
    with pytest.raises(skfemntv.AssemblyMemoryBudgetError):
        skfemntv.NativeCrossBilinearForm(
            test_basis,
            trial_basis,
            memory_limit_bytes=estimate.construction_peak_total_bytes_upper_bound,
        )
    with pytest.raises(ValueError, match="memory_safety_factor"):
        skfemntv.NativeBilinearForm(
            test_basis,
            memory_limit_bytes=10**9,
            memory_safety_factor=0.5,
        )


def test_public_asm_enforces_budget_even_when_assembler_is_cached():
    basis = _tet_basis()

    @skfemntv.BilinearForm
    def form(u, v, _w):
        return ddot(grad(u), grad(v))

    matrix = skfemntv.asm(form, basis)
    assert matrix.shape == (basis.N, basis.N)
    estimate = skfemntv.estimate_bilinear_memory(basis)
    with pytest.raises(skfemntv.AssemblyMemoryBudgetError):
        skfemntv.asm(
            form,
            basis,
            memory_limit_bytes=estimate.construction_peak_total_bytes_upper_bound,
        )


def _cut_basis(components=1):
    mesh = skfemntv.MeshTri.init_tensor(
        np.linspace(-1.0, 1.0, 6), np.linspace(-1.0, 1.0, 5)
    )
    regular = skfemntv.Basis(
        mesh,
        skfemntv.ElementVector(skfemntv.ElementTriP1(), dim=components),
    )
    quadrature = skfemntv.LevelSet(
        lambda x: x[0] ** 2 + x[1] ** 2 - 0.7**2
    ).cut_quadrature(mesh, intorder=2)
    return skfemntv.CutCellBasis(regular, quadrature)


def test_cut_bilinear_preflight_uses_active_cells_and_flat_points():
    cut = _cut_basis(components=2)
    estimate = skfemntv.estimate_bilinear_memory(cut)
    cells = cut.active_cell_dofs.shape[0]
    local = int(np.prod(cut.active_cell_dofs.shape[1:]))

    assert estimate.kind == "cut-bilinear"
    assert estimate.entity_count == cells
    assert estimate.quadrature_points_per_entity == cut.npoints
    assert estimate.row_local_dofs == local
    assert estimate.scatter_bytes == cells * local**2 * 8
    assert estimate.dof_map_bytes == (cells * local + cells + 1) * 8


def test_cut_bilinear_and_cross_budget_guards_run_before_native_allocation():
    scalar = _cut_basis(components=1)
    vector = _cut_basis(components=2)
    estimate = skfemntv.estimate_bilinear_memory(vector)
    with pytest.raises(skfemntv.AssemblyMemoryBudgetError):
        skfemntv.NativeBilinearForm(
            vector,
            memory_limit_bytes=estimate.construction_peak_total_bytes_upper_bound,
        )
    cross = skfemntv.estimate_bilinear_memory(vector, scalar)
    assert cross.kind == "cut-cross-bilinear"
    with pytest.raises(skfemntv.AssemblyMemoryBudgetError):
        skfemntv.NativeCrossBilinearForm(
            vector,
            scalar,
            memory_limit_bytes=cross.construction_peak_total_bytes_upper_bound,
        )
