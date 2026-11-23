import numpy as np
import pytest

skfem = pytest.importorskip("skfem")
import skfemntv


def test_affine_temperature_conductivity_tangent_matches_finite_difference():
    law = skfemntv.AffineTemperatureConductivity(
        reference_conductivity=3.0,
        reference_temperature=2.0,
        relative_slope=0.05,
    )
    kernel = skfemntv.AffineTemperatureConductivityKernel(law)
    shape = np.asarray([[[0.6, 0.2, 0.2], [0.2, 0.6, 0.2]]])
    gradients = np.asarray([[[[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]],
                             [[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]]]])
    weights = np.asarray([[0.4, 0.6]])
    temperature = np.asarray([[2.3, 4.1, 5.6]])
    response = kernel.integrate(shape, gradients, weights, temperature)

    direction = np.asarray([[0.3, -0.4, 0.2]])
    step = 1.0e-6
    plus = kernel.integrate(shape, gradients, weights, temperature + step * direction)
    minus = kernel.integrate(shape, gradients, weights, temperature - step * direction)
    finite_difference = (plus.residual - minus.residual) / (2.0 * step)
    tangent_action = np.einsum("eij,ej->ei", response.tangent, direction)

    np.testing.assert_allclose(tangent_action, finite_difference, rtol=1.0e-8, atol=1.0e-10)


def test_affine_temperature_conductivity_parameter_derivative_matches_finite_difference():
    reference_conductivity = 3.0
    reference_temperature = 2.0
    relative_slope = 0.05
    step = 1.0e-6
    shape = np.asarray([[[0.6, 0.2, 0.2], [0.2, 0.6, 0.2]]])
    gradients = np.asarray([[[[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]],
                             [[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]]]])
    weights = np.asarray([[0.4, 0.6]])
    temperature = np.asarray([[2.3, 4.1, 5.6]])

    def kernel(slope):
        law = skfemntv.AffineTemperatureConductivity(
            reference_conductivity=reference_conductivity,
            reference_temperature=reference_temperature,
            relative_slope=slope,
        )
        return skfemntv.AffineTemperatureConductivityKernel(law)

    derivative = kernel(relative_slope).integrate_relative_slope_residual_derivative(
        shape, gradients, weights, temperature
    )
    plus = kernel(relative_slope + step).integrate(
        shape, gradients, weights, temperature
    ).residual
    minus = kernel(relative_slope - step).integrate(
        shape, gradients, weights, temperature
    ).residual

    np.testing.assert_allclose(
        derivative,
        (plus - minus) / (2.0 * step),
        rtol=1.0e-9,
        atol=1.0e-10,
    )


def test_kelvin_voigt_dissipation_velocity_tangent_matches_finite_difference():
    rng = np.random.default_rng(513)
    elements = 2
    quadrature = 3
    nodes = 4
    shape = rng.uniform(0.0, 1.0, size=(elements, quadrature, nodes))
    shape /= np.sum(shape, axis=2, keepdims=True)
    gradients = rng.normal(size=(elements, quadrature, nodes, 3))
    gradients -= np.mean(gradients, axis=2, keepdims=True)
    weights = rng.uniform(0.1, 0.5, size=(elements, quadrature))
    velocity = rng.normal(size=(elements, nodes, 3))
    direction = rng.normal(size=velocity.shape)
    kernel = skfemntv.KelvinVoigtDissipationKernel(
        shear_viscosity=0.7,
        bulk_viscosity=0.3,
    )
    evaluation = kernel.integrate(
        shape,
        gradients,
        weights,
        velocity,
        compute_velocity_tangent=True,
    )
    step = 1.0e-6
    plus = kernel.integrate(shape, gradients, weights, velocity + step * direction)
    minus = kernel.integrate(shape, gradients, weights, velocity - step * direction)
    assert evaluation.velocity_tangent is not None
    tangent_action = np.einsum(
        "eabk,ebk->ea", evaluation.velocity_tangent, direction, optimize=True
    )
    finite_difference = (plus.heat_source - minus.heat_source) / (2.0 * step)

    np.testing.assert_allclose(
        tangent_action, finite_difference, rtol=1.0e-8, atol=2.0e-10
    )
    assert evaluation.total_power == pytest.approx(
        float(np.sum(evaluation.heat_source)), rel=0.0, abs=0.0
    )


@pytest.mark.parametrize(
    "cell_type",
    ("tet", "hex"),
)
def test_shared_elasticity_weak_form_assembles_identically(cell_type):
    material = skfemntv.IsotropicLinearElasticity(210.0, 0.3)
    weak_form = skfemntv.isotropic_linear_elasticity_weak_form(material)

    if cell_type == "tet":
        points = np.asarray(
            ((0.0, 1.0, 0.0, 0.0),
             (0.0, 0.0, 1.0, 0.0),
             (0.0, 0.0, 0.0, 1.0)),
            dtype=np.float64,
        )
        cells = np.asarray(((0,), (1,), (2,), (3,)), dtype=np.int64)
        reference_mesh = skfem.MeshTet(points, cells)
        native_mesh = skfemntv.MeshTet(points, cells)
        reference_element = skfem.ElementTetP1()
        native_element = skfemntv.ElementTetP1()
    else:
        reference_mesh = skfem.MeshHex.init_tensor(
            np.asarray((0.0, 1.0)),
            np.asarray((0.0, 1.0)),
            np.asarray((0.0, 1.0)),
        )
        # Use the same canonical cell connectivity in both backends.
        native_mesh = skfemntv.MeshHex(reference_mesh.p, reference_mesh.t)
        reference_element = skfem.ElementHex1()
        native_element = skfemntv.ElementHex1()
    reference_basis = skfem.Basis(
        reference_mesh, skfem.ElementVector(reference_element, dim=3)
    )
    reference = skfem.asm(skfem.BilinearForm(weak_form), reference_basis).tocsr()

    native_basis = skfemntv.Basis(
        native_mesh, skfemntv.ElementVector(native_element, dim=3)
    )
    assembled = skfemntv.asm(
        skfemntv.BilinearForm(weak_form), native_basis
    ).tocsr()

    assert assembled.shape == reference.shape
    np.testing.assert_allclose(
        assembled.toarray(), reference.toarray(), rtol=2.0e-13, atol=2.0e-13
    )


def _tetrahedron_bases():
    points = np.asarray(
        ((0.0, 1.0, 0.0, 0.0),
         (0.0, 0.0, 1.0, 0.0),
         (0.0, 0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    cells = np.asarray(((0,), (1,), (2,), (3,)), dtype=np.int64)
    reference_mesh = skfem.MeshTet(points, cells)
    native_mesh = skfemntv.MeshTet(points, cells)
    return (
        skfem.Basis(reference_mesh, skfem.ElementTetP1()),
        skfemntv.Basis(native_mesh, skfemntv.ElementTetP1()),
        skfem.Basis(
            reference_mesh,
            skfem.ElementVector(skfem.ElementTetP1(), dim=3),
        ),
        skfemntv.Basis(
            native_mesh,
            skfemntv.ElementVector(skfemntv.ElementTetP1(), dim=3),
        ),
    )


def test_shared_thermal_weak_forms_assemble_identically():
    scalar_reference, scalar_native, _, _ = _tetrahedron_bases()
    material = skfemntv.IsotropicThermalMaterial(3.0, 2.0, 4.0)
    reference_forms = (
        skfem.BilinearForm(
            skfemntv.scalar_thermal_conductivity_weak_form(material)
        ),
        skfem.BilinearForm(
            skfemntv.scalar_thermal_capacity_weak_form(material)
        ),
    )
    native_forms = (
        skfemntv.BilinearForm(
            skfemntv.scalar_thermal_conductivity_weak_form(material)
        ),
        skfemntv.BilinearForm(
            skfemntv.scalar_thermal_capacity_weak_form(material)
        ),
    )
    for reference_form, native_form in zip(
        reference_forms, native_forms, strict=True
    ):
        reference = skfem.asm(reference_form, scalar_reference).tocsr()
        native = skfemntv.asm(native_form, scalar_native).tocsr()
        np.testing.assert_allclose(
            native.toarray(), reference.toarray(), rtol=1.0e-13, atol=1.0e-13
        )

    source = skfemntv.volumetric_heat_source_weak_form(5.0)
    reference_load = skfem.asm(skfem.LinearForm(source), scalar_reference)
    native_load = skfemntv.asm(
        skfemntv.LinearForm(source), scalar_native
    )
    np.testing.assert_allclose(native_load, reference_load, rtol=1.0e-13, atol=1.0e-13)

    reference_coordinates = np.asarray(
        scalar_reference.global_coordinates(), dtype=np.float64
    )
    native_coordinates = np.asarray(
        scalar_native.global_coordinates, dtype=np.float64
    )
    reference_source_values = 1.0 + reference_coordinates[0] + 2.0 * reference_coordinates[1]
    native_source_values = (
        1.0 + native_coordinates[..., 0] + 2.0 * native_coordinates[..., 1]
    )
    reference_source = skfem.asm(
        skfem.LinearForm(
            skfemntv.volumetric_heat_source_weak_form(reference_source_values)
        ),
        scalar_reference,
    )
    native_source = skfemntv.asm(
        skfemntv.LinearForm(
            skfemntv.volumetric_heat_source_weak_form(native_source_values)
        ),
        scalar_native,
    )
    np.testing.assert_allclose(
        native_source, reference_source, rtol=1.0e-13, atol=1.0e-13
    )


def test_shared_thermoelastic_coupling_assembles_identically():
    scalar_reference, scalar_native, vector_reference, vector_native = (
        _tetrahedron_bases()
    )
    weak_form = skfemntv.thermal_expansion_coupling_weak_form(2.5)
    reference = skfem.asm(
        skfem.BilinearForm(weak_form), scalar_reference, vector_reference
    ).tocsr()
    native = skfemntv.asm(
        skfemntv.BilinearForm(weak_form), scalar_native, vector_native
    ).tocsr()
    assert native.shape == reference.shape
    np.testing.assert_allclose(
        native.toarray(), reference.toarray(), rtol=1.0e-13, atol=1.0e-13
    )


def test_shared_mechanical_load_and_mass_weak_forms_assemble_identically():
    scalar_reference, scalar_native, vector_reference, vector_native = (
        _tetrahedron_bases()
    )
    reference_mass = skfem.asm(
        skfem.BilinearForm(skfemntv.isotropic_mass_weak_form(2.0)),
        vector_reference,
    ).tocsr()
    native_mass = skfemntv.asm(
        skfemntv.BilinearForm(skfemntv.isotropic_mass_weak_form(2.0)),
        vector_native,
    ).tocsr()
    np.testing.assert_allclose(
        native_mass.toarray(), reference_mass.toarray(),
        rtol=1.0e-13, atol=1.0e-13,
    )

    force_density = np.asarray((1.0, -2.0, 3.0), dtype=np.float64)
    reference_body_force = skfem.asm(
        skfem.LinearForm(skfemntv.body_force_weak_form(force_density)),
        vector_reference,
    )
    native_body_force = skfemntv.asm(
        skfemntv.LinearForm(skfemntv.body_force_weak_form(force_density)),
        vector_native,
    )
    np.testing.assert_allclose(
        native_body_force, reference_body_force, rtol=1.0e-13, atol=1.0e-13
    )

    reference_surface_basis = skfem.FacetBasis(
        vector_reference.mesh, vector_reference.elem
    )
    native_surface_basis = skfemntv.FacetBasis(
        vector_native.mesh, vector_native.elem
    )
    traction = np.asarray((0.5, 1.5, -0.25), dtype=np.float64)
    reference_traction = skfem.asm(
        skfem.LinearForm(skfemntv.surface_traction_weak_form(traction)),
        reference_surface_basis,
    )
    native_traction = skfemntv.asm(
        skfemntv.LinearForm(skfemntv.surface_traction_weak_form(traction)),
        native_surface_basis,
    )
    np.testing.assert_allclose(
        native_traction, reference_traction, rtol=1.0e-13, atol=1.0e-13
    )

    reference_pressure = skfem.asm(
        skfem.LinearForm(skfemntv.normal_pressure_weak_form(4.0)),
        reference_surface_basis,
    )
    native_pressure = skfemntv.asm(
        skfemntv.LinearForm(skfemntv.normal_pressure_weak_form(4.0)),
        native_surface_basis,
    )
    np.testing.assert_allclose(
        native_pressure, reference_pressure, rtol=1.0e-13, atol=1.0e-13
    )
