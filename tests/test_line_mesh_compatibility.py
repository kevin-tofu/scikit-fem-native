"""One-dimensional mesh and volume-basis compatibility tests."""

import numpy as np
import pytest
import skfem
from skfem.helpers import ddot as skfem_ddot
from skfem.helpers import dot as skfem_dot
from skfem.helpers import grad as skfem_grad

import skfemntv
from skfemntv.helpers import ddot, dot, grad
from skfemntv import utils as native_utils


def test_meshline_topology_refinement_and_restriction_match_skfem():
    native = skfemntv.MeshLine.init_tensor([0., .3, 1.]).with_boundaries({
        "left": lambda x: np.isclose(x[0], 0.),
        "right": lambda x: np.isclose(x[0], 1.),
    }).with_subdomains({"first": lambda x: x[0] < .5})
    reference = skfem.MeshLine.init_tensor(np.array([0., .3, 1.])).with_boundaries({
        "left": lambda x: np.isclose(x[0], 0.),
        "right": lambda x: np.isclose(x[0], 1.),
    }).with_subdomains({"first": lambda x: x[0] < .5})
    np.testing.assert_array_equal(native.facets, reference.facets)
    np.testing.assert_array_equal(native.boundary_facets(), reference.boundary_facets())
    np.testing.assert_array_equal(native.boundaries["left"], reference.boundaries["left"])
    refined = native.refined(2)
    expected_refined = reference.refined(2)
    np.testing.assert_allclose(refined.p, expected_refined.p)
    np.testing.assert_array_equal(refined.t, expected_refined.t)
    np.testing.assert_array_equal(
        refined.subdomains["first"], expected_refined.subdomains["first"]
    )
    native_restricted = native.restrict([1])
    reference_restricted = reference.restrict([1])
    np.testing.assert_allclose(native_restricted.p, reference_restricted.p)
    np.testing.assert_array_equal(native_restricted.t, reference_restricted.t)
    np.testing.assert_array_equal(native.remove_elements([0]).t, reference.remove_elements([0]).t)


@pytest.mark.parametrize("order", [0, 1, 2])
def test_line_basis_geometry_projection_and_point_evaluation_match_skfem(order):
    native_mesh = skfemntv.MeshLine.init_tensor(np.linspace(0., 1., 5)).with_boundaries({
        "left": lambda x: np.isclose(x[0], 0.),
        "right": lambda x: np.isclose(x[0], 1.),
    })
    reference_mesh = skfem.MeshLine(native_mesh.p, native_mesh.t).with_boundaries({
        "left": lambda x: np.isclose(x[0], 0.),
        "right": lambda x: np.isclose(x[0], 1.),
    })
    native_element = (skfemntv.ElementLineP0() if order == 0 else
                      skfemntv.ElementLineP1() if order == 1 else
                      skfemntv.ElementLineP2())
    reference_element = (skfem.ElementLineP0() if order == 0 else
                         skfem.ElementLineP1() if order == 1 else
                         skfem.ElementLineP2())
    native = skfemntv.Basis(native_mesh, native_element, intorder=4)
    reference = skfem.Basis(reference_mesh, reference_element, intorder=4)
    np.testing.assert_array_equal(native.element_dofs, reference.element_dofs)
    np.testing.assert_allclose(native.doflocs, reference.doflocs)
    np.testing.assert_allclose(native.dx, reference.dx)
    np.testing.assert_allclose(
        native.project(lambda x: x[0] ** 2 + x[0]),
        reference.project(lambda x: x[0] ** 2 + x[0]),
        rtol=2e-12, atol=2e-12,
    )
    points = np.array([[.1, .26, .7, .95]])
    if order == 0:
        coefficients = np.arange(native.N, dtype=float) + .5
        actual = native.evaluate_at(points, coefficients)
        expected = reference.probes(points) @ coefficients
        np.testing.assert_allclose(actual.value, expected, atol=2e-13)
        np.testing.assert_allclose(
            native.probes(points) @ coefficients, expected, atol=2e-13,
        )
    else:
        coefficients = native.doflocs[0] ** 2
        actual = native.evaluate_at(points, coefficients)
        expected = reference.probes(points) @ coefficients
        np.testing.assert_allclose(actual.value, expected, atol=2e-13)
        np.testing.assert_allclose(
            native.probes(points) @ coefficients, expected, atol=2e-13
        )
        if order == 2:
            np.testing.assert_allclose(actual.value, points[0] ** 2, atol=2e-13)
            np.testing.assert_allclose(actual.grad[0], 2. * points[0], atol=2e-13)
            native_plot, native_values = native.refinterp(coefficients, nrefs=2)
            reference_plot, reference_values = reference.refinterp(
                coefficients, Nrefs=2,
            )
            np.testing.assert_allclose(native_plot.p, reference_plot.p, atol=2e-11)
            np.testing.assert_array_equal(native_plot.t, reference_plot.t)
            np.testing.assert_allclose(
                native_values, reference_values, rtol=3e-13, atol=3e-11,
            )
        np.testing.assert_array_equal(native.get_dofs().all(), reference.get_dofs().all())
        np.testing.assert_array_equal(
            native.get_dofs(facets="left").all(),
            reference.get_dofs(facets="left").all(),
        )


@pytest.mark.parametrize("order", [0, 1, 2])
def test_line_stiffness_and_mass_assembly_match_skfem(order):
    native_mesh = skfemntv.MeshLine.init_tensor(np.linspace(0., 1., 6))
    reference_mesh = skfem.MeshLine(native_mesh.p, native_mesh.t)
    native_element = (skfemntv.ElementLineP0() if order == 0 else
                      skfemntv.ElementLineP1() if order == 1 else
                      skfemntv.ElementLineP2())
    reference_element = (skfem.ElementLineP0() if order == 0 else
                         skfem.ElementLineP1() if order == 1 else
                         skfem.ElementLineP2())
    native = skfemntv.Basis(native_mesh, native_element, intorder=5)
    reference = skfem.Basis(reference_mesh, reference_element, intorder=5)

    @skfemntv.BilinearForm
    def native_stiffness(u, v, w):
        return ddot(grad(u), grad(v))

    @skfemntv.BilinearForm
    def native_mass(u, v, w):
        return dot(u, v)

    @skfem.BilinearForm
    def reference_stiffness(u, v, w):
        return skfem_dot(skfem_grad(u), skfem_grad(v))

    @skfem.BilinearForm
    def reference_mass(u, v, w):
        return u * v

    np.testing.assert_allclose(
        skfemntv.asm(native_stiffness, native).toarray(),
        skfem.asm(reference_stiffness, reference).toarray(),
        rtol=2e-13, atol=2e-13,
    )
    np.testing.assert_allclose(
        skfemntv.asm(native_mass, native).toarray(),
        skfem.asm(reference_mass, reference).toarray(),
        rtol=2e-13, atol=2e-13,
    )


@pytest.mark.parametrize("order", [1, 2])
def test_line_poisson_workflow_matches_skfem(order):
    points = np.linspace(0., 1., 9)
    native = skfemntv.Basis(
        skfemntv.MeshLine.init_tensor(points),
        skfemntv.ElementLineP1() if order == 1 else skfemntv.ElementLineP2(),
        intorder=5,
    )
    reference = skfem.Basis(
        skfem.MeshLine.init_tensor(points),
        skfem.ElementLineP1() if order == 1 else skfem.ElementLineP2(), intorder=5,
    )

    @skfemntv.BilinearForm
    def native_stiffness(u, v, w):
        return ddot(grad(u), grad(v))

    @skfem.BilinearForm
    def reference_stiffness(u, v, w):
        return skfem_dot(skfem_grad(u), skfem_grad(v))

    @skfemntv.LinearForm
    def native_load(v, w):
        return dot(w.x[0] * 0. + 1., v)

    @skfem.LinearForm
    def reference_load(v, w):
        return v

    native_matrix = skfemntv.asm(native_stiffness, native)
    reference_matrix = skfem.asm(reference_stiffness, reference)
    native_rhs = skfemntv.asm(native_load, native)
    reference_rhs = skfem.asm(reference_load, reference)
    native_dofs = native.get_dofs().all()
    reference_dofs = reference.get_dofs().all()
    native_solution = native_utils.solve(*native_utils.condense(
        native_matrix, native_rhs, D=native_dofs,
    ))
    reference_solution = skfem.solve(*skfem.condense(
        reference_matrix, reference_rhs, D=reference_dofs,
    ))
    np.testing.assert_allclose(native_solution, reference_solution, atol=2e-13)
    np.testing.assert_allclose(
        native_solution, native.doflocs[0] * (1. - native.doflocs[0]) / 2.,
        atol=2e-13,
    )


@pytest.mark.parametrize("order", [0, 1, 2])
def test_line_endpoint_facet_basis_assembly_matches_skfem(order):
    points = np.array([0., .3, 1.])
    native_mesh = skfemntv.MeshLine.init_tensor(points).with_boundaries({
        "ends": lambda x: np.isclose(x[0], 0.) | np.isclose(x[0], 1.),
    })
    reference_mesh = skfem.MeshLine.init_tensor(points).with_boundaries({
        "ends": lambda x: np.isclose(x[0], 0.) | np.isclose(x[0], 1.),
    })
    native_element = (skfemntv.ElementLineP0() if order == 0 else
                      skfemntv.ElementLineP1() if order == 1 else
                      skfemntv.ElementLineP2())
    reference_element = (skfem.ElementLineP0() if order == 0 else
                         skfem.ElementLineP1() if order == 1 else
                         skfem.ElementLineP2())
    native = skfemntv.FacetBasis(native_mesh, native_element, facets="ends")
    reference = skfem.FacetBasis(reference_mesh, reference_element, facets="ends")

    np.testing.assert_array_equal(native.facet_ids, reference.find)
    np.testing.assert_array_equal(native.parent_elements, reference.tind)
    np.testing.assert_array_equal(native.element_dofs, reference.element_dofs)
    np.testing.assert_allclose(native.dx, reference.dx)
    np.testing.assert_allclose(
        native.global_coordinates,
        np.asarray(reference.global_coordinates()).transpose(1, 2, 0),
    )
    np.testing.assert_allclose(
        native.normals,
        reference.normals.transpose(1, 2, 0),
    )
    np.testing.assert_allclose(
        native.tabulated_shape,
        np.stack([np.asarray(field[0]) for field in reference.basis], axis=-1),
    )
    np.testing.assert_allclose(
        native.tabulated_gradients[..., 0],
        np.stack([field[0].grad[0] for field in reference.basis], axis=-1),
    )

    @skfemntv.BilinearForm
    def native_boundary_mass(u, v, w):
        return dot(u, v)

    @skfem.BilinearForm
    def reference_boundary_mass(u, v, w):
        return u * v

    @skfemntv.LinearForm
    def native_boundary_load(v, w):
        return dot(w.x[0] * 0. + 1., v)

    @skfem.LinearForm
    def reference_boundary_load(v, w):
        return v

    np.testing.assert_allclose(
        skfemntv.asm(native_boundary_mass, native).toarray(),
        skfem.asm(reference_boundary_mass, reference).toarray(),
    )
    np.testing.assert_allclose(
        skfemntv.asm(native_boundary_load, native),
        skfem.asm(reference_boundary_load, reference),
    )


@pytest.mark.parametrize("side", [0, 1])
def test_line_interior_facet_traces_match_skfem(side):
    points = np.array([0., .3, 1.])
    native_mesh = skfemntv.MeshLine.init_tensor(points)
    reference_mesh = skfem.MeshLine.init_tensor(points)
    native = skfemntv.InteriorFacetBasis(
        native_mesh, skfemntv.ElementLineP1(), side=side,
    )
    reference = skfem.InteriorFacetBasis(
        reference_mesh, skfem.ElementLineP1(), side=side,
    )
    np.testing.assert_array_equal(native.facet_ids, reference.find)
    np.testing.assert_array_equal(native.parent_elements, reference.tind)
    np.testing.assert_array_equal(native.element_dofs, reference.element_dofs)
    np.testing.assert_allclose(native.normals, reference.normals.transpose(1, 2, 0))
    np.testing.assert_allclose(
        native.tabulated_shape,
        np.stack([np.asarray(field[0]) for field in reference.basis], axis=-1),
    )


def test_line_poisson_neumann_boundary_workflow_matches_skfem():
    points = np.linspace(0., 1., 9)
    native_mesh = skfemntv.MeshLine.init_tensor(points).with_boundaries({
        "left": lambda x: np.isclose(x[0], 0.),
        "right": lambda x: np.isclose(x[0], 1.),
    })
    reference_mesh = skfem.MeshLine.init_tensor(points).with_boundaries({
        "left": lambda x: np.isclose(x[0], 0.),
        "right": lambda x: np.isclose(x[0], 1.),
    })
    native_basis = skfemntv.Basis(
        native_mesh, skfemntv.ElementLineP1(), intorder=4,
    )
    reference_basis = skfem.Basis(
        reference_mesh, skfem.ElementLineP1(), intorder=4,
    )
    native_right = skfemntv.FacetBasis(
        native_mesh, skfemntv.ElementLineP1(), facets="right",
    )
    reference_right = skfem.FacetBasis(
        reference_mesh, skfem.ElementLineP1(), facets="right",
    )

    @skfemntv.BilinearForm
    def native_stiffness(u, v, w):
        return ddot(grad(u), grad(v))

    @skfem.BilinearForm
    def reference_stiffness(u, v, w):
        return skfem_dot(skfem_grad(u), skfem_grad(v))

    @skfemntv.LinearForm
    def native_flux(v, w):
        return dot(w.n, v)

    @skfem.LinearForm
    def reference_flux(v, w):
        return skfem_dot(w.n, v)

    native_matrix = skfemntv.asm(native_stiffness, native_basis)
    reference_matrix = skfem.asm(reference_stiffness, reference_basis)
    native_rhs = skfemntv.asm(native_flux, native_right)
    reference_rhs = skfem.asm(reference_flux, reference_right)
    native_dofs = native_basis.get_dofs(facets="left").all()
    reference_dofs = reference_basis.get_dofs(facets="left").all()
    native_solution = native_utils.solve(*native_utils.condense(
        native_matrix, native_rhs, D=native_dofs,
    ))
    reference_solution = skfem.solve(*skfem.condense(
        reference_matrix, reference_rhs, D=reference_dofs,
    ))
    np.testing.assert_allclose(native_rhs, reference_rhs, atol=2e-13)
    np.testing.assert_allclose(native_solution, reference_solution, atol=2e-13)
    np.testing.assert_allclose(native_solution, points, atol=2e-13)
