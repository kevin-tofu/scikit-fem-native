from __future__ import annotations

import numpy as np

from skfemntv import (
    Basis,
    ElementHex20,
    ElementVector,
    Hex20LinearElasticityKernel,
    IsotropicLinearElasticity,
    MeshHex,
    MeshHex20,
    NativeCrossBilinearForm,
)


def test_hex20_kernel_returns_element_stiffness_and_material_fields() -> None:
    mesh = MeshHex20.from_mesh(MeshHex())
    points = mesh.p.T
    elements = mesh.t.T
    material = IsotropicLinearElasticity(120.0, 0.25)
    kernel = Hex20LinearElasticityKernel(
        points, elements, material, integration_order=4
    )

    assert kernel.element_stiffness.shape == (1, 60, 60)
    np.testing.assert_allclose(
        kernel.element_stiffness[0], kernel.element_stiffness[0].T, atol=1e-12
    )
    basis = Basis(
        mesh, ElementVector(ElementHex20(), dim=3), intorder=4
    )
    reference = NativeCrossBilinearForm(basis, basis).assemble_tensor(
        "gradient", "gradient", material.stiffness_tensor
    )
    np.testing.assert_allclose(
        kernel.element_stiffness[0], reference.toarray(), rtol=2e-13, atol=1e-13
    )

    gradient = np.asarray([[0.1, 0.2, -0.1], [-0.3, 0.4, 0.05], [0.2, -0.15, 0.3]])
    displacement = (gradient @ points.T).T
    response = kernel.evaluate_element(0, displacement)
    expected_strain = 0.5 * (gradient + gradient.T)
    np.testing.assert_allclose(
        response.strain,
        np.broadcast_to(expected_strain, response.strain.shape),
        rtol=3e-13,
        atol=3e-13,
    )
    np.testing.assert_allclose(
        response.stress,
        material.stress(response.strain),
        rtol=3e-13,
        atol=3e-13,
    )


def test_hex20_surface_load_kernel_returns_local_contributions() -> None:
    mesh = MeshHex20.from_mesh(MeshHex())
    kernel = Hex20LinearElasticityKernel(
        mesh.p.T,
        mesh.t.T,
        IsotropicLinearElasticity(120.0, 0.25),
        integration_order=4,
    )
    facets = mesh._facet_connectivity(mesh.boundary_facets(), full=True).T
    force = np.asarray([2.0, -3.0, 5.0])

    contributions = kernel.integrate_surface_total_force(facets, force)

    assert contributions.global_dofs.shape == (facets.shape[0], 60)
    assert contributions.local_force.shape == (facets.shape[0], 60)
    np.testing.assert_allclose(
        contributions.local_force.reshape((-1, 20, 3)).sum(axis=(0, 1)),
        force,
        atol=2e-13,
    )
