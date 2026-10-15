import numpy as np
import pytest
import skfem
import skfemntv


@pytest.mark.parametrize(
    "mesh,element,reference_mesh,reference_element",
    [
        (skfemntv.MeshTri.init_tensor([0.0, 0.5, 1.0], [0.0, 0.5, 1.0]),
         skfemntv.ElementTriP0(), skfem.MeshTri, skfem.ElementTriP0),
        (skfemntv.MeshTri.init_tensor([0.0, 0.5, 1.0], [0.0, 0.5, 1.0]),
         skfemntv.ElementTriP1(), skfem.MeshTri, skfem.ElementTriP1),
        (skfemntv.MeshTri.init_tensor([0.0, 0.5, 1.0], [0.0, 0.5, 1.0]),
         skfemntv.ElementTriP2(), skfem.MeshTri, skfem.ElementTriP2),
        (skfemntv.MeshTet.init_tensor([0.0, 1.0], [0.0, 1.0], [0.0, 1.0]),
         skfemntv.ElementTetP0(), skfem.MeshTet, skfem.ElementTetP0),
        (skfemntv.MeshTet.init_tensor([0.0, 1.0], [0.0, 1.0], [0.0, 1.0]),
         skfemntv.ElementTetP1(), skfem.MeshTet, skfem.ElementTetP1),
        (skfemntv.MeshTet.init_tensor([0.0, 1.0], [0.0, 1.0], [0.0, 1.0]),
         skfemntv.ElementTetP2(), skfem.MeshTet, skfem.ElementTetP2),
    ],
)
def test_l2_projection_matches_scikit_fem(mesh, element, reference_mesh, reference_element):
    basis = skfemntv.Basis(mesh, element, intorder=5)
    reference = skfem.Basis(
        reference_mesh(mesh.p, mesh.t), reference_element(), intorder=5
    )

    def field(x):
        if x.shape[0] == 2:
            return x[0] ** 2 + x[0] * x[1] + 2.0 * x[1]
        return x[0] ** 2 + x[1] * x[2] + 2.0 * x[2]

    np.testing.assert_allclose(
        basis.project(field), reference.project(field), rtol=2e-11, atol=2e-11
    )


@pytest.mark.parametrize("dimension,order", [(2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2)])
def test_vector_l2_projection_matches_scikit_fem(dimension, order):
    if dimension == 2:
        mesh = skfemntv.MeshTri.init_tensor([0.0, 0.5, 1.0], [0.0, 0.5, 1.0])
        reference_mesh = skfem.MeshTri(mesh.p, mesh.t)
        native_element, reference_element = (
            (skfemntv.ElementTriP0(), skfem.ElementTriP0()),
            (skfemntv.ElementTriP1(), skfem.ElementTriP1()),
            (skfemntv.ElementTriP2(), skfem.ElementTriP2()),
        )[order]
    else:
        mesh = skfemntv.MeshTet.init_tensor([0.0, 1.0], [0.0, 1.0], [0.0, 1.0])
        reference_mesh = skfem.MeshTet(mesh.p, mesh.t)
        native_element, reference_element = (
            (skfemntv.ElementTetP0(), skfem.ElementTetP0()),
            (skfemntv.ElementTetP1(), skfem.ElementTetP1()),
            (skfemntv.ElementTetP2(), skfem.ElementTetP2()),
        )[order]
    native_basis = skfemntv.Basis(
        mesh, skfemntv.ElementVector(native_element, dim=dimension), intorder=5
    )
    reference_basis = skfem.Basis(
        reference_mesh, skfem.ElementVector(reference_element, dim=dimension), intorder=5,
    )

    def field(x):
        return np.asarray([x[i] + (i + 1) * x[0] * x[-1] for i in range(dimension)])

    np.testing.assert_allclose(
        native_basis.project(field), reference_basis.project(field),
        rtol=2e-11, atol=2e-11,
    )


def test_projection_preserves_complex_values_when_dtype_is_requested():
    mesh = skfemntv.MeshTri.init_tensor([0.0, 1.0], [0.0, 1.0])
    basis = skfemntv.Basis(mesh, skfemntv.ElementTriP1())
    result = basis.project(1.0 + 2.0j, dtype=np.complex128)
    np.testing.assert_allclose(result, 1.0 + 2.0j, atol=1e-12)


def test_projection_rejects_unimplemented_composite_paths():
    mesh = skfemntv.MeshTri.init_tensor([0.0, 1.0], [0.0, 1.0])
    basis = skfemntv.Basis(mesh, skfemntv.ElementTriP1())
    composite = skfemntv.Basis(
        mesh,
        skfemntv.ElementComposite(
            skfemntv.ElementTriP1(), skfemntv.ElementTriP1()
        ),
    )
    with pytest.raises(NotImplementedError, match="composite bases"):
        composite.project(1.0)


@pytest.mark.parametrize("dimension,order,vector", [
    (dimension, order, vector)
    for dimension in (2, 3)
    for order in (0, 1, 2)
    for vector in (False, True)
])
def test_element_subset_projection_matches_scikit_fem(
    dimension, order, vector
):
    if dimension == 2:
        mesh = skfemntv.MeshTri.init_tensor([0.0, 0.5, 1.0], [0.0, 0.5, 1.0])
        reference_mesh = skfem.MeshTri(mesh.p, mesh.t)
        native_elements = (
            skfemntv.ElementTriP0(), skfemntv.ElementTriP1(),
            skfemntv.ElementTriP2(),
        )
        reference_elements = (
            skfem.ElementTriP0(), skfem.ElementTriP1(), skfem.ElementTriP2(),
        )
    else:
        mesh = skfemntv.MeshTet.init_tensor([0.0, 1.0], [0.0, 1.0], [0.0, 1.0])
        reference_mesh = skfem.MeshTet(mesh.p, mesh.t)
        native_elements = (
            skfemntv.ElementTetP0(), skfemntv.ElementTetP1(),
            skfemntv.ElementTetP2(),
        )
        reference_elements = (
            skfem.ElementTetP0(), skfem.ElementTetP1(), skfem.ElementTetP2(),
        )
    selected = np.arange(mesh.nelements)[::2]
    mesh = mesh.with_subdomains({"selected": selected})
    reference_mesh = reference_mesh.with_subdomains({"selected": selected})
    native_element, reference_element = native_elements[order], reference_elements[order]
    if vector:
        native_element = skfemntv.ElementVector(native_element, dim=dimension)
        reference_element = skfem.ElementVector(reference_element, dim=dimension)
    native_basis = skfemntv.Basis(mesh, native_element, intorder=5)
    reference_basis = skfem.Basis(reference_mesh, reference_element, intorder=5)
    field = lambda x: (
        np.asarray([x[i] + x[0] ** 2 for i in range(dimension)])
        if vector else x[0] ** 2 + x[-1]
    )
    mask = np.zeros(mesh.nelements, dtype=bool)
    mask[selected] = True
    selectors = (
        selected,
        mask,
        "selected",
        lambda x: x[0] < 0.5,
    )
    for selector in selectors:
        np.testing.assert_allclose(
            native_basis.project(field, elements=selector),
            reference_basis.project(field, elements=selector),
            rtol=2e-10,
            atol=2e-10,
        )


def test_projection_on_with_elements_basis_matches_scikit_fem():
    mesh = skfemntv.MeshTri.init_tensor([0.0, 0.5, 1.0], [0.0, 0.5, 1.0])
    native_basis = skfemntv.Basis(
        mesh, skfemntv.ElementTriP2(), intorder=5
    ).with_elements([0, 1])
    reference_mesh = skfem.MeshTri(mesh.p, mesh.t)
    reference_basis = skfem.Basis(
        reference_mesh, skfem.ElementTriP2(), intorder=5
    ).with_elements([0, 1])
    field = lambda x: x[0] ** 2 + x[1]
    np.testing.assert_allclose(
        native_basis.project(field), reference_basis.project(field),
        rtol=2e-10, atol=2e-10,
    )
