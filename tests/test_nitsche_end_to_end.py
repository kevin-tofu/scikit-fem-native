import numpy as np
import pytest
from scipy.sparse import block_diag
from scipy.sparse.linalg import spsolve

import skfemntv


YOUNG = 100.
POISSON = 0.
STRAIN = .01
LAME = (0., YOUNG / 2.)


def _basis(mesh, element):
    return skfemntv.Basis(
        mesh, skfemntv.ElementVector(element), intorder=4
    )


def _facets(mesh, element, axis, value):
    selected = mesh.facets_satisfying(
        lambda x: np.isclose(x[axis], value), boundaries_only=True
    )
    return skfemntv.FacetBasis(
        mesh,
        skfemntv.ElementVector(element),
        facets=selected,
        intorder=4,
    )


def _stiffness(basis):
    return skfemntv.NativeAssembler.from_basis(
        basis, skfemntv.LinearElasticity(YOUNG, POISSON)
    ).assemble(np.zeros(basis.N), None).tangent


def _solve_bar(left_mesh, left_element, right_mesh, right_element, factor=10.):
    left = _basis(left_mesh, left_element)
    right = _basis(right_mesh, right_element)
    interface_x = left_mesh.p[0].max()
    left_interface = _facets(left_mesh, left_element, 0, interface_x)
    right_interface = _facets(right_mesh, right_element, 0, interface_x)
    nitsche = skfemntv.assemble_symmetric_nitsche(
        left_interface,
        right_interface,
        master_lame=LAME,
        slave_lame=LAME,
        stabilization_factor=factor,
    )
    tangent = (
        block_diag((_stiffness(left), _stiffness(right)), format="csr")
        + nitsche.matrix
    ).tocsr()
    right_boundary = _facets(
        right_mesh, right_element, 0, right_mesh.p[0].max()
    )
    right_load = skfemntv.NativeLinearForm(right_boundary).assemble(
        value=np.array([YOUNG * STRAIN, 0., 0.])
    )[0]
    load = np.concatenate((np.zeros(left.N), right_load))
    fixed_nodes = np.flatnonzero(np.isclose(left_mesh.p[0], left_mesh.p[0].min()))
    fixed = (
        3 * fixed_nodes[:, None] + np.arange(3)[None, :]
    ).ravel()
    free = np.setdiff1d(np.arange(left.N + right.N), fixed)
    displacement = np.zeros(left.N + right.N)
    displacement[free] = spsolve(tangent[free][:, free], load[free])
    exact = np.concatenate((
        np.column_stack((
            STRAIN * left_mesh.p[0],
            np.zeros((left_mesh.p.shape[1], 2)),
        )).ravel(),
        np.column_stack((
            STRAIN * right_mesh.p[0],
            np.zeros((right_mesh.p.shape[1], 2)),
        )).ravel(),
    ))
    return tangent, load, free, displacement, exact, nitsche


@pytest.mark.parametrize("level", (1, 2))
def test_nonmatching_hex8_bar_recovers_exact_uniaxial_solution(level):
    left = skfemntv.MeshHex.init_tensor(
        [0., .5], np.linspace(0., 1., level + 1),
        np.linspace(0., 1., level + 1),
    )
    right = skfemntv.MeshHex.init_tensor(
        [.5, 1.], np.linspace(0., 1., 2 * level + 1),
        np.linspace(0., 1., 2 * level + 1),
    )
    tangent, load, free, displacement, exact, nitsche = _solve_bar(
        left, skfemntv.ElementHex1(), right, skfemntv.ElementHex1()
    )
    np.testing.assert_allclose(
        displacement, exact, rtol=2e-10, atol=2e-12
    )
    np.testing.assert_allclose(
        (tangent @ displacement)[free], load[free], rtol=2e-10, atol=2e-11
    )
    assert load @ displacement == pytest.approx(YOUNG * STRAIN**2)
    assert nitsche.stabilization.minimum_penalty > 0.


@pytest.mark.parametrize(
    "mesh_factory, element_factory, tolerance",
    [
        (
            lambda: skfemntv.MeshHex20.from_mesh(skfemntv.MeshHex()),
            skfemntv.ElementHex20,
            2e-11,
        ),
        (
            lambda: skfemntv.MeshHex2.from_mesh(skfemntv.MeshHex()),
            skfemntv.ElementHex2,
            3e-11,
        ),
    ],
)
def test_high_order_two_body_bar_recovers_exact_solution(
    mesh_factory, element_factory, tolerance,
):
    left = mesh_factory()
    right_template = mesh_factory()
    right_points = right_template.p.copy()
    right_points[0] += 1.
    right = type(right_template)(right_points, right_template.t)
    _, _, _, displacement, exact, _ = _solve_bar(
        left, element_factory(), right, element_factory()
    )
    np.testing.assert_allclose(displacement, exact, rtol=tolerance, atol=2e-12)


def test_automatic_stabilization_makes_nonmatching_bar_positive_definite():
    left = skfemntv.MeshHex.init_tensor([0., .5], [0., 1.], [0., 1.])
    right = skfemntv.MeshHex.init_tensor(
        [.5, 1.], [0., .5, 1.], [0., .5, 1.]
    )
    tangent, _, free, _, _, _ = _solve_bar(
        left, skfemntv.ElementHex1(), right, skfemntv.ElementHex1(), factor=10.
    )
    eigenvalues = np.linalg.eigvalsh(tangent[free][:, free].toarray())
    assert eigenvalues[0] > 0.
    weak, _, weak_free, _, _, _ = _solve_bar(
        left, skfemntv.ElementHex1(), right, skfemntv.ElementHex1(), factor=.01
    )
    weak_eigenvalues = np.linalg.eigvalsh(
        weak[weak_free][:, weak_free].toarray()
    )
    assert weak_eigenvalues[0] < 0.
