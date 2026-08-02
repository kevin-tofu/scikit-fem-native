import numpy as np
import skfem
from scipy.sparse.linalg import spsolve
from skfem.models.elasticity import linear_elasticity

import skfemntv


def meshes():
    axis_x = np.linspace(0., 1., 3)
    axis_yz = np.linspace(0., 1., 2)
    linear = skfemntv.MeshTet.init_tensor(axis_x, axis_yz, axis_yz)
    return linear, skfemntv.MeshTet2.from_mesh(linear)


def reference_basis(linear_mesh):
    reference_linear = skfem.MeshTet(linear_mesh.p, linear_mesh.t)
    reference_mesh = skfem.MeshTet2.from_mesh(reference_linear)
    return skfem.Basis(
        reference_mesh,
        skfem.ElementVector(skfem.ElementTetP2()),
        intorder=4,
    )


def coordinate_permutation(native_basis, reference):
    lookup = {
        tuple(np.round(reference.doflocs[:, 3*node], 14)): node
        for node in range(reference.N // 3)
    }
    permutation = []
    for node in range(native_basis.N // 3):
        key = tuple(np.round(native_basis.mesh.p[:, node], 14))
        reference_node = lookup[key]
        permutation.extend(3*reference_node + component for component in range(3))
    return np.asarray(permutation)


def test_independent_tet10_matches_skfem_reference():
    linear_mesh, mesh = meshes()
    basis = skfemntv.Basis(
        mesh, skfemntv.ElementVector(skfemntv.ElementTetP2()), intorder=4
    )
    reference = reference_basis(linear_mesh)
    young, poisson = 29., .22
    lmbda = young*poisson/((1+poisson)*(1-2*poisson))
    mu = young/(2*(1+poisson))
    expected = linear_elasticity(Lambda=lmbda, Mu=mu).assemble(reference)
    permutation = coordinate_permutation(basis, reference)
    expected = expected[permutation][:, permutation]
    actual = skfemntv.NativeAssembler.from_basis(
        basis, skfemntv.LinearElasticity(young, poisson)
    ).evaluate(np.zeros(basis.N)).tangent
    np.testing.assert_allclose(
        actual.toarray(), expected.toarray(), rtol=5e-12, atol=5e-12
    )


def test_tet10_reproduces_quadratic_manufactured_solution():
    _, mesh = meshes()
    basis = skfemntv.Basis(
        mesh, skfemntv.ElementVector(skfemntv.ElementTetP2()), intorder=4
    )
    young, poisson = 10., .25
    lmbda = young*poisson/((1+poisson)*(1-2*poisson))
    mu = young/(2*(1+poisson))
    tangent = skfemntv.NativeAssembler.from_basis(
        basis, skfemntv.LinearElasticity(young, poisson)
    ).evaluate(np.zeros(basis.N)).tangent
    body = -2*lmbda-4*mu
    load, _ = skfemntv.NativeLinearForm(basis).assemble(
        value=np.array([body, body, body])
    )
    exact = np.array([
        mesh.p[dof % 3, dof // 3]**2 for dof in range(basis.N)
    ])
    boundary_nodes = np.flatnonzero(
        np.any(np.isclose(mesh.p, 0.) | np.isclose(mesh.p, 1.), axis=0)
    )
    boundary = (
        3*boundary_nodes[:, None] + np.arange(3)[None, :]
    ).ravel()
    free = np.setdiff1d(np.arange(basis.N), boundary)
    solution = exact.copy()
    solution[free] = spsolve(
        tangent[free][:, free],
        load[free] - tangent[free][:, boundary] @ exact[boundary],
    )
    np.testing.assert_allclose(solution, exact, rtol=2e-12, atol=2e-12)
