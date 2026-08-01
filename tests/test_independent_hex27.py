import numpy as np
import skfem
from scipy.sparse.linalg import spsolve
from skfem.models.elasticity import linear_elasticity

import skfemntv


def meshes():
    linear = skfemntv.MeshHex.init_tensor(
        np.linspace(0., 1., 3),
        np.linspace(0., 1., 2),
        np.linspace(0., 1., 2),
    )
    return linear, skfemntv.MeshHex2.from_mesh(linear)


def reference_basis(linear):
    reference_mesh = skfem.MeshHex2.from_mesh(
        skfem.MeshHex(linear.p, linear.t)
    )
    return skfem.Basis(
        reference_mesh,
        skfem.ElementVector(skfem.ElementHex2()),
        intorder=4,
    )


def permutation(native, reference):
    lookup = {
        tuple(np.round(reference.doflocs[:, 3*node], 14)): node
        for node in range(reference.N // 3)
    }
    result = []
    for node in range(native.N // 3):
        other = lookup[tuple(np.round(native.mesh.p[:, node], 14))]
        result.extend(3*other + component for component in range(3))
    return np.asarray(result)


def test_independent_hex27_matches_skfem_reference():
    linear, mesh = meshes()
    basis = skfemntv.Basis(
        mesh, skfemntv.ElementVector(skfemntv.ElementHex2()), intorder=4
    )
    reference = reference_basis(linear)
    young, poisson = 37., .19
    lmbda = young*poisson/((1+poisson)*(1-2*poisson))
    mu = young/(2*(1+poisson))
    expected = linear_elasticity(Lambda=lmbda, Mu=mu).assemble(reference)
    order = permutation(basis, reference)
    expected = expected[order][:, order]
    actual = skfemntv.NativeAssembler.from_basis(
        basis, skfemntv.LinearElasticity(young, poisson)
    ).assemble(np.zeros(basis.N), None).tangent
    np.testing.assert_allclose(
        actual.toarray(), expected.toarray(), rtol=8e-12, atol=8e-12
    )


def test_hex27_reproduces_quadratic_solution():
    _, mesh = meshes()
    basis = skfemntv.Basis(
        mesh, skfemntv.ElementVector(skfemntv.ElementHex2()), intorder=4
    )
    young, poisson = 10., .25
    lmbda = young*poisson/((1+poisson)*(1-2*poisson))
    mu = young/(2*(1+poisson))
    tangent = skfemntv.NativeAssembler.from_basis(
        basis, skfemntv.LinearElasticity(young, poisson)
    ).assemble(np.zeros(basis.N), None).tangent
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
    np.testing.assert_allclose(solution, exact, rtol=3e-12, atol=3e-12)
