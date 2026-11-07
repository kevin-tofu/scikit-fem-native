"""Compare the native Hex ordering with scikit-fem at the FE-operator level."""

import numpy as np
import skfem
import skfemntv
from skfem.helpers import ddot as ref_ddot
from skfem.helpers import dot as ref_dot
from skfem.helpers import div as ref_div
from skfem.helpers import sym_grad as ref_sym_grad
from skfemntv.helpers import ddot as native_ddot
from skfemntv.helpers import dot as native_dot
from skfemntv.helpers import div as native_div
from skfemntv.helpers import sym_grad as native_sym_grad


def _reference_mesh_with_scitopt_hex_orientation(x, y, z):
    mesh = skfem.MeshHex.init_tensor(x, y, z)
    cells = mesh.t.copy()
    for element in range(cells.shape[1]):
        corners = mesh.p[:, cells[:, element]]
        v1 = corners[:, 1] - corners[:, 0]
        v2 = corners[:, 3] - corners[:, 0]
        v3 = corners[:, 4] - corners[:, 0]
        if np.dot(np.cross(v1, v2), v3) < 0.0:
            cells[1, element], cells[3, element] = (
                cells[3, element], cells[1, element]
            )
    return skfem.MeshHex(mesh.p, cells)


def _native_to_reference_dofs(native_basis, reference_basis):
    reference_nodes = {
        tuple(point): index
        for index, point in enumerate(reference_basis.mesh.p.T)
    }
    node_map = np.asarray([
        reference_nodes[tuple(point)] for point in native_basis.mesh.p.T
    ])
    result = np.empty(native_basis.N, dtype=np.int64)
    for component in range(native_basis.nodal_dofs.shape[0]):
        for native_node, reference_node in enumerate(node_map):
            result[native_basis.nodal_dofs[component, native_node]] = (
                reference_basis.nodal_dofs[component, reference_node]
            )
    return result


def test_hex_local_and_tensor_mesh_order_matches_scikit_fem():
    x = np.asarray([0.0, 1.0, 2.0])
    y = np.asarray([0.0, 0.5, 1.0])
    z = np.asarray([0.0, 1.0])
    native_mesh = skfemntv.MeshHex.init_tensor(x, y, z)
    reference_mesh = skfem.MeshHex.init_tensor(x, y, z)

    np.testing.assert_array_equal(native_mesh.p, reference_mesh.p)
    np.testing.assert_array_equal(native_mesh.t, reference_mesh.t)
    np.testing.assert_array_equal(
        skfemntv.ElementHex1().doflocs,
        skfem.ElementHex1().doflocs,
    )
    np.testing.assert_array_equal(
        skfemntv.ElementHex2().doflocs,
        skfem.ElementHex2().doflocs,
    )
    np.testing.assert_array_equal(
        skfemntv.MeshHex().p, skfem.MeshHex().p
    )
    np.testing.assert_array_equal(
        skfemntv.MeshHex().t, skfem.MeshHex().t
    )

    native_basis = skfemntv.Basis(native_mesh, skfemntv.ElementHex1())
    reference_basis = skfem.Basis(reference_mesh, skfem.ElementHex1())
    np.testing.assert_array_equal(
        native_basis.element_dofs, reference_basis.element_dofs
    )

    native_quadratic = skfemntv.Basis(
        skfemntv.MeshHex2.from_mesh(skfemntv.MeshHex()),
        skfemntv.ElementHex2(),
    )
    reference_quadratic = skfem.Basis(
        skfem.MeshHex2.from_mesh(skfem.MeshHex()),
        skfem.ElementHex2(),
    )
    np.testing.assert_allclose(
        native_quadratic.mesh.p[:, native_quadratic.element_dofs[:, 0]],
        reference_quadratic.mesh.p[
            :, reference_quadratic.element_dofs[:, 0]
        ],
    )


def test_tensor_hex_orderings_produce_same_vector_operators():
    x = np.asarray([0.0, 1.0, 2.0])
    y = np.asarray([0.0, 0.5, 1.0])
    z = np.asarray([0.0, 1.0])
    native_mesh = skfemntv.MeshHex.init_tensor(x, y, z)
    reference_mesh = _reference_mesh_with_scitopt_hex_orientation(x, y, z)
    native_basis = skfemntv.Basis(
        native_mesh,
        skfemntv.ElementVector(skfemntv.ElementHex1()),
        intorder=3,
    )
    reference_basis = skfem.Basis(
        reference_mesh,
        skfem.ElementVector(skfem.ElementHex1()),
        intorder=3,
    )

    @skfemntv.BilinearForm
    def native_mass(u, v, w):
        return native_dot(u, v)

    @skfem.BilinearForm
    def reference_mass(u, v, w):
        return ref_dot(u, v)

    @skfemntv.BilinearForm
    def native_elasticity(u, v, w):
        strain_u = native_sym_grad(u)
        strain_v = native_sym_grad(v)
        return 1.4 * native_ddot(strain_u, strain_v) + 2.3 * native_div(u) * native_div(v)

    @skfem.BilinearForm
    def reference_elasticity(u, v, w):
        strain_u = ref_sym_grad(u)
        strain_v = ref_sym_grad(v)
        return 1.4 * ref_ddot(strain_u, strain_v) + 2.3 * ref_div(u) * ref_div(v)

    native_to_reference = _native_to_reference_dofs(
        native_basis, reference_basis
    )
    for native_form, reference_form in (
        (native_mass, reference_mass),
        (native_elasticity, reference_elasticity),
    ):
        native_matrix = skfemntv.asm(native_form, native_basis).toarray()
        reference_matrix = skfem.asm(reference_form, reference_basis).toarray()
        np.testing.assert_allclose(
            native_matrix,
            reference_matrix[np.ix_(native_to_reference, native_to_reference)],
            rtol=5.0e-13,
            atol=5.0e-13,
        )
