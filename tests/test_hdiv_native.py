import numpy as np
import pytest
import skfem
from skfem.helpers import div,dot

import skfemntv


@pytest.mark.parametrize("dimension", [2, 3])
def test_rt0_native_cell_integration_matches_scikit_fem(dimension):
    if dimension == 2:
        reference_mesh = skfem.MeshTri.init_tensor(
            np.linspace(0., 1., 3), np.linspace(0., 1., 3)
        )
        native_mesh = skfemntv.MeshTri(reference_mesh.p, reference_mesh.t)
        native_basis = skfemntv.AffineTriRT0Basis(native_mesh, intorder=4)
        reference_basis = skfem.Basis(
            reference_mesh, skfem.ElementTriRT0(), intorder=4
        )
    else:
        reference_mesh = skfem.MeshTet.init_tensor(
            np.linspace(0., 1., 2), np.linspace(0., 1., 2),
            np.linspace(0., 1., 2),
        )
        native_mesh = skfemntv.MeshTet(reference_mesh.p, reference_mesh.t)
        native_basis = skfemntv.AffineTetRT0Basis(native_mesh, intorder=4)
        reference_basis = skfem.Basis(
            reference_mesh, skfem.ElementTetRT0(), intorder=4
        )

    # The same mesh construction uses the same global facet numbering.
    np.testing.assert_array_equal(native_basis.element_dofs, reference_basis.element_dofs)
    native_assembler = skfemntv.SimplexRT0Assembler(native_basis)

    @skfem.BilinearForm
    def mass(u, v, w):
        return dot(u, v)

    @skfem.BilinearForm
    def divergence(u, v, w):
        return div(u) * div(v)

    native_mass = native_assembler.assemble(value=1.7).toarray()
    native_divergence = native_assembler.assemble(divergence=0.8).toarray()
    reference_mass = skfem.asm(mass, reference_basis) * 1.7
    reference_divergence = skfem.asm(divergence, reference_basis) * 0.8
    np.testing.assert_allclose(native_mass, reference_mass.toarray(), atol=3e-13)
    np.testing.assert_allclose(
        native_divergence, reference_divergence.toarray(), atol=3e-13
    )

    force = np.arange(1., dimension + 1.)

    @skfem.LinearForm
    def load(v, w):
        return dot(force, v)

    native_load = native_assembler.assemble_vector_load(force)
    reference_load = skfem.asm(load, reference_basis)
    np.testing.assert_allclose(native_load, reference_load, atol=3e-13)
