import numpy as np
import pytest
import skfem
from skfem.helpers import dot as reference_dot,mul as reference_mul

import skfemntv
from skfemntv.helpers import dot,mul


@pytest.mark.parametrize("family", ["tri", "tet"])
def test_rt0_tensor_mass_matches_scikit_fem(family):
    if family == "tri":
        mesh=skfemntv.MeshTri.init_tensor(
            [0.,.5,1.],[0.,.5,1.]
        )
        reference_mesh=skfem.MeshTri(mesh.p,mesh.t)
        element=skfemntv.ElementTriRT0()
        reference_element=skfem.ElementTriRT0()
    else:
        mesh=skfemntv.MeshTet.init_tensor([0.,1.],[0.,1.],[0.,1.])
        reference_mesh=skfem.MeshTet(mesh.p,mesh.t)
        element=skfemntv.ElementTetRT0()
        reference_element=skfem.ElementTetRT0()
    dimension=mesh.dim()
    basis=skfemntv.Basis(mesh,element,intorder=5)
    reference_basis=skfem.Basis(
        reference_mesh,reference_element,quadrature=basis.quadrature
    )

    @skfemntv.BilinearForm
    def tensor_mass(u,v,w):
        return dot(mul(w.tensor,u),v)

    @skfem.BilinearForm
    def reference_tensor_mass(u,v,w):
        return reference_dot(reference_mul(w.tensor,u),v)

    constant=np.eye(dimension)*1.3
    constant[0,-1]+=0.17
    constant[-1,0]-=0.09
    actual=skfemntv.asm(tensor_mass,basis,tensor=constant)
    expected=skfem.asm(reference_tensor_mass,reference_basis,tensor=constant)
    np.testing.assert_allclose(
        actual.toarray(),expected.toarray(),rtol=4e-13,atol=4e-13
    )

    x=np.moveaxis(basis.global_coordinates,-1,0)
    coefficient=np.empty((dimension,dimension)+basis.dx.shape)
    for i in range(dimension):
        for j in range(dimension):
            coefficient[i,j]=(1.1 if i==j else .13*(i+1)-.07*j)
            coefficient[i,j]+=0.08*(i+1)*x[j]
    actual=skfemntv.asm(tensor_mass,basis,tensor=coefficient)
    expected=skfem.asm(
        reference_tensor_mass,reference_basis,tensor=coefficient
    )
    np.testing.assert_allclose(
        actual.toarray(),expected.toarray(),rtol=5e-13,atol=5e-13
    )


def test_rt0_tensor_mass_can_be_combined_with_scalar_mass():
    mesh=skfemntv.MeshTri.init_tensor([0.,1.],[0.,1.])
    basis=skfemntv.Basis(mesh,skfemntv.ElementTriRT0(),intorder=4)
    reference=skfem.Basis(
        skfem.MeshTri(mesh.p,mesh.t),skfem.ElementTriRT0(),
        quadrature=basis.quadrature,
    )
    tensor=np.array([[1.4,.2],[-.1,.7]])

    @skfemntv.BilinearForm
    def native(u,v,w):
        return dot(mul(w.tensor,u),v)+.25*dot(u,v)

    @skfem.BilinearForm
    def upstream(u,v,w):
        return reference_dot(reference_mul(w.tensor,u),v)+.25*reference_dot(u,v)

    np.testing.assert_allclose(
        skfemntv.asm(native,basis,tensor=tensor).toarray(),
        skfem.asm(upstream,reference,tensor=tensor).toarray(),
        rtol=3e-13,atol=3e-13,
    )
