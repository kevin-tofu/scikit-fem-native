import numpy as np
import skfem
from skfem.helpers import ddot as reference_ddot
from skfem.helpers import grad as reference_grad

import skfn
from skfn.helpers import ddot,grad


def test_volume_functional_of_interpolated_field_matches_skfem():
    linear=skfn.MeshTet.init_tensor(
        np.linspace(0.,1.,3),
        np.linspace(0.,1.,2),
        np.linspace(0.,1.,2),
    )
    mesh=skfn.MeshTet2.from_mesh(linear)
    basis=skfn.Basis(
        mesh,skfn.ElementVector(skfn.ElementTetP2()),intorder=4
    )
    component=np.arange(basis.N)%3
    coefficients=(
        (component+1.)*basis.doflocs[0]
        +(.2*component-.3)*basis.doflocs[1]
        +.1*basis.doflocs[2]
    )
    field=basis.interpolate(coefficients)

    @skfn.Functional
    def energy(w):
        return (
            .5*ddot(grad(w.u),grad(w.u))
            +(1.+w.x[0])*np.sum(w.u*w.u,axis=0)
        )

    actual=skfn.asm(energy,basis,u=field)

    reference_mesh=skfem.MeshTet2.from_mesh(
        skfem.MeshTet(linear.p,linear.t)
    )
    reference_basis=skfem.Basis(
        reference_mesh,
        skfem.ElementVector(skfem.ElementTetP2()),
        intorder=4,
    )
    reference_component=np.arange(reference_basis.N)%3
    reference_coefficients=(
        (reference_component+1.)*reference_basis.doflocs[0]
        +(.2*reference_component-.3)*reference_basis.doflocs[1]
        +.1*reference_basis.doflocs[2]
    )
    reference_field=reference_basis.interpolate(reference_coefficients)

    @skfem.Functional
    def reference(w):
        return (
            .5*reference_ddot(
                reference_grad(w.u),reference_grad(w.u)
            )
            +(1.+w.x[0])*np.sum(w.u*w.u,axis=0)
        )

    expected=skfem.asm(reference,reference_basis,u=reference_field)
    np.testing.assert_allclose(actual,expected,rtol=2e-12,atol=2e-12)


def test_facet_functional_and_interpolation_match_skfem():
    mesh=skfn.MeshHex()
    basis=skfn.FacetBasis(
        mesh,skfn.ElementVector(skfn.ElementHex1()),intorder=4
    )
    coefficients=np.arange(basis.N,dtype=float)/basis.N
    field=basis.interpolate(coefficients)

    @skfn.Functional
    def surface(w):
        return (
            1.+w.x[0]+w.n[0]**2
            +np.sum(w.u*w.u,axis=0)
        )

    actual=skfn.asm(surface,basis,u=field)

    reference_mesh=skfem.MeshHex(mesh.p,mesh.t)
    reference_basis=skfem.FacetBasis(
        reference_mesh,
        skfem.ElementVector(skfem.ElementHex1()),
        intorder=4,
    )
    reference_field=reference_basis.interpolate(coefficients)

    @skfem.Functional
    def reference(w):
        return (
            1.+w.x[0]+w.n[0]**2
            +np.sum(w.u*w.u,axis=0)
        )

    expected=skfem.asm(reference,reference_basis,u=reference_field)
    np.testing.assert_allclose(actual,expected,rtol=2e-12,atol=2e-12)


def test_composite_functional_accepts_split_fields():
    linear=skfn.MeshTet()
    mesh=skfn.MeshTet2.from_mesh(linear)
    basis=skfn.Basis(
        mesh,
        skfn.ElementVector(skfn.ElementTetP2())
        *skfn.ElementTetP1(),
        intorder=4,
    )
    coefficients=np.linspace(-.2,.7,basis.N)
    velocity,pressure=basis.interpolate(coefficients)

    @skfn.Functional
    def functional(w):
        return (
            ddot(grad(w.velocity),grad(w.velocity))
            +w.pressure*w.pressure
        )

    actual=skfn.asm(
        functional,basis,velocity=velocity,pressure=pressure
    )
    expected=np.sum(
        (
            np.einsum(
                "ijeq,ijeq->eq",velocity.grad,velocity.grad
            )
            +pressure.value**2
        )*basis.dx
    )
    np.testing.assert_allclose(actual,expected,rtol=2e-15,atol=2e-15)


def test_functional_rejects_vector_integrand():
    basis=skfn.Basis(
        skfn.MeshTet(),
        skfn.ElementVector(skfn.ElementTetP1()),
    )
    field=basis.interpolate(np.zeros(basis.N))

    @skfn.Functional
    def vector_result(w):
        return w.u

    try:
        skfn.asm(vector_result,basis,u=field)
    except skfn.UnsupportedNativeForm:
        pass
    else:
        raise AssertionError("vector Functional result must be rejected")
