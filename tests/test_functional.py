import numpy as np
import skfem
from skfem.helpers import ddot as reference_ddot
from skfem.helpers import grad as reference_grad

import skfemntv
from skfemntv.helpers import ddot,dot,grad


def test_volume_functional_of_interpolated_field_matches_skfem():
    linear=skfemntv.MeshTet.init_tensor(
        np.linspace(0.,1.,3),
        np.linspace(0.,1.,2),
        np.linspace(0.,1.,2),
    )
    mesh=skfemntv.MeshTet2.from_mesh(linear)
    basis=skfemntv.Basis(
        mesh,skfemntv.ElementVector(skfemntv.ElementTetP2()),intorder=4
    )
    component=np.arange(basis.N)%3
    coefficients=(
        (component+1.)*basis.doflocs[0]
        +(.2*component-.3)*basis.doflocs[1]
        +.1*basis.doflocs[2]
    )
    field=basis.interpolate(coefficients)

    @skfemntv.Functional
    def energy(w):
        return (
            .5*ddot(grad(w.u),grad(w.u))
            +(1.+w.x[0])*np.sum(w.u*w.u,axis=0)
        )

    actual=skfemntv.asm(energy,basis,u=field)

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
    mesh=skfemntv.MeshHex()
    basis=skfemntv.FacetBasis(
        mesh,skfemntv.ElementVector(skfemntv.ElementHex1()),intorder=4
    )
    coefficients=np.arange(basis.N,dtype=float)/basis.N
    field=basis.interpolate(coefficients)

    @skfemntv.Functional
    def surface(w):
        return (
            1.+w.x[0]+w.n[0]**2
            +np.sum(w.u*w.u,axis=0)
        )

    actual=skfemntv.asm(surface,basis,u=field)

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
    linear=skfemntv.MeshTet()
    mesh=skfemntv.MeshTet2.from_mesh(linear)
    basis=skfemntv.Basis(
        mesh,
        skfemntv.ElementVector(skfemntv.ElementTetP2())
        *skfemntv.ElementTetP1(),
        intorder=4,
    )
    coefficients=np.linspace(-.2,.7,basis.N)
    velocity,pressure=basis.interpolate(coefficients)

    @skfemntv.Functional
    def functional(w):
        return (
            ddot(grad(w.velocity),grad(w.velocity))
            +w.pressure*w.pressure
        )

    actual=skfemntv.asm(
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
    basis=skfemntv.Basis(
        skfemntv.MeshTet(),
        skfemntv.ElementVector(skfemntv.ElementTetP1()),
    )
    field=basis.interpolate(np.zeros(basis.N))

    @skfemntv.Functional
    def vector_result(w):
        return w.u

    try:
        skfemntv.asm(vector_result,basis,u=field)
    except skfemntv.UnsupportedNativeForm:
        pass
    else:
        raise AssertionError("vector Functional result must be rejected")


def test_supermesh_functional_integrates_geometry_context():
    points=np.array([
        [0.,1.,0.],
        [0.,0.,1.],
        [0.,0.,0.],
    ])
    slave_points=points.copy()
    slave_points[2]+=.01
    triangles=np.array([[0],[1],[2]])
    integration=skfemntv.TriangleSupermesh(
        points,triangles,slave_points,triangles,
        projection_tolerance=.02,
    )

    @skfemntv.Functional
    def geometry(w):
        return (
            1.+w.gap
            +w.n_master[2]**2
            +w.n_slave[2]**2
        )

    actual=skfemntv.asm(geometry,integration=integration)
    expected=.5*(3.+.01)
    np.testing.assert_allclose(actual,expected,rtol=2e-15,atol=2e-15)


def test_supermesh_interpolated_jump_functional():
    points=np.array([
        [0.,1.,0.],
        [0.,0.,1.],
        [0.,0.,0.],
    ])
    triangles=np.array([[0],[1],[2]])
    integration=skfemntv.TriangleSupermesh(
        points,triangles,points,triangles
    )
    master=np.array([0.,1.,1.])
    slave=master+2.
    master_field,slave_field=integration.interpolate(master,slave)
    assert master_field.grad is None
    assert slave_field.grad is None

    @skfemntv.Functional
    def jump_energy(w):
        jump_value=w.slave-w.master
        return jump_value*jump_value

    actual=skfemntv.asm(
        jump_energy,integration=integration,
        master=master_field,slave=slave_field,
    )
    np.testing.assert_allclose(actual,2.,rtol=2e-15,atol=2e-15)


def test_facet_supermesh_functional_accepts_trace_gradients():
    mesh=skfemntv.MeshTet()
    facets=skfemntv.FacetBasis(
        mesh,skfemntv.ElementVector(skfemntv.ElementTetP1(),dim=1)
    )
    integration=skfemntv.TriangleSupermesh.from_facets(facets,facets)
    coefficients=facets.doflocs[0]+2.*facets.doflocs[1]
    master,slave=integration.interpolate(coefficients,coefficients)
    assert master.grad is not None
    assert slave.grad is not None

    @skfemntv.Functional
    def gradient_energy(w):
        return dot(grad(w.master),grad(w.slave))

    actual=skfemntv.asm(
        gradient_energy,integration=integration,
        master=master,slave=slave,
    )
    expected=np.sum(
        np.einsum("ieq,ieq->eq",master.grad,slave.grad)
        *integration._weights
    )
    np.testing.assert_allclose(actual,expected,rtol=2e-15,atol=2e-15)
