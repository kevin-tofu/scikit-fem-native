import numpy as np
import pytest
import skfem
from skfem.helpers import ddot as reference_ddot
from skfem.helpers import dot as reference_dot
from skfem.helpers import grad as reference_grad

import skfemntv
from skfemntv.helpers import ddot,dot,grad


def _reference_mesh(mesh):
    if isinstance(mesh,skfemntv.MeshHex2):
        return skfem.MeshHex2.from_mesh(skfem.MeshHex())
    if isinstance(mesh,skfemntv.MeshHex):
        return skfem.MeshHex(mesh.p,mesh.t)
    if isinstance(mesh,skfemntv.MeshTet2):
        return skfem.MeshTet2(mesh.p,mesh.t)
    return skfem.MeshTet(mesh.p,mesh.t)


def _coefficients(basis,components):
    component=np.arange(basis.N)%components
    x=basis.doflocs
    return (
        (component+1.)*x[0]
        +(.25*component-.4)*x[1]
        +(.3-.1*component)*x[2]
        +.2*component
    )


def _quadrature_order(coordinates):
    return [
        np.lexsort(np.round(points,14).T[::-1])
        for points in coordinates
    ]


@pytest.mark.parametrize(
    "mesh,element,reference_element,intorder",
    [
        (skfemntv.MeshTet(),skfemntv.ElementTetP1(),skfem.ElementTetP1(),2),
        (skfemntv.MeshTet2(),skfemntv.ElementTetP2(),skfem.ElementTetP2(),4),
        (skfemntv.MeshHex(),skfemntv.ElementHex1(),skfem.ElementHex1(),2),
        (skfemntv.MeshHex2(),skfemntv.ElementHex2(),skfem.ElementHex2(),4),
    ],
)
def test_vector_interpolation_matches_skfem(
    mesh,element,reference_element,intorder
):
    basis=skfemntv.Basis(
        mesh,skfemntv.ElementVector(element),intorder=intorder
    )
    coefficients=_coefficients(basis,3)
    actual=basis.interpolate(coefficients)

    reference_basis=skfem.Basis(
        _reference_mesh(mesh),
        skfem.ElementVector(reference_element),
        intorder=intorder,
    )
    reference_coefficients=_coefficients(reference_basis,3)
    expected=reference_basis.interpolate(reference_coefficients)
    actual_order=_quadrature_order(basis.global_coordinates)
    reference_coordinates=np.moveaxis(
        np.asarray(reference_basis.global_coordinates()),0,-1
    )
    expected_order=_quadrature_order(reference_coordinates)
    actual_value=np.stack([
        actual.value[:,entity,order]
        for entity,order in enumerate(actual_order)
    ],axis=1)
    expected_value=np.stack([
        np.asarray(expected)[:,entity,order]
        for entity,order in enumerate(expected_order)
    ],axis=1)
    actual_gradient=np.stack([
        actual.grad[:,:,entity,order]
        for entity,order in enumerate(actual_order)
    ],axis=2)
    expected_gradient=np.stack([
        expected.grad[:,:,entity,order]
        for entity,order in enumerate(expected_order)
    ],axis=2)
    np.testing.assert_allclose(
        actual_value,expected_value,rtol=2e-13,atol=2e-13
    )
    np.testing.assert_allclose(
        actual_gradient,expected_gradient,rtol=5e-13,atol=5e-13
    )
    np.testing.assert_allclose(
        np.einsum("iieq->eq",actual_gradient),
        np.einsum("iieq->eq",expected_gradient),
        rtol=5e-13,atol=5e-13,
    )


def test_composite_interpolation_and_form_parameter_match_skfem():
    linear=skfemntv.MeshTet.init_tensor(
        np.linspace(0.,1.,3),
        np.linspace(0.,1.,2),
        np.linspace(0.,1.,2),
    )
    mesh=skfemntv.MeshTet2.from_mesh(linear)
    basis=skfemntv.Basis(
        mesh,
        skfemntv.ElementVector(skfemntv.ElementTetP2())
        *skfemntv.ElementTetP1(),
        intorder=4,
    )
    velocity_indices,pressure_indices=basis.split_indices()
    coefficients=np.zeros(basis.N)
    velocity_basis,pressure_basis=basis.split_bases()
    coefficients[velocity_indices]=_coefficients(velocity_basis,3)
    coefficients[pressure_indices]=(
        pressure_basis.doflocs[0]
        -2.*pressure_basis.doflocs[1]
        +.5*pressure_basis.doflocs[2]
    )
    velocity,pressure=basis.interpolate(coefficients)
    assert velocity.value.shape[0]==3
    assert pressure.value.shape==basis.dx.shape

    @skfemntv.LinearForm
    def residual(v,q,w):
        return ddot(grad(w.velocity),grad(v))+w.pressure*q

    actual=skfemntv.asm(
        residual,basis,velocity=velocity,pressure=pressure
    )

    reference_mesh=skfem.MeshTet2.from_mesh(
        skfem.MeshTet(linear.p,linear.t)
    )
    reference_basis=skfem.Basis(
        reference_mesh,
        skfem.ElementVector(skfem.ElementTetP2())
        *skfem.ElementTetP1(),
        intorder=4,
    )
    reference_coefficients=np.zeros(reference_basis.N)
    reference_fields=reference_basis.split_bases()
    reference_indices=reference_basis.split_indices()
    reference_coefficients[reference_indices[0]]=_coefficients(
        reference_fields[0],3
    )
    reference_coefficients[reference_indices[1]]=(
        reference_fields[1].doflocs[0]
        -2.*reference_fields[1].doflocs[1]
        +.5*reference_fields[1].doflocs[2]
    )
    reference_velocity,reference_pressure=reference_basis.interpolate(
        reference_coefficients
    )

    @skfem.LinearForm
    def reference(v,q,w):
        return (
            reference_ddot(
                reference_grad(w.velocity),reference_grad(v)
            )
            +w.pressure*q
        )

    expected=skfem.asm(
        reference,reference_basis,
        velocity=reference_velocity,
        pressure=reference_pressure,
    )
    permutation=np.empty(basis.N,dtype=np.int64)
    for native_dofs,reference_dofs,components in zip(
        basis.split_indices(),reference_basis.split_indices(),(3,1)
    ):
        lookup={}
        for dof in reference_dofs:
            coordinate=tuple(np.round(reference_basis.doflocs[:,dof],14))
            lookup.setdefault(coordinate,[]).append(int(dof))
        for offset,native_dof in enumerate(native_dofs):
            coordinate=tuple(np.round(
                basis.doflocs[:,native_dof],14
            ))
            permutation[native_dof]=lookup[coordinate][offset%components]
    np.testing.assert_allclose(
        actual,expected[permutation],rtol=2e-12,atol=2e-12
    )


def test_interpolated_scalar_coefficient_in_bilinear_form():
    mesh=skfemntv.MeshTet.init_tensor(
        np.linspace(0.,1.,3),
        np.linspace(0.,1.,2),
        np.linspace(0.,1.,2),
    )
    basis=skfemntv.Basis(
        mesh,skfemntv.ElementVector(skfemntv.ElementTetP1(),dim=1)
    )
    coefficient=basis.interpolate(
        1.+basis.doflocs[0]+.5*basis.doflocs[1]
    )

    @skfemntv.BilinearForm
    def mass(u,v,w):
        return w.coefficient*dot(u,v)

    actual=skfemntv.asm(mass,basis,coefficient=coefficient)

    reference_mesh=skfem.MeshTet(mesh.p,mesh.t)
    reference_basis=skfem.Basis(
        reference_mesh,skfem.ElementTetP1()
    )
    reference_coefficient=reference_basis.interpolate(
        1.+reference_basis.doflocs[0]
        +.5*reference_basis.doflocs[1]
    )

    @skfem.BilinearForm
    def reference(u,v,w):
        return w.coefficient*u*v

    expected=skfem.asm(
        reference,reference_basis,
        coefficient=reference_coefficient,
    )
    np.testing.assert_allclose(
        actual.toarray(),expected.toarray(),rtol=5e-13,atol=5e-13
    )
