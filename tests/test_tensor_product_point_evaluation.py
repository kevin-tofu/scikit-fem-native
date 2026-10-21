"""Point evaluation and projection parity for tensor-product H1 bases."""

import numpy as np
import pytest
import skfem

import skfemntv


def _meshes(dimension, curved):
    if dimension == 2:
        linear = skfemntv.MeshQuad.init_tensor([0., 1.], [0., 1.])
        reference_linear = skfem.MeshQuad(linear.p, linear.t)
        if not curved:
            return linear, reference_linear
        native = skfemntv.MeshQuad2.from_mesh(linear)
        reference = skfem.MeshQuad2.from_mesh(reference_linear)
        native_points = native.p.copy()
        native_vertices=np.any(np.all(
            np.isclose(native_points.T[:,None,:],linear.p.T[None,:,:]),axis=2
        ),axis=1)
        native_points[1,~native_vertices] += .04*np.sin(
            np.pi*native_points[0,~native_vertices]
        )
        reference_points = reference.doflocs.copy()
        reference_vertices=np.any(np.all(
            np.isclose(reference_points.T[:,None,:],reference_linear.p.T[None,:,:]),axis=2
        ),axis=1)
        reference_points[1,~reference_vertices] += .04*np.sin(
            np.pi*reference_points[0,~reference_vertices]
        )
        return (
            skfemntv.MeshQuad2(native_points, native.t.copy()),
            skfem.MeshQuad2(doflocs=reference_points,t=reference.t.copy()),
        )
    linear = skfemntv.MeshHex.init_tensor([0., 1.], [0., 1.], [0., 1.])
    reference_linear = skfem.MeshHex(linear.p, linear.t)
    if not curved:
        return linear, reference_linear
    native = skfemntv.MeshHex2.from_mesh(linear)
    reference = skfem.MeshHex2.from_mesh(reference_linear)
    native_points = native.p.copy()
    native_vertices=np.any(np.all(
        np.isclose(native_points.T[:,None,:],linear.p.T[None,:,:]),axis=2
    ),axis=1)
    native_points[2,~native_vertices] += .025*np.sin(
        np.pi*native_points[0,~native_vertices]
    )
    reference_points = reference.doflocs.copy()
    reference_vertices=np.any(np.all(
        np.isclose(reference_points.T[:,None,:],reference_linear.p.T[None,:,:]),axis=2
    ),axis=1)
    reference_points[2,~reference_vertices] += .025*np.sin(
        np.pi*reference_points[0,~reference_vertices]
    )
    return (
        skfemntv.MeshHex2(native_points, native.t.copy()),
        skfem.MeshHex2(doflocs=reference_points,t=reference.t.copy()),
    )


@pytest.mark.parametrize("dimension,order", [(2, 1), (2, 2), (3, 1), (3, 2)])
def test_tensor_point_evaluation_probes_and_gradients_match_skfem(
    dimension, order
):
    native_mesh, reference_mesh = _meshes(dimension, curved=order==2)
    if dimension == 2:
        native_element = skfemntv.ElementQuad1() if order==1 else skfemntv.ElementQuad2()
        reference_element = skfem.ElementQuad1() if order==1 else skfem.ElementQuad2()
        reference = np.array(((.2, .55, .35), (.25, .2, .6)))
    else:
        native_element = skfemntv.ElementHex1() if order==1 else skfemntv.ElementHex2()
        reference_element = skfem.ElementHex1() if order==1 else skfem.ElementHex2()
        reference = np.array((
            (.2, .55, .35), (.25, .2, .35), (.3, .35, .2),
        ))
    native_basis = skfemntv.Basis(native_mesh,native_element,intorder=5)
    reference_basis = skfem.Basis(reference_mesh,reference_element,intorder=5)
    native_coefficients = (
        native_basis.doflocs[0]**2
        + .7*native_basis.doflocs[1]
        - .3*native_basis.doflocs[-1]
    )
    reference_coefficients = (
        reference_basis.doflocs[0]**2
        + .7*reference_basis.doflocs[1]
        - .3*reference_basis.doflocs[-1]
    )
    points = reference_basis.mapping.F(
        reference[:,None,:],tind=np.zeros(reference.shape[1],dtype=int)
    )[:,0,:]

    local_coefficients=reference_coefficients[
        reference_basis.element_dofs[:,0]
    ]

    def reference_values(query):
        reference_points=reference_basis.mapping.invF(
            query[:,None,:],
            tind=np.zeros(query.shape[1],dtype=int),
        )[:,0,:]
        result=np.zeros(query.shape[1])
        for local_dof,coefficient in enumerate(local_coefficients):
            result+=coefficient*reference_basis.elem.lbasis(
                reference_points,local_dof
            )[0]
        return result

    result = native_basis.evaluate_at(points,native_coefficients)
    expected = reference_values(points)
    np.testing.assert_allclose(result.value,expected,rtol=3e-11,atol=3e-11)
    np.testing.assert_allclose(
        native_basis.probes(points)@native_coefficients,
        result.value,rtol=2e-12,atol=2e-12,
    )
    np.testing.assert_allclose(
        native_basis.interpolator(native_coefficients)(points),
        result.value,rtol=2e-12,atol=2e-12,
    )

    step=1e-6
    expected_gradient=np.vstack([
        (
            reference_values(points+np.eye(dimension)[:,[axis]]*step)
            -reference_values(points-np.eye(dimension)[:,[axis]]*step)
        )/(2.*step)
        for axis in range(dimension)
    ])
    np.testing.assert_allclose(
        result.grad,expected_gradient,rtol=4e-7,atol=4e-7
    )


@pytest.mark.parametrize("dimension", [2, 3])
def test_tensor_l2_projection_matches_skfem(dimension):
    native_mesh,reference_mesh=_meshes(dimension,curved=True)
    if dimension==2:
        native_element=skfemntv.ElementQuad2()
        reference_element=skfem.ElementQuad2()
    else:
        native_element=skfemntv.ElementHex2()
        reference_element=skfem.ElementHex2()
    native=skfemntv.Basis(native_mesh,native_element,intorder=8)
    reference=skfem.Basis(reference_mesh,reference_element,intorder=8)

    def field(x):
        return x[0]**2 + x[1]*x[-1] + .4*x[-1]

    native_values=native.project(field)
    reference_values=reference.project(field)
    reference_by_point={
        tuple(np.round(point,13)):dof
        for dof,point in enumerate(reference.doflocs.T)
    }
    permutation=np.asarray([
        reference_by_point[tuple(np.round(point,13))]
        for point in native.doflocs.T
    ])
    np.testing.assert_allclose(
        native_values,reference_values[permutation],rtol=3e-10,atol=3e-11
    )
