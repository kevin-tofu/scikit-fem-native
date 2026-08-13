import numpy as np
import pytest
import skfem
from skfem.element import ElementTriN1
from skfem.mapping import MappingAffine

from skfemntv._hcurl_mapping import (
    covariant_piola,
    covariant_piola_curl,
    covariant_piola_vector_curl,
    jacobian_determinant,
    tetrahedron_affine_jacobian,
    triangle_affine_jacobian,
)
from skfemntv._nedelec_reference import (
    TET_DIRECTED_EDGES,TET_VERTICES,tet_n1_basis,tet_n1_curl,
    tri_n1_basis,tri_n1_curl,
)


def _physical_triangle():
    return np.array(((0.2,1.7,-0.1),(-0.3,0.1,1.4)))


def test_covariant_piola_preserves_directed_tangential_line_integrals():
    vertices=_physical_triangle()
    jacobian=triangle_affine_jacobian(vertices)
    points,weights=np.polynomial.legendre.leggauss(4)
    parameter=.5*(points+1.)
    weights=.5*weights
    reference_vertices=np.array(((0.,1.,0.),(0.,0.,1.)))
    edges=((0,1),(1,2),(2,0))

    for edge,(start,end) in enumerate(edges):
        ref_tangent=reference_vertices[:,end]-reference_vertices[:,start]
        ref_points=(
            reference_vertices[:,start,None]
            +ref_tangent[:,None]*parameter[None]
        )
        reference=tri_n1_basis(ref_points)
        physical=covariant_piola(reference,jacobian[...,None])
        physical_tangent=jacobian@ref_tangent
        moments=np.einsum("biq,i,q->b",physical,physical_tangent,weights)
        np.testing.assert_allclose(moments,np.eye(3)[edge],atol=2e-15)


def test_mapped_values_and_curls_match_scikit_fem():
    vertices=_physical_triangle()
    mesh=skfem.MeshTri(vertices,np.array(((0,),(1,),(2,))))
    mapping=MappingAffine(mesh)
    element=ElementTriN1()
    points=np.array(((0.1,0.6,0.2),(0.2,0.1,0.5)))
    jacobian=triangle_affine_jacobian(vertices)[...,None]
    actual_values=covariant_piola(tri_n1_basis(points),jacobian)
    actual_curls=covariant_piola_curl(tri_n1_curl(points),jacobian)
    signs=np.array((-1.,-1.,1.))

    for basis,sign in enumerate(signs):
        field,=element.gbasis(mapping,points,basis)
        np.testing.assert_allclose(actual_values[basis],sign*np.asarray(field)[:,0])
        np.testing.assert_allclose(actual_curls[basis],sign*field.curl[0])


def test_signed_determinant_controls_curl_on_reversed_triangle():
    vertices=_physical_triangle()
    forward=triangle_affine_jacobian(vertices)
    reversed_jacobian=triangle_affine_jacobian(vertices[:,[0,2,1]])
    assert jacobian_determinant(forward)>0.
    assert jacobian_determinant(reversed_jacobian)<0.
    np.testing.assert_allclose(
        covariant_piola_curl(np.array((2.,2.,2.)),reversed_jacobian),
        -covariant_piola_curl(np.array((2.,2.,2.)),forward),
    )


def test_hcurl_mapping_rejects_invalid_or_singular_geometry():
    with pytest.raises(ValueError,match="vertices"):
        triangle_affine_jacobian(np.zeros((3,3)))
    singular=triangle_affine_jacobian(
        np.array(((0.,1.,2.),(0.,0.,0.)))
    )
    with pytest.raises(ValueError,match="nonsingular"):
        covariant_piola(np.zeros((3,2)),singular)
    with pytest.raises(ValueError,match="nonsingular"):
        covariant_piola_curl(np.zeros(3),singular)


def test_tetrahedral_piola_preserves_directed_tangential_moments():
    physical_vertices=np.array((
        (.2,1.4,-.1,.3),(-.3,.2,1.6,.1),(.1,.4,.2,1.8)
    ))
    jacobian=tetrahedron_affine_jacobian(physical_vertices)
    nodes,weights=np.polynomial.legendre.leggauss(4)
    parameter=.5*(nodes+1.)
    weights=.5*weights
    for edge,(start,end) in enumerate(TET_DIRECTED_EDGES):
        reference_tangent=TET_VERTICES[end]-TET_VERTICES[start]
        points=(
            TET_VERTICES[start,:,None]
            +reference_tangent[:,None]*parameter[None,:]
        )
        values=covariant_piola(tet_n1_basis(points),jacobian[...,None])
        physical_tangent=jacobian@reference_tangent
        moments=np.einsum("biq,i,q->b",values,physical_tangent,weights)
        np.testing.assert_allclose(moments,np.eye(6)[edge],atol=3e-15)


def test_tetrahedral_vector_curl_uses_jacobian_over_determinant():
    vertices=np.array((
        (.2,1.4,-.1,.3),(-.3,.2,1.6,.1),(.1,.4,.2,1.8)
    ))
    jacobian=tetrahedron_affine_jacobian(vertices)
    points=np.array(((.1,.2),(.2,.1),(.3,.15)))
    expected=np.einsum(
        "ij,bjq->biq",jacobian,tet_n1_curl(points)
    )/jacobian_determinant(jacobian)
    np.testing.assert_allclose(
        covariant_piola_vector_curl(tet_n1_curl(points),jacobian[...,None]),
        expected,
    )


def test_tetrahedral_mapping_rejects_invalid_or_singular_geometry():
    with pytest.raises(ValueError,match="tetrahedron vertices"):
        tetrahedron_affine_jacobian(np.zeros((2,4)))
    singular=tetrahedron_affine_jacobian(np.array((
        (0.,1.,0.,1.),(0.,0.,1.,1.),(0.,0.,0.,0.)
    )))
    with pytest.raises(ValueError,match="nonsingular"):
        covariant_piola(np.zeros((6,3)),singular)
    with pytest.raises(ValueError,match="nonsingular"):
        covariant_piola_vector_curl(np.zeros((6,3)),singular)
