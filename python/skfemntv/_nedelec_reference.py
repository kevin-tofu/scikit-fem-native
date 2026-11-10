"""Reference Nédélec bases on unit simplices through third order on triangles."""

from __future__ import annotations

import numpy as np
from ._tet_n2_coefficients import TET_N2_COEFFICIENTS,TET_N3_COEFFICIENTS


TRI_VERTICES=np.array(((0.,0.),(1.,0.),(0.,1.)),dtype=np.float64)
TRI_DIRECTED_EDGES=((0,1),(1,2),(2,0))
TET_VERTICES=np.array(
    ((0.,0.,0.),(1.,0.,0.),(0.,1.,0.),(0.,0.,1.)),dtype=np.float64
)
TET_DIRECTED_EDGES=((0,1),(1,2),(2,0),(0,3),(1,3),(2,3))
_TET_BARYCENTRIC_GRADIENTS=np.array(
    ((-1.,-1.,-1.),(1.,0.,0.),(0.,1.,0.),(0.,0.,1.)),dtype=np.float64
)


def tri_n1_basis(points):
    """Evaluate edge-dual Whitney basis functions in component-first layout.

    ``points`` has shape ``(2, ...)`` and the result has shape ``(3, 2, ...)``:
    one vector basis function for each directed reference edge.
    """
    points=np.asarray(points,dtype=np.float64)
    if points.ndim<1 or points.shape[0]!=2:
        raise ValueError("triangle reference points must have shape (2, ...)")
    x,y=points
    result=np.empty((3,2)+points.shape[1:],dtype=np.float64)
    result[0,0]=1.-y
    result[0,1]=x
    result[1,0]=-y
    result[1,1]=x
    result[2,0]=-y
    result[2,1]=x-1.
    return result


def tri_n1_curl(points):
    """Evaluate scalar reference curls, one constant curl per basis."""
    points=np.asarray(points,dtype=np.float64)
    if points.ndim<1 or points.shape[0]!=2:
        raise ValueError("triangle reference points must have shape (2, ...)")
    return np.full((3,)+points.shape[1:],2.,dtype=np.float64)


def tri_n2_basis(points):
    """Second-order first-family Nedelec basis on the reference triangle."""
    points=np.asarray(points,dtype=np.float64)
    if points.ndim<1 or points.shape[0]!=2:
        raise ValueError("triangle reference points must have shape (2, ...)")
    x,y=points
    result=np.empty((8,2)+points.shape[1:],dtype=np.float64)
    result[0,0]=8.*x*y-6.*x+8.*y*y-12.*y+4.
    result[0,1]=2.*x*(-4.*x-4.*y+3.)
    result[1,0]=-8.*x*y+6.*x+2.*y-2.
    result[1,1]=4.*x*(2.*x-1.)
    result[2,0]=2.*y*(1.-4.*x)
    result[2,1]=4.*x*(2.*x-1.)
    result[3,0]=4.*y*(1.-2.*y)
    result[3,1]=2.*x*(4.*y-1.)
    result[4,0]=2.*y*(-4.*x-4.*y+3.)
    result[4,1]=8.*x*x+8.*x*y-12.*x-6.*y+4.
    result[5,0]=4.*y*(2.*y-1.)
    result[5,1]=-8.*x*y+2.*x+6.*y-2.
    result[6,0]=8.*y*(-x-2.*y+2.)
    result[6,1]=8.*x*(x+2.*y-1.)
    result[7,0]=8.*y*(2.*x+y-1.)
    result[7,1]=8.*x*(-2.*x-y+2.)
    return result


def tri_n2_curl(points):
    """Scalar curls of the second-order reference-triangle basis."""
    points=np.asarray(points,dtype=np.float64)
    if points.ndim<1 or points.shape[0]!=2:
        raise ValueError("triangle reference points must have shape (2, ...)")
    x,y=points
    values=(-24.*x-24.*y+18.,24.*x-6.,24.*x-6.,24.*y-6.,
            24.*x+24.*y-18.,6.-24.*y,48.*y+24.*x-24.,
            -24.*y-48.*x+24.)
    return np.stack(values,axis=0)


def tri_n2_gradients(points):
    """Reference-coordinate gradients in (basis, component, axis, point)."""
    points=np.asarray(points,dtype=np.float64)
    if points.ndim!=2 or points.shape[0]!=2:
        raise ValueError("triangle reference points must have shape (2, nq)")
    x,y=points
    grad=np.empty((8,2,2,points.shape[1]),dtype=np.float64)
    grad[:,0,0]=np.stack((8.*y-6.,-8.*y+6.,-8.*y,np.zeros_like(x),
                          -8.*y,np.zeros_like(x),-8.*y,16.*y),axis=0)
    grad[:,0,1]=np.stack((8.*x+16.*y-12.,-8.*x+2.,2.-8.*x,
                          4.-16.*y,-8.*x-16.*y+6.,16.*y-4.,
                          -8.*x-32.*y+16.,16.*x+16.*y-8.),axis=0)
    grad[:,1,0]=np.stack((-16.*x-8.*y+6.,16.*x-4.,16.*x-4.,
                          8.*y-2.,16.*x+8.*y-12.,-8.*y+2.,
                          16.*x+16.*y-8.,-32.*x-8.*y+16.),axis=0)
    grad[:,1,1]=np.stack((-8.*x,np.zeros_like(x),np.zeros_like(x),8.*x,
                          8.*x-6.,-8.*x+6.,16.*x, -8.*x),axis=0)
    return grad


def tri_n1_edge_moments(*,reversed_edges=(),quadrature_order=4):
    """Integrate each basis tangentially along every directed reference edge."""
    reversed_edges=set(map(int,reversed_edges))
    if not reversed_edges <= {0,1,2}:
        raise ValueError("reversed edge indices must be drawn from 0, 1, 2")
    points,weights=np.polynomial.legendre.leggauss(quadrature_order)
    parameter=.5*(points+1.)
    weights=.5*weights
    moments=np.empty((3,3),dtype=np.float64)
    for edge_index,(start,end) in enumerate(TRI_DIRECTED_EDGES):
        first=TRI_VERTICES[start]
        second=TRI_VERTICES[end]
        if edge_index in reversed_edges:
            first,second=second,first
        tangent=second-first
        edge_points=(
            first[:,None]+tangent[:,None]*parameter[None,:]
        )
        values=tri_n1_basis(edge_points)
        moments[edge_index]=np.einsum(
            "biq,iq,q->b",values,tangent[:,None]*np.ones_like(parameter),weights
        )
    return moments


def tet_n1_basis(points):
    """Evaluate edge-dual Whitney basis functions on the unit tetrahedron."""
    points=np.asarray(points,dtype=np.float64)
    if points.ndim<1 or points.shape[0]!=3:
        raise ValueError("tetrahedron reference points must have shape (3, ...)")
    lambdas=np.concatenate(
        ((1.-np.sum(points,axis=0))[None,...],points),axis=0
    )
    result=np.empty((6,3)+points.shape[1:],dtype=np.float64)
    vector_axes=(slice(None),)+(None,)*(points.ndim-1)
    for edge,(first,second) in enumerate(TET_DIRECTED_EDGES):
        result[edge]=(
            lambdas[first][None,...]*
            _TET_BARYCENTRIC_GRADIENTS[second][vector_axes]
            -lambdas[second][None,...]*
            _TET_BARYCENTRIC_GRADIENTS[first][vector_axes]
        )
    return result


def tet_n1_curl(points):
    """Evaluate constant vector reference curls for all six edge bases."""
    points=np.asarray(points,dtype=np.float64)
    if points.ndim<1 or points.shape[0]!=3:
        raise ValueError("tetrahedron reference points must have shape (3, ...)")
    constants=np.asarray([
        2.*np.cross(
            _TET_BARYCENTRIC_GRADIENTS[first],
            _TET_BARYCENTRIC_GRADIENTS[second],
        )
        for first,second in TET_DIRECTED_EDGES
    ])
    return np.broadcast_to(
        constants[(slice(None),slice(None))+(None,)*(points.ndim-1)],
        (6,3)+points.shape[1:],
    ).copy()


def tet_n2_basis(points):
    """Second-order first-family Nedelec basis on the reference tetrahedron.

    The polynomial coefficients are tabulated from the Basix N1E degree-2
    element.  The returned shape is ``(20, 3, ...)``.
    """
    points=np.asarray(points,dtype=np.float64)
    if points.ndim<1 or points.shape[0]!=3:
        raise ValueError("tetrahedron reference points must have shape (3, ...)")
    x,y,z=points
    monomials=np.stack((np.ones_like(x),x,y,z,x*x,x*y,x*z,y*y,y*z,z*z))
    return np.einsum("bcm,m...->bc...",TET_N2_COEFFICIENTS,monomials)


def tet_n2_gradients(points):
    """Reference gradients in ``(basis, component, axis, point)`` layout."""
    points=np.asarray(points,dtype=np.float64)
    if points.ndim!=2 or points.shape[0]!=3:
        raise ValueError("tetrahedron reference points must have shape (3, nq)")
    x,y,z=points
    zero=np.zeros_like(x)
    one=np.ones_like(x)
    monomial_gradients=np.stack((
        (zero,one,zero,zero,2*x,y,z,zero,zero,zero),
        (zero,zero,one,zero,zero,x,zero,2*y,z,zero),
        (zero,zero,zero,one,zero,zero,x,zero,y,2*z),
    ))
    return np.einsum("bcm,amq->bcaq",TET_N2_COEFFICIENTS,monomial_gradients)


def tet_n2_curl(points):
    """Vector curls of the second-order reference-tetrahedron basis."""
    gradients=tet_n2_gradients(np.asarray(points,dtype=np.float64))
    return np.stack((
        gradients[:,2,1]-gradients[:,1,2],
        gradients[:,0,2]-gradients[:,2,0],
        gradients[:,1,0]-gradients[:,0,1],
    ),axis=1)


_TET_N3_EXPONENTS=(
    (0,0,0),(1,0,0),(0,1,0),(0,0,1),
    (2,0,0),(1,1,0),(1,0,1),(0,2,0),(0,1,1),(0,0,2),
    (3,0,0),(2,1,0),(2,0,1),(1,2,0),(1,1,1),(1,0,2),
    (0,3,0),(0,2,1),(0,1,2),(0,0,3),
)


def _tet_n3_monomials(points):
    x,y,z=points
    return np.stack(tuple(x**a*y**b*z**c for a,b,c in _TET_N3_EXPONENTS))


def tet_n3_basis(points):
    """Third-order first-family Nedelec basis on the unit tetrahedron."""
    points=np.asarray(points,dtype=np.float64)
    if points.ndim<1 or points.shape[0]!=3:
        raise ValueError("tetrahedron reference points must have shape (3, ...)")
    return np.einsum("bcm,m...->bc...",TET_N3_COEFFICIENTS,_tet_n3_monomials(points))


def tet_n3_gradients(points):
    """Reference gradients in ``(basis, component, axis, point)`` layout."""
    points=np.asarray(points,dtype=np.float64)
    if points.ndim!=2 or points.shape[0]!=3:
        raise ValueError("tetrahedron reference points must have shape (3, nq)")
    x,y,z=points
    coordinates=(x,y,z)
    gradients=[]
    for axis in range(3):
        derivative=[]
        for exponents in _TET_N3_EXPONENTS:
            powers=list(exponents)
            factor=powers[axis]
            if factor==0:
                derivative.append(np.zeros_like(x))
                continue
            powers[axis]-=1
            derivative.append(factor*coordinates[0]**powers[0]
                              *coordinates[1]**powers[1]
                              *coordinates[2]**powers[2])
        gradients.append(np.stack(derivative))
    return np.einsum(
        "bcm,amq->bcaq",TET_N3_COEFFICIENTS,np.stack(gradients),
    )


def tet_n3_curl(points):
    """Vector curls of the third-order reference-tetrahedron basis."""
    gradients=tet_n3_gradients(np.asarray(points,dtype=np.float64))
    return np.stack((
        gradients[:,2,1]-gradients[:,1,2],
        gradients[:,0,2]-gradients[:,2,0],
        gradients[:,1,0]-gradients[:,0,1],
    ),axis=1)


def tet_n1_edge_moments(*,reversed_edges=(),quadrature_order=4):
    """Integrate each tetrahedral basis along every directed reference edge."""
    reversed_edges=set(map(int,reversed_edges))
    if not reversed_edges <= set(range(6)):
        raise ValueError("reversed edge indices must be drawn from 0 through 5")
    nodes,weights=np.polynomial.legendre.leggauss(quadrature_order)
    parameter=.5*(nodes+1.)
    weights=.5*weights
    moments=np.empty((6,6),dtype=np.float64)
    for edge_index,(start,end) in enumerate(TET_DIRECTED_EDGES):
        first=TET_VERTICES[start]
        second=TET_VERTICES[end]
        if edge_index in reversed_edges:
            first,second=second,first
        tangent=second-first
        edge_points=first[:,None]+tangent[:,None]*parameter[None,:]
        moments[edge_index]=np.einsum(
            "biq,i,q->b",tet_n1_basis(edge_points),tangent,weights
        )
    return moments


__all__=[
    "TET_DIRECTED_EDGES",
    "TET_VERTICES",
    "TRI_DIRECTED_EDGES",
    "TRI_VERTICES",
    "tet_n1_basis",
    "tet_n1_curl",
    "tet_n1_edge_moments",
    "tet_n2_basis",
    "tet_n2_curl",
    "tet_n2_gradients",
    "tet_n3_basis",
    "tet_n3_curl",
    "tet_n3_gradients",
    "tri_n1_basis",
    "tri_n1_curl",
    "tri_n1_edge_moments",
]


def tri_n3_lbasis(X, i):
        x, y = X

        x2 = x**2
        y2 = y**2
        xy = x * y

        if i == 0:
            phi = np.array([
                -8.0*x2*y + 8.0*x2 - 16.0*x*y2 + 26.0*xy -
                10.0*x - 8.0*y**3 + 18.0*y2 - 13.0*y + 3.0,
                x*(8.0*x2 + 16.0*xy - 10.0*x + 8.0*y2 - 10.0*y + 3.0)
            ])
            dphi = 32.0*x2 + 64.0*xy - 46.0*x + 32.0*y2 - 46.0*y + 16.0

        elif i == 1:
            phi = np.array([
                16.0*x2*y - 16.0*x2 + 16.0*x*y2 - 32.0*xy + 16.0*x
                - 4.0*y2 + 7.0*y - 3.0,
                x*(-16.0*x2 - 16.0*xy + 16.0*x + 4.0*y - 3.0)
            ])
            dphi = -64.0*x2 - 64.0*xy + 64.0*x + 12.0*y - 10.0

        elif i == 2:
            phi = np.array([
                -8.0*x2*y + 8.0*x2 + 6.0*xy - 6.0*x - 1.0*y + 1.0,
                x*(8.0*x2 - 6.0*x + 1.0)
            ])
            dphi = 32.0*x2 - 18.0*x + 2.0

        elif i == 3:
            phi = np.array([
                y*(-8.0*x2 + 6.0*x - 1.0),
                x*(8.0*x2 - 6.0*x + 1.0)
            ])
            dphi = 32.0*x2 - 18.0*x + 2.0

        elif i == 4:
            phi = np.array([
                y*(-16.0*xy + 4.0*x + 4.0*y - 1.0),
                x*(16.0*xy - 4.0*x - 4.0*y + 1.0)
            ])
            dphi = 64.0*xy - 12.0*x - 12.0*y + 2.0

        elif i == 5:
            phi = np.array([
                y*(-8.0*y2 + 6.0*y - 1.0),
                x*(8.0*y2 - 6.0*y + 1.0)
            ])
            dphi = 32.0*y2 - 18.0*y + 2.0

        elif i == 6:
            phi = np.array([
                y*(-8.0*y2 + 6.0*y - 1.0),
                8.0*x*y2 - 6.0*xy + 1.0*x - 8.0*y2 + 6.0*y - 1.0
            ])
            dphi = 32.0*y2 - 18.0*y + 2.0

        elif i == 7:
            phi = np.array([
                y*(16.0*xy - 4.0*x + 16.0*y2 - 16.0*y + 3.0),
                -16.0*x2*y + 4.0*x2 - 16.0*x*y2 + 32.0*xy
                - 7.0*x + 16.0*y2 - 16.0*y + 3.0
            ])
            dphi = -64.0*xy + 12.0*x - 64.0*y2 + 64.0*y - 10.0

        elif i == 8:
            phi = np.array([
                y*(-8.0*x2 - 16.0*xy + 10.0*x - 8.0*y2 + 10.0*y - 3.0),
                8.0*x**3 + 16.0*x2*y - 18.0*x2 + 8.0*x*y2 -
                26.0*xy + 13.0*x - 8.0*y2 + 10.0*y - 3.0
            ])
            dphi = 32.0*x2 + 64.0*xy - 46.0*x + 32.0*y2 - 46.0*y + 16.0

        elif i == 9:
            phi = np.array([
                y*(8.0*x2 + 32.0*xy - 30.0*x + 24.0*y2 - 42.0*y + 18.0),
                x*(-8.0*x2 - 32.0*xy + 14.0*x - 24.0*y2 + 26.0*y - 6.0)
            ])
            dphi = -32.0*x2 - 128.0*xy + 58.0*x - 96.0*y2 + 110.0*y - 24.0

        elif i == 10:
            phi = np.array([
                y*(-24.0*x2 - 32.0*xy + 26.0*x - 8.0*y2 + 14.0*y - 6.0),
                x*(24.0*x2 + 32.0*xy - 42.0*x + 8.0*y2 - 30.0*y + 18.0)
            ])
            dphi = 96.0*x2 + 128.0*xy - 110.0*x + 32.0*y2 - 58.0*y + 24.0

        elif i == 11:
            phi = np.array([
                y*(-16.0*x2 - 32.0*xy + 36.0*x + 8.0*y - 8.0),
                x*(16.0*x2 + 32.0*xy - 20.0*x - 8.0*y + 4.0)
            ])
            dphi = 64.0*x2 + 128.0*xy - 76.0*x - 24.0*y + 12.0

        elif i == 12:
            phi = np.array([
                y*(24.0*x2 + 16.0*xy - 22.0*x - 4.0*y + 4.0),
                x*(-24.0*x2 - 16.0*xy + 30.0*x + 4.0*y - 6.0)
            ])
            dphi = -96.0*x2 - 64.0*xy + 82.0*x + 12.0*y - 10.0

        elif i == 13:
            phi = np.array([
                y*(-16.0*xy + 4.0*x - 24.0*y2 + 30.0*y - 6.0),
                x*(16.0*xy - 4.0*x + 24.0*y2 - 22.0*y + 4.0)
            ])
            dphi = 64.0*xy - 12.0*x + 96.0*y2 - 82.0*y + 10.0

        elif i == 14:
            phi = np.array([
                y*(32.0*xy - 8.0*x + 16.0*y2 - 20.0*y + 4.0),
                x*(-32.0*xy + 8.0*x - 16.0*y2 + 36.0*y - 8.0)
            ])
            dphi = -128.0*xy + 24.0*x - 64.0*y2 + 76.0*y - 12.0

        else:
            raise IndexError("N3 basis index must be in 0..14")

        return phi, dphi


def tri_n3_basis(points):
    """Evaluate the third-order first-family basis on the reference triangle."""
    points=np.asarray(points)
    if points.ndim<1 or points.shape[0]!=2:
        raise ValueError("triangle reference points must have shape (2, ...)")
    return np.stack([tri_n3_lbasis(points,i)[0] for i in range(15)],axis=0)


def tri_n3_curl(points):
    """Scalar curls of the third-order reference-triangle basis."""
    points=np.asarray(points)
    if points.ndim<1 or points.shape[0]!=2:
        raise ValueError("triangle reference points must have shape (2, ...)")
    return np.stack([tri_n3_lbasis(points,i)[1] for i in range(15)],axis=0)


def tri_n3_gradients(points):
    """Reference gradients in (basis, component, axis, point) layout."""
    points=np.asarray(points,dtype=np.float64)
    if points.ndim!=2 or points.shape[0]!=2:
        raise ValueError("triangle reference points must have shape (2, nq)")
    step=1.e-30
    gradients=np.empty((15,2,2,points.shape[1]),dtype=np.float64)
    for axis in range(2):
        perturbed=points.astype(np.complex128)
        perturbed[axis]+=1j*step
        gradients[:,:,axis]=np.imag(tri_n3_basis(perturbed))/step
    return gradients
