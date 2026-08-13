"""Reference-only lowest-order Nédélec bases on unit simplices."""

from __future__ import annotations

import numpy as np


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
    "tri_n1_basis",
    "tri_n1_curl",
    "tri_n1_edge_moments",
]
