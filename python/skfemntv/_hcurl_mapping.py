"""Affine covariant Piola primitives for two-dimensional H(curl)."""

from __future__ import annotations

import numpy as np


def triangle_affine_jacobian(vertices):
    """Return component-first Jacobians for reference-to-physical triangles.

    ``vertices`` has shape ``(2, 3, ...)`` and the result ``(2, 2, ...)``.
    """
    vertices=np.asarray(vertices,dtype=np.float64)
    if vertices.ndim<2 or vertices.shape[:2]!=(2,3):
        raise ValueError("triangle vertices must have shape (2, 3, ...)")
    return np.stack(
        (vertices[:,1]-vertices[:,0],vertices[:,2]-vertices[:,0]),axis=1
    )


def jacobian_determinant(jacobian):
    jacobian=np.asarray(jacobian,dtype=np.float64)
    if jacobian.ndim<2 or jacobian.shape[:2]!=(2,2):
        raise ValueError("triangle Jacobian must have shape (2, 2, ...)")
    return jacobian[0,0]*jacobian[1,1]-jacobian[0,1]*jacobian[1,0]


def covariant_piola(reference_values,jacobian):
    """Map reference vectors by ``J^{-T}`` in component-first layout."""
    values=np.asarray(reference_values,dtype=np.float64)
    jacobian=np.asarray(jacobian,dtype=np.float64)
    if values.ndim<2 or values.shape[1]!=2:
        raise ValueError("reference basis values must have shape (basis, 2, ...)")
    determinant=jacobian_determinant(jacobian)
    if np.any(np.isclose(determinant,0.)):
        raise ValueError("covariant Piola mapping requires nonsingular Jacobians")
    inverse=np.linalg.inv(np.moveaxis(jacobian,(0,1),(-2,-1)))
    inverse=np.moveaxis(inverse,(-2,-1),(0,1))
    try:
        return np.einsum("rp...,br...->bp...",inverse,values)
    except ValueError as error:
        raise ValueError(
            "reference values and Jacobian batch axes must broadcast"
        ) from error


def covariant_piola_curl(reference_curls,jacobian):
    """Map two-dimensional scalar curls by signed ``1 / det(J)``."""
    curls=np.asarray(reference_curls,dtype=np.float64)
    if curls.ndim<1:
        raise ValueError("reference curls must have shape (basis, ...)")
    determinant=jacobian_determinant(jacobian)
    if np.any(np.isclose(determinant,0.)):
        raise ValueError("curl mapping requires nonsingular Jacobians")
    try:
        return curls/determinant
    except ValueError as error:
        raise ValueError(
            "reference curls and Jacobian batch axes must broadcast"
        ) from error


__all__=[
    "covariant_piola",
    "covariant_piola_curl",
    "jacobian_determinant",
    "triangle_affine_jacobian",
]
