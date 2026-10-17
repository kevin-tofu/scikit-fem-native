"""Reference lowest-order Raviart--Thomas bases on simplices."""

from __future__ import annotations

import numpy as np


def tri_rt0_basis(points):
    x,y=np.asarray(points,dtype=np.float64)
    values=np.empty((3,2)+x.shape,dtype=np.float64)
    values[0,0]=x; values[0,1]=y-1.
    values[1,0]=x; values[1,1]=y
    values[2,0]=x-1.; values[2,1]=y
    divergence=np.full((3,)+x.shape,2.,dtype=np.float64)
    return values,divergence


def tet_rt0_basis(points):
    x,y,z=np.asarray(points,dtype=np.float64)
    values=np.empty((4,3)+x.shape,dtype=np.float64)
    values[0,0]=x; values[0,1]=y; values[0,2]=z-1.
    values[1,0]=x; values[1,1]=y-1.; values[1,2]=z
    values[2,0]=x-1.; values[2,1]=y; values[2,2]=z
    values[3,0]=x; values[3,1]=y; values[3,2]=z
    divergence=np.full((4,)+x.shape,3.,dtype=np.float64)
    return values,divergence


__all__=["tet_rt0_basis","tri_rt0_basis"]
