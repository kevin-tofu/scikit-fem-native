"""Compare TetN1/TetN2/TetN3 errors for a smooth PEC Maxwell solution."""

import numpy as np
from scipy.sparse.linalg import spsolve

import skfemntv
from skfemntv.helpers import curl,dot


ALPHA=.05
INTORDER=8
REFINEMENTS=(1,2,3,4)
ELEMENTS=(
    ("N1",skfemntv.ElementTetN1),
    ("N2",skfemntv.ElementTetN2),
    ("N3",skfemntv.ElementTetN3),
)


@skfemntv.BilinearForm
def maxwell(u,v,w):
    return dot(u,v)+ALPHA*dot(curl(u),curl(v))


@skfemntv.LinearForm
def manufactured_load(v,w):
    return dot(w.force,v)


def exact_field(coordinates):
    x,y,z=coordinates
    sx,sy,sz=(np.sin(np.pi*q) for q in (x,y,z))
    return np.stack((sy*sz,sz*sx,sx*sy))


def solve_level(element_type,subdivisions):
    axis=np.linspace(0.,1.,subdivisions+1)
    mesh=skfemntv.MeshTet.init_tensor(axis,axis,axis)
    basis=skfemntv.Basis(mesh,element_type(),intorder=INTORDER)
    coordinates=np.moveaxis(basis.global_coordinates,-1,0)
    exact=exact_field(coordinates)
    force=(1.+2.*ALPHA*np.pi**2)*exact
    matrix=skfemntv.asm(maxwell,basis).tocsr()
    load=skfemntv.asm(manufactured_load,basis,force=force)
    boundary=basis.get_dofs().all()
    free=np.setdiff1d(np.arange(basis.N),boundary)
    coefficients=np.zeros(basis.N)
    coefficients[free]=spsolve(matrix[free][:,free],load[free])
    numerical=basis.interpolate(coefficients)
    value_error=numerical.value-exact
    sx,sy,sz=(np.sin(np.pi*q) for q in coordinates)
    cx,cy,cz=(np.cos(np.pi*q) for q in coordinates)
    exact_curl=np.pi*np.stack((
        sx*(cy-cz),sy*(cz-cx),sz*(cx-cy),
    ))
    curl_error=numerical.curl-exact_curl
    l2_error=np.sqrt(np.sum(np.sum(value_error**2,axis=0)*basis.dx))
    curl_norm=np.sqrt(np.sum(np.sum(curl_error**2,axis=0)*basis.dx))
    return basis.N,l2_error,curl_norm


def main():
    print(f"Manufactured tetrahedral Maxwell solution; alpha={ALPHA:g}, intorder={INTORDER}")
    print("order  n    DOFs       L2 error   rate      curl error rate")
    for label,element_type in ELEMENTS:
        previous=None
        for n in REFINEMENTS:
            ndofs,l2_error,curl_error=solve_level(element_type,n)
            if previous is None:
                l2_rate=curl_rate=np.nan
            else:
                old_n,old_l2,old_curl=previous
                ratio=np.log(n/old_n)
                l2_rate=np.log(old_l2/l2_error)/ratio
                curl_rate=np.log(old_curl/curl_error)/ratio
            print(
                f"{label:>4} {n:3d} {ndofs:7d}  {l2_error:10.3e} "
                f"{l2_rate:5.2f}  {curl_error:10.3e} {curl_rate:5.2f}"
            )
            previous=(n,l2_error,curl_error)


if __name__=="__main__":
    main()
