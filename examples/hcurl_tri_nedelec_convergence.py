"""Compare N1/N2/N3 H(curl) errors for a manufactured Maxwell solution."""

import numpy as np
from scipy.sparse.linalg import spsolve

import skfemntv
from skfemntv.helpers import curl,dot


ALPHA=.05
INTORDER=8
REFINEMENTS=(2,4,8,16)
ELEMENTS=(
    ("N1",skfemntv.ElementTriN1),
    ("N2",skfemntv.ElementTriN2),
    ("N3",skfemntv.ElementTriN3),
)


@skfemntv.BilinearForm
def maxwell(u,v,w):
    return dot(u,v)+ALPHA*curl(u)*curl(v)


@skfemntv.LinearForm
def manufactured_load(v,w):
    return dot(w.force,v)


def solve_level(element_type,subdivisions):
    mesh=skfemntv.MeshTri.init_tensor(
        np.linspace(0.,1.,subdivisions+1),
        np.linspace(0.,1.,subdivisions+1),
    )
    basis=skfemntv.Basis(mesh,element_type(),intorder=INTORDER)
    x=basis.global_coordinates[...,0]
    y=basis.global_coordinates[...,1]
    sin_x=np.sin(np.pi*x)
    sin_y=np.sin(np.pi*y)
    force=np.stack((
        ALPHA*np.pi**2*np.cos(np.pi*x)*np.cos(np.pi*y),
        (1.+ALPHA*np.pi**2)*sin_x*sin_y,
    ))
    matrix=skfemntv.asm(maxwell,basis).tocsr()
    load=skfemntv.asm(manufactured_load,basis,force=force)
    boundary=basis.get_dofs().all()
    free=np.setdiff1d(np.arange(basis.N),boundary)
    coefficients=np.zeros(basis.N)
    coefficients[free]=spsolve(matrix[free][:,free],load[free])

    numerical=basis.interpolate(coefficients)
    exact=np.zeros_like(numerical.value)
    exact[1]=sin_x*sin_y
    exact_curl=np.pi*np.cos(np.pi*x)*sin_y
    value_error=numerical.value-exact
    curl_error=numerical.curl-exact_curl
    l2_error=np.sqrt(np.einsum(
        "ieq,ieq,eq->",value_error,value_error,basis.dx,
    ))
    curl_norm=np.sqrt(np.sum(curl_error**2*basis.dx))
    return basis.N,l2_error,curl_norm


def main():
    print(f"Manufactured Maxwell solution; alpha={ALPHA:g}, intorder={INTORDER}")
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
