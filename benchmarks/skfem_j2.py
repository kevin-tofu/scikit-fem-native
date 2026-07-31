"""Vectorized scikit-fem reference forms for small-strain J2 plasticity."""

import numpy as np
from skfem import BilinearForm,LinearForm
from skfem.helpers import sym_grad


_METRIC=np.array([1.,1.,1.,2.,2.,2.])


def voigt(tensor):
    return np.stack((
        tensor[0,0],tensor[1,1],tensor[2,2],
        tensor[0,1],tensor[1,2],tensor[0,2],
    ))


def update(
    strain,young,poisson,yield_stress,hardening,
    plastic_strain=None,equivalent_plastic_strain=None,
):
    """Vectorized radial return matching ``skfn.J2Plasticity``."""
    if plastic_strain is None:
        plastic_strain=np.zeros_like(strain)
    if equivalent_plastic_strain is None:
        equivalent_plastic_strain=np.zeros(strain.shape[1:],dtype=strain.dtype)
    mu=young/(2.*(1.+poisson))
    lmbda=young*poisson/((1.+poisson)*(1.-2.*poisson))
    bulk=young/(3.*(1.-2.*poisson))
    elastic=strain-plastic_strain
    trace=elastic[0]+elastic[1]+elastic[2]
    trial=2.*mu*elastic
    trial[:3]+=lmbda*trace
    mean=(trial[0]+trial[1]+trial[2])/3.
    deviator=trial.copy();deviator[:3]-=mean
    equivalent=np.sqrt(1.5*np.einsum(
        "i...,i...->...",_METRIC[:,None,None]*deviator,deviator
    ))
    current_yield=yield_stress+hardening*equivalent_plastic_strain
    plastic=equivalent>current_yield
    safe=np.maximum(equivalent,1e-30)
    increment=np.where(
        plastic,(equivalent-current_yield)/(3.*mu+hardening),0.
    )
    scale=1.-3.*mu*increment/safe
    stress=deviator*scale
    stress[:3]+=mean

    tangent=np.empty((6,6)+strain.shape[1:],dtype=strain.dtype)
    denominator=3.*mu+hardening
    tangent_scale=np.where(
        plastic,
        hardening/denominator
        +3.*mu*current_yield/(denominator*safe),
        1.,
    )
    directional=np.where(
        plastic,
        9.*mu*mu*current_yield/(denominator*safe**3),
        0.,
    )
    for column in range(6):
        basis=np.zeros(6);basis[column]=1.
        basis_trace=basis[:3].sum()
        basis_deviator=basis.copy();basis_deviator[:3]-=basis_trace/3.
        projection=np.einsum(
            "i...,i->...",deviator,_METRIC*basis_deviator
        )
        for row in range(6):
            tangent[row,column]=(
                (bulk*basis_trace if row<3 else 0.)
                +2.*mu*tangent_scale*basis_deviator[row]
                -directional*deviator[row]*projection
            )
    flow=1.5*deviator/safe
    trial_plastic=plastic_strain+np.where(
        plastic[None,...],increment[None,...]*flow,0.
    )
    trial_alpha=equivalent_plastic_strain+increment
    return stress,tangent,trial_plastic,trial_alpha


def forms():
    @LinearForm
    def residual(v,w):
        test=voigt(sym_grad(v))
        return np.einsum(
            "i...,i...->...",_METRIC[:,None,None]*test,w["stress"]
        )

    @BilinearForm
    def tangent(increment,v,w):
        trial=voigt(sym_grad(increment))
        test=voigt(sym_grad(v))
        return np.einsum(
            "i...,ij...,j...->...",
            _METRIC[:,None,None]*test,w["constitutive"],trial,
        )

    return residual,tangent
