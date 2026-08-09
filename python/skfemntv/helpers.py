"""Form helpers implemented independently by skfemntv."""

import numpy as np

from .forms import ddot, div, dot, grad
from .interface import avg, jump, normal_grad


def trace(value):
    return np.einsum("ii...->...", value)


def transpose(value):
    return np.einsum("ij...->ji...", value)


def sym_grad(value):
    gradient = grad(value)
    if gradient.__class__.__name__ in {"_TrialGradient","_TestGradient"}:
        from .forms import _SymmetricGradient
        role="trial" if gradient.__class__.__name__=="_TrialGradient" else "test"
        return _SymmetricGradient(role,gradient.factor)
    return 0.5 * (gradient + transpose(gradient))


def isotropic_traction_tensor(normal,lame_lambda,lame_mu):
    """Return the coefficient mapping ``grad(u)`` to ``sigma(u) @ n``.

    The returned trailing axes are ``(traction component, displacement
    component, derivative direction)``.  Leading entity/quadrature axes are
    retained, so the result can be used directly in an interface form.
    """
    normal=np.asarray(normal,dtype=float)
    if normal.ndim<1:
        raise ValueError("normal must have a spatial component axis")
    dimension=normal.shape[0]
    normal=np.moveaxis(normal,0,-1)
    lame_lambda=np.asarray(lame_lambda,dtype=float)
    lame_mu=np.asarray(lame_mu,dtype=float)
    leading=np.broadcast_shapes(
        normal.shape[:-1],lame_lambda.shape,lame_mu.shape
    )
    normal=np.broadcast_to(normal,leading+(dimension,))
    lame_lambda=np.broadcast_to(lame_lambda,leading)
    lame_mu=np.broadcast_to(lame_mu,leading)
    identity=np.eye(dimension)
    return (
        lame_lambda[...,None,None,None]
        *normal[..., :,None,None]*identity[None,...]
        +lame_mu[...,None,None,None]
        *identity[:, :,None]*normal[...,None,None,:]
        +lame_mu[...,None,None,None]
        *identity[:,None,:]*normal[...,None,:,None]
    )
