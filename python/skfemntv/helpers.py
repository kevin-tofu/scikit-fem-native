"""Form helpers implemented independently by skfemntv."""

import numpy as np

from .forms import ddot, div, dot, grad, mul
from ._h1_fields import (
    Curl,SymmetricGradient,TestGradient,TestValue,TrialGradient,TrialValue,
)
from .interface import avg, jump, normal_grad
from ._errors import UnsupportedNativeForm


def _reject_symbolic(*values):
    symbolic=(TestGradient,TestValue,TrialGradient,TrialValue)
    if any(isinstance(value,symbolic) for value in values):
        raise UnsupportedNativeForm(
            "this helper supports numeric arrays, not symbolic nonlinear form expressions"
        )


def det(matrix):
    """Compute 2x2 or 3x3 determinants with component axes first.

    Trailing axes are preserved, matching the layout of interpolated fields
    and quadrature coefficients in scikit-fem.
    """
    _reject_symbolic(matrix)
    matrix=np.asarray(matrix)
    if matrix.ndim<2 or matrix.shape[0]!=matrix.shape[1]:
        raise ValueError("det expects a square matrix with component axes first")
    if matrix.shape[0]==2:
        return matrix[0,0]*matrix[1,1]-matrix[0,1]*matrix[1,0]
    if matrix.shape[0]==3:
        return (
            matrix[0,0]*(matrix[1,1]*matrix[2,2]-matrix[1,2]*matrix[2,1])
            -matrix[0,1]*(matrix[1,0]*matrix[2,2]-matrix[1,2]*matrix[2,0])
            +matrix[0,2]*(matrix[1,0]*matrix[2,1]-matrix[1,1]*matrix[2,0])
        )
    raise ValueError("det supports only 2x2 and 3x3 matrices")


def inv(matrix):
    """Invert 2x2 or 3x3 matrices with component axes first."""
    _reject_symbolic(matrix)
    matrix=np.asarray(matrix)
    determinant=det(matrix)
    if matrix.shape[0]==2:
        result=np.empty_like(matrix,dtype=np.result_type(matrix,determinant,float))
        result[0,0]=matrix[1,1]/determinant
        result[0,1]=-matrix[0,1]/determinant
        result[1,0]=-matrix[1,0]/determinant
        result[1,1]=matrix[0,0]/determinant
        return result
    result=np.empty_like(matrix,dtype=np.result_type(matrix,determinant,float))
    result[0,0]=matrix[1,1]*matrix[2,2]-matrix[1,2]*matrix[2,1]
    result[1,0]=matrix[1,2]*matrix[2,0]-matrix[1,0]*matrix[2,2]
    result[2,0]=matrix[1,0]*matrix[2,1]-matrix[1,1]*matrix[2,0]
    result[0,1]=matrix[0,2]*matrix[2,1]-matrix[0,1]*matrix[2,2]
    result[1,1]=matrix[0,0]*matrix[2,2]-matrix[0,2]*matrix[2,0]
    result[2,1]=matrix[0,1]*matrix[2,0]-matrix[0,0]*matrix[2,1]
    result[0,2]=matrix[0,1]*matrix[1,2]-matrix[0,2]*matrix[1,1]
    result[1,2]=matrix[0,2]*matrix[1,0]-matrix[0,0]*matrix[1,2]
    result[2,2]=matrix[0,0]*matrix[1,1]-matrix[0,1]*matrix[1,0]
    return result/determinant


def eye(weight,n):
    """Return a diagonal ``n`` by ``n`` matrix scaled by ``weight``."""
    _reject_symbolic(weight)
    if isinstance(n,bool) or not isinstance(n,(int,np.integer)) or n<0:
        raise ValueError("n must be a nonnegative integer")
    weight=np.asarray(weight)
    result=np.zeros(
        (int(n),int(n))+weight.shape,dtype=np.result_type(weight,float)
    )
    for index in range(int(n)):
        result[index,index]=weight
    return result


def cross(left,right):
    """Compute component-first 2D scalar or 3D vector cross products."""
    _reject_symbolic(left,right)
    left=np.asarray(left)
    right=np.asarray(right)
    if left.shape!=right.shape or left.ndim<1:
        raise ValueError("cross operands must have matching component-first shapes")
    if left.shape[0]==2:
        return left[0]*right[1]-left[1]*right[0]
    if left.shape[0]==3:
        return np.asarray((
            left[1]*right[2]-left[2]*right[1],
            left[2]*right[0]-left[0]*right[2],
            left[0]*right[1]-left[1]*right[0],
        ))
    raise ValueError("cross supports 2D and 3D vectors")


def curl(value):
    """Return the curl of a discrete scalar or vector field.

    This matches ``skfem.helpers.curl`` for two-dimensional scalar/vector
    fields and three-dimensional vector fields. In bilinear forms, the
    supported symbolic uses are 2D scalar curl products and 3D curl dot
    products on vector H1 fields.
    """
    if isinstance(value,TrialValue):
        return Curl("trial",value.factor)
    if isinstance(value,TestValue):
        return Curl("test",value.factor)
    existing=getattr(value,"curl",None)
    if existing is not None:
        return existing
    gradient=getattr(value,"grad",None)
    if gradient is None:
        raise NotImplementedError("curl requires a field with gradients")
    gradient=np.asarray(gradient)
    if gradient.ndim==3 and gradient.shape[0]==2:
        return np.array([gradient[1],-gradient[0]])
    if gradient.ndim==4 and gradient.shape[0]==2 and gradient.shape[1]==2:
        return gradient[1,0]-gradient[0,1]
    if gradient.ndim==4 and gradient.shape[0]==3 and gradient.shape[1]==3:
        return np.array([
            gradient[2,1]-gradient[1,2],
            gradient[0,2]-gradient[2,0],
            gradient[1,0]-gradient[0,1],
        ])
    raise NotImplementedError(
        "curl supports scalar 2D and vector 2D/3D discrete fields"
    )


def trace(value):
    return np.einsum("ii...->...", value)


def transpose(value):
    return np.einsum("ij...->ji...", value)


def sym_grad(value):
    gradient = grad(value)
    if isinstance(gradient,(TrialGradient,TestGradient)):
        role="trial" if isinstance(gradient,TrialGradient) else "test"
        return SymmetricGradient(role,gradient.factor)
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
