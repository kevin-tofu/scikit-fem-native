"""Equivalent scikit-fem forms used as a nonlinear reference."""

import numpy as np
from skfem import BilinearForm, LinearForm
from skfem.helpers import ddot, det, eye, grad, inv, transpose


def forms(mu, lmbda):
    @LinearForm
    def residual(v, w):
        field = w["displacement"]
        deformation_gradient = (
            eye(1.0 + 0.0 * field.grad[0, 0], 3) + field.grad
        )
        inverse_transpose = transpose(inv(deformation_gradient))
        first_piola = (
            mu * (deformation_gradient - inverse_transpose)
            + lmbda
            * np.log(det(deformation_gradient))
            * inverse_transpose
        )
        return ddot(first_piola, grad(v))

    @BilinearForm
    def tangent(increment, v, w):
        field = w["displacement"]
        deformation_gradient = (
            eye(1.0 + 0.0 * field.grad[0, 0], 3) + field.grad
        )
        inverse_transpose = transpose(inv(deformation_gradient))
        log_jacobian = np.log(det(deformation_gradient))
        increment_gradient = grad(increment)
        test_gradient = grad(v)
        return (
            mu * ddot(increment_gradient, test_gradient)
            + (mu - lmbda * log_jacobian)
            * np.einsum(
                "il...,kj...,kl...,ij...",
                inverse_transpose,
                inverse_transpose,
                increment_gradient,
                test_gradient,
            )
            + lmbda
            * np.einsum(
                "ij...,kl...,kl...,ij...",
                inverse_transpose,
                inverse_transpose,
                increment_gradient,
                test_gradient,
            )
        )

    return residual, tangent
