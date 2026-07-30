"""Form helpers implemented independently by skfn."""

import numpy as np

from .forms import ddot, dot, grad
from .interface import avg, jump, normal_grad


def trace(value):
    return np.einsum("ii...->...", value)


def transpose(value):
    return np.einsum("ij...->ji...", value)


def sym_grad(value):
    gradient = grad(value)
    return 0.5 * (gradient + transpose(gradient))
