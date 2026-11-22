"""Isotropic small-strain elasticity constitutive model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IsotropicLinearElasticity:
    """Three-dimensional isotropic Hooke material.

    The returned fourth-order tensor maps the displacement gradient to the
    stress in a ``grad(u):grad(v)`` bilinear form. This is a constitutive law,
    independent of any particular finite-element API.
    """

    young_modulus: float
    poisson_ratio: float

    def __post_init__(self) -> None:
        young = float(self.young_modulus)
        poisson = float(self.poisson_ratio)
        if not np.isfinite(young) or young <= 0.0:
            raise ValueError("young_modulus must be positive and finite")
        if not np.isfinite(poisson) or not -1.0 < poisson < 0.5:
            raise ValueError("poisson_ratio must be finite and in (-1, 0.5)")
        object.__setattr__(self, "young_modulus", young)
        object.__setattr__(self, "poisson_ratio", poisson)

    @property
    def lame_lambda(self) -> float:
        e, nu = self.young_modulus, self.poisson_ratio
        return e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    @property
    def lame_mu(self) -> float:
        return self.young_modulus / (2.0 * (1.0 + self.poisson_ratio))

    @property
    def stiffness_tensor(self) -> np.ndarray:
        identity = np.eye(3)
        return (
            self.lame_lambda * np.einsum("ij,kl->ijkl", identity, identity)
            + self.lame_mu
            * (
                np.einsum("ik,jl->ijkl", identity, identity)
                + np.einsum("il,jk->ijkl", identity, identity)
            )
        )

    @property
    def voigt_stiffness(self) -> np.ndarray:
        """Matrix for engineering strain order ``xx, yy, zz, xy, yz, xz``."""
        lam, mu = self.lame_lambda, self.lame_mu
        matrix = np.zeros((6, 6), dtype=np.float64)
        matrix[:3, :3] = lam
        matrix[np.arange(3), np.arange(3)] += 2.0 * mu
        matrix[3:, 3:] = np.eye(3) * mu
        return matrix

    def stress(self, strain: np.ndarray) -> np.ndarray:
        values = np.asarray(strain, dtype=np.float64)
        if values.ndim < 2 or values.shape[-2:] != (3, 3):
            raise ValueError("strain must have trailing shape (3, 3)")
        if not np.all(np.isfinite(values)):
            raise ValueError("strain must be finite")
        trace = np.trace(values, axis1=-2, axis2=-1)
        return (
            self.lame_lambda * trace[..., None, None] * np.eye(3)
            + 2.0 * self.lame_mu * values
        )


def isotropic_linear_elasticity_form(material: IsotropicLinearElasticity):
    """Return the weak form for 3D isotropic small-strain elasticity.

    The constitutive expression lives with the material model. The caller owns
    global assembly and coupling to constraints or other physical fields.
    """
    if not isinstance(material, IsotropicLinearElasticity):
        raise TypeError("material must be an IsotropicLinearElasticity")

    from .forms import BilinearForm
    from .helpers import ddot, div, sym_grad

    @BilinearForm
    def elasticity(u, v, _w):
        return (
            2.0 * material.lame_mu * ddot(sym_grad(u), sym_grad(v))
            + material.lame_lambda * div(u) * div(v)
        )

    return elasticity


__all__ = ["IsotropicLinearElasticity", "isotropic_linear_elasticity_form"]
