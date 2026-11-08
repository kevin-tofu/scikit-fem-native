"""Experimental finite-strain isotropic plasticity material points.

The implementation uses a multiplicative elastic/plastic split and a
logarithmic-strain J2 return map, with a plane-strain finite-element assembler
for the large-rotation sheet-forming path. The default material-point tangent
is an analytic consistent derivative; centered finite differences remain
available as a verification mode.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .basis import Basis, ElementTriP1, ElementTriP2, ElementVector


@dataclass(frozen=True)
class FiniteStrainJ2Evaluation:
    """Batched trial response; stress is Cauchy and tangent is ``dP/dF``."""

    cauchy_stress: np.ndarray
    first_piola_stress: np.ndarray
    tangent: np.ndarray | None
    trial_state: np.ndarray
    plastic_points: np.ndarray
    equivalent_stress: np.ndarray
    plastic_multiplier: np.ndarray


class FiniteStrainJ2Material:
    """Multiplicative J2 plasticity with Hencky elasticity and isotropic hardening.

    Deformation gradients have shape ``(..., 3, 3)``.  Plane strain is
    represented by embedding a 2D in-plane gradient in 3D with
    ``F[2, 2] = 1`` and zero out-of-plane shear.  History stores the plastic
    deformation gradient (nine entries) and accumulated equivalent plastic
    strain.  Evaluation is transactional: the input history is never changed.

    The default tangent is the consistent directional derivative ``dP/dF``
    at fixed committed history.  A centered finite-difference mode is retained
    as a reference for material-point verification.  In the plastic regime the
    exact first-Piola tangent can be nonsymmetric; callers must preserve that
    Jacobian instead of replacing it with its symmetric part.
    """

    def __init__(
        self,
        young_modulus: float,
        poisson_ratio: float,
        yield_stress: float,
        hardening_modulus: float = 0.0,
        *,
        hardening_curve=None,
        tangent_step: float = 2e-7,
        tangent_method: str = "analytic",
    ):
        values = np.asarray(
            [young_modulus, poisson_ratio, yield_stress, hardening_modulus,
             tangent_step], dtype=float
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("material parameters must be finite")
        if young_modulus <= 0. or not -1. < poisson_ratio < 0.5:
            raise ValueError("require E > 0 and -1 < poisson_ratio < 0.5")
        if yield_stress <= 0. or hardening_modulus < 0. or tangent_step <= 0.:
            raise ValueError("invalid yield stress, hardening, or tangent step")
        if tangent_method not in {"analytic", "finite_difference"}:
            raise ValueError("tangent_method must be analytic or finite_difference")
        self.young_modulus = float(young_modulus)
        self.poisson_ratio = float(poisson_ratio)
        self.yield_stress = float(yield_stress)
        self.hardening_modulus = float(hardening_modulus)
        self.tangent_step = float(tangent_step)
        self.tangent_method = tangent_method
        self.mu = self.young_modulus / (2. * (1. + self.poisson_ratio))
        self.bulk_modulus = self.young_modulus / (3. * (1. - 2. * self.poisson_ratio))
        self._hardening_curve = None
        if hardening_curve is not None:
            curve = np.asarray(hardening_curve, dtype=float)
            if (curve.ndim != 2 or curve.shape[1] != 2 or curve.shape[0] < 2
                    or not np.all(np.isfinite(curve))
                    or np.any(np.diff(curve[:, 0]) <= 0.)
                    or np.any(np.diff(curve[:, 1]) < 0.)
                    or curve[0, 0] != 0. or np.any(curve[:, 1] <= 0.)):
                raise ValueError(
                    "hardening_curve must contain increasing (plastic_strain, yield_stress) rows"
                )
            if not np.isclose(curve[0, 1], yield_stress, rtol=1e-10, atol=1e-12):
                raise ValueError("yield_stress must match the first hardening-curve value")
            self._hardening_curve = curve.copy()

    @staticmethod
    def _dev(tensor: np.ndarray) -> np.ndarray:
        return tensor - np.trace(tensor) * np.eye(3) / 3.

    def _yield(self, equivalent_plastic_strain: float) -> float:
        if self._hardening_curve is None:
            return self.yield_stress + self.hardening_modulus * equivalent_plastic_strain
        curve = self._hardening_curve
        if equivalent_plastic_strain <= curve[-1, 0]:
            return float(np.interp(equivalent_plastic_strain, curve[:, 0], curve[:, 1]))
        slope = ((curve[-1, 1] - curve[-2, 1])
                 / (curve[-1, 0] - curve[-2, 0]))
        return float(curve[-1, 1] + slope * (equivalent_plastic_strain - curve[-1, 0]))

    @staticmethod
    def _symmetric_spectral(matrix: np.ndarray, function) -> np.ndarray:
        eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
        values = function(eigenvalues)
        return (eigenvectors * values) @ eigenvectors.T

    @staticmethod
    def _symmetric_spectral_direction(
        eigenvalues: np.ndarray,
        eigenvectors: np.ndarray,
        direction: np.ndarray,
        function,
        derivative,
    ) -> np.ndarray:
        """Frechet derivative of a symmetric matrix function in one direction."""
        values = eigenvalues[:, None]
        other_values = eigenvalues[None, :]
        delta = values - other_values
        scale = np.maximum(np.maximum(np.abs(values), np.abs(other_values)), 1.)
        separated = np.abs(delta) > 1e-8 * scale
        divided = np.empty_like(delta)
        function_values = function(eigenvalues)
        numerator = function_values[:, None] - function_values[None, :]
        np.divide(numerator, delta, out=divided, where=separated)
        midpoint = 0.5 * (values + other_values)
        divided[~separated] = derivative(midpoint[~separated])
        local_direction = eigenvectors.T @ direction @ eigenvectors
        result = eigenvectors @ (divided * local_direction) @ eigenvectors.T
        return 0.5 * (result + result.T)

    def _hardening_slope(self, equivalent_plastic_strain: float) -> float:
        if self._hardening_curve is None:
            return self.hardening_modulus
        curve = self._hardening_curve
        segment = int(np.searchsorted(
            curve[:, 0], equivalent_plastic_strain, side="right"
        ) - 1)
        segment = min(max(segment, 0), curve.shape[0] - 2)
        return float(
            (curve[segment + 1, 1] - curve[segment, 1])
            / (curve[segment + 1, 0] - curve[segment, 0])
        )

    def _consistent_tangent(
        self, deformation_gradient: np.ndarray, committed: np.ndarray,
        tangent_components: tuple[int, ...],
        tangent_direction: np.ndarray | None = None,
    ) -> np.ndarray:
        """Differentiate the Hencky radial-return stress with fixed history."""
        f = deformation_gradient
        fp_inverse = np.linalg.inv(committed[:9].reshape(3, 3))
        fe = f @ fp_inverse
        ce = fe.T @ fe
        eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (ce + ce.T))
        if np.min(eigenvalues) <= 1e-14:
            raise ValueError("elastic right Cauchy-Green tensor is not positive definite")
        log_ce = (eigenvectors * np.log(eigenvalues)) @ eigenvectors.T
        log_strain = 0.5 * log_ce
        stretch = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T
        rotation = fe @ np.linalg.inv(stretch)

        alpha = float(committed[9])
        deviatoric_strain = self._dev(log_strain)
        deviatoric_trial = 2. * self.mu * deviatoric_strain
        q_trial = float(np.sqrt(max(
            0., 1.5 * np.sum(deviatoric_trial ** 2)
        )))
        yield_trial = self._yield(alpha)
        yield_distance = q_trial - yield_trial
        yield_tolerance = 1e-10 * max(1., yield_trial)
        plastic = yield_distance > yield_tolerance
        # At first yield the constitutive map is only directionally
        # differentiable.  Select the generalized derivative from the
        # requested Newton direction when one is available.  The stress and
        # trial history always follow the exact return-map branch below.
        plastic_tangent = plastic
        if abs(yield_distance) <= yield_tolerance:
            if tangent_direction is None:
                # Keep the historical loading-side choice for callers that
                # request a state-independent matrix tangent.
                plastic_tangent = True
            else:
                d_fe = tangent_direction @ fp_inverse
                d_ce = d_fe.T @ fe + fe.T @ d_fe
                d_log_strain = 0.5 * self._symmetric_spectral_direction(
                    eigenvalues, eigenvectors, d_ce, np.log,
                    lambda value: 1. / value,
                )
                d_deviatoric_strain = self._dev(d_log_strain)
                d_q = (
                    6. * self.mu ** 2
                    * np.sum(deviatoric_strain * d_deviatoric_strain)
                    / max(q_trial, 1e-30)
                )
                plastic_tangent = d_q > 0.
        gamma = self._plastic_multiplier(q_trial, alpha) if plastic else 0.
        elastic_log_strain = log_strain.copy()
        if plastic:
            elastic_log_strain -= (3. * self.mu * gamma / q_trial) * deviatoric_strain

        tau_corotational = (
            2. * self.mu * self._dev(elastic_log_strain)
            + self.bulk_modulus * np.trace(elastic_log_strain) * np.eye(3)
        )
        tau = rotation @ tau_corotational @ rotation.T
        inverse_f_transpose = np.linalg.inv(f).T
        tangent = np.zeros((3, 3, 3, 3), dtype=float)

        for k in tangent_components:
            for ell in tangent_components:
                d_f = np.zeros((3, 3), dtype=float)
                d_f[k, ell] = 1.
                d_fe = d_f @ fp_inverse
                d_ce = d_fe.T @ fe + fe.T @ d_fe
                d_log_strain = 0.5 * self._symmetric_spectral_direction(
                    eigenvalues, eigenvectors, d_ce, np.log,
                    lambda value: 1. / value,
                )
                d_deviatoric_strain = self._dev(d_log_strain)

                if plastic_tangent:
                    d_q = (
                        6. * self.mu ** 2
                        * np.sum(deviatoric_strain * d_deviatoric_strain)
                        / q_trial
                    )
                    hardening_slope = self._hardening_slope(alpha + gamma)
                    d_gamma = d_q / (3. * self.mu + hardening_slope)
                    scale_dev = 1. - 3. * self.mu * gamma / q_trial
                    d_scale_dev = -3. * self.mu * (
                        d_gamma / q_trial - gamma * d_q / q_trial ** 2
                    )
                    d_elastic_log_strain = (
                        d_log_strain
                        - (1. - scale_dev) * d_deviatoric_strain
                        + d_scale_dev * deviatoric_strain
                    )
                else:
                    d_elastic_log_strain = d_log_strain

                d_tau_corotational = (
                    2. * self.mu * self._dev(d_elastic_log_strain)
                    + self.bulk_modulus * np.trace(d_log_strain) * np.eye(3)
                )
                d_stretch = self._symmetric_spectral_direction(
                    eigenvalues, eigenvectors, d_ce, np.sqrt,
                    lambda value: 0.5 / np.sqrt(value),
                )
                inverse_stretch = np.linalg.inv(stretch)
                d_rotation = d_fe @ inverse_stretch - (
                    rotation @ d_stretch @ inverse_stretch
                )
                d_tau = (
                    d_rotation @ tau_corotational @ rotation.T
                    + rotation @ d_tau_corotational @ rotation.T
                    + rotation @ tau_corotational @ d_rotation.T
                )
                d_first_piola = (
                    d_tau @ inverse_f_transpose
                    - tau @ inverse_f_transpose @ d_f.T @ inverse_f_transpose
                )
                tangent[:, :, k, ell] = d_first_piola

        return tangent

    def _update(self, deformation_gradient: np.ndarray, committed: np.ndarray):
        f = deformation_gradient
        fp = committed[:9].reshape(3, 3)
        alpha = float(committed[9])
        determinant = float(np.linalg.det(f))
        determinant_fp = float(np.linalg.det(fp))
        if determinant <= 1e-12 or determinant_fp <= 1e-12:
            raise ValueError("deformation and plastic gradients must have positive determinants")

        fe_trial = f @ np.linalg.inv(fp)
        ce_trial = fe_trial.T @ fe_trial
        eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (ce_trial + ce_trial.T))
        if np.min(eigenvalues) <= 1e-14:
            raise ValueError("elastic right Cauchy-Green tensor is not positive definite")
        log_strain_trial = (eigenvectors * (0.5 * np.log(eigenvalues))) @ eigenvectors.T
        stretch_trial = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T
        rotation = fe_trial @ np.linalg.inv(stretch_trial)

        deviatoric_trial = 2. * self.mu * self._dev(log_strain_trial)
        q_trial = float(np.sqrt(max(0., 1.5 * np.sum(deviatoric_trial**2))))
        yield_trial = self._yield(alpha)
        plastic = q_trial - yield_trial > 1e-10 * max(1., yield_trial)
        gamma = 0.
        log_strain_elastic = log_strain_trial
        trial_state = committed.copy()

        if plastic:
            gamma = self._plastic_multiplier(q_trial, alpha)
            flow = 1.5 * deviatoric_trial / q_trial
            log_strain_elastic = log_strain_trial - gamma * flow
            stretch_elastic = self._symmetric_spectral(
                log_strain_elastic, np.exp
            )
            fe_elastic = rotation @ stretch_elastic
            fp_new = np.linalg.solve(fe_elastic, f)
            trial_state[:9] = fp_new.reshape(-1)
            trial_state[9] = alpha + gamma

        tau_corotational = (
            2. * self.mu * self._dev(log_strain_elastic)
            + self.bulk_modulus * np.trace(log_strain_elastic) * np.eye(3)
        )
        tau = rotation @ tau_corotational @ rotation.T
        cauchy = tau / determinant
        first_piola = tau @ np.linalg.inv(f).T
        equivalent_stress = float(np.sqrt(max(
            0., 1.5 * np.sum(self._dev(tau)**2)
        )))
        return cauchy, first_piola, trial_state, plastic, equivalent_stress, gamma


    def _plastic_multiplier(
        self, trial_equivalent_stress: float, equivalent_plastic_strain: float
    ) -> float:
        """Solve the scalar J2 consistency equation for piecewise-linear hardening.

        On each tabulated hardening interval the consistency equation is linear
        in the plastic multiplier.  Solving that segment directly avoids an
        iterative root finder in every material evaluation and tangent
        linearization.
        """
        if self._hardening_curve is None:
            return float(
                (trial_equivalent_stress - self.yield_stress
                 - self.hardening_modulus * equivalent_plastic_strain)
                / (3. * self.mu + self.hardening_modulus)
            )

        curve = self._hardening_curve
        segment = int(np.searchsorted(
            curve[:, 0], equivalent_plastic_strain, side="right"
        ) - 1)
        segment = min(max(segment, 0), curve.shape[0] - 2)
        while True:
            lower_strain, lower_yield = curve[segment]
            upper_strain, upper_yield = curve[segment + 1]
            slope = (upper_yield - lower_yield) / (upper_strain - lower_strain)
            intercept = lower_yield - slope * lower_strain
            increment = (
                trial_equivalent_stress
                - intercept - slope * equivalent_plastic_strain
            ) / (3. * self.mu + slope)
            updated_strain = equivalent_plastic_strain + increment
            segment_tolerance = 1e-13 * max(1., abs(upper_strain))
            if (updated_strain <= upper_strain + segment_tolerance
                    or segment == curve.shape[0] - 2):
                return float(increment)
            segment += 1

    def initial_state(self, point_shape=()) -> np.ndarray:
        """Return identity plastic gradients and zero accumulated plastic strain."""
        if isinstance(point_shape, (int, np.integer)):
            point_shape = (int(point_shape),)
        shape = tuple(point_shape)
        if any(not isinstance(size, (int, np.integer)) or size < 0 for size in shape):
            raise ValueError("point_shape must contain nonnegative integers")
        state = np.zeros(tuple(int(size) for size in shape) + (10,), dtype=float)
        state[..., 0] = state[..., 4] = state[..., 8] = 1.
        return state

    @staticmethod
    def _validate_state(state: np.ndarray) -> None:
        if np.any(state[..., 9] < 0.):
            raise ValueError("committed equivalent plastic strain must be nonnegative")
        plastic_gradients = state[..., :9].reshape(-1, 3, 3)
        determinants = np.linalg.det(plastic_gradients)
        if (np.any(determinants <= 0.)
                or not np.allclose(determinants, 1., rtol=1e-8, atol=1e-10)):
            raise ValueError("committed plastic deformation gradients must be isochoric")

    def evaluate(
        self, deformation_gradient, committed_state, *, tangent=True,
        tangent_components=None, tangent_direction=None,
    ):
        """Evaluate trial states without changing committed history.

        ``tangent_direction`` optionally selects the loading or unloading
        generalized derivative at points on the J2 yield surface. Its shape
        is the deformation-gradient batch shape followed by ``(3, 3)``.
        Away from the yield surface it does not change the consistent tangent.
        """
        f = np.asarray(deformation_gradient, dtype=float)
        if f.ndim < 2 or f.shape[-2:] != (3, 3) or not np.all(np.isfinite(f)):
            raise ValueError("deformation_gradient must be finite with trailing shape (3, 3)")
        batch = f.shape[:-2]
        state = np.asarray(committed_state, dtype=float)
        if state.shape != batch + (10,) or not np.all(np.isfinite(state)):
            raise ValueError(f"committed_state must have finite shape {batch + (10,)}")
        direction = None
        if tangent_direction is not None:
            direction = np.asarray(tangent_direction, dtype=float)
            if (direction.shape != batch + (3, 3)
                    or not np.all(np.isfinite(direction))):
                raise ValueError(
                    "tangent_direction must be finite with shape "
                    f"{batch + (3, 3)}"
                )
        self._validate_state(state)
        flat_f = np.ascontiguousarray(f.reshape(-1, 3, 3))
        flat_state = np.ascontiguousarray(state.reshape(-1, 10))
        responses = [self._update(point_f, point_state)
                     for point_f, point_state in zip(flat_f, flat_state)]
        cauchy = np.asarray([item[0] for item in responses]).reshape(batch + (3, 3))
        first_piola = np.asarray([item[1] for item in responses]).reshape(batch + (3, 3))
        trial_state = np.asarray([item[2] for item in responses]).reshape(batch + (10,))
        plastic = np.asarray([item[3] for item in responses], dtype=bool).reshape(batch)
        equivalent = np.asarray([item[4] for item in responses]).reshape(batch)
        multiplier = np.asarray([item[5] for item in responses]).reshape(batch)

        algorithmic_tangent = None
        if tangent:
            components = (0, 1, 2) if tangent_components is None else tuple(
                int(component) for component in tangent_components
            )
            if (not components or len(set(components)) != len(components)
                    or any(component < 0 or component > 2 for component in components)):
                raise ValueError("tangent_components must be unique indices from 0 to 2")
            algorithmic_tangent = np.empty(batch + (3, 3, 3, 3), dtype=float)
            tangent_flat = algorithmic_tangent.reshape(-1, 3, 3, 3, 3)
            if self.tangent_method == "analytic":
                flat_direction = (
                    None if direction is None
                    else np.ascontiguousarray(direction.reshape(-1, 3, 3))
                )
                for point, (point_f, point_state) in enumerate(zip(flat_f, flat_state)):
                    tangent_flat[point] = self._consistent_tangent(
                        point_f, point_state, components,
                        None if flat_direction is None else flat_direction[point],
                    )
            else:
                tangent_flat.fill(0.)
                for point, (point_f, point_state) in enumerate(zip(flat_f, flat_state)):
                    tangent_point = np.empty((3, 3, 3, 3), dtype=float)
                    tangent_point.fill(0.)
                    for k in components:
                        for ell in components:
                            step = self.tangent_step * max(1., abs(point_f[k, ell]))
                            plus = point_f.copy()
                            minus = point_f.copy()
                            plus[k, ell] += step
                            minus[k, ell] -= step
                            p_plus = self._update(plus, point_state)[1]
                            p_minus = self._update(minus, point_state)[1]
                            tangent_point[:, :, k, ell] = (
                                (p_plus - p_minus) / (2. * step)
                            )
                    tangent_flat[point] = tangent_point
        return FiniteStrainJ2Evaluation(
            cauchy, first_piola, algorithmic_tangent, trial_state, plastic,
            equivalent, multiplier,
        )

    def commit(self, evaluation: FiniteStrainJ2Evaluation) -> np.ndarray:
        """Copy a trial history after the caller has accepted equilibrium."""
        if not isinstance(evaluation, FiniteStrainJ2Evaluation):
            raise TypeError("evaluation must be a FiniteStrainJ2Evaluation")
        if (evaluation.trial_state.ndim < 1 or evaluation.trial_state.shape[-1] != 10
                or not np.all(np.isfinite(evaluation.trial_state))):
            raise ValueError("evaluation has an invalid trial state")
        self._validate_state(evaluation.trial_state)
        return np.array(evaluation.trial_state, dtype=float, copy=True)


class FiniteStrainHill48Material(FiniteStrainJ2Material):
    """Finite Hencky strain plasticity with an associated Hill48 yield surface.

    This reuses the transactional multiplicative-plasticity state layout of
    :class:`FiniteStrainJ2Material` (nine plastic-gradient entries and one
    accumulated equivalent plastic strain). The local Hill48 return is solved
    in the elastic corotational Hencky frame. The first-Piola tangent combines
    the implicit Hill48 return-map tangent with the spectral Hencky and polar
    rotation derivatives.
    """

    def __init__(
        self,
        young_modulus: float,
        poisson_ratio: float,
        yield_stress: float,
        *,
        stress_ratios=(1., 1., 1., 1., 1., 1.),
        hardening_modulus: float = 0.,
        hardening_curve=None,
        tangent_step: float = 2e-7,
        tangent_method: str = "analytic",
    ):
        super().__init__(
            young_modulus,
            poisson_ratio,
            yield_stress,
            hardening_modulus,
            hardening_curve=hardening_curve,
            tangent_step=tangent_step,
            tangent_method=tangent_method,
        )
        from .hill48 import Hill48PlasticityMaterial

        self._hill48 = Hill48PlasticityMaterial(
            young_modulus,
            poisson_ratio,
            yield_stress=yield_stress,
            hardening_modulus=hardening_modulus,
            stress_ratios=stress_ratios,
            hardening_curve=hardening_curve,
        )
        self.stress_ratios = np.asarray(stress_ratios, dtype=float).copy()

    def _consistent_tangent(
        self, deformation_gradient: np.ndarray, committed: np.ndarray,
        tangent_components: tuple[int, ...],
        tangent_direction: np.ndarray | None = None,
    ) -> np.ndarray:
        f = deformation_gradient
        fp_inverse = np.linalg.inv(committed[:9].reshape(3, 3))
        fe = f @ fp_inverse
        ce = fe.T @ fe
        eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (ce + ce.T))
        if np.min(eigenvalues) <= 1e-14:
            raise ValueError(
                "elastic right Cauchy-Green tensor is not positive definite"
            )
        log_strain = 0.5 * (
            eigenvectors * np.log(eigenvalues)
        ) @ eigenvectors.T
        stretch = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T
        rotation = fe @ np.linalg.inv(stretch)

        alpha = float(committed[9])
        local_history = np.zeros(7, dtype=float)
        local_history[6] = alpha
        stress_corotational, updated_history, _, plastic = (
            self._hill48._stress_update(log_strain, local_history)
        )
        gamma = float(updated_history[6] - alpha)
        trial_stress = np.einsum(
            "ijkl,kl->ij", self._hill48._elastic, log_strain
        )
        trial_equivalent, trial_gradient = (
            self._hill48._equivalent_and_gradient(trial_stress)
        )
        yield_value = self._hill48._yield_stress(alpha)
        yield_tolerance = 1e-10 * max(1., yield_value)
        plastic_tangent = plastic
        if abs(trial_equivalent - yield_value) <= yield_tolerance:
            if tangent_direction is None:
                plastic_tangent = True
            else:
                d_fe = tangent_direction @ fp_inverse
                d_ce = d_fe.T @ fe + fe.T @ d_fe
                d_log_strain = 0.5 * self._symmetric_spectral_direction(
                    eigenvalues, eigenvectors, d_ce, np.log,
                    lambda value: 1. / value,
                )
                d_trial_stress = np.einsum(
                    "ijkl,kl->ij", self._hill48._elastic, d_log_strain
                )
                d_equivalent = float(np.sum(trial_gradient * d_trial_stress))
                plastic_tangent = d_equivalent > 0.

        if plastic_tangent:
            corotational_tangent = self._hill48._consistent_tangent(
                log_strain, np.zeros((3, 3)), alpha,
                stress_corotational, gamma,
            )
        else:
            corotational_tangent = self._hill48._elastic

        tau = rotation @ stress_corotational @ rotation.T
        inverse_f_transpose = np.linalg.inv(f).T
        tangent = np.zeros((3, 3, 3, 3), dtype=float)
        for k in tangent_components:
            for ell in tangent_components:
                d_f = np.zeros((3, 3), dtype=float)
                d_f[k, ell] = 1.
                d_fe = d_f @ fp_inverse
                d_ce = d_fe.T @ fe + fe.T @ d_fe
                d_log_strain = 0.5 * self._symmetric_spectral_direction(
                    eigenvalues, eigenvectors, d_ce, np.log,
                    lambda value: 1. / value,
                )
                d_tau_corotational = np.einsum(
                    "ijkl,kl->ij", corotational_tangent, d_log_strain
                )
                d_stretch = self._symmetric_spectral_direction(
                    eigenvalues, eigenvectors, d_ce, np.sqrt,
                    lambda value: 0.5 / np.sqrt(value),
                )
                inverse_stretch = np.linalg.inv(stretch)
                d_rotation = d_fe @ inverse_stretch - (
                    rotation @ d_stretch @ inverse_stretch
                )
                d_tau = (
                    d_rotation @ stress_corotational @ rotation.T
                    + rotation @ d_tau_corotational @ rotation.T
                    + rotation @ stress_corotational @ d_rotation.T
                )
                tangent[:, :, k, ell] = (
                    d_tau @ inverse_f_transpose
                    - tau @ inverse_f_transpose @ d_f.T @ inverse_f_transpose
                )
        return tangent

    def _update(self, deformation_gradient: np.ndarray, committed: np.ndarray):
        f = deformation_gradient
        fp = committed[:9].reshape(3, 3)
        determinant = float(np.linalg.det(f))
        if determinant <= 1e-12 or np.linalg.det(fp) <= 1e-12:
            raise ValueError(
                "deformation and plastic gradients must have positive determinants"
            )

        fe_trial = f @ np.linalg.inv(fp)
        ce_trial = fe_trial.T @ fe_trial
        eigenvalues, eigenvectors = np.linalg.eigh(
            0.5 * (ce_trial + ce_trial.T)
        )
        if np.min(eigenvalues) <= 1e-14:
            raise ValueError("elastic right Cauchy-Green tensor is not positive definite")
        log_strain_trial = (
            eigenvectors * (0.5 * np.log(eigenvalues))
        ) @ eigenvectors.T
        stretch_trial = (
            eigenvectors * np.sqrt(eigenvalues)
        ) @ eigenvectors.T
        rotation = fe_trial @ np.linalg.inv(stretch_trial)

        local_history = np.zeros(7, dtype=float)
        local_history[6] = committed[9]
        stress_corotational, updated_local_history, equivalent_stress, plastic = (
            self._hill48._stress_update(log_strain_trial, local_history)
        )
        elastic_log_strain = log_strain_trial - self._hill48._plastic_tensor(
            updated_local_history
        )
        elastic_stretch = self._symmetric_spectral(elastic_log_strain, np.exp)
        fe_elastic = rotation @ elastic_stretch
        fp_new = np.linalg.solve(fe_elastic, f)

        tau = rotation @ stress_corotational @ rotation.T
        cauchy = tau / determinant
        first_piola = tau @ np.linalg.inv(f).T
        trial_state = committed.copy()
        trial_state[:9] = fp_new.reshape(-1)
        trial_state[9] = updated_local_history[6]
        plastic_multiplier = float(updated_local_history[6] - committed[9])
        return (
            cauchy,
            first_piola,
            trial_state,
            bool(plastic),
            float(equivalent_stress),
            plastic_multiplier,
        )


@dataclass(frozen=True)
class PlaneStrainFiniteStrainPlasticEvaluation:
    """Finite-strain element trial with full 3D integration-point stresses."""

    residual: np.ndarray
    tangent: sparse.csr_matrix | None
    trial_state: np.ndarray
    plastic_points: np.ndarray
    cauchy_stress: np.ndarray
    equivalent_stress: np.ndarray
    plastic_multiplier: np.ndarray


class PlaneStrainFiniteStrainJ2Assembler:
    """Total-Lagrangian plane-strain assembler for finite-strain plasticity.

    The 2D deformation gradient is embedded in 3D with the out-of-plane stretch
    fixed to one. TriP1 and TriP2 vector displacement fields are supported.
    The material tangent is assembled without symmetrization; plastic
    configurations can therefore produce a nonsymmetric global Jacobian.
    ``FiniteStrainHill48Material`` is accepted through the common transactional
    material interface; its algorithmic tangent differentiates the local
    Hill48 return map and finite-strain kinematics.
    """

    def __init__(self, basis: Basis, material: FiniteStrainJ2Material):
        if (not isinstance(basis, Basis) or not isinstance(basis.elem, ElementVector)
                or basis.elem._dim != 2
                or not isinstance(basis.elem.elem, (ElementTriP1, ElementTriP2))):
            raise TypeError("basis must use a vector TriP1 or TriP2 displacement field")
        if not isinstance(material, FiniteStrainJ2Material):
            raise TypeError("material must be a FiniteStrainJ2Material")
        self.basis = basis
        self.material = material
        nodes = basis.element_dofs.shape[0] // 2
        gradients = np.asarray(basis.tabulated_gradients)
        if gradients.shape[2:] != (nodes, 2):
            raise ValueError("basis gradients do not match a 2D vector DOF layout")
        self._gradients = np.ascontiguousarray(gradients)
        self._dofs = np.ascontiguousarray(
            basis.element_dofs.T.reshape(basis.nelems, nodes, 2), dtype=np.int64
        )
        self._weights = np.ascontiguousarray(basis.dx)
        from .nonlinear import _prepare_tangent_scatter
        self._tangent_graph = _prepare_tangent_scatter(
            self._dofs.reshape(basis.nelems, -1), basis.N
        )

    def initial_state(self) -> np.ndarray:
        """Return identity plastic gradients at every quadrature point."""
        return self.material.initial_state(
            (self.basis.nelems, self._weights.shape[1])
        )

    def evaluate(
        self, displacement, committed_state, *, tangent: bool = True,
        tangent_direction=None,
    ) -> PlaneStrainFiniteStrainPlasticEvaluation:
        """Assemble internal force and optionally its trial algorithmic tangent.

        ``tangent_direction`` is an optional displacement-space direction used
        only to select the generalized derivative at yield-surface points.
        """
        u = np.ascontiguousarray(displacement, dtype=float)
        state = np.asarray(committed_state, dtype=float)
        expected = (self.basis.nelems, self._weights.shape[1], 10)
        if u.shape != (self.basis.N,) or not np.all(np.isfinite(u)):
            raise ValueError(f"displacement must be a finite vector of shape ({self.basis.N},)")
        if state.shape != expected or not np.all(np.isfinite(state)):
            raise ValueError(f"committed_state must be finite with shape {expected}")
        self.material._validate_state(state)
        direction = None
        if tangent_direction is not None:
            direction = np.ascontiguousarray(tangent_direction, dtype=float)
            if direction.shape != (self.basis.N,) or not np.all(np.isfinite(direction)):
                raise ValueError(
                    f"tangent_direction must be a finite vector of shape ({self.basis.N},)"
                )

        element_count, quadrature_count, node_count, _ = self._gradients.shape
        local_dofs = 2 * node_count
        local_residual = np.zeros((element_count, local_dofs), dtype=float)
        local_tangent = (
            np.zeros((element_count, local_dofs, local_dofs), dtype=float)
            if tangent else None
        )
        material_tangent = (
            np.empty((element_count, quadrature_count, 2, 2, 2, 2), dtype=float)
            if tangent else None
        )
        trial_state = np.empty_like(state)
        plastic_points = np.zeros((element_count, quadrature_count), dtype=bool)
        cauchy_stress = np.empty((element_count, quadrature_count, 3, 3), dtype=float)
        equivalent_stress = np.empty((element_count, quadrature_count), dtype=float)
        plastic_multiplier = np.empty((element_count, quadrature_count), dtype=float)

        for element in range(element_count):
            local_u = u[self._dofs[element]]
            local_direction = (
                None if direction is None
                else direction[self._dofs[element]]
            )
            for quadrature in range(quadrature_count):
                gradients = self._gradients[element, quadrature]
                deformation_2d = np.eye(2) + local_u.T @ gradients
                deformation_3d = np.eye(3)
                deformation_3d[:2, :2] = deformation_2d
                direction_3d = None
                if local_direction is not None:
                    direction_3d = np.zeros((3, 3), dtype=float)
                    direction_3d[:2, :2] = local_direction.T @ gradients
                response = self.material.evaluate(
                    deformation_3d, state[element, quadrature], tangent=tangent,
                    tangent_components=(0, 1) if tangent else None,
                    tangent_direction=direction_3d,
                )
                first_piola = response.first_piola_stress[:2, :2]
                weight = self._weights[element, quadrature]
                local_residual[element].reshape(node_count, 2)[:] += (
                    weight * (first_piola @ gradients.T).T
                )
                if tangent:
                    material_tangent[element, quadrature] = (
                        response.tangent[:2, :2, :2, :2]
                    )
                trial_state[element, quadrature] = response.trial_state
                plastic_points[element, quadrature] = response.plastic_points
                cauchy_stress[element, quadrature] = response.cauchy_stress
                equivalent_stress[element, quadrature] = response.equivalent_stress
                plastic_multiplier[element, quadrature] = response.plastic_multiplier

        if tangent:
            # Contract all element/quadrature tangent blocks at once.  The
            # previous scalar a-i-b-k loops issued one small einsum call for
            # every local matrix entry; this equivalent contraction keeps the
            # same (possibly nonsymmetric) algorithmic tangent while avoiding
            # that Python-level assembly overhead.
            local_tangent[:] = np.einsum(
                "eq,eqaj,eqijkl,eqbl->eaibk",
                self._weights, self._gradients, material_tangent,
                self._gradients, optimize=True,
            ).reshape(element_count, local_dofs, local_dofs)

        dofs = self._dofs.reshape(element_count, -1)
        if tangent:
            from .nonlinear import _assemble_global_response
            residual, tangent_matrix = _assemble_global_response(
                local_residual, local_tangent, dofs,
                (self.basis.N, self.basis.N), self._tangent_graph,
            )
        else:
            residual = np.zeros(self.basis.N, dtype=float)
            np.add.at(residual, dofs.reshape(-1), local_residual.reshape(-1))
            tangent_matrix = None
        return PlaneStrainFiniteStrainPlasticEvaluation(
            residual, tangent_matrix, trial_state, plastic_points, cauchy_stress,
            equivalent_stress, plastic_multiplier,
        )

    def commit(self, evaluation: PlaneStrainFiniteStrainPlasticEvaluation) -> np.ndarray:
        """Commit material history from an accepted equilibrium evaluation."""
        if not isinstance(evaluation, PlaneStrainFiniteStrainPlasticEvaluation):
            raise TypeError("evaluation must be a PlaneStrainFiniteStrainPlasticEvaluation")
        expected = (self.basis.nelems, self._weights.shape[1], 10)
        if evaluation.trial_state.shape != expected or not np.all(np.isfinite(evaluation.trial_state)):
            raise ValueError("evaluation has an invalid trial state")
        self.material._validate_state(evaluation.trial_state)
        return np.array(evaluation.trial_state, dtype=float, copy=True)


class PlaneStrainFiniteStrainHill48Assembler(PlaneStrainFiniteStrainJ2Assembler):
    """Plane-strain assembler bound to :class:`FiniteStrainHill48Material`."""

    def __init__(self, basis: Basis, material: FiniteStrainHill48Material):
        if not isinstance(material, FiniteStrainHill48Material):
            raise TypeError("material must be a FiniteStrainHill48Material")
        super().__init__(basis, material)


__all__ = [
    "FiniteStrainJ2Evaluation", "FiniteStrainJ2Material",
    "FiniteStrainHill48Material",
    "PlaneStrainFiniteStrainJ2Assembler",
    "PlaneStrainFiniteStrainHill48Assembler",
    "PlaneStrainFiniteStrainPlasticEvaluation",
]
