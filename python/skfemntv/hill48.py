"""Small-strain Hill48 material response and plane-strain assembly prototype."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import root

from .basis import (
    Basis, ElementHex1, ElementHex2, ElementHex20, ElementTetP1, ElementTetP2,
    ElementTriP1, ElementTriP2, ElementVector,
)
from .nonlinear import (
    PlasticEvaluation,
    _assemble_global_response,
    _prepare_tangent_scatter,
)


@dataclass(frozen=True)
class Hill48MaterialEvaluation:
    """Transactional Hill48 stress update for a batch of material points."""

    stress: np.ndarray
    tangent: np.ndarray
    trial_state: np.ndarray
    plastic_points: np.ndarray
    equivalent_stress: np.ndarray
    energy: np.ndarray


def _elastic_tensor(lam: float, mu: float) -> np.ndarray:
    eye = np.eye(3)
    return lam * np.einsum("ij,kl->ijkl", eye, eye) + mu * (
        np.einsum("ik,jl->ijkl", eye, eye)
        + np.einsum("il,jk->ijkl", eye, eye)
    )


def _pack_tensor(value: np.ndarray) -> np.ndarray:
    return np.array([value[0, 0], value[1, 1], value[2, 2],
                     value[0, 1], value[1, 2], value[0, 2]])


def _unpack_stress(value: np.ndarray) -> np.ndarray:
    result = np.zeros((3, 3))
    result[0, 0], result[1, 1], result[2, 2] = value[:3]
    result[0, 1] = result[1, 0] = value[3]
    result[1, 2] = result[2, 1] = value[4]
    result[0, 2] = result[2, 0] = value[5]
    return result


class Hill48PlasticityMaterial:
    """Associated Hill48 plasticity with isotropic linear hardening.

    ``stress_ratios`` follows Abaqus' order ``(R11, R22, R33, R12, R13,
    R23)``. The equivalent stress is normalized so six unit ratios recover
    von Mises plasticity. State has the existing J2 layout
    ``(ep11, ep22, ep33, ep12, ep23, ep13, ep_eq)``.

    This Python material-point implementation uses a local nonlinear solve
    and forms its algorithmic tangent by implicit differentiation of the
    return equations. The local Jacobian is analytic; a native batched kernel
    remains future work.
    """

    def __init__(self, young_modulus: float, poisson_ratio: float,
                 yield_stress: float | None = None, hardening_modulus: float = 0.0,
                 stress_ratios=(1., 1., 1., 1., 1., 1.),
                 hardening_curve=None):
        values = np.asarray([young_modulus, poisson_ratio, hardening_modulus], dtype=float)
        ratios = np.asarray(stress_ratios, dtype=float)
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(ratios)):
            raise ValueError("material parameters must be finite")
        if young_modulus <= 0. or not -1. < poisson_ratio < 0.5:
            raise ValueError("require young_modulus > 0 and -1 < poisson_ratio < 0.5")
        if hardening_modulus < 0.:
            raise ValueError("hardening_modulus must be nonnegative")
        if hardening_curve is None:
            if yield_stress is None or not np.isfinite(yield_stress) or yield_stress <= 0.:
                raise ValueError("yield_stress must be positive when no hardening curve is supplied")
            if hardening_modulus < 0.:
                raise ValueError("hardening_modulus must be nonnegative")
            self.hardening_curve = None
            self.yield_stress = float(yield_stress)
            self.hardening_modulus = float(hardening_modulus)
        else:
            curve = np.asarray(hardening_curve, dtype=float)
            if (curve.ndim != 2 or curve.shape[1] != 2 or curve.shape[0] < 2
                    or not np.all(np.isfinite(curve))):
                raise ValueError("hardening_curve must be a finite (n>=2, 2) array of (ep, stress)")
            if (curve[0, 0] != 0. or np.any(np.diff(curve[:, 0]) <= 0.)
                    or np.any(curve[:, 1] <= 0.) or np.any(np.diff(curve[:, 1]) < 0.)):
                raise ValueError("hardening curve must start at ep=0 and be monotone")
            if hardening_modulus != 0.:
                raise ValueError("specify either hardening_modulus or hardening_curve, not both")
            if (yield_stress is not None
                    and not np.isclose(yield_stress, curve[0, 1], rtol=1e-12, atol=1e-12)):
                raise ValueError("yield_stress must match the first hardening_curve stress")
            self.hardening_curve = curve.copy()
            self.yield_stress = float(curve[0, 1])
            self.hardening_modulus = 0.0
        if ratios.shape != (6,) or np.any(ratios <= 0.):
            raise ValueError("stress_ratios must contain six positive R values")
        self.young_modulus = float(young_modulus)
        self.poisson_ratio = float(poisson_ratio)
        self.stress_ratios = ratios.copy()
        r11, r22, r33, r12, r13, r23 = ratios
        self.F = 0.5 * (r22**-2 + r33**-2 - r11**-2)
        self.G = 0.5 * (r33**-2 + r11**-2 - r22**-2)
        self.H = 0.5 * (r11**-2 + r22**-2 - r33**-2)
        self.L = 1.5 / r23**2
        self.M = 1.5 / r13**2
        self.N = 1.5 / r12**2
        normal = np.array([
            [self.G + self.H, -self.H, -self.G],
            [-self.H, self.F + self.H, -self.F],
            [-self.G, -self.F, self.F + self.G],
        ])
        if np.linalg.eigvalsh(normal).min() < -1e-12:
            raise ValueError("stress ratios produce a nonconvex Hill48 surface")
        self.mu = self.young_modulus / (2. * (1. + self.poisson_ratio))
        self.lambda_ = self.young_modulus * self.poisson_ratio / (
            (1. + self.poisson_ratio) * (1. - 2. * self.poisson_ratio)
        )
        self._elastic = _elastic_tensor(self.lambda_, self.mu)
        self._elastic_voigt = np.diag([2. * self.mu, 2. * self.mu, 2. * self.mu,
                                       2. * self.mu, 2. * self.mu, 2. * self.mu])
        self._elastic_voigt[:3, :3] = self.lambda_
        self._elastic_voigt[np.arange(3), np.arange(3)] += 2. * self.mu
        self._hill_voigt = np.zeros((6, 6))
        self._hill_voigt[:3, :3] = normal
        self._hill_voigt[3, 3] = 2. * self.N
        self._hill_voigt[4, 4] = 2. * self.L
        self._hill_voigt[5, 5] = 2. * self.M
        self._tensor_voigt_flow = np.diag([1., 1., 1., 0.5, 0.5, 0.5])

    def yield_stress_at(self, plastic_strain):
        """Return isotropic yield stress at equivalent plastic strain."""
        alpha = np.asarray(plastic_strain, dtype=float)
        if not np.all(np.isfinite(alpha)) or np.any(alpha < 0.):
            raise ValueError("plastic_strain must be finite and nonnegative")
        if alpha.ndim == 0:
            return self._yield_stress(float(alpha))
        return np.asarray([self._yield_stress(float(value)) for value in alpha.flat]).reshape(alpha.shape)

    def _yield_stress(self, plastic_strain: float) -> float:
        if self.hardening_curve is None:
            return self.yield_stress + self.hardening_modulus * plastic_strain
        curve = self.hardening_curve
        if plastic_strain <= curve[-1, 0]:
            return float(np.interp(plastic_strain, curve[:, 0], curve[:, 1]))
        slope = (curve[-1, 1] - curve[-2, 1]) / (curve[-1, 0] - curve[-2, 0])
        return float(curve[-1, 1] + slope * (plastic_strain - curve[-1, 0]))

    def _hardening_slope(self, plastic_strain: float) -> float:
        if self.hardening_curve is None:
            return self.hardening_modulus
        curve = self.hardening_curve
        index = min(max(np.searchsorted(curve[:, 0], plastic_strain, side="right") - 1, 0),
                    curve.shape[0] - 2)
        return float((curve[index + 1, 1] - curve[index, 1]) /
                     (curve[index + 1, 0] - curve[index, 0]))

    def _hardening_work(self, start: float, end: float) -> float:
        if end <= start:
            return 0.0
        if self.hardening_curve is None:
            return (self.yield_stress * (end - start)
                    + 0.5 * self.hardening_modulus * (end**2 - start**2))
        work = 0.0
        current = start
        curve = self.hardening_curve
        while current < end:
            index = min(max(np.searchsorted(curve[:, 0], current, side="right") - 1, 0),
                        curve.shape[0] - 2)
            next_knot = curve[index + 1, 0]
            stop = min(end, next_knot) if current < curve[-1, 0] else end
            if stop <= current:
                stop = end
            work += 0.5 * (self._yield_stress(current) + self._yield_stress(stop)) * (stop - current)
            current = stop
        return float(work)

    def initial_state(self, point_shape=()) -> np.ndarray:
        if isinstance(point_shape, (int, np.integer)):
            point_shape = (int(point_shape),)
        shape = tuple(point_shape)
        return np.zeros(shape + (7,), dtype=float)

    def equivalent_stress(self, stress) -> np.ndarray:
        """Evaluate Hill48 equivalent stress for finite symmetric tensors."""
        sigma = np.asarray(stress, dtype=float)
        if sigma.ndim < 2 or sigma.shape[-2:] != (3, 3) or not np.all(np.isfinite(sigma)):
            raise ValueError("stress must be finite with trailing shape (3, 3)")
        if not np.allclose(sigma, sigma.swapaxes(-1, -2), rtol=1e-12, atol=1e-14):
            raise ValueError("stress tensors must be symmetric")
        values = np.empty(sigma.shape[:-2])
        for index in np.ndindex(values.shape):
            values[index] = self._equivalent_and_gradient(sigma[index])[0]
        if not values.shape:
            return np.asarray(self._equivalent_and_gradient(sigma)[0])
        return values

    def _equivalent_and_gradient(self, stress: np.ndarray):
        s11, s22, s33 = stress[0, 0], stress[1, 1], stress[2, 2]
        s12, s23, s13 = stress[0, 1], stress[1, 2], stress[0, 2]
        q2 = (
            self.F * (s22 - s33)**2
            + self.G * (s33 - s11)**2
            + self.H * (s11 - s22)**2
            + 2. * self.L * s23**2
            + 2. * self.M * s13**2
            + 2. * self.N * s12**2
        )
        q = float(np.sqrt(max(q2, 0.)))
        if q <= np.finfo(float).tiny:
            return q, np.zeros((3, 3))
        gradient = np.zeros((3, 3))
        gradient[0, 0] = (self.G * (s11 - s33) + self.H * (s11 - s22)) / q
        gradient[1, 1] = (self.F * (s22 - s33) + self.H * (s22 - s11)) / q
        gradient[2, 2] = (self.F * (s33 - s22) + self.G * (s33 - s11)) / q
        gradient[0, 1] = gradient[1, 0] = self.N * s12 / q
        gradient[1, 2] = gradient[2, 1] = self.L * s23 / q
        gradient[0, 2] = gradient[2, 0] = self.M * s13 / q
        return q, gradient

    def _return_residual(self, unknown, strain, plastic_strain, alpha0, scale):
        stress = _unpack_stress(unknown[:6])
        gamma = unknown[6]
        q, flow = self._equivalent_and_gradient(stress)
        updated_strain = strain - plastic_strain - gamma * flow
        predicted = np.einsum("ijkl,kl->ij", self._elastic, updated_strain)
        yield_value = q - self._yield_stress(float(alpha0 + gamma))
        return np.r_[_pack_tensor(stress - predicted) / scale, yield_value / scale]

    def _return_jacobian(self, unknown, strain, plastic_strain, alpha0, scale):
        """Analytic Jacobian of the seven local Hill48 return equations."""
        stress_vector = np.asarray(unknown[:6], dtype=float)
        gamma = float(unknown[6])
        metric_stress = self._hill_voigt @ stress_vector
        q_squared = float(stress_vector @ metric_stress)
        q = float(np.sqrt(max(q_squared, np.finfo(float).tiny)))
        q_gradient = metric_stress / q
        q_hessian = self._hill_voigt / q - np.outer(
            metric_stress, metric_stress
        ) / q**3
        flow = self._tensor_voigt_flow @ q_gradient
        flow_jacobian = self._tensor_voigt_flow @ q_hessian
        jacobian = np.zeros((7, 7))
        jacobian[:6, :6] = (
            np.eye(6) + gamma * self._elastic_voigt @ flow_jacobian
        ) / scale
        jacobian[:6, 6] = self._elastic_voigt @ flow / scale
        jacobian[6, :6] = q_gradient / scale
        jacobian[6, 6] = -self._hardening_slope(float(alpha0 + gamma)) / scale
        return jacobian

    def _consistent_tangent(self, strain, plastic_strain, alpha0, stress, gamma):
        """Differentiate the converged local return equations implicitly.

        The local 7x7 return Jacobian is analytic, so the same linearization
        supports both the local Newton solve and the global material tangent.
        """
        scale = max(self.young_modulus, self.yield_stress, 1.)
        unknown = np.r_[_pack_tensor(stress), gamma]

        jacobian = self._return_jacobian(
            unknown, strain, plastic_strain, alpha0, scale
        )

        tangent = np.empty((3, 3, 3, 3))
        for k in range(3):
            for l in range(k, 3):
                strain_direction = np.zeros((3, 3))
                strain_direction[k, l] = 1.
                strain_direction[l, k] = 1.
                elastic_increment = np.einsum(
                    "ijkl,kl->ij", self._elastic, strain_direction
                )
                source = np.r_[_pack_tensor(elastic_increment) / scale, 0.]
                sensitivity = np.linalg.solve(jacobian, source)
                stress_sensitivity = _unpack_stress(sensitivity[:6])
                if k != l:
                    stress_sensitivity *= 0.5
                tangent[:, :, k, l] = stress_sensitivity
                tangent[:, :, l, k] = stress_sensitivity
        return 0.5 * (tangent + tangent.transpose(2, 3, 0, 1))

    def _stress_update(self, strain: np.ndarray, state: np.ndarray):
        ep = np.zeros((3, 3))
        ep[0, 0], ep[1, 1], ep[2, 2] = state[:3]
        ep[0, 1] = ep[1, 0] = state[3]
        ep[1, 2] = ep[2, 1] = state[4]
        ep[0, 2] = ep[2, 0] = state[5]
        elastic_strain = strain - ep
        trial_stress = np.einsum("ijkl,kl->ij", self._elastic, elastic_strain)
        q_trial, _ = self._equivalent_and_gradient(trial_stress)
        current_yield = self._yield_stress(float(state[6]))
        if q_trial <= current_yield + 1e-12 * self.yield_stress:
            return trial_stress, state.copy(), q_trial, False

        scale = max(self.young_modulus, self.yield_stress, 1.)
        initial_gamma = max(
            (q_trial - current_yield) /
            (3. * self.mu + self._hardening_slope(float(state[6]))),
            1e-14,
        )

        def residual(unknown):
            return self._return_residual(unknown, strain, ep, float(state[6]), scale)

        initial = np.r_[_pack_tensor(trial_stress), initial_gamma]
        def jacobian(unknown):
            return self._return_jacobian(
                unknown, strain, ep, float(state[6]), scale
            )

        solution = root(residual, initial, jac=jacobian, method="hybr",
                         options={"xtol": 1e-11, "maxfev": 100})
        error = np.linalg.norm(residual(solution.x), ord=np.inf)
        if not solution.success and error > 2e-9:
            raise RuntimeError(f"Hill48 return mapping failed: {solution.message}; residual={error:.3e}")
        stress = _unpack_stress(solution.x[:6])
        gamma = float(solution.x[6])
        if gamma < -1e-12:
            raise RuntimeError("Hill48 return mapping produced a negative multiplier")
        gamma = max(gamma, 0.)
        _, flow = self._equivalent_and_gradient(stress)
        delta_ep = gamma * flow
        delta_alpha = gamma
        updated = state.copy()
        updated[0] += delta_ep[0, 0]
        updated[1] += delta_ep[1, 1]
        updated[2] += delta_ep[2, 2]
        updated[3] += delta_ep[0, 1]
        updated[4] += delta_ep[1, 2]
        updated[5] += delta_ep[0, 2]
        updated[6] += delta_alpha
        q, _ = self._equivalent_and_gradient(stress)
        return stress, updated, q, True

    def _point_response(self, strain: np.ndarray, state: np.ndarray):
        stress, updated, q, plastic = self._stress_update(strain, state)
        if not plastic:
            tangent = self._elastic.copy()
        else:
            gamma = float(updated[6] - state[6])
            tangent = self._consistent_tangent(
                strain, self._plastic_tensor(state), float(state[6]), stress, gamma
            )
        elastic_strain = strain.copy()
        elastic_strain[0, 0] -= updated[0]
        elastic_strain[1, 1] -= updated[1]
        elastic_strain[2, 2] -= updated[2]
        elastic_strain[0, 1] -= updated[3]
        elastic_strain[1, 0] -= updated[3]
        elastic_strain[1, 2] -= updated[4]
        elastic_strain[2, 1] -= updated[4]
        elastic_strain[0, 2] -= updated[5]
        elastic_strain[2, 0] -= updated[5]
        energy = 0.5 * np.einsum(
            "ij,ijkl,kl", elastic_strain, self._elastic, elastic_strain
        )
        energy += self._hardening_work(float(state[6]), float(updated[6]))
        return stress, tangent, updated, plastic, q, energy

    @staticmethod
    def _plastic_tensor(state):
        ep = np.zeros((3, 3))
        ep[0, 0], ep[1, 1], ep[2, 2] = state[:3]
        ep[0, 1] = ep[1, 0] = state[3]
        ep[1, 2] = ep[2, 1] = state[4]
        ep[0, 2] = ep[2, 0] = state[5]
        return ep

    def evaluate(self, strain, committed_state) -> Hill48MaterialEvaluation:
        eps = np.asarray(strain, dtype=float)
        if eps.ndim < 2 or eps.shape[-2:] != (3, 3) or not np.all(np.isfinite(eps)):
            raise ValueError("strain must be finite with trailing shape (3, 3)")
        if not np.allclose(eps, eps.swapaxes(-1, -2), rtol=1e-12, atol=1e-14):
            raise ValueError("strain tensors must be symmetric")
        batch = eps.shape[:-2]
        state = np.asarray(committed_state, dtype=float)
        if state.shape != batch + (7,) or not np.all(np.isfinite(state)):
            raise ValueError(f"committed_state must have shape {batch + (7,)}")
        if np.any(state[..., 6] < 0.):
            raise ValueError("committed equivalent plastic strain must be nonnegative")
        if not batch:
            response = self._point_response(eps, state)
            return Hill48MaterialEvaluation(*response[:3], np.asarray(response[3]),
                                            np.asarray(response[4]), np.asarray(response[5]))
        stress = np.empty(batch + (3, 3))
        tangent = np.empty(batch + (3, 3, 3, 3))
        trial = np.empty(batch + (7,))
        plastic = np.zeros(batch, dtype=bool)
        equivalent = np.empty(batch)
        energy = np.empty(batch)
        for index in np.ndindex(batch):
            response = self._point_response(eps[index], state[index])
            stress[index], tangent[index], trial[index], plastic[index], equivalent[index], energy[index] = response
        return Hill48MaterialEvaluation(stress, tangent, trial, plastic, equivalent, energy)

    def commit(self, evaluation: Hill48MaterialEvaluation) -> np.ndarray:
        if not isinstance(evaluation, Hill48MaterialEvaluation):
            raise TypeError("evaluation must be a Hill48MaterialEvaluation")
        if evaluation.trial_state.ndim < 1 or evaluation.trial_state.shape[-1] != 7:
            raise ValueError("evaluation has an invalid trial_state")
        if (not np.all(np.isfinite(evaluation.trial_state))
                or np.any(evaluation.trial_state[..., 6] < 0.)):
            raise ValueError("evaluation has invalid plastic history")
        return np.array(evaluation.trial_state, dtype=float, copy=True)


class PlaneStrainHill48Assembler:
    """Experimental plane-strain TriP1/TriP2 assembler for Hill48 plasticity."""

    def __init__(self, basis: Basis, young_modulus: float, poisson_ratio: float,
                 yield_stress: float | None = None, hardening_modulus: float = 0.0,
                 stress_ratios=(1., 1., 1., 1., 1., 1.), hardening_curve=None):
        if (not isinstance(basis, Basis) or not isinstance(basis.elem, ElementVector)
                or basis.elem._dim != 2
                or not isinstance(basis.elem.elem, (ElementTriP1, ElementTriP2))):
            raise TypeError("basis must use a two-component vector TriP1 or TriP2 field")
        self.basis = basis
        self.material = Hill48PlasticityMaterial(
            young_modulus, poisson_ratio, yield_stress, hardening_modulus,
            stress_ratios, hardening_curve,
        )
        self.young_modulus = self.material.young_modulus
        self.poisson_ratio = self.material.poisson_ratio
        self.yield_stress = self.material.yield_stress
        self.hardening_modulus = self.material.hardening_modulus
        self._gradients = np.ascontiguousarray(basis.tabulated_gradients)
        self._weights = np.ascontiguousarray(basis.dx)
        nodes = basis.element_dofs.shape[0] // 2
        self._dofs = np.ascontiguousarray(
            basis.element_dofs.T.reshape(basis.nelems, nodes, 2), dtype=np.int64
        )
        self._tangent_graph = _prepare_tangent_scatter(
            self._dofs.reshape(basis.nelems, -1), basis.N
        )

    def initial_state(self) -> np.ndarray:
        return self.material.initial_state((self.basis.nelems, self._weights.shape[1]))

    def evaluate(self, displacement, committed_state) -> PlasticEvaluation:
        u = np.asarray(displacement, dtype=float)
        state = np.asarray(committed_state, dtype=float)
        expected = (self.basis.nelems, self._weights.shape[1], 7)
        if u.shape != (self.basis.N,) or not np.all(np.isfinite(u)):
            raise ValueError(f"displacement must be finite with shape ({self.basis.N},)")
        if state.shape != expected or not np.all(np.isfinite(state)):
            raise ValueError(f"committed_state must be finite with shape {expected}")
        if np.any(state[..., 6] < 0.):
            raise ValueError("committed equivalent plastic strain must be nonnegative")
        local_u = u[self._dofs]
        grad_u = np.einsum("eai,eqaj->eqij", local_u, self._gradients)
        strain = np.zeros(grad_u.shape[:2] + (3, 3))
        strain[:, :, :2, :2] = 0.5 * (grad_u + grad_u.swapaxes(-1, -2))
        response = self.material.evaluate(strain, state)
        local_residual = np.einsum(
            "eqij,eqaj,eq->eai", response.stress[:, :, :2, :2],
            self._gradients, self._weights
        )
        local_tangent = np.einsum(
            "eqaj,eqijkl,eqbl,eq->eaibk",
            self._gradients, response.tangent[:, :, :2, :2, :2, :2],
            self._gradients, self._weights,
        ).reshape(self.basis.nelems, self._dofs.shape[1] * 2, -1)
        dofs = self._dofs.reshape(self.basis.nelems, -1)
        residual, tangent = _assemble_global_response(
            local_residual.reshape(self.basis.nelems, -1), local_tangent, dofs,
            (self.basis.N, self.basis.N), self._tangent_graph,
        )
        return PlasticEvaluation(
            residual, tangent, response.trial_state, response.plastic_points,
            response.stress, response.equivalent_stress,
            float(np.sum(response.energy * self._weights)),
        )

    def commit(self, evaluation: PlasticEvaluation) -> np.ndarray:
        if not isinstance(evaluation, PlasticEvaluation):
            raise TypeError("evaluation must be a PlasticEvaluation")
        if (evaluation.trial_state.shape != self.initial_state().shape
                or not np.all(np.isfinite(evaluation.trial_state))
                or np.any(evaluation.trial_state[..., 6] < 0.)):
            raise ValueError("evaluation has an invalid trial_state shape")
        return np.array(evaluation.trial_state, copy=True)


class Hill48PlasticityAssembler:
    """Small-strain 3D Hill48 plasticity on Tet4/Tet10 and Hex8/Hex20/Hex27.

    This shares the J2 assembler's transactional history layout, so existing
    Newton/contact drivers can use it without changing their commit protocol.
    The constitutive tangent is obtained by implicit differentiation with the
    analytic local return Jacobian.
    """

    _supported_elements = (ElementTetP1, ElementTetP2, ElementHex1,
                           ElementHex20, ElementHex2)

    def __init__(self, basis: Basis, young_modulus: float, poisson_ratio: float,
                 yield_stress: float | None = None, hardening_modulus: float = 0.0,
                 stress_ratios=(1., 1., 1., 1., 1., 1.), hardening_curve=None):
        if (not isinstance(basis, Basis) or not isinstance(basis.elem, ElementVector)
                or basis.elem._dim != 3
                or not isinstance(basis.elem.elem, self._supported_elements)):
            raise TypeError(
                "basis must use a three-component vector Tet4, Tet10, Hex8, Hex20, or Hex27 field"
            )
        self.basis = basis
        self.material = Hill48PlasticityMaterial(
            young_modulus, poisson_ratio, yield_stress, hardening_modulus,
            stress_ratios, hardening_curve,
        )
        self._gradients = np.ascontiguousarray(basis.tabulated_gradients)
        nodes = basis.element_dofs.shape[0] // 3
        if self._gradients.shape[2:] != (nodes, 3):
            raise ValueError("basis gradients do not match a three-dimensional vector DOF layout")
        self._weights = np.ascontiguousarray(basis.dx)
        self._dofs = np.ascontiguousarray(
            basis.element_dofs.T.reshape(basis.nelems, nodes, 3), dtype=np.int64
        )
        self._tangent_graph = _prepare_tangent_scatter(
            self._dofs.reshape(basis.nelems, -1), basis.N
        )

    def initial_state(self) -> np.ndarray:
        return self.material.initial_state((self.basis.nelems, self._weights.shape[1]))

    def evaluate(self, displacement, committed_state) -> PlasticEvaluation:
        u = np.asarray(displacement, dtype=float)
        state = np.asarray(committed_state, dtype=float)
        expected = (self.basis.nelems, self._weights.shape[1], 7)
        if u.shape != (self.basis.N,) or not np.all(np.isfinite(u)):
            raise ValueError(f"displacement must be finite with shape ({self.basis.N},)")
        if state.shape != expected or not np.all(np.isfinite(state)):
            raise ValueError(f"committed_state must be finite with shape {expected}")
        if np.any(state[..., 6] < 0.):
            raise ValueError("committed equivalent plastic strain must be nonnegative")
        local_u = u[self._dofs]
        grad_u = np.einsum("eai,eqaj->eqij", local_u, self._gradients)
        strain = 0.5 * (grad_u + grad_u.swapaxes(-1, -2))
        response = self.material.evaluate(strain, state)
        local_residual = np.einsum(
            "eqij,eqaj,eq->eai", response.stress, self._gradients, self._weights
        )
        local_tangent = np.einsum(
            "eqaj,eqijkl,eqbl,eq->eaibk", self._gradients, response.tangent,
            self._gradients, self._weights,
        ).reshape(self.basis.nelems, self._dofs.shape[1] * 3, -1)
        dofs = self._dofs.reshape(self.basis.nelems, -1)
        residual, tangent = _assemble_global_response(
            local_residual.reshape(self.basis.nelems, -1), local_tangent, dofs,
            (self.basis.N, self.basis.N), self._tangent_graph,
        )
        return PlasticEvaluation(
            residual, tangent, response.trial_state, response.plastic_points,
            response.stress, response.equivalent_stress,
            float(np.sum(response.energy * self._weights)),
        )

    def commit(self, evaluation: PlasticEvaluation) -> np.ndarray:
        if not isinstance(evaluation, PlasticEvaluation):
            raise TypeError("evaluation must be a PlasticEvaluation")
        if (evaluation.trial_state.shape != self.initial_state().shape
                or not np.all(np.isfinite(evaluation.trial_state))
                or np.any(evaluation.trial_state[..., 6] < 0.)):
            raise ValueError("evaluation has an invalid trial_state shape")
        return np.array(evaluation.trial_state, copy=True)


__all__ = [
    "Hill48MaterialEvaluation", "Hill48PlasticityMaterial",
    "PlaneStrainHill48Assembler", "Hill48PlasticityAssembler",
]
