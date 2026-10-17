"""Focused nonlinear material and finite-deformation assembly kernels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix

from ._skfn import (
    assemble_nonlinear_global,evaluate_j2_material,evaluate_j2_plasticity,evaluate_neo_hookean,
    evaluate_neo_hookean_material,
)
from .basis import (
    Basis,ElementHex1,ElementHex2,ElementHex20,ElementTetP1,ElementTetP2,
    ElementVector,
)


def _prepare_tangent_scatter(dofs: np.ndarray, ndofs: int):
    """Precompute local-matrix to sorted CSR data positions for a fixed graph."""
    local_size = dofs.shape[1]
    rows = np.broadcast_to(dofs[:, :, None], (dofs.shape[0], local_size, local_size))
    columns = np.broadcast_to(dofs[:, None, :], rows.shape)
    keys = rows.reshape(-1) * np.int64(ndofs) + columns.reshape(-1)
    unique_keys, inverse = np.unique(keys, return_inverse=True)
    csr_rows = unique_keys // ndofs
    counts = np.bincount(csr_rows, minlength=ndofs)
    indptr = np.empty(ndofs + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])
    indices = (unique_keys % ndofs).astype(np.int32 if unique_keys.size < 2**31 else np.int64)
    scatter = inverse.reshape(rows.shape).astype(
        np.int32 if unique_keys.size < 2**31 else np.int64
    )
    return indptr, indices, scatter


def _assemble_global_response(local_residual, local_tangent, dofs, shape, graph):
    indptr, indices, scatter = graph
    residual, data = assemble_nonlinear_global(
        dofs, local_residual, local_tangent, scatter, shape[0], indices.size
    )
    tangent = csr_matrix((data, indices.copy(), indptr.copy()), shape=shape)
    return residual, tangent


@dataclass(frozen=True)
class NonlinearEvaluation:
    """Neo-Hookean assembly and quadrature-point finite-strain response.

    ``stress`` stores the first Piola stress and both point tensors have shape
    ``(elements, quadrature, 3, 3)``. ``energy`` is the integrated total.
    """

    residual: np.ndarray
    tangent: object
    energy: float
    stress: np.ndarray
    deformation_gradient: np.ndarray
    cauchy_stress: np.ndarray
    equivalent_stress: np.ndarray


@dataclass(frozen=True)
class PlasticEvaluation:
    """Trial assembly and quadrature-point material response.

    ``stress`` has shape ``(elements, quadrature, 3, 3)`` and ``equivalent_stress``
    has shape ``(elements, quadrature)``. ``plastic_points`` marks points that
    underwent plastic flow during this trial, not points with prior history.
    ``energy`` is the integrated incremental potential for this trial.
    """

    residual: np.ndarray
    tangent: object
    trial_state: np.ndarray
    plastic_points: np.ndarray
    stress: np.ndarray
    equivalent_stress: np.ndarray
    energy: float


@dataclass(frozen=True)
class J2MaterialEvaluation:
    """Constitutive response for a batch of integration-point strains."""

    stress: np.ndarray
    tangent: np.ndarray
    trial_state: np.ndarray
    plastic_points: np.ndarray
    equivalent_stress: np.ndarray
    energy: np.ndarray


class J2PlasticityMaterial:
    """3D small-strain J2 material point with isotropic linear hardening.

    Strain and stress use symmetric tensor components; plastic history stores
    ``(11, 22, 33, 12, 23, 13, equivalent_plastic_strain)``. Every evaluation
    is a trial update and leaves the supplied committed state unchanged.
    """

    def __init__(self,young_modulus: float,poisson_ratio: float,
                 yield_stress: float,hardening_modulus: float=0.):
        vals=np.asarray([young_modulus,poisson_ratio,yield_stress,hardening_modulus],dtype=float)
        if not np.all(np.isfinite(vals)):
            raise ValueError("material parameters must be finite")
        self.young_modulus=float(young_modulus)
        self.poisson_ratio=float(poisson_ratio)
        self.yield_stress=float(yield_stress)
        self.hardening_modulus=float(hardening_modulus)
        if self.young_modulus<=0. or not -1.<self.poisson_ratio<.5:
            raise ValueError("require young_modulus > 0 and -1 < poisson_ratio < 0.5")
        if self.yield_stress<=0. or self.hardening_modulus<0.:
            raise ValueError("yield_stress must be positive and hardening_modulus nonnegative")
        self.mu=self.young_modulus/(2.*(1.+self.poisson_ratio))
        self.lambda_=self.young_modulus*self.poisson_ratio/(
            (1.+self.poisson_ratio)*(1.-2.*self.poisson_ratio))

    def initial_state(self,point_shape=()) -> np.ndarray:
        """Create zero history for one material point or a batch of points."""
        if isinstance(point_shape,(int,np.integer)):
            point_shape=(int(point_shape),)
        shape=tuple(point_shape)
        if any(not isinstance(size,(int,np.integer)) or size<0 for size in shape):
            raise ValueError("point_shape must contain nonnegative integers")
        return np.zeros(tuple(int(size) for size in shape)+(7,),dtype=np.float64)

    def evaluate(self,strain,committed_state) -> J2MaterialEvaluation:
        """Return stress, algorithmic tangent, and a fresh trial history."""
        eps=np.asarray(strain,dtype=np.float64)
        if eps.ndim<2 or eps.shape[-2:]!=(3,3) or not np.all(np.isfinite(eps)):
            raise ValueError("strain must be finite with trailing shape (3, 3)")
        if not np.allclose(eps,eps.swapaxes(-1,-2),rtol=1e-12,atol=1e-14):
            raise ValueError("strain tensors must be symmetric")
        batch=eps.shape[:-2]
        state=np.asarray(committed_state,dtype=np.float64)
        if state.shape!=batch+(7,) or not np.all(np.isfinite(state)):
            raise ValueError(f"committed_state must be finite with shape {batch+(7,)}")
        if np.any(state[...,6]<0.):
            raise ValueError("committed equivalent plastic strain must be nonnegative")
        flat_strain=np.ascontiguousarray(eps.reshape(-1,3,3))
        flat_state=np.ascontiguousarray(state.reshape(-1,7))
        stress,tangent,trial,plastic,equivalent,energy=evaluate_j2_material(
            flat_strain,flat_state,self.lambda_,self.mu,
            self.yield_stress,self.hardening_modulus)
        return J2MaterialEvaluation(
            stress.reshape(batch+(3,3)),
            tangent.reshape(batch+(3,3,3,3)),
            trial.reshape(batch+(7,)),plastic.astype(bool).reshape(batch),
            equivalent.reshape(batch),energy.reshape(batch))

    def commit(self,evaluation: J2MaterialEvaluation) -> np.ndarray:
        """Adopt trial material history by making an independent copy."""
        if not isinstance(evaluation,J2MaterialEvaluation):
            raise TypeError("evaluation must be a J2MaterialEvaluation")
        if evaluation.trial_state.ndim<1 or evaluation.trial_state.shape[-1]!=7:
            raise ValueError("evaluation has an invalid trial_state shape")
        if not np.all(np.isfinite(evaluation.trial_state)) or np.any(evaluation.trial_state[...,6]<0.):
            raise ValueError("evaluation has an invalid trial_state")
        return np.array(evaluation.trial_state,dtype=np.float64,copy=True)


class J2PlasticityAssembler:
    """Small-strain 3D J2 plasticity with isotropic linear hardening.

    Supported vector H1 fields are Tet4, Tet10, Hex8, Hex20, and Hex27.

    State values at each integration point are tensor plastic strain
    ``(11, 22, 33, 12, 23, 13)`` and accumulated equivalent plastic strain.
    ``evaluate`` is transactional: it never changes the supplied state.
    """

    def __init__(self,basis: Basis,young_modulus: float,poisson_ratio: float,
                 yield_stress: float,hardening_modulus: float=0.):
        if not isinstance(basis,Basis) or not isinstance(basis.elem,ElementVector) \
                or basis.elem._dim!=3 or not isinstance(
                    basis.elem.elem,
                    (ElementTetP1,ElementTetP2,ElementHex1,ElementHex20,ElementHex2),
                ):
            raise TypeError(
                "basis must use a three-component vector Tet4, Tet10, Hex8, Hex20, or Hex27 field"
            )
        self.basis=basis
        self.material=J2PlasticityMaterial(
            young_modulus,poisson_ratio,yield_stress,hardening_modulus)
        self.young_modulus=self.material.young_modulus
        self.poisson_ratio=self.material.poisson_ratio
        self.yield_stress=self.material.yield_stress
        self.hardening_modulus=self.material.hardening_modulus
        self.mu=self.material.mu
        self.lambda_=self.material.lambda_
        nodes=basis.element_dofs.shape[0]//3
        gradients=np.asarray(basis.tabulated_gradients)
        self._gradients=np.ascontiguousarray(gradients)
        if self._gradients.shape[2:]!=(nodes,3):
            raise ValueError("basis gradients do not match vector DOF layout")
        self._dofs=np.ascontiguousarray(basis.element_dofs.T.reshape(basis.nelems,nodes,3),dtype=np.int64)
        self._tangent_graph=_prepare_tangent_scatter(
            self._dofs.reshape(basis.nelems,-1),basis.N)
        self._weights=np.ascontiguousarray(basis.dx)

    def initial_state(self) -> np.ndarray:
        """Return zero plastic strain/history for every cell quadrature point."""
        return self.material.initial_state((self.basis.nelems,self._weights.shape[1]))

    def evaluate(self,displacement,committed_state) -> PlasticEvaluation:
        """Compute a trial response without changing ``committed_state``."""
        u=np.ascontiguousarray(displacement,dtype=np.float64)
        state=np.asarray(committed_state,dtype=np.float64)
        expected=(self.basis.nelems,self._weights.shape[1],7)
        if u.shape!=(self.basis.N,):
            raise ValueError(f"displacement must have shape ({self.basis.N},)")
        if state.shape!=expected or not np.all(np.isfinite(state)):
            raise ValueError(f"committed_state must be finite with shape {expected}")
        if np.any(state[...,6]<0.):
            raise ValueError("committed equivalent plastic strain must be nonnegative")
        r,k,trial,plastic,stress,equivalent_stress,energy=evaluate_j2_plasticity(
            self._dofs,self._gradients,self._weights,u,
            np.ascontiguousarray(state),self.lambda_,self.mu,
            self.yield_stress,self.hardening_modulus)
        dofs=self._dofs.reshape(self.basis.nelems,-1)
        residual,tangent=_assemble_global_response(
            r,k,dofs,(self.basis.N,self.basis.N),self._tangent_graph)
        return PlasticEvaluation(
            residual,tangent,trial,plastic.astype(bool),stress,equivalent_stress,
            float(np.sum(energy*self._weights)))

    def commit(self,evaluation: PlasticEvaluation) -> np.ndarray:
        """Adopt a trial history by returning an independent committed copy."""
        if not isinstance(evaluation,PlasticEvaluation):
            raise TypeError("evaluation must be a PlasticEvaluation")
        expected=(self.basis.nelems,self._weights.shape[1],7)
        if evaluation.trial_state.shape!=expected or not np.all(np.isfinite(evaluation.trial_state)):
            raise ValueError("evaluation has an invalid trial_state")
        return np.array(evaluation.trial_state,dtype=np.float64,copy=True)


class J2PlasticityTet4Assembler(J2PlasticityAssembler):
    """Compatibility-specific constructor restricted to linear Tet4 fields."""

    def __init__(self,basis: Basis,young_modulus: float,poisson_ratio: float,
                 yield_stress: float,hardening_modulus: float=0.):
        if not isinstance(basis,Basis) or not isinstance(basis.elem,ElementVector) \
                or not isinstance(basis.elem.elem,ElementTetP1):
            raise TypeError("J2PlasticityTet4Assembler requires a vector Tet4 basis")
        super().__init__(basis,young_modulus,poisson_ratio,yield_stress,hardening_modulus)


@dataclass(frozen=True)
class NeoHookeanMaterialEvaluation:
    """Point response with first Piola and Cauchy stresses and dP/dF."""

    stress: np.ndarray
    tangent: np.ndarray
    energy: np.ndarray
    cauchy_stress: np.ndarray
    equivalent_stress: np.ndarray


class NeoHookeanMaterial:
    """Compressible Neo-Hookean material-point law in the reference frame."""

    def __init__(self,mu: float,lambda_: float):
        self.mu=float(mu)
        self.lambda_=float(lambda_)
        if not np.isfinite(self.mu) or self.mu<=0.:
            raise ValueError("mu must be positive and finite")
        if not np.isfinite(self.lambda_) or self.lambda_<0.:
            raise ValueError("lambda_ must be nonnegative and finite")

    def evaluate(self,deformation_gradient) -> NeoHookeanMaterialEvaluation:
        """Evaluate batched deformation gradients with trailing shape (3, 3)."""
        f=np.asarray(deformation_gradient,dtype=np.float64)
        if f.ndim<2 or f.shape[-2:]!=(3,3) or not np.all(np.isfinite(f)):
            raise ValueError("deformation_gradient must be finite with trailing shape (3, 3)")
        batch=f.shape[:-2]
        stress,tangent,energy,cauchy,equivalent=evaluate_neo_hookean_material(
            np.ascontiguousarray(f.reshape(-1,3,3)),self.mu,self.lambda_)
        return NeoHookeanMaterialEvaluation(
            stress.reshape(batch+(3,3)),
            tangent.reshape(batch+(3,3,3,3)),
            energy.reshape(batch),cauchy.reshape(batch+(3,3)),
            equivalent.reshape(batch))


class NeoHookeanAssembler:
    """Total-Lagrangian compressible Neo-Hookean assembler.

    Supported displacement cells are Tet4, Tet10, Hex8, Hex20, and Hex27.
    The kernel evaluates the deformation gradient ``F = I + grad(u)`` at each
    basis quadrature point and rejects states with ``det(F) <= 0``.
    """

    def __init__(self,basis: Basis,mu: float,lambda_: float):
        if not isinstance(basis,Basis):
            raise TypeError("basis must be a skfemntv.Basis")
        if not isinstance(basis.elem,ElementVector) or basis.elem._dim!=3:
            raise TypeError("basis must use a three-component ElementVector")
        if not isinstance(
            basis.elem.elem,
            (ElementTetP1,ElementTetP2,ElementHex1,ElementHex20,ElementHex2),
        ):
            raise TypeError(
                "basis must use Tet4, Tet10, Hex8, Hex20, or Hex27 fields"
            )
        self.basis=basis
        self.material=NeoHookeanMaterial(mu,lambda_)
        self.mu=self.material.mu
        self.lambda_=self.material.lambda_

        nodes=basis.element_dofs.shape[0]//3
        self._dofs=np.ascontiguousarray(
            basis.element_dofs.T.reshape(basis.nelems,nodes,3),dtype=np.int64
        )
        self._tangent_graph=_prepare_tangent_scatter(
            self._dofs.reshape(basis.nelems,-1),basis.N)
        gradients=np.asarray(basis.tabulated_gradients)
        if gradients.shape[2:]!=(nodes,3):
            raise ValueError("basis gradients do not match vector DOF layout")
        self._gradients=np.ascontiguousarray(gradients)
        self._weights=np.ascontiguousarray(basis.dx)

    def evaluate(self,displacement) -> NonlinearEvaluation:
        """Evaluate internal force, material/geometric tangent, and energy."""
        u=np.ascontiguousarray(displacement,dtype=np.float64)
        if u.shape!=(self.basis.N,):
            raise ValueError(f"displacement must have shape ({self.basis.N},)")
        (local_residual,local_tangent,element_energy,stress,deformation_gradient,
         cauchy_stress,equivalent_stress)=evaluate_neo_hookean(
            self._dofs,self._gradients,self._weights,u,self.mu,self.lambda_
        )
        dofs=self._dofs.reshape(self.basis.nelems,-1)
        residual,tangent=_assemble_global_response(
            local_residual,local_tangent,dofs,
            (self.basis.N,self.basis.N),self._tangent_graph)
        return NonlinearEvaluation(
            residual=residual,tangent=tangent,
            energy=float(np.sum(element_energy)),
            stress=stress,deformation_gradient=deformation_gradient,
            cauchy_stress=cauchy_stress,equivalent_stress=equivalent_stress,
        )


class NeoHookeanTet4Assembler(NeoHookeanAssembler):
    """Compatibility-specific constructor restricted to linear Tet4 bases."""

    def __init__(self,basis: Basis,mu: float,lambda_: float):
        if not isinstance(basis.elem,ElementVector) or not isinstance(
            basis.elem.elem,ElementTetP1
        ) or basis.mesh.t.shape[0]!=4:
            raise TypeError("NeoHookeanTet4Assembler requires vector Tet4 basis")
        super().__init__(basis,mu,lambda_)


__all__=[
    "J2MaterialEvaluation","J2PlasticityAssembler","J2PlasticityMaterial",
    "J2PlasticityTet4Assembler","NeoHookeanAssembler","NeoHookeanMaterial",
    "NeoHookeanMaterialEvaluation","NeoHookeanTet4Assembler",
    "NonlinearEvaluation","PlasticEvaluation",
]
