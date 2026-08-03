from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix

from ._skfn import (
    J2GlobalAssembler,LinearElasticMaterialAssembler,evaluate_j2_state,
    StandardLinearSolidAssembler,evaluate_standard_linear_solid,
)
from .evaluation import EvaluationDiagnostics,NativeEvaluation
from .kernels import LinearElasticity
from .runtime import available_num_threads


class MaterialState:
    """Contiguous native state with shape ``(points, state_size)``."""

    def __init__(self,storage: np.ndarray):
        storage=np.asarray(storage,dtype=np.float64)
        if storage.ndim!=2:
            raise ValueError("material state must be a two-dimensional array")
        self._storage=np.ascontiguousarray(storage)

    @classmethod
    def _from_storage(cls,storage: np.ndarray):
        obj=cls.__new__(cls)
        obj._storage=storage
        return obj

    @property
    def storage(self):
        return self._storage

    def copy(self,*,readonly=False):
        storage=np.array(self._storage,dtype=np.float64,order="C",copy=True)
        if readonly:
            storage.flags.writeable=False
        return type(self)._from_storage(storage)


class MaterialStateHistory:
    """Explicit committed/trial state owner for nonlinear load stepping."""

    def __init__(self,initial_state: MaterialState):
        if not isinstance(initial_state,MaterialState):
            raise TypeError("initial_state must be a MaterialState")
        self._state_type=type(initial_state)
        self._committed=initial_state.copy(readonly=True)
        self._trial=None
        self._commit_count=0

    @property
    def committed(self):
        return self._committed

    @property
    def trial(self):
        return self._trial

    @property
    def commit_count(self):
        return self._commit_count

    def stage(self,trial_state: MaterialState):
        if not isinstance(trial_state,self._state_type):
            raise TypeError(
                f"trial_state must be a {self._state_type.__name__}"
            )
        if trial_state.storage.shape!=self._committed.storage.shape:
            raise ValueError(
                "trial_state shape differs from committed state"
            )
        self._trial=trial_state.copy(readonly=True)
        return self._trial

    def commit(self):
        if self._trial is None:
            raise RuntimeError("no trial state has been staged")
        self._committed=self._trial
        self._trial=None
        self._commit_count+=1
        return self._committed

    def rollback(self):
        self._trial=None
        return self._committed


class J2State(MaterialState):
    """Named views over a contiguous ``(points, 7)`` native state buffer."""

    def __init__(
        self,plastic_strain: np.ndarray,
        equivalent_plastic_strain: np.ndarray,
    ):
        plastic=np.asarray(plastic_strain,dtype=np.float64)
        alpha=np.asarray(equivalent_plastic_strain,dtype=np.float64)
        if plastic.ndim!=2 or plastic.shape[1]!=6:
            raise ValueError("plastic_strain must have shape (points, 6)")
        if alpha.shape!=(plastic.shape[0],):
            raise ValueError(
                "equivalent_plastic_strain must have shape (points,)"
            )
        self._storage=np.empty((plastic.shape[0],7),dtype=np.float64)
        self._storage[:,:6]=plastic
        self._storage[:,6]=alpha

    @classmethod
    def _from_storage(cls,storage: np.ndarray) -> J2State:
        obj=cls.__new__(cls)
        obj._storage=storage
        return obj

    @property
    def plastic_strain(self):
        return self._storage[:,:6]

    @property
    def equivalent_plastic_strain(self):
        return self._storage[:,6]


class StandardLinearSolidState(MaterialState):
    @property
    def viscous_strain(self):
        return self._storage


@dataclass(frozen=True)
class J2Evaluation:
    stress: np.ndarray
    tangent: np.ndarray
    trial_state: J2State


@dataclass(frozen=True)
class StandardLinearSolidEvaluation:
    stress: np.ndarray
    tangent: np.ndarray
    trial_state: StandardLinearSolidState


@dataclass(frozen=True)
class J2Plasticity:
    kernel_name="j2"
    state_size=7
    state_fields=("plastic_strain","equivalent_plastic_strain")
    young_modulus: float
    poisson_ratio: float
    yield_stress: float
    hardening_modulus: float = 0.

    def initial_state(self,count: int) -> J2State:
        if count<0:
            raise ValueError("state count must be nonnegative")
        return J2State._from_storage(
            np.zeros((count,self.state_size),dtype=np.float64)
        )

    def evaluate(
        self,strain: np.ndarray,state: J2State,*,num_threads: int=0
    ) -> J2Evaluation:
        strain=np.ascontiguousarray(strain,dtype=np.float64)
        if not isinstance(state,J2State):
            raise TypeError("state must be a J2State")
        if state.storage.shape!=(strain.shape[0],self.state_size):
            raise ValueError(
                f"state must have shape ({strain.shape[0]}, {self.state_size})"
            )
        if isinstance(num_threads,bool) or not isinstance(num_threads,int) or num_threads<0:
            raise ValueError("num_threads must be a nonnegative integer")
        effective=(
            min(num_threads,available_num_threads()) if num_threads else 0
        )
        stress,tangent,trial_storage=evaluate_j2_state(
            strain,state.storage,self.young_modulus,self.poisson_ratio,
            self.yield_stress,self.hardening_modulus,effective,
        )
        return J2Evaluation(
            stress,tangent,J2State._from_storage(trial_storage)
        )


@dataclass(frozen=True)
class StandardLinearSolid:
    kernel_name="standard_linear_solid"
    state_size=6
    state_fields=("viscous_strain",)
    equilibrium_modulus: float
    branch_modulus: float
    poisson_ratio: float
    relaxation_time: float
    time_step: float

    def __post_init__(self):
        if self.equilibrium_modulus<=0.:
            raise ValueError("equilibrium_modulus must be positive")
        if self.branch_modulus<0.:
            raise ValueError("branch_modulus must be nonnegative")
        if not -1.<self.poisson_ratio<.5:
            raise ValueError("poisson_ratio must be in (-1, .5)")
        if self.relaxation_time<=0. or self.time_step<=0.:
            raise ValueError("relaxation_time and time_step must be positive")

    def initial_state(self,count: int) -> StandardLinearSolidState:
        if count<0:
            raise ValueError("state count must be nonnegative")
        return StandardLinearSolidState._from_storage(
            np.zeros((count,self.state_size),dtype=np.float64)
        )

    def evaluate(
        self,strain: np.ndarray,state: StandardLinearSolidState,*,
        num_threads: int=0,time_step: float | None=None,
    ) -> StandardLinearSolidEvaluation:
        strain=np.ascontiguousarray(strain,dtype=np.float64)
        if strain.ndim!=2 or strain.shape[1]!=6:
            raise ValueError("strain must have shape (points, 6)")
        if not isinstance(state,StandardLinearSolidState):
            raise TypeError("state must be a StandardLinearSolidState")
        if state.storage.shape!=strain.shape:
            raise ValueError(f"state must have shape {strain.shape}")
        if (isinstance(num_threads,bool) or not isinstance(num_threads,int)
                or num_threads<0):
            raise ValueError("num_threads must be a nonnegative integer")
        if time_step is not None and (
            isinstance(time_step,bool) or not np.isscalar(time_step)
            or time_step<=0.
        ):
            raise ValueError("time_step must be positive")
        effective=min(num_threads,available_num_threads()) if num_threads else 0
        stress,tangent,trial=evaluate_standard_linear_solid(
            strain,state.storage,self.equilibrium_modulus,
            self.branch_modulus,self.poisson_ratio,self.relaxation_time,
            self.time_step,effective,0. if time_step is None else time_step,
        )
        return StandardLinearSolidEvaluation(
            stress,tangent,StandardLinearSolidState._from_storage(trial)
        )


class MaterialAssembler:
    """Fused stateful material assembly using a native material kernel.

    Material dispatch occurs once during construction.  Constitutive updates
    inside the quadrature loop are statically dispatched in C++.
    """

    def __init__(
        self,basis,
        material: J2Plasticity | LinearElasticity | StandardLinearSolid,
    ):
        if isinstance(material,J2Plasticity):
            native_type=J2GlobalAssembler
            self._state_type=J2State
        elif isinstance(material,LinearElasticity):
            native_type=LinearElasticMaterialAssembler
            self._state_type=MaterialState
        elif isinstance(material,StandardLinearSolid):
            native_type=StandardLinearSolidAssembler
            self._state_type=StandardLinearSolidState
        else:
            raise TypeError(
                "unsupported material kernel; currently available: "
                "J2Plasticity, LinearElasticity, StandardLinearSolid"
            )
        if getattr(basis.elem,"_dim",None)!=3:
            raise ValueError(
                "MaterialAssembler requires a three-component Basis"
            )
        nodes=len(basis.elem.elem.doflocs)
        entities,quadrature=basis.dx.shape
        dofs=basis.element_dofs.T.reshape(entities,nodes,3)
        self.material=material
        arguments=[
            np.ascontiguousarray(dofs,dtype=np.int64),
            np.ascontiguousarray(basis.tabulated_gradients,dtype=np.float64),
            np.ascontiguousarray(basis.dx,dtype=np.float64),
        ]
        if isinstance(material,J2Plasticity):
            arguments.extend((
                material.young_modulus,material.poisson_ratio,
                material.yield_stress,material.hardening_modulus,
            ))
        elif isinstance(material,LinearElasticity):
            arguments.extend((material.young_modulus,material.poisson_ratio))
        elif isinstance(material,StandardLinearSolid):
            arguments.extend((
                material.equilibrium_modulus,
                material.branch_modulus,material.poisson_ratio,
                material.relaxation_time,material.time_step,
            ))
        self._native=native_type(*arguments)
        self._tangent=csr_matrix((
            self._native.values,self._native.indices,self._native.indptr,
        ),shape=(basis.N,basis.N),copy=False)

    @property
    def ndofs(self):
        return self._native.ndofs

    @property
    def state_count(self):
        return self._native.state_count

    @property
    def tangent(self):
        return self._tangent

    def initial_state(self):
        if isinstance(self.material,J2Plasticity):
            return self.material.initial_state(self.state_count)
        if isinstance(self.material,StandardLinearSolid):
            return self.material.initial_state(self.state_count)
        return MaterialState._from_storage(
            np.zeros(
                (self.state_count,self.material.state_size),dtype=np.float64
            )
        )

    def initial_history(self):
        return MaterialStateHistory(self.initial_state())

    def assemble_trial(
        self,u: np.ndarray,history: MaterialStateHistory,**kwargs,
    ) -> NativeEvaluation:
        """Evaluate from committed state and stage, but do not commit, trial state."""
        if not isinstance(history,MaterialStateHistory):
            raise TypeError("history must be a MaterialStateHistory")
        if not isinstance(history.committed,self._state_type):
            raise TypeError(
                "history state type is incompatible with this assembler"
            )
        result=self.assemble(u,history.committed,**kwargs)
        history.stage(result.trial_state)
        return result

    def assemble(
        self,u: np.ndarray,state: MaterialState,*,
        mode: str="residual_tangent",num_threads: int=0,
        time_step: float | None=None,
    ) -> NativeEvaluation:
        if mode not in {"residual_tangent","residual"}:
            raise ValueError(f"unsupported evaluation mode: {mode!r}")
        u=np.asarray(u)
        if u.dtype!=np.float64 or not u.flags.c_contiguous:
            raise TypeError("u must be a C-contiguous float64 array")
        if u.shape!=(self.ndofs,):
            raise ValueError(f"u must have shape ({self.ndofs},)")
        if not isinstance(state,self._state_type):
            raise TypeError(f"state must be a {self._state_type.__name__}")
        if state.storage.shape!=(self.state_count,self.material.state_size):
            raise ValueError(
                "state must have shape "
                f"({self.state_count}, {self.material.state_size})"
            )
        if isinstance(num_threads,bool) or not isinstance(num_threads,int) or num_threads<0:
            raise ValueError("num_threads must be a nonnegative integer")
        if time_step is not None and (
            isinstance(time_step,bool) or not np.isscalar(time_step)
            or time_step<=0.
        ):
            raise ValueError("time_step must be positive")
        effective=(min(num_threads,available_num_threads()) if num_threads else 0)
        residual,_,trial_storage,seconds=self._native.evaluate(
            u,state.storage,mode=="residual_tangent",effective,
            0. if time_step is None else time_step,
        )
        return NativeEvaluation(
            residual=residual,
            tangent=self._tangent if mode=="residual_tangent" else None,
            trial_state=self._state_type._from_storage(trial_storage),
            diagnostics=EvaluationDiagnostics(
                element_count=self._native.nelements,
                quadrature_evaluations=self.state_count,
                assembly_seconds=seconds,
            ),
        )


# Backward-compatible material-specific name.
J2Assembler=MaterialAssembler
