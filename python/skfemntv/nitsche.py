"""High-level symmetric Nitsche assembly on nonmatching interfaces."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.sparse import csr_matrix

from .helpers import isotropic_traction_tensor


@dataclass(frozen=True)
class NitscheStabilizationDiagnostics:
    """Resolved facet scales and coefficients used by automatic stabilization."""

    master_characteristic_length: np.ndarray
    slave_characteristic_length: np.ndarray
    master_average_weight: np.ndarray
    slave_average_weight: np.ndarray
    penalty: np.ndarray
    stabilization_factor: float

    @property
    def minimum_penalty(self):
        return float(np.min(self.penalty))

    @property
    def maximum_penalty(self):
        return float(np.max(self.penalty))


@dataclass(frozen=True)
class SymmetricNitscheResult:
    """Interface contributions for the symmetric Nitsche formulation."""

    matrix: csr_matrix
    penalty: csr_matrix
    consistency: csr_matrix
    adjoint_consistency: csr_matrix
    average_weights: tuple[float, float] | tuple[np.ndarray, np.ndarray]
    stabilization: NitscheStabilizationDiagnostics | None = None


def _validate_lame(name, values):
    try:
        lame_lambda, lame_mu = (float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain (lame_lambda, lame_mu)") from error
    if not np.isfinite((lame_lambda, lame_mu)).all():
        raise ValueError(f"{name} must be finite")
    if lame_mu <= 0. or 3. * lame_lambda + 2. * lame_mu <= 0.:
        raise ValueError(f"{name} must define a stable isotropic material")
    return lame_lambda, lame_mu


def _readonly(values):
    result = np.asarray(values, dtype=float)
    result.flags.writeable = False
    return result


def _automatic_stabilization(
    master, slave, integration, master_lame, slave_lame, factor,
):
    factor = float(factor)
    if not np.isfinite(factor) or factor <= 0.:
        raise ValueError("stabilization_factor must be finite and positive")
    master_area = np.sum(np.asarray(master.dx, dtype=float), axis=1)
    slave_area = np.sum(np.asarray(slave.dx, dtype=float), axis=1)
    master_parent = np.asarray(integration.master_trace.parent_facets, dtype=np.int64)
    slave_parent = np.asarray(integration.slave_trace.parent_facets, dtype=np.int64)
    if (
        np.any(master_parent < 0) or np.any(master_parent >= len(master_area))
        or np.any(slave_parent < 0) or np.any(slave_parent >= len(slave_area))
    ):
        raise ValueError("integration parent facets do not match the supplied bases")
    master_h = np.sqrt(master_area[master_parent])
    slave_h = np.sqrt(slave_area[slave_parent])
    if np.any(master_h <= 0.) or np.any(slave_h <= 0.):
        raise ValueError("interface facets must have positive area")
    master_scale = (master_lame[0] + 2. * master_lame[1]) / master_h
    slave_scale = (slave_lame[0] + 2. * slave_lame[1]) / slave_h
    scale_sum = master_scale + slave_scale
    master_weight = slave_scale / scale_sum
    slave_weight = master_scale / scale_sum
    penalty = factor * 2. * master_scale * slave_scale / scale_sum
    expand = (slice(None),) + (None,) * (len(integration._coefficient_shape) - 1)
    master_weight = np.broadcast_to(
        master_weight[expand], integration._coefficient_shape
    )
    slave_weight = np.broadcast_to(
        slave_weight[expand], integration._coefficient_shape
    )
    penalty = np.broadcast_to(penalty[expand], integration._coefficient_shape)
    diagnostics = NitscheStabilizationDiagnostics(
        _readonly(master_h), _readonly(slave_h),
        _readonly(master_weight), _readonly(slave_weight),
        _readonly(penalty), factor,
    )
    return penalty, master_weight, slave_weight, diagnostics


def assemble_symmetric_nitsche(
    master,
    slave,
    *,
    penalty=None,
    master_lame,
    slave_lame=None,
    average_weights=None,
    stabilization_factor=10.,
    integration=None,
    num_threads=None,
):
    """Assemble a two-sided symmetric Nitsche interface matrix.

    The returned contribution is

    ``penalty * <[u], [v]> - <{sigma(u)n}, [v]>
    - <{sigma(v)n}, [u]>``.

    ``master`` and ``slave`` are vector ``FacetBasis`` objects.  The Lamé
    pairs may differ between the two sides.  With ``penalty=None``, facet
    length, material-weighted traction averages, and a harmonic penalty are
    resolved automatically.  Explicit ``average_weights`` override the
    automatic traction weights and must sum to one.
    """
    from .supermesh import InterfaceSupermesh

    master_lame = _validate_lame("master_lame", master_lame)
    slave_lame = _validate_lame(
        "slave_lame", master_lame if slave_lame is None else slave_lame
    )
    if integration is None:
        integration = InterfaceSupermesh.from_facets(master, slave)
    expected_sizes = (int(master.N), int(slave.N))
    actual_sizes = (int(integration.master_size), int(integration.slave_size))
    if actual_sizes != expected_sizes:
        raise ValueError(
            "integration trace sizes do not match the supplied FacetBasis objects"
        )
    stabilization = None
    if penalty is None:
        penalty_coefficient, master_weight, slave_weight, stabilization = (
            _automatic_stabilization(
                master, slave, integration, master_lame, slave_lame,
                stabilization_factor,
            )
        )
    else:
        penalty = float(penalty)
        if not np.isfinite(penalty) or penalty <= 0.:
            raise ValueError("penalty must be finite and positive")
        penalty_coefficient = penalty
        master_weight = slave_weight = None
    if average_weights is not None:
        weights = np.asarray(average_weights, dtype=float)
        if weights.shape != (2,) or not np.isfinite(weights).all():
            raise ValueError("average_weights must contain two finite values")
        if not np.isclose(weights.sum(), 1.):
            raise ValueError("average_weights must sum to one")
        master_weight, slave_weight = float(weights[0]), float(weights[1])
        if stabilization is not None:
            stabilization = replace(
                stabilization,
                master_average_weight=_readonly(np.broadcast_to(
                    master_weight, integration._coefficient_shape
                )),
                slave_average_weight=_readonly(np.broadcast_to(
                    slave_weight, integration._coefficient_shape
                )),
            )
    elif master_weight is None:
        master_weight, slave_weight = .5, .5

    jump_weights = (1., -1.)
    penalty_matrix = integration.assemble_traces(
        jump_weights,
        jump_weights,
        coefficient=penalty_coefficient,
        num_threads=num_threads,
    )
    normal = np.moveaxis(integration.master_normals, -1, 0)
    master_traction = isotropic_traction_tensor(normal, *master_lame) * (
        np.asarray(master_weight)[..., None, None, None]
    )
    slave_traction = isotropic_traction_tensor(normal, *slave_lame) * (
        np.asarray(slave_weight)[..., None, None, None]
    )
    consistency = (
        integration.assemble_traces(
            jump_weights,
            (1., 0.),
            row_kind="value",
            column_kind="gradient",
            coefficient=master_traction,
            num_threads=num_threads,
        )
        + integration.assemble_traces(
            jump_weights,
            (0., 1.),
            row_kind="value",
            column_kind="gradient",
            coefficient=slave_traction,
            num_threads=num_threads,
        )
    ).tocsr()
    adjoint = consistency.T.tocsr()
    matrix = (penalty_matrix - consistency - adjoint).tocsr()
    return SymmetricNitscheResult(
        matrix=matrix,
        penalty=penalty_matrix.tocsr(),
        consistency=consistency,
        adjoint_consistency=adjoint,
        average_weights=(master_weight, slave_weight),
        stabilization=stabilization,
    )


__all__ = [
    "NitscheStabilizationDiagnostics",
    "SymmetricNitscheResult",
    "assemble_symmetric_nitsche",
]
