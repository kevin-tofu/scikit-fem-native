"""Contact path using the NUMISHEET '93 blank and rigid profiles.

This integrates the source deck's die, punch, and blank-holder geometry with
the contact/Newton kernel. The sharp analytical tool corners are preserved by
default; an explicit holder fillet is available for sensitivity studies. The
default material model is small-strain Hill48; pass
``--finite-strain-j2`` for multiplicative finite-strain isotropic J2 with the
official deck's tabular hardening curve, or add ``--puso-isotropic-hardening``
to select Puso's separately reported isotropic law. These are partial-stroke
integration checks, not reproductions of the 70 mm shell benchmark.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import tempfile
import warnings
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.linalg import LinAlgWarning
from scipy.optimize import lsq_linear

import skfemntv
from skfemntv.mortar_contact import RigidProfileMortarContact2D
from scikit_fvm_native.kktkit_bridge import (
    solve_kktkit_contact_step, solve_kktkit_equality_step,
    solve_sparse_contact_step, solve_sparse_equality_step,
)

# Use the analytical corner from the NUMISHEET input by default.  An optional
# fillet is retained only for explicit geometry-sensitivity studies; it must
# not silently alter the reference tool shape.
HOLDER_CORNER_SMOOTHING_MM = float(os.environ.get(
    "SCIKIT_FVM_FORMING_HOLDER_CORNER_SMOOTHING_MM", "0.0"
))
if (not np.isfinite(HOLDER_CORNER_SMOOTHING_MM)
        or HOLDER_CORNER_SMOOTHING_MM < 0.):
    raise ValueError(
        "SCIKIT_FVM_FORMING_HOLDER_CORNER_SMOOTHING_MM must be finite "
        "and nonnegative"
    )
# Use the exact signed-distance Hessian by default so the Newton tangent
# differentiates the same rounded profile as the gap and contact normal.
# A positive blend remains available as an explicit quasi-Newton option for
# runs that need extra robustness while crossing a line/arc curvature jump.
HOLDER_CONTACT_CURVATURE_BLEND_LENGTH_MM = float(os.environ.get(
    "SCIKIT_FVM_FORMING_HOLDER_CONTACT_CURVATURE_BLEND_LENGTH_MM", "0.0"
))
PUNCH_CONTACT_CURVATURE_BLEND_LENGTH_MM = 0.6


def _is_symmetric_matrix(matrix, *, rtol=1e-10, atol=1e-12):
    difference = (matrix - matrix.T).tocsr()
    scale = float(np.max(np.abs(matrix.data), initial=0.))
    return difference.nnz == 0 or bool(
        np.max(np.abs(difference.data), initial=0.) <= atol + rtol * scale
    )


def load_profiles(reference_path: Path):
    reference = json.loads(reference_path.read_text())
    data = reference["analytical_tool_profiles_from_reference_input"]
    axis = np.asarray(data["surface_axis_line"], dtype=float)
    generator = np.asarray(data["surface_reference_direction"], dtype=float)
    transform, offset = skfemntv.AnalyticalRigidProfile2D.keyword_local_transform_xz(
        axis, generator
    )
    scale = 1000.0  # source deck uses metres; this example uses mm, N, MPa
    profiles = {}
    for name, body in data["rigid_bodies"].items():
        local = skfemntv.AnalyticalRigidProfile2D.from_keyword_profile(body["profile"])
        profile = local.transformed(scale * transform, scale * offset)
        if name == "blank_holder":
            # Keep the source's sharp inner corner unless a geometry
            # sensitivity run explicitly requests a fillet.
            profile = profile.with_rounded_corners(
                HOLDER_CORNER_SMOOTHING_MM
            )
        profiles[name] = {
            "profile": profile,
            "reference_node_mm": np.asarray(body["reference_node"], dtype=float) * scale,
        }
    return reference, {name: value["profile"] for name, value in profiles.items()}


def run(nx: int = 70, nz: int = 4, maximum_stroke_mm: float = 70.0,
        maximum_increment_mm: float = 0.1,
        refine_punch_arc: bool = False,
        refine_holder_corner: bool = False,
        refine_tool_contacts: bool = False,
        finite_strain_j2: bool = False,
        checkpoint_path: str | Path | None = None,
        resume: bool = False,
        minimum_unload_increment: float = 1e-5,
        resume_forming_from_springback: bool = False,
        forming_only: bool = False,
        friction_coefficient: float = 0.0,
        forming_equilibrium_tolerance: float = 2.0e-1,
        forming_complementarity_tolerance: float = 1.0e-2,
        forming_equilibrium_relative_tolerance: float = 5.0e-3,
        forming_slow_progress_limit: int | None = None,
        holder_position_control_mm: float | None = None,
        refine_holder_return: bool = False,
        mesh_refinement_level: int = 0,
        re_equilibrate_checkpoint: bool = False,
        puso_isotropic_hardening: bool = False,
        initial_unload_increment: float = 0.1,
        maximum_unload_increment: float = 0.2,
        withdraw_punch_after_forming: bool = False) -> dict:
    minimum_nx = 14 if refine_holder_return else (11 if refine_tool_contacts else 8)
    if nx < minimum_nx or nz < 2 or maximum_stroke_mm < 0. or maximum_increment_mm <= 0.:
        raise ValueError(
            f"require nx>={minimum_nx}, nz>=2, nonnegative stroke, positive increment"
        )
    if mesh_refinement_level < 0:
        raise ValueError("mesh_refinement_level must be nonnegative")
    if puso_isotropic_hardening and not finite_strain_j2:
        raise ValueError("Puso isotropic hardening requires --finite-strain-j2")
    if re_equilibrate_checkpoint and not resume:
        raise ValueError("re-equilibrating a checkpoint requires --resume")
    refinement_modes = sum((refine_punch_arc, refine_holder_corner,
                            refine_tool_contacts, refine_holder_return))
    if (refinement_modes > 1
            and not (refine_punch_arc and refine_holder_return
                     and refinement_modes == 2)):
        raise ValueError(
            "only punch-arc and holder-return refinements may be combined"
        )
    # Contact normal changes at the rounded holder return can require stroke
    # cutbacks well below 1/1024 of the nominal increment.  Keep the adaptive
    # floor small enough to resolve that branch instead of repeatedly
    # restoring a step known to fail.
    forming_increment_floor_mm = float(os.environ.get(
        "SCIKIT_FVM_FORMING_INCREMENT_FLOOR_MM",
        str(max(maximum_increment_mm / 16384., 1e-8)),
    ))
    if (not np.isfinite(forming_increment_floor_mm)
            or forming_increment_floor_mm <= 0.
            or forming_increment_floor_mm > maximum_increment_mm):
        raise ValueError(
            "SCIKIT_FVM_FORMING_INCREMENT_FLOOR_MM must be finite, positive, "
            "and no larger than maximum_increment_mm"
        )
    forming_increment_recovery_factor = float(os.environ.get(
        "SCIKIT_FVM_FORMING_INCREMENT_RECOVERY_FACTOR", "1.5"
    ))
    if (not np.isfinite(forming_increment_recovery_factor)
            or forming_increment_recovery_factor <= 1.):
        raise ValueError(
            "SCIKIT_FVM_FORMING_INCREMENT_RECOVERY_FACTOR must be finite "
            "and greater than one"
        )
    if not np.isfinite(friction_coefficient) or friction_coefficient < 0.:
        raise ValueError("friction_coefficient must be finite and nonnegative")
    if (not np.isfinite(forming_equilibrium_tolerance)
            or forming_equilibrium_tolerance <= 0.
            or not np.isfinite(forming_complementarity_tolerance)
            or forming_complementarity_tolerance <= 0.
            or not np.isfinite(forming_equilibrium_relative_tolerance)
            or forming_equilibrium_relative_tolerance < 0.):
        raise ValueError("forming convergence tolerances must be finite and positive")
    if resume_forming_from_springback and not resume:
        raise ValueError("--resume-forming-from-springback requires --resume")
    if (holder_position_control_mm is not None
            and not np.isfinite(holder_position_control_mm)):
        raise ValueError("holder_position_control_mm must be finite")
    reference_path = Path(__file__).with_name("numisheet93_draw_bending_reference.json")
    reference, tools = load_profiles(reference_path)
    punch_contact_curvature_blend_length_mm = float(os.environ.get(
        "SCIKIT_FVM_FORMING_PUNCH_CONTACT_CURVATURE_BLEND_LENGTH_MM",
        str(PUNCH_CONTACT_CURVATURE_BLEND_LENGTH_MM),
    ))
    if (not np.isfinite(punch_contact_curvature_blend_length_mm)
            or punch_contact_curvature_blend_length_mm < 0.):
        raise ValueError(
            "SCIKIT_FVM_FORMING_PUNCH_CONTACT_CURVATURE_BLEND_LENGTH_MM "
            "must be finite and nonnegative"
        )
    if (not np.isfinite(HOLDER_CONTACT_CURVATURE_BLEND_LENGTH_MM)
            or HOLDER_CONTACT_CURVATURE_BLEND_LENGTH_MM < 0.):
        raise ValueError(
            "SCIKIT_FVM_FORMING_HOLDER_CONTACT_CURVATURE_BLEND_LENGTH_MM "
            "must be finite and nonnegative"
        )
    material_data = reference["material"]
    geometry = reference["blank"]
    half_length = float(geometry["half_model_length"])
    thickness = float(geometry["thickness"])
    # Contact feasibility is measured at a fraction of the physical sheet
    # thickness.  Keep a strict default, but allow mesh/contact studies to set
    # a physically scaled tolerance when exact pointwise closure stalls an
    # otherwise equilibrated increment.
    forming_contact_gap_tolerance_mm = float(os.environ.get(
        "SCIKIT_FVM_FORMING_CONTACT_GAP_TOLERANCE_MM",
        str(max(2e-7, 1e-4 * thickness)),
    ))
    if (not np.isfinite(forming_contact_gap_tolerance_mm)
            or forming_contact_gap_tolerance_mm <= 0.):
        raise ValueError(
            "SCIKIT_FVM_FORMING_CONTACT_GAP_TOLERANCE_MM must be finite "
            "and positive"
        )
    die_contact_feature_blend_temperature_mm = float(os.environ.get(
        "SCIKIT_FVM_FORMING_DIE_FEATURE_BLEND_TEMPERATURE_MM",
        str(max(1e-9, 0.05 * forming_contact_gap_tolerance_mm)),
    ))
    if (not np.isfinite(die_contact_feature_blend_temperature_mm)
            or die_contact_feature_blend_temperature_mm < 0.):
        raise ValueError(
            "SCIKIT_FVM_FORMING_DIE_FEATURE_BLEND_TEMPERATURE_MM must be "
            "finite and nonnegative"
        )
    die_support_gap_limit_mm = float(os.environ.get(
        "SCIKIT_FVM_FORMING_DIE_SUPPORT_GAP_LIMIT_MM",
        str(1.5 * thickness),
    ))
    if (not np.isfinite(die_support_gap_limit_mm)
            or die_support_gap_limit_mm <= 0.):
        raise ValueError(
            "SCIKIT_FVM_FORMING_DIE_SUPPORT_GAP_LIMIT_MM must be finite "
            "and positive"
        )
    # Use the same physical gap tolerance in the linear contact solve and the
    # nonlinear acceptance check. The inner KKT solve can use a separate
    # complementarity tolerance to return a useful Newton direction; the
    # outer line search still checks the physical complementarity before an
    # iterate or material history can be committed.
    linear_contact_tolerance = forming_contact_gap_tolerance_mm
    # Contact gap and force-gap complementarity have different physical
    # units, so use an independent force-gap tolerance for N·mm.
    linear_contact_complementarity_tolerance = float(os.environ.get(
        "SCIKIT_FVM_FORMING_LINEAR_CONTACT_COMPLEMENTARITY_TOLERANCE_N_MM",
        str(forming_complementarity_tolerance),
    ))
    if (not np.isfinite(linear_contact_complementarity_tolerance)
            or linear_contact_complementarity_tolerance <= 0.):
        raise ValueError(
            "SCIKIT_FVM_FORMING_LINEAR_CONTACT_COMPLEMENTARITY_TOLERANCE_N_MM "
            "must be finite and positive"
        )
    default_holder_jump_tolerance_mm = 0.25 * thickness
    holder_jump_tolerance_mm = float(os.environ.get(
        "SCIKIT_FVM_FORMING_HOLDER_JUMP_TOLERANCE_MM",
        str(default_holder_jump_tolerance_mm),
    ))
    if (not np.isfinite(holder_jump_tolerance_mm)
            or holder_jump_tolerance_mm <= 0.):
        raise ValueError(
            "SCIKIT_FVM_FORMING_HOLDER_JUMP_TOLERANCE_MM must be finite "
            "and positive"
        )
    holder_support_search_radius_mm = float(os.environ.get(
        "SCIKIT_FVM_FORMING_HOLDER_SUPPORT_SEARCH_RADIUS_MM",
        str(max(0.25, 2.5 * thickness,
                10. * forming_contact_gap_tolerance_mm)),
    ))
    if (not np.isfinite(holder_support_search_radius_mm)
            or holder_support_search_radius_mm <= 0.):
        raise ValueError(
            "SCIKIT_FVM_FORMING_HOLDER_SUPPORT_SEARCH_RADIUS_MM must be "
            "finite and positive"
        )
    holder_seed_tensile_release_threshold = float(os.environ.get(
        "SCIKIT_FVM_FORMING_HOLDER_SEED_TENSILE_RELEASE_N_PER_MM", "1e-2"
    ))
    if (not np.isfinite(holder_seed_tensile_release_threshold)
            or holder_seed_tensile_release_threshold < 0.):
        raise ValueError(
            "SCIKIT_FVM_FORMING_HOLDER_SEED_TENSILE_RELEASE_N_PER_MM must "
            "be finite and nonnegative"
        )
    default_force_resolution_floor_mm = max(
        maximum_increment_mm / 1024., 1e-8
    )
    force_resolution_floor_setting = os.environ.get(
        "SCIKIT_FVM_FORMING_FORCE_RESOLUTION_FLOOR_MM"
    )
    force_resolution_floor_mm = (
        default_force_resolution_floor_mm
        if force_resolution_floor_setting is None
        else float(force_resolution_floor_setting)
    )
    if (not np.isfinite(force_resolution_floor_mm)
            or force_resolution_floor_mm <= 0.):
        raise ValueError(
            "SCIKIT_FVM_FORMING_FORCE_RESOLUTION_FLOOR_MM must be finite "
            "and positive"
        )
    default_force_change_max_subdivisions = max(
        1,
        int(np.ceil(np.log2(
            maximum_increment_mm / force_resolution_floor_mm
        ))),
    )
    forming_force_change_max_subdivisions = int(os.environ.get(
        "SCIKIT_FVM_FORMING_FORCE_CHANGE_MAX_SUBDIVISIONS",
        str(default_force_change_max_subdivisions),
    ))
    if forming_force_change_max_subdivisions < 1:
        raise ValueError(
            "SCIKIT_FVM_FORMING_FORCE_CHANGE_MAX_SUBDIVISIONS must be positive"
        )
    contact_collocation_points_per_facet = int(os.environ.get(
        "SCIKIT_FVM_FORMING_CONTACT_COLLOCATION_POINTS", "2"
    ))
    default_contact_method = (
        "point-collocation" if friction_coefficient > 0. else "mortar-p0"
    )
    contact_method = os.environ.get(
        "SCIKIT_FVM_FORMING_CONTACT_METHOD", default_contact_method
    ).strip().lower()
    if contact_method not in {"point-collocation", "mortar-p0"}:
        raise ValueError(
            "SCIKIT_FVM_FORMING_CONTACT_METHOD must be "
            "point-collocation or mortar-p0"
        )
    if contact_method == "mortar-p0" and friction_coefficient > 0.:
        raise ValueError("mortar-p0 contact is currently frictionless only")
    holder_contact_collocation_points_per_facet = int(os.environ.get(
        "SCIKIT_FVM_FORMING_HOLDER_COLLOCATION_POINTS",
        str(16 if contact_method == "mortar-p0"
            else contact_collocation_points_per_facet),
    ))
    mortar_subfacets_per_facet = int(os.environ.get(
        "SCIKIT_FVM_FORMING_MORTAR_SUBFACETS_PER_FACET", "1"
    ))
    mortar_gap_reduction = os.environ.get(
        "SCIKIT_FVM_FORMING_MORTAR_GAP_REDUCTION", "softmin"
    ).strip().lower()
    if mortar_gap_reduction not in {"average", "minimum", "softmin"}:
        raise ValueError(
            "SCIKIT_FVM_FORMING_MORTAR_GAP_REDUCTION must be average, "
            "minimum, or softmin"
        )
    mortar_gap_smoothing_temperature_mm = float(os.environ.get(
        "SCIKIT_FVM_FORMING_MORTAR_GAP_TEMPERATURE_MM", "1e-7"
    ))
    if (not np.isfinite(mortar_gap_smoothing_temperature_mm)
            or mortar_gap_smoothing_temperature_mm <= 0.):
        raise ValueError(
            "SCIKIT_FVM_FORMING_MORTAR_GAP_TEMPERATURE_MM must be finite "
            "and positive"
        )
    contact_rank_tolerance = float(os.environ.get(
        "SCIKIT_FVM_FORMING_CONTACT_RANK_TOLERANCE", "1e-6"
    ))
    contact_solver_max_iterations = int(os.environ.get(
        "SCIKIT_FVM_FORMING_CONTACT_MAX_ITERATIONS", "160"
    ))
    contact_dual_tolerance = float(os.environ.get(
        "SCIKIT_FVM_FORMING_CONTACT_DUAL_TOLERANCE", "1e-2"
    ))
    contact_restoration_equilibrium_tolerance_multiplier = float(
        os.environ.get(
            "SCIKIT_FVM_FORMING_CONTACT_RESTORATION_EQUILIBRIUM_TOLERANCES",
            "100",
        )
    )
    if (not np.isfinite(
            contact_restoration_equilibrium_tolerance_multiplier
    ) or contact_restoration_equilibrium_tolerance_multiplier <= 0.):
        raise ValueError(
            "SCIKIT_FVM_FORMING_CONTACT_RESTORATION_EQUILIBRIUM_TOLERANCES "
            "must be finite and positive"
        )
    contact_restoration_holder_tolerance_multiplier = float(os.environ.get(
        "SCIKIT_FVM_FORMING_CONTACT_RESTORATION_HOLDER_TOLERANCE_MULTIPLIER",
        "5",
    ))
    if (not np.isfinite(contact_restoration_holder_tolerance_multiplier)
            or contact_restoration_holder_tolerance_multiplier <= 0.):
        raise ValueError(
            "SCIKIT_FVM_FORMING_CONTACT_RESTORATION_HOLDER_TOLERANCE_MULTIPLIER "
            "must be finite and positive"
        )
    soft_holder_support_seed = os.environ.get(
        "SCIKIT_FVM_FORMING_SOFT_HOLDER_SUPPORT_SEED", ""
    ).lower() in {"1", "true", "yes"}
    soft_punch_support_seed = os.environ.get(
        "SCIKIT_FVM_FORMING_SOFT_PUNCH_SUPPORT_SEED", ""
    ).lower() in {"1", "true", "yes"}
    semismooth_first_setting = os.environ.get(
        "SCIKIT_FVM_FORMING_CONTACT_SEMISMOOTH_FIRST",
        "1" if finite_strain_j2 else "0",
    ).lower()
    contact_semismooth_first = (
        finite_strain_j2
        and semismooth_first_setting in {"1", "true", "yes"}
    )
    directional_tangent_refinements = int(os.environ.get(
        # At first yield the constitutive Jacobian is direction dependent.
        # Refresh once along the contact-constrained Newton direction by
        # default; setting this to zero opts into the loading-side tangent.
        "SCIKIT_FVM_FORMING_DIRECTIONAL_TANGENT_REFINEMENTS", "1"
    ))
    if not 0 <= directional_tangent_refinements <= 3:
        raise ValueError(
            "SCIKIT_FVM_FORMING_DIRECTIONAL_TANGENT_REFINEMENTS must be in [0, 3]"
        )
    feasible_contact_equilibrium_reduction_fraction = float(os.environ.get(
        "SCIKIT_FVM_FORMING_FEASIBLE_CONTACT_EQUILIBRIUM_REDUCTION", "0.01"
    ))
    if (not np.isfinite(feasible_contact_equilibrium_reduction_fraction)
            or not 0. < feasible_contact_equilibrium_reduction_fraction <= 1.):
        raise ValueError(
            "SCIKIT_FVM_FORMING_FEASIBLE_CONTACT_EQUILIBRIUM_REDUCTION must "
            "be finite and in (0, 1]"
        )
    allow_feasible_holder_residual_tradeoff = os.environ.get(
        "SCIKIT_FVM_FORMING_FEASIBLE_HOLDER_RESIDUAL_TRADEOFF", "0"
    ).lower() in {"1", "true", "yes"}
    holder_newton_step_limit_mm = float(os.environ.get(
        "SCIKIT_FVM_FORMING_HOLDER_NEWTON_STEP_LIMIT_MM", "0.1"
    ))
    displacement_newton_step_limit_mm = float(os.environ.get(
        "SCIKIT_FVM_FORMING_DISPLACEMENT_NEWTON_STEP_LIMIT_MM",
        str(max(1e-6, 0.025 * thickness)),
    ))
    if (contact_collocation_points_per_facet < 1
            or holder_contact_collocation_points_per_facet < 1
            or mortar_subfacets_per_facet < 1):
        raise ValueError(
            "contact collocation point and mortar subfacet counts must be positive"
        )
    if (not np.isfinite(holder_newton_step_limit_mm)
            or holder_newton_step_limit_mm <= 0.):
        raise ValueError(
            "SCIKIT_FVM_FORMING_HOLDER_NEWTON_STEP_LIMIT_MM must be finite "
            "and positive"
        )
    if (not np.isfinite(displacement_newton_step_limit_mm)
            or displacement_newton_step_limit_mm <= 0.):
        raise ValueError(
            "SCIKIT_FVM_FORMING_DISPLACEMENT_NEWTON_STEP_LIMIT_MM must be "
            "finite and positive"
        )
    if (not np.isfinite(contact_rank_tolerance)
            or contact_rank_tolerance <= 0.
            or contact_rank_tolerance >= 1.):
        raise ValueError(
            "SCIKIT_FVM_FORMING_CONTACT_RANK_TOLERANCE must be finite "
            "and strictly between zero and one"
        )
    if contact_solver_max_iterations < 1:
        raise ValueError(
            "SCIKIT_FVM_FORMING_CONTACT_MAX_ITERATIONS must be positive"
        )
    if (not np.isfinite(contact_dual_tolerance)
            or contact_dual_tolerance <= 0.):
        raise ValueError(
            "SCIKIT_FVM_FORMING_CONTACT_DUAL_TOLERANCE must be finite "
            "and positive"
        )
    if refine_holder_return:
        # The 35x4 continuation exposed a 21.5 mm top-surface facet that
        # rotated around the blank-holder return. Split that material path
        # into short segments while retaining a graded mesh over the blank.
        bounds = np.array([
            0., 18., 25., 29.5, 30.4, 32.2, 33.5, 36., 40., 48.,
            60., 69.5, 80., 91., half_length,
        ])
        # At nx=35 these weights give cell counts
        # [3,2,1,1,2,1,1,2,3,3,2,2,2,10].
        extra_cells = np.array([2, 1, 0, 0, 1, 0, 0, 1, 2, 2, 1, 1, 1, 9])
        weights = extra_cells / float(np.sum(extra_cells))
        exact = (nx - len(weights)) * weights
        counts = 1 + np.floor(exact).astype(int)
        remainder = nx - int(np.sum(counts))
        counts[np.argsort(-(exact - np.floor(exact)))[:remainder]] += 1
        regions = list(zip(bounds[:-1], bounds[1:], counts))
        x_nodes = np.concatenate([
            np.linspace(left, right, int(cells) + 1)[:-1]
            for left, right, cells in regions
        ] + [bounds[-1:]])
    elif refine_tool_contacts:
        # Resolve both the blank-holder fillet and the nearby die-entry
        # transition.  These are the moving contact zones observed during
        # full-stroke continuation; without the extra breakpoints, one long
        # boundary edge can carry a contact point across an entire tool arc.
        bounds = np.array([
            0., 18., 25., 29.5, 30.4, 32.2, 33.5, 36., 40., 48., 91.,
            half_length,
        ])
        weights = np.array([
            0.12, 0.16, 0.08, 0.07, 0.12, 0.07,
            0.06, 0.12, 0.09, 0.07, 0.04,
        ])
        exact = (nx - len(weights)) * weights
        counts = 1 + np.floor(exact).astype(int)
        remainder = nx - int(np.sum(counts))
        counts[np.argsort(-(exact - np.floor(exact)))[:remainder]] += 1
        regions = list(zip(bounds[:-1], bounds[1:], counts))
        x_nodes = np.concatenate([
            np.linspace(left, right, int(cells) + 1)[:-1]
            for left, right, cells in regions
        ] + [bounds[-1:]])
    elif refine_holder_corner:
        # Resolve the blank-holder's rounded leading corner explicitly.  The
        # coarse mesh lets one edge sample move from the lower face onto the
        # fillet, where its contact normal rotates rapidly.
        bounds = np.array([
            0., 18., 25., 29.5, 30.4, 32.2, 33.5, 36., 91., half_length,
        ])
        weights = np.array([0.14, 0.18, 0.10, 0.08, 0.15, 0.08, 0.07, 0.14, 0.06])
        exact = (nx - len(weights)) * weights
        counts = 1 + np.floor(exact).astype(int)
        remainder = nx - int(np.sum(counts))
        counts[np.argsort(-(exact - np.floor(exact)))[:remainder]] += 1
        regions = list(zip(bounds[:-1], bounds[1:], counts))
        x_nodes = np.concatenate([
            np.linspace(left, right, int(cells) + 1)[:-1]
            for left, right, cells in regions
        ] + [bounds[-1:]])
    elif refine_punch_arc:
        bounds = np.array([0., 18., 25., 36., 91., half_length])
        weights = np.array([0.15, 0.24, 0.12, 0.24, 0.25])
        exact = (nx - len(weights)) * weights
        counts = 1 + np.floor(exact).astype(int)
        remainder = nx - int(np.sum(counts))
        counts[np.argsort(-(exact - np.floor(exact)))[:remainder]] += 1
        regions = list(zip(bounds[:-1], bounds[1:], counts))
        x_nodes = np.concatenate([
            np.linspace(left, right, int(cells) + 1)[:-1]
            for left, right, cells in regions
        ] + [bounds[-1:]])
    else:
        # Concentrate boundary facets around the punch, die shoulder, and
        # holder radii; the long free flange can use larger cells.
        n_punch_die = max(1, int(round(0.45 * nx)))
        n_holder = max(1, int(round(0.35 * nx)))
        n_flange = nx - n_punch_die - n_holder
        if n_flange < 1:
            n_flange = 1
            n_punch_die = max(1, int(round(0.4 * (nx - 1))))
            n_holder = nx - n_punch_die - n_flange
        regions = [(0., 36., n_punch_die), (36., 91., n_holder),
                   (91., half_length, n_flange)]
        x_nodes = np.r_[
            np.linspace(0., 36., n_punch_die + 1),
            np.linspace(36., 91., n_holder + 1)[1:],
            np.linspace(91., half_length, n_flange + 1)[1:],
        ]
    if refine_holder_return and refine_punch_arc:
        # Preserve the holder-return partition and selectively bisect its
        # material edges over the punch arc. This produces a controlled mesh
        # comparison with the holder-return-only branch while resolving the
        # punch contact-point transfers observed during forming.
        left = x_nodes[:-1]
        right = x_nodes[1:]
        punch_arc_edges = (left >= 25. - 1e-12) & (right <= 36. + 1e-12)
        x_nodes = np.sort(np.r_[
            x_nodes, 0.5 * (left[punch_arc_edges] + right[punch_arc_edges])
        ])
    z_nodes = np.linspace(-0.5 * thickness, 0.5 * thickness, nz + 1)
    mesh = skfemntv.MeshTri.init_tensor(x_nodes, z_nodes)
    for _ in range(mesh_refinement_level):
        mesh = mesh.refined()
        x_nodes = np.sort(np.r_[x_nodes, 0.5 * (x_nodes[:-1] + x_nodes[1:])])
        z_nodes = np.sort(np.r_[z_nodes, 0.5 * (z_nodes[:-1] + z_nodes[1:])])
    # TriP1 has a constant deformation gradient/stress per affine triangle, so
    # the centroid rule integrates the internal force and tangent exactly.
    p1_centroid_rule = (
        np.array([[1. / 3.], [1. / 3.]]), np.array([0.5])
    )
    basis = skfemntv.Basis(
        mesh, skfemntv.ElementVector(skfemntv.ElementTriP1(), dim=2),
        intorder=2, quadrature=p1_centroid_rule,
    )
    hardening = material_data[
        "hardening_curve_equivalent_plastic_strain_yield_stress_MPa"
    ]
    if puso_isotropic_hardening:
        puso = reference["puso_isotropic_hardening"]
        plastic_strain = np.linspace(0.0, 0.5, 501)
        hardening = np.column_stack((
            plastic_strain,
            float(puso["initial_yield_stress_MPa"])
            + sum(
                float(term["Q_MPa"])
                * (1.0 - np.exp(-float(term["b"])*plastic_strain))
                for term in puso["saturation_terms"]
            ),
        ))
    material_tangent_method = os.environ.get(
        "SCIKIT_FVM_FORMING_MATERIAL_TANGENT_METHOD", "analytic"
    ).lower()
    if finite_strain_j2:
        material = skfemntv.FiniteStrainJ2Material(
            material_data["young_modulus"],
            material_data["poisson_ratio"],
            float(hardening[0][1]),
            hardening_curve=hardening,
            tangent_method=material_tangent_method,
        )
        assembler = skfemntv.PlaneStrainFiniteStrainJ2Assembler(basis, material)
        equivalent_plastic_strain_index = 9
    else:
        assembler = skfemntv.PlaneStrainHill48Assembler(
            basis,
            young_modulus=material_data["young_modulus"],
            poisson_ratio=material_data["poisson_ratio"],
            yield_stress=material_data["initial_yield_stress"],
            stress_ratios=material_data[
                "hill48_coefficients_R11_R22_R33_R12_R13_R23"
            ],
            hardening_curve=hardening,
        )
        equivalent_plastic_strain_index = 6

    def evaluate_candidate(displacement, committed_state):
        if finite_strain_j2:
            return assembler.evaluate(displacement, committed_state, tangent=False)
        return assembler.evaluate(displacement, committed_state)

    node_count = basis.N // 2
    dof_x = 2 * np.arange(node_count)
    dof_z = dof_x + 1
    node_x = basis.doflocs[0, dof_x]
    node_z = basis.doflocs[1, dof_z]
    symmetry_nodes = np.flatnonzero(np.isclose(node_x, 0.))
    symmetry_dofs = dof_x[symmetry_nodes]
    def boundary_quadrature(
        z_value: float, x_min: float, x_max: float, points_per_facet: int
    ):
        boundary_nodes = np.flatnonzero(np.isclose(node_z, z_value))
        boundary_nodes = boundary_nodes[np.argsort(node_x[boundary_nodes])]
        samples_nodes, samples_weights, samples_points = [], [], []
        facet_nodes = []
        collocation_coordinates, _ = np.polynomial.legendre.leggauss(
            points_per_facet
        )
        for left, right in zip(boundary_nodes[:-1], boundary_nodes[1:]):
            xa, xb = node_x[[left, right]]
            if xa < x_min - 1e-10 or xb > x_max + 1e-10:
                continue
            nodes = np.array([left, right], dtype=np.int64)
            facet_nodes.append(nodes)
            for coordinate in collocation_coordinates:
                weights = np.array([
                    0.5 * (1. - coordinate),
                    0.5 * (1. + coordinate),
                ])
                samples_nodes.append(nodes)
                samples_weights.append(weights)
                samples_points.append(
                    np.column_stack((node_x[nodes], node_z[nodes])).T @ weights
                )
        return {
            "nodes": np.asarray(samples_nodes, dtype=np.int64),
            "weights": np.asarray(samples_weights, dtype=float),
            "points": np.asarray(samples_points, dtype=float),
            "facet_nodes": np.asarray(facet_nodes, dtype=np.int64),
        }

    punch_quadrature = boundary_quadrature(
        0.5 * thickness, 0., 27., contact_collocation_points_per_facet
    )
    holder_quadrature = boundary_quadrature(
        0.5 * thickness, 30., 88., holder_contact_collocation_points_per_facet
    )
    die_quadrature = boundary_quadrature(
        -0.5 * thickness, 25., half_length,
        contact_collocation_points_per_facet,
    )
    if min(punch_quadrature["nodes"].shape[0],
           holder_quadrature["nodes"].shape[0],
           die_quadrature["nodes"].shape[0]) == 0:
        raise ValueError("mesh does not resolve all three source tool contact regions")

    def facet_midpoint_samples(samples):
        nodes = samples["facet_nodes"]
        weights = np.full((nodes.shape[0], 2), 0.5)
        points = np.sum(
            np.stack((node_x[nodes], node_z[nodes]), axis=-1)
            * weights[:, :, None], axis=1,
        )
        return {"nodes": nodes, "weights": weights, "points": points}

    if contact_method == "mortar-p0":
        punch_mortar = RigidProfileMortarContact2D(
            punch_quadrature["facet_nodes"], basis.doflocs[:, dof_x].T,
            dof_x, dof_z, variable_count=basis.N + 1,
            quadrature_order=contact_collocation_points_per_facet,
            subfacets_per_facet=mortar_subfacets_per_facet,
            gap_reduction=mortar_gap_reduction,
            gap_smoothing_temperature=mortar_gap_smoothing_temperature_mm,
        )
        die_mortar = RigidProfileMortarContact2D(
            die_quadrature["facet_nodes"], basis.doflocs[:, dof_x].T,
            dof_x, dof_z, variable_count=basis.N + 1,
            quadrature_order=contact_collocation_points_per_facet,
            subfacets_per_facet=mortar_subfacets_per_facet,
            gap_reduction=mortar_gap_reduction,
            gap_smoothing_temperature=mortar_gap_smoothing_temperature_mm,
        )
        holder_mortar = RigidProfileMortarContact2D(
            holder_quadrature["facet_nodes"], basis.doflocs[:, dof_x].T,
            dof_x, dof_z, variable_count=basis.N + 1,
            tool_translation_dof=basis.N,
            tool_translation_direction=(0., 1.),
            quadrature_order=holder_contact_collocation_points_per_facet,
            subfacets_per_facet=mortar_subfacets_per_facet,
            gap_reduction=mortar_gap_reduction,
            gap_smoothing_temperature=mortar_gap_smoothing_temperature_mm,
        )
        punch = facet_midpoint_samples(punch_quadrature)
        die = facet_midpoint_samples(die_quadrature)
        holder = facet_midpoint_samples(holder_quadrature)
    else:
        punch_mortar = die_mortar = holder_mortar = None
        punch = punch_quadrature
        die = die_quadrature
        holder = holder_quadrature

    def sample_coordinates(samples, u):
        current_x = node_x + u[dof_x]
        current_z = node_z + u[dof_z]
        coordinates = np.stack((
            current_x[samples["nodes"]], current_z[samples["nodes"]],
        ), axis=-1)
        return np.sum(samples["weights"][..., None] * coordinates, axis=1)

    symmetry_equality_count = symmetry_dofs.size
    holder_position_equality_row = (
        symmetry_equality_count
        if holder_position_control_mm is not None else None
    )
    equality_count = symmetry_equality_count + int(
        holder_position_control_mm is not None
    )
    punch_row_count = (
        punch_mortar.facet_count if contact_method == "mortar-p0"
        else punch["nodes"].shape[0]
    )
    die_row_count = (
        die_mortar.facet_count if contact_method == "mortar-p0"
        else die["nodes"].shape[0]
    )
    holder_row_count = (
        holder_mortar.facet_count if contact_method == "mortar-p0"
        else holder["nodes"].shape[0]
    )
    punch_rows = np.arange(equality_count, equality_count + punch_row_count)
    die_start = equality_count + punch_rows.size
    die_rows = np.arange(die_start, die_start + die_row_count)
    holder_start = die_start + die_rows.size
    holder_rows = np.arange(holder_start, holder_start + holder_row_count)
    row_count = equality_count + punch_rows.size + die_rows.size + holder_rows.size
    unilateral = np.r_[np.zeros(equality_count, dtype=bool),
                       np.ones(punch_rows.size + die_rows.size + holder_rows.size,
                               dtype=bool)]
    variable_count = basis.N + 1  # final variable is the holder's vertical position

    def tool_state(u: np.ndarray, holder_z: float, stroke: float):
        if contact_method == "mortar-p0":
            full_u = np.r_[u, holder_z]
            punch_eval = punch_mortar.evaluate(
                full_u, tools["punch"], translation=[0., -stroke],
                curvature_blend_length=punch_contact_curvature_blend_length_mm,
            )
            die_eval = die_mortar.evaluate(
                full_u, tools["die"],
                feature_blend_temperature=(
                    die_contact_feature_blend_temperature_mm
                ),
            )
            holder_eval = holder_mortar.evaluate(
                full_u, tools["blank_holder"], translation=[0., holder_z],
                curvature_blend_length=HOLDER_CONTACT_CURVATURE_BLEND_LENGTH_MM,
            )
            gaps = np.r_[punch_eval.gaps, die_eval.gaps, holder_eval.gaps]
            constraints = np.zeros((row_count, variable_count))
            for row, dof in enumerate(symmetry_dofs):
                constraints[row, dof] = 1.
            if holder_position_equality_row is not None:
                constraints[holder_position_equality_row, -1] = 1.
            constraints[punch_rows] = punch_eval.jacobian.toarray()
            constraints[die_rows] = die_eval.jacobian.toarray()
            constraints[holder_rows] = holder_eval.jacobian.toarray()
            return gaps, constraints, ("mortar-p0", (punch_eval, die_eval, holder_eval))

        points_top = sample_coordinates(punch, u)
        points_die = sample_coordinates(die, u)
        points_holder = sample_coordinates(holder, u)
        punch_projection = tools["punch"].translated([0., -stroke]).project(
            points_top,
            curvature_blend_length=punch_contact_curvature_blend_length_mm,
        )
        die_projection = tools["die"].project(
            points_die,
            feature_blend_temperature=(
                die_contact_feature_blend_temperature_mm
            ),
        )
        holder_projection = tools["blank_holder"].translated(
            [0., holder_z]
        ).project(
            points_holder,
            curvature_blend_length=HOLDER_CONTACT_CURVATURE_BLEND_LENGTH_MM,
        )
        gaps = np.r_[punch_projection.gaps, die_projection.gaps,
                     holder_projection.gaps]
        constraints = np.zeros((row_count, variable_count))
        for row, dof in enumerate(symmetry_dofs):
            constraints[row, dof] = 1.
        if holder_position_equality_row is not None:
            constraints[holder_position_equality_row, -1] = 1.
        start = equality_count
        constraints[punch_rows[:, None], dof_x[punch["nodes"]]] = (
            punch["weights"] * punch_projection.normals[:, 0, None]
        )
        constraints[punch_rows[:, None], dof_z[punch["nodes"]]] = (
            punch["weights"] * punch_projection.normals[:, 1, None]
        )
        constraints[die_rows[:, None], dof_x[die["nodes"]]] = (
            die["weights"] * die_projection.normals[:, 0, None]
        )
        constraints[die_rows[:, None], dof_z[die["nodes"]]] = (
            die["weights"] * die_projection.normals[:, 1, None]
        )
        constraints[holder_rows[:, None], dof_x[holder["nodes"]]] = (
            holder["weights"] * holder_projection.normals[:, 0, None]
        )
        constraints[holder_rows[:, None], dof_z[holder["nodes"]]] = (
            holder["weights"] * holder_projection.normals[:, 1, None]
        )
        constraints[holder_rows, -1] = -holder_projection.normals[:, 1]
        gap_hessians = np.concatenate((
            punch_projection.gap_hessians,
            die_projection.gap_hessians,
            holder_projection.gap_hessians,
        ), axis=0)
        return gaps, constraints, gap_hessians

    def minimum_tool_gap(gaps, geometry) -> float:
        """Include mortar quadrature samples in unilateral feasibility checks."""
        minimum = float(np.min(gaps, initial=0.))
        if (isinstance(geometry, tuple) and len(geometry) == 2
                and geometry[0] == "mortar-p0"):
            for evaluation in geometry[1]:
                minimum = min(
                    minimum,
                    float(np.min(evaluation.minimum_sample_gaps, initial=0.)),
                )
        return minimum

    def approach_holder_support(u, holder_z, stroke):
        """Move a force-controlled holder to its nearest feasible support.

        Before contact exists, the holder force has no contact reaction to
        balance it.  Do not ask the structural Newton solve to find that
        missing support through a nearly singular rigid-tool degree of
        freedom.  Instead, use the exact contact geometry and its local gap
        derivative to approach the nearest force-opposing row.  This only
        changes the trial tool position; the blank and its committed material
        history remain untouched.  A backtracking search checks every tool
        gap, including mortar quadrature samples, before accepting the move.
        """
        if (holder_position_control_mm is not None
                or holder_force_per_width <= 0.):
            return float(holder_z), False, 0.
        gaps, constraints, geometry = tool_state(u, holder_z, stroke)
        support_sensitivity = constraints[holder_rows, -1]
        # Ignore nearly tangential normals: they nominally oppose the load,
        # but need an unbounded rigid-tool translation to close a small gap.
        support_rows = holder_rows[support_sensitivity > 0.1]
        if not support_rows.size:
            return float(holder_z), False, 0.
        support_gaps = gaps[support_rows - equality_count]
        support_motion = support_gaps / constraints[support_rows, -1]
        nearest_index = int(np.argmin(support_motion))
        row = int(support_rows[nearest_index])
        gap = float(support_gaps[nearest_index])
        # Start just inside the contact branch.  At a positive gap with a zero
        # multiplier, the Fischer--Burmeister generalized derivative has no
        # gap-to-reaction coupling, even when that gap is below the outer
        # physical feasibility tolerance.  A tiny bounded penetration selects
        # the contact-side derivative and lets the holder force enter the KKT
        # correction.
        contact_activation_gap_mm = min(
            1e-8, forming_contact_gap_tolerance_mm
        )
        if gap <= -contact_activation_gap_mm:
            return float(holder_z), False, gap
        # Let the normal complementarity projection release a stale reaction
        # before changing the tool geometry.  Otherwise the old multiplier
        # would be carried to a different support location during approach.
        if np.any(multipliers[support_rows] < -1e-2):
            return float(holder_z), False, gap
        sensitivity = float(constraints[row, -1])
        if not np.isfinite(sensitivity) or sensitivity <= 1e-8:
            return float(holder_z), False, gap
        correction = float(np.clip(
            (-contact_activation_gap_mm - gap) / sensitivity,
            -holder_newton_step_limit_mm,
            holder_newton_step_limit_mm,
        ))
        if abs(correction) <= 1e-14:
            return float(holder_z), False, gap
        for backtrack in range(12):
            candidate_z = float(holder_z + correction * (0.5 ** backtrack))
            candidate_gaps, _, candidate_geometry = tool_state(
                u, candidate_z, stroke
            )
            candidate_gap = float(
                candidate_gaps[row - equality_count]
            )
            candidate_minimum_gap = minimum_tool_gap(
                candidate_gaps, candidate_geometry
            )
            if (candidate_gap < gap - max(1e-10, 1e-6 * gap)
                    and candidate_minimum_gap
                    >= -forming_contact_gap_tolerance_mm):
                return candidate_z, True, candidate_gap
        return float(holder_z), False, gap

    # A small elastic stick range regularizes Coulomb friction, then the trial
    # tangential multiplier is capped by mu times the compressive normal force.
    # The stored scalar multiplier is resolved along each contact's current
    # tangent and is committed only with an accepted forming increment.
    friction_slip_scale = max(1e-4, 1e-2 * thickness)

    def friction_response(
        u, holder_z, stroke, base_u, base_holder_z, base_stroke,
        normal_multipliers, *, tangent=True,
    ):
        response = np.zeros(variable_count)
        stiffness_rows, stiffness_columns, stiffness_values = [], [], []
        trial_multipliers = np.zeros(row_count)
        if friction_coefficient <= 0.:
            return response, sparse.csr_matrix((variable_count, variable_count)), trial_multipliers
        groups = (
            (punch, punch_rows, tools["punch"], np.array([0., -stroke]),
             np.array([0., -base_stroke]), False),
            (die, die_rows, tools["die"], np.zeros(2), np.zeros(2), False),
            (holder, holder_rows, tools["blank_holder"], np.array([0., holder_z]),
             np.array([0., base_holder_z]), True),
        )
        for samples, rows_for_tool, profile, tool_shift, base_shift, movable in groups:
            current_points = sample_coordinates(samples, u)
            base_points = sample_coordinates(samples, base_u)
            projection = profile.translated(tool_shift).project(current_points)
            relative_increment = current_points - base_points - (tool_shift - base_shift)
            normals = projection.normals
            tangents = np.column_stack((-normals[:, 1], normals[:, 0]))
            slip = np.einsum("ij,ij->i", tangents, relative_increment)
            normal_capacity = np.maximum(-normal_multipliers[rows_for_tool], 0.)
            friction_capacity = friction_coefficient * normal_capacity
            elastic_stiffness = friction_capacity / friction_slip_scale
            raw_multiplier = (
                committed_friction_multipliers[rows_for_tool]
                + elastic_stiffness * slip
            )
            tangential_multiplier = np.clip(
                raw_multiplier, -friction_capacity, friction_capacity
            )
            trial_multipliers[rows_for_tool] = tangential_multiplier
            for local_row, row in enumerate(rows_for_tool):
                tau = float(tangential_multiplier[local_row])
                if tau == 0. and (not tangent or elastic_stiffness[local_row] == 0.):
                    continue
                dofs, coefficients = [], []
                for node, weight in zip(
                    samples["nodes"][local_row], samples["weights"][local_row],
                    strict=True,
                ):
                    dofs.extend((int(dof_x[node]), int(dof_z[node])))
                    coefficients.extend((
                        float(weight * tangents[local_row, 0]),
                        float(weight * tangents[local_row, 1]),
                    ))
                if movable:
                    dofs.append(basis.N)
                    coefficients.append(float(-tangents[local_row, 1]))
                np.add.at(response, dofs, tau * np.asarray(coefficients))
                if tangent and abs(raw_multiplier[local_row]) < friction_capacity[local_row]:
                    coefficient_array = np.asarray(coefficients)
                    local_matrix = elastic_stiffness[local_row] * np.outer(
                        coefficient_array, coefficient_array
                    )
                    rr, cc = np.nonzero(local_matrix)
                    stiffness_rows.extend(np.asarray(dofs, dtype=np.int64)[rr].tolist())
                    stiffness_columns.extend(np.asarray(dofs, dtype=np.int64)[cc].tolist())
                    stiffness_values.extend(local_matrix[rr, cc].tolist())
        matrix = sparse.coo_matrix(
            (stiffness_values, (stiffness_rows, stiffness_columns)),
            shape=(variable_count, variable_count),
        ).tocsr()
        return response, matrix, trial_multipliers

    def contact_geometric_tangent(gap_hessians, multipliers):
        """Assemble the signed-distance curvature contribution to the KKT tangent."""
        if (isinstance(gap_hessians, tuple) and len(gap_hessians) == 2
                and gap_hessians[0] == "mortar-p0"):
            rows, columns, values = [], [], []
            for row_slice, evaluation in zip(
                (punch_rows, die_rows, holder_rows), gap_hessians[1], strict=True
            ):
                for local, contact_row in enumerate(row_slice):
                    multiplier = float(multipliers[contact_row])
                    if abs(multiplier) <= 1e-14:
                        continue
                    dofs = evaluation.facet_dofs[local]
                    keep = dofs >= 0
                    local_tangent = multiplier * evaluation.facet_hessians[local]
                    local_tangent = local_tangent[np.ix_(keep, keep)]
                    local_dofs = dofs[keep]
                    rr, cc = np.nonzero(local_tangent)
                    rows.extend(local_dofs[rr].tolist())
                    columns.extend(local_dofs[cc].tolist())
                    values.extend(local_tangent[rr, cc].tolist())
            return sparse.coo_matrix(
                (values, (rows, columns)), shape=(variable_count, variable_count)
            ).tocsr()
        rows, columns, values = [], [], []
        for local_row, contact_row in enumerate(range(equality_count, row_count)):
            multiplier = float(multipliers[contact_row])
            if abs(multiplier) <= 1e-14:
                continue
            if contact_row < die_rows[0]:
                samples, sample_index, tool_vertical_dof = (
                    punch, contact_row - punch_rows[0], None
                )
            elif contact_row < holder_rows[0]:
                samples, sample_index, tool_vertical_dof = (
                    die, contact_row - die_rows[0], None
                )
            else:
                samples, sample_index, tool_vertical_dof = (
                    holder, contact_row - holder_rows[0], basis.N
                )
            hgap = gap_hessians[local_row]
            node_ids = samples["nodes"][sample_index]
            shape = samples["weights"][sample_index]
            local_dofs = []
            map_rows = []
            map_values = []
            for a, node in enumerate(node_ids):
                local_dofs.extend((int(dof_x[node]), int(dof_z[node])))
                map_rows.extend((0, 1))
                map_values.extend((shape[a], shape[a]))
            if tool_vertical_dof is not None:
                local_dofs.append(int(tool_vertical_dof))
                map_rows.append(1)
                map_values.append(-1.)
            mapping = np.zeros((2, len(local_dofs)))
            mapping[np.asarray(map_rows), np.arange(len(local_dofs))] = map_values
            local_tangent = multiplier * mapping.T @ hgap @ mapping
            rr, cc = np.nonzero(local_tangent)
            rows.extend(np.asarray(local_dofs, dtype=np.int64)[rr].tolist())
            columns.extend(np.asarray(local_dofs, dtype=np.int64)[cc].tolist())
            values.extend(local_tangent[rr, cc].tolist())
        return sparse.coo_matrix(
            (values, (rows, columns)), shape=(variable_count, variable_count)
        ).tocsr()

    # Scale the total force to the half model's 5 mm out-of-plane width.
    modeled_width = 5.0
    half_model_holder_force = (
        0.5 * reference["process"]["blank_holder_force_total"]
        * modeled_width / reference["blank"]["width"]
    )
    punch_force_change_reference_N = max(half_model_holder_force, 1.0)
    holder_force_per_width = half_model_holder_force / modeled_width
    holder_external_force_per_width = (
        0. if holder_position_control_mm is not None
        else holder_force_per_width
    )

    def equilibrium_tolerance_for(multipliers, constraints):
        """Combine an absolute floor with a load-scaled residual tolerance.

        The infinity norm is in N/mm. Scale the relative component with the
        largest current or last-accepted punch reaction and the applied
        blank-holder force per modeled width. Holding the accepted reaction
        scale through one Newton solve avoids an abrupt tolerance collapse
        when an iterate crosses a punch-detachment branch; the scale updates
        after that state is committed.
        """
        punch_vertical_jacobian = np.asarray(
            constraints[punch_rows][:, dof_z].sum(axis=1)
        ).reshape(-1)
        punch_force_per_width = abs(float(
            np.sum(multipliers[punch_rows] * punch_vertical_jacobian)
        ))
        accepted_punch_force_per_width = (
            abs(float(records[-1].get("punch_force_N_half_model", 0.)))
            / modeled_width
            if records else 0.
        )
        load_scale_per_width = max(
            punch_force_per_width,
            accepted_punch_force_per_width,
            holder_force_per_width,
        )
        return max(
            forming_equilibrium_tolerance,
            forming_equilibrium_relative_tolerance * load_scale_per_width,
        )

    def retained_contact_gap_limits():
        """Return the temporary gap neighborhood for retained reactions.

        Preserve unrequired reactions only while their sample remains within
        the geometric contact tolerance. A force-gap product can be small
        while a weak reaction acts across a visibly open gap; retaining that
        pair can pin a stale support row and block equilibrium progress.
        Neighboring die and load-bearing support rows are selected by their
        dedicated transfer logic.
        """
        return np.full(
            row_count - equality_count,
            forming_contact_gap_tolerance_mm,
            dtype=float,
        )


    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    checkpoint_state_current_layout = not resume
    if resume and checkpoint is None:
        raise ValueError("resume requires checkpoint_path")
    checkpoint_lock_stream = None
    if checkpoint is not None:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        lock_path = checkpoint.with_name(checkpoint.name + ".lock")
        checkpoint_lock_stream = lock_path.open("a+")
        try:
            fcntl.flock(
                checkpoint_lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
            )
        except BlockingIOError as error:
            checkpoint_lock_stream.close()
            raise RuntimeError(
                f"checkpoint is already in use by another forming run: {checkpoint}"
            ) from error
    mesh_digest = hashlib.sha256(b"numisheet93-tri-mesh-v1")
    for mesh_array in (mesh.p, mesh.t):
        contiguous = np.ascontiguousarray(mesh_array)
        mesh_digest.update(str(contiguous.shape).encode("ascii"))
        mesh_digest.update(contiguous.dtype.str.encode("ascii"))
        mesh_digest.update(contiguous.view(np.uint8))
    contact_layout_digest = hashlib.sha256(b"numisheet93-contact-layout-v1")
    for group_name, samples in (
        ("punch", punch), ("die", die), ("holder", holder),
    ):
        contact_layout_digest.update(group_name.encode("ascii"))
        for key in ("nodes", "weights", "points", "facet_nodes"):
            if key not in samples:
                continue
            contiguous = np.ascontiguousarray(samples[key])
            contact_layout_digest.update(key.encode("ascii"))
            contact_layout_digest.update(str(contiguous.shape).encode("ascii"))
            contact_layout_digest.update(contiguous.dtype.str.encode("ascii"))
            contact_layout_digest.update(contiguous.view(np.uint8))
    checkpoint_metadata = {
        "nx": nx,
        "nz": nz,
        "mesh_refinement_level": mesh_refinement_level,
        "mesh_signature": mesh_digest.hexdigest(),
        "contact_layout_signature": contact_layout_digest.hexdigest(),
        "blank_holder_corner_smoothing_radius_mm": (
            HOLDER_CORNER_SMOOTHING_MM
        ),
        "contact_row_count": row_count,
        "contact_method": contact_method,
        "mortar_gap_reduction": mortar_gap_reduction,
        "mortar_gap_smoothing_temperature_mm": (
            mortar_gap_smoothing_temperature_mm
        ),
        "die_contact_feature_blend_temperature_mm": (
            die_contact_feature_blend_temperature_mm
        ),
        "maximum_increment_mm": maximum_increment_mm,
        "refine_punch_arc": refine_punch_arc,
        "refine_tool_contacts": refine_tool_contacts,
        "refine_holder_return": refine_holder_return,
        "finite_strain_j2": finite_strain_j2,
        "puso_isotropic_hardening": puso_isotropic_hardening,
        "material_tangent_method": material_tangent_method,
        "friction_coefficient": float(friction_coefficient),
        "modeled_width_mm": modeled_width,
        "blank_holder_force_N_half_model": half_model_holder_force,
        "quadrature_points_per_element": int(basis.dx.shape[1]),
        "punch_force_increment_relative_tolerance": float(os.environ.get(
            "SCIKIT_FVM_FORMING_FORCE_CHANGE_TOLERANCE", "0.02"
        )),
        "maximum_plastic_increment_per_step": float(os.environ.get(
            "SCIKIT_FVM_FORMING_MAX_PLASTIC_INCREMENT_PER_STEP", "6e-3"
        )),
        "punch_force_increment_max_subdivisions": (
            forming_force_change_max_subdivisions
        ),
        "punch_force_increment_reference_N": punch_force_change_reference_N,
        "contact_collocation_points_per_facet": (
            contact_collocation_points_per_facet
        ),
        "mortar_subfacets_per_facet": mortar_subfacets_per_facet,
        "holder_contact_collocation_points_per_facet": (
            holder_contact_collocation_points_per_facet
        ),
        "contact_rank_tolerance": contact_rank_tolerance,
        "contact_solver_max_iterations": contact_solver_max_iterations,
        "contact_dual_tolerance": contact_dual_tolerance,
        "linear_contact_complementarity_tolerance_N_mm": (
            linear_contact_complementarity_tolerance
        ),
        "contact_semismooth_first": contact_semismooth_first,
        "directional_tangent_refinements": directional_tangent_refinements,
        "feasible_contact_equilibrium_reduction_fraction": (
            feasible_contact_equilibrium_reduction_fraction
        ),
        "allow_feasible_holder_residual_tradeoff": (
            allow_feasible_holder_residual_tradeoff
        ),
        "contact_restoration_equilibrium_tolerance_multiplier": (
            contact_restoration_equilibrium_tolerance_multiplier
        ),
        "contact_restoration_holder_tolerance_multiplier": (
            contact_restoration_holder_tolerance_multiplier
        ),
        "holder_newton_step_limit_mm": holder_newton_step_limit_mm,
        "displacement_newton_step_limit_mm": displacement_newton_step_limit_mm,
        "holder_jump_tolerance_mm": holder_jump_tolerance_mm,
        "holder_support_search_radius_mm": holder_support_search_radius_mm,
        "holder_seed_tensile_release_N_per_mm": (
            holder_seed_tensile_release_threshold
        ),
        "holder_position_control_mm": holder_position_control_mm,
        "soft_holder_support_seed": soft_holder_support_seed,
        "soft_punch_support_seed": soft_punch_support_seed,
        "soft_die_support_seed": os.environ.get(
            "SCIKIT_FVM_FORMING_SOFT_DIE_SUPPORT_SEED", ""
        ).lower() in {"1", "true", "yes"},
    }
    displacement = np.zeros(basis.N)
    holder_z = 0.0
    previous_displacement = None
    previous_holder_z = None
    previous_stroke = None
    committed = assembler.initial_state()
    records = []
    failure_log = []
    committed_contact_multipliers = np.zeros(row_count)
    committed_friction_multipliers = np.zeros(row_count)
    checkpoint_has_contact_multipliers = False
    checkpoint_extra = {}
    accepted_stroke = 0.0
    suggested_increment_mm = maximum_increment_mm
    saved_regularization_floor = 0.0
    springback_checkpoint_active = False
    checkpoint_holder_position_changed = False
    if resume:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
        with np.load(checkpoint, allow_pickle=False) as saved:
            saved_metadata = json.loads(str(saved["metadata_json"].item()))
            contact_multiplier_mesh_level = int(saved_metadata.get(
                "contact_multiplier_mesh_refinement_level",
                saved_metadata.get("mesh_refinement_level", 0),
            ))
            contact_mesh_level_changed = (
                contact_multiplier_mesh_level != mesh_refinement_level
            )
            contact_multipliers_need_rebuild = contact_mesh_level_changed
            contact_layout_signature_changed = (
                saved_metadata.get("contact_layout_signature")
                != checkpoint_metadata["contact_layout_signature"]
            )
            contact_multipliers_need_rebuild |= (
                contact_layout_signature_changed
            )
            saved_contact_method = saved_metadata.get(
                "contact_method", "point-collocation"
            )
            contact_method_changed = saved_contact_method != contact_method
            contact_multipliers_need_rebuild |= contact_method_changed
            saved_contact_collocation_count = int(saved_metadata.get(
                "contact_collocation_points_per_facet", 2
            ))
            saved_mortar_subfacets = int(saved_metadata.get(
                "mortar_subfacets_per_facet", 1
            ))
            contact_sampling_changed = (
                saved_contact_collocation_count
                != contact_collocation_points_per_facet
                or ((saved_contact_method == "mortar-p0"
                     or contact_method == "mortar-p0")
                    and (saved_mortar_subfacets != mortar_subfacets_per_facet
                         or saved_metadata.get(
                             "mortar_gap_reduction", "average"
                         ) != mortar_gap_reduction
                         or float(saved_metadata.get(
                             "mortar_gap_smoothing_temperature_mm", 1e-5
                         )) != mortar_gap_smoothing_temperature_mm))
            )
            checkpoint_state_current_layout = not (
                contact_multipliers_need_rebuild or contact_sampling_changed
            )
            displacement = np.asarray(saved["displacement"], dtype=float)
            committed = np.asarray(saved["committed"], dtype=float)
            saved_quadrature_count = int(saved_metadata.get(
                "quadrature_points_per_element", committed.shape[1]
            ))
            comparable_saved_metadata = {
                key: value for key, value in saved_metadata.items()
                if key != "quadrature_points_per_element"
            }
            comparable_requested_metadata = {
                key: value for key, value in checkpoint_metadata.items()
                if key != "quadrature_points_per_element"
            }
            comparable_saved_metadata.setdefault("friction_coefficient", 0.0)
            comparable_saved_metadata.setdefault("puso_isotropic_hardening", False)
            comparable_saved_metadata.setdefault("holder_position_control_mm", None)
            comparable_saved_metadata.setdefault(
                "holder_newton_step_limit_mm", 0.1
            )
            comparable_saved_metadata.setdefault(
                "displacement_newton_step_limit_mm",
                displacement_newton_step_limit_mm,
            )
            comparable_saved_metadata.setdefault(
                "contact_restoration_equilibrium_tolerance_multiplier",
                comparable_requested_metadata[
                    "contact_restoration_equilibrium_tolerance_multiplier"
                ],
            )
            comparable_saved_metadata.setdefault(
                "contact_restoration_holder_tolerance_multiplier",
                comparable_requested_metadata[
                    "contact_restoration_holder_tolerance_multiplier"
                ],
            )
            comparable_saved_metadata.setdefault(
                "holder_jump_tolerance_mm", 0.25 * thickness
            )
            comparable_saved_metadata.setdefault(
                "holder_support_search_radius_mm",
                comparable_requested_metadata[
                    "holder_support_search_radius_mm"
                ],
            )
            comparable_saved_metadata.setdefault("mesh_refinement_level", 0)
            # Older checkpoints did not record this mesh-generation switch.
            # The mesh signature remains authoritative when the flag is absent.
            comparable_saved_metadata.setdefault(
                "refine_tool_contacts",
                comparable_requested_metadata.get("refine_tool_contacts", False),
            )
            # Earlier checkpoints predate explicit profile geometry metadata;
            # their driver used the 0.6 mm holder-corner fillet.
            comparable_saved_metadata.setdefault(
                "blank_holder_corner_smoothing_radius_mm", 0.6
            )
            # Holder and die active-set seed modes were introduced after
            # early checkpoints were written. Treat absent fields as the mode
            # explicitly selected for this resume, then persist them on save.
            for support_seed_key in (
                "soft_holder_support_seed", "soft_die_support_seed",
            ):
                comparable_saved_metadata.setdefault(
                    support_seed_key,
                    comparable_requested_metadata[support_seed_key],
                )
            # Checkpoints predating the optional punch seed mode used a
            # required punch reaction seed. Preserve that legacy default so
            # enabling the soft mode rebuilds the contact multipliers.
            comparable_saved_metadata.setdefault(
                "soft_punch_support_seed", False
            )
            checkpoint_support_seed_changed = any(
                bool(comparable_saved_metadata[key])
                != bool(comparable_requested_metadata[key])
                for key in (
                    "soft_holder_support_seed", "soft_punch_support_seed",
                    "soft_die_support_seed",
                )
            )
            comparable_saved_metadata.setdefault(
                "punch_force_increment_max_subdivisions", 4
            )
            comparable_saved_metadata.setdefault(
                "holder_contact_collocation_points_per_facet",
                comparable_saved_metadata.get(
                    "contact_collocation_points_per_facet", 2
                ),
            )
            saved_holder_collocation_count = int(
                comparable_saved_metadata[
                    "holder_contact_collocation_points_per_facet"
                ]
            )
            saved_holder_position_control = comparable_saved_metadata.get(
                "holder_position_control_mm"
            )
            saved_contact_row_count = int(comparable_saved_metadata.get(
                "contact_row_count", 0,
            ))
            # A checkpoint may seed a new punch/die or holder contact
            # discretization, or prescribed holder position, when the blank
            # mesh and constitutive model are unchanged. Rebuild contact
            # reactions whenever the row identities differ.
            for flexible_key in (
                "contact_row_count",
                "contact_layout_signature",
                "contact_method",
                "contact_multiplier_mesh_refinement_level",
                "contact_collocation_points_per_facet",
                "mortar_subfacets_per_facet",
                "mortar_gap_reduction",
                "mortar_gap_smoothing_temperature_mm",
                # This local signed-gap smoothing changes the contact
                # regularization only; the checkpoint is re-equilibrated
                # before any increment is committed.
                "die_contact_feature_blend_temperature_mm",
                "holder_contact_collocation_points_per_facet",
                "contact_rank_tolerance",
                "contact_solver_max_iterations",
                "contact_dual_tolerance",
                "linear_contact_complementarity_tolerance_N_mm",
                "contact_semismooth_first",
                "directional_tangent_refinements",
                "feasible_contact_equilibrium_reduction_fraction",
                "allow_feasible_holder_residual_tradeoff",
                "contact_restoration_equilibrium_tolerance_multiplier",
                "contact_restoration_holder_tolerance_multiplier",
                "material_tangent_method",
                # The nominal stroke increment is an integration control,
                # not model physics. A checkpoint stores the committed
                # displacement and constitutive history, so a continuation
                # may change this cap and let the existing force/plastic
                # increment controls subdivide steps where needed.
                "maximum_increment_mm",
                "holder_position_control_mm",
                "holder_support_search_radius_mm",
                "holder_support_seed_gap_limit_mm",
                "holder_seed_tensile_release_N_per_mm",
                "holder_newton_step_limit_mm",
                "displacement_newton_step_limit_mm",
                "punch_force_increment_max_subdivisions",
                "punch_force_increment_relative_tolerance",
                "maximum_plastic_increment_per_step",
                # These are active-set initialization strategies, not model
                # physics. Permit changing them on resume; their multipliers
                # are discarded below so the new strategy starts from the
                # saved displacement and constitutive history cleanly.
                "soft_holder_support_seed",
                "soft_punch_support_seed",
                "soft_die_support_seed",
                # Curvature blending changes only the approximate contact
                # tangent, not the signed gaps or normals. Allow a checkpoint
                # to resume with a more consistent tangent and re-equilibrate.
                "holder_contact_curvature_blend_length_mm",
                "punch_contact_curvature_blend_length_mm",
            ):
                comparable_saved_metadata.pop(flexible_key, None)
                comparable_requested_metadata.pop(flexible_key, None)
            if comparable_saved_metadata != comparable_requested_metadata:
                raise ValueError(
                    "checkpoint model/mesh metadata does not match this run; "
                    f"saved={comparable_saved_metadata}, "
                    f"requested={comparable_requested_metadata}"
                )
            holder_z = float(saved["holder_z"])
            checkpoint_holder_position_changed = (
                holder_position_control_mm is not None
                and abs(holder_z - holder_position_control_mm) > 1e-10
            )
            if checkpoint_holder_position_changed:
                holder_z = float(holder_position_control_mm)
            accepted_stroke = float(saved["accepted_stroke"])
            records = json.loads(str(saved["records_json"].item()))
            if "failure_log_json" in saved.files:
                failure_log = json.loads(str(saved["failure_log_json"].item()))
            if "suggested_increment_mm" in saved.files:
                suggested_increment_mm = min(
                    float(saved["suggested_increment_mm"]), maximum_increment_mm
                )
            # A saved floor belongs to a failed trial, not to the committed
            # equilibrium state.  Restart each new continuation attempt from
            # the unregularized consistent tangent.
            saved_regularization_floor = 0.0
            saved_contact_multipliers = np.asarray(
                saved["contact_multipliers"], dtype=float
            ) if "contact_multipliers" in saved.files else np.zeros(0)
            saved_friction_multipliers = np.asarray(
                saved["friction_multipliers"], dtype=float
            ) if "friction_multipliers" in saved.files else np.zeros(0)
            contact_layout_changed = (
                saved_holder_collocation_count
                != holder_contact_collocation_points_per_facet
                or saved_contact_row_count != row_count
            )
            if contact_multipliers_need_rebuild or contact_sampling_changed:
                # A transferred checkpoint preserves displacement and
                # constitutive history, but contact multipliers belong to the
                # source mesh's boundary rows and collocation points. Let the
                # current contact solve reconstruct reactions from its current
                # geometry instead of assigning old forces to different rows.
                committed_contact_multipliers = np.zeros(row_count)
                committed_friction_multipliers = np.zeros(row_count)
                same_contact_mesh = (
                    not contact_mesh_level_changed
                    and saved_metadata.get(
                        "mesh_signature", checkpoint_metadata["mesh_signature"]
                    ) == checkpoint_metadata["mesh_signature"]
                )
                same_holder_control = (
                    saved_holder_position_control == holder_position_control_mm
                )
                if (contact_sampling_changed
                        and not contact_method_changed
                        and same_contact_mesh
                        and same_holder_control
                        and saved_contact_multipliers.shape
                        == (saved_contact_row_count,)):
                    # Contact multipliers are facet-local generalized forces.
                    # Keep their support when the sampling layout changes
                    # within the same contact formulation. A point-collocation
                    # to P0 switch changes the gap function itself, so old
                    # point reactions are not a valid warm start; the saved
                    # geometry below reconstructs reactions on closed P0 rows.
                    old_equality_count = symmetry_equality_count + int(
                        saved_holder_position_control is not None
                    )
                    old_punch_points_per_facet = (
                        saved_mortar_subfacets
                        if saved_contact_method == "mortar-p0"
                        else saved_contact_collocation_count
                    )
                    old_die_points_per_facet = old_punch_points_per_facet
                    old_holder_points_per_facet = (
                        saved_mortar_subfacets
                        if saved_contact_method == "mortar-p0"
                        else saved_holder_collocation_count
                    )
                    punch_facet_count = punch_quadrature["facet_nodes"].shape[0]
                    die_facet_count = die_quadrature["facet_nodes"].shape[0]
                    holder_facet_count = holder_quadrature["facet_nodes"].shape[0]
                    new_punch_points_per_facet = (
                        punch_mortar.facet_count // punch_facet_count
                        if contact_method == "mortar-p0" else
                        punch["nodes"].shape[0] // punch_facet_count
                    )
                    new_die_points_per_facet = (
                        die_mortar.facet_count // die_facet_count
                        if contact_method == "mortar-p0" else
                        die["nodes"].shape[0] // die_facet_count
                    )
                    new_holder_points_per_facet = (
                        holder_mortar.facet_count // holder_facet_count
                        if contact_method == "mortar-p0" else
                        holder["nodes"].shape[0] // holder_facet_count
                    )
                    old_punch_count = (
                        punch_facet_count * old_punch_points_per_facet
                    )
                    old_die_count = die_facet_count * old_die_points_per_facet
                    old_holder_count = (
                        holder_facet_count * old_holder_points_per_facet
                    )
                    old_counts = (
                        old_equality_count + old_punch_count
                        + old_die_count + old_holder_count
                    )
                    new_layout_matches = (
                        new_punch_points_per_facet * punch_facet_count
                        == punch_rows.size
                        and new_die_points_per_facet * die_facet_count
                        == die_rows.size
                        and new_holder_points_per_facet * holder_facet_count
                        == holder_rows.size
                    )
                    if (old_counts == saved_contact_row_count
                            and new_layout_matches):
                        groups = (
                            (punch, old_equality_count, int(punch_rows[0]),
                             punch_facet_count, old_punch_points_per_facet,
                             new_punch_points_per_facet),
                            (die, old_equality_count + old_punch_count,
                             int(die_rows[0]), die_facet_count,
                             old_die_points_per_facet,
                             new_die_points_per_facet),
                            (holder, old_equality_count + old_punch_count
                             + old_die_count, int(holder_rows[0]),
                             holder_facet_count,
                             old_holder_points_per_facet,
                             new_holder_points_per_facet),
                        )
                        old_friction_compatible = (
                            saved_friction_multipliers.shape
                            == (saved_contact_row_count,)
                        )
                        def contact_sample_coordinates(method, count):
                            if method == "mortar-p0":
                                return -1. + (2. * np.arange(count) + 1.) / count
                            return np.polynomial.legendre.leggauss(count)[0]

                        for (tool, old_start, new_start, facet_count,
                             old_count, new_count) in groups:
                            old_points = contact_sample_coordinates(
                                saved_contact_method, old_count
                            )
                            new_points = contact_sample_coordinates(
                                contact_method, new_count
                            )
                            for facet in range(facet_count):
                                for old_point, coordinate in enumerate(old_points):
                                    new_point = int(np.argmin(
                                        np.abs(new_points - coordinate)
                                    ))
                                    old_row = (
                                        old_start + facet * old_count + old_point
                                    )
                                    new_row = (
                                        new_start + facet * new_count + new_point
                                    )
                                    committed_contact_multipliers[new_row] += (
                                        saved_contact_multipliers[old_row]
                                    )
                                    if old_friction_compatible:
                                        committed_friction_multipliers[new_row] += (
                                            saved_friction_multipliers[old_row]
                                        )
                        checkpoint_has_contact_multipliers = True
                else:
                    checkpoint_has_contact_multipliers = False
            elif (saved_holder_collocation_count
                    != holder_contact_collocation_points_per_facet
                    and saved_holder_position_control
                    == holder_position_control_mm
                    and saved_contact_multipliers.shape
                    == (saved_contact_row_count,)):
                holder_facet_count = (
                    holder["nodes"].shape[0]
                    // holder_contact_collocation_points_per_facet
                )
                old_holder_row_count = (
                    holder_facet_count * saved_holder_collocation_count
                )
                old_holder_row_start = (
                    saved_contact_row_count - old_holder_row_count
                )
                new_holder_row_start = int(holder_rows[0])
                if old_holder_row_start != new_holder_row_start:
                    raise ValueError(
                        "cannot remap holder contact rows: punch/die row "
                        "layout also changed"
                    )
                committed_contact_multipliers = np.zeros(row_count)
                committed_contact_multipliers[:new_holder_row_start] = (
                    saved_contact_multipliers[:old_holder_row_start]
                )
                old_gauss_points = np.polynomial.legendre.leggauss(
                    saved_holder_collocation_count
                )[0]
                new_gauss_points = np.polynomial.legendre.leggauss(
                    holder_contact_collocation_points_per_facet
                )[0]
                for facet in range(holder_facet_count):
                    for old_point, coordinate in enumerate(old_gauss_points):
                        new_point = int(np.argmin(
                            np.abs(new_gauss_points - coordinate)
                        ))
                        old_row = (
                            old_holder_row_start
                            + facet * saved_holder_collocation_count
                            + old_point
                        )
                        new_row = (
                            new_holder_row_start
                            + facet * holder_contact_collocation_points_per_facet
                            + new_point
                        )
                        committed_contact_multipliers[new_row] += (
                            saved_contact_multipliers[old_row]
                        )
                committed_friction_multipliers = np.zeros(row_count)
                if saved_friction_multipliers.shape == (
                    saved_contact_row_count,
                ):
                    committed_friction_multipliers[:new_holder_row_start] = (
                        saved_friction_multipliers[:old_holder_row_start]
                    )
                    for facet in range(holder_facet_count):
                        for old_point, coordinate in enumerate(old_gauss_points):
                            new_point = int(np.argmin(
                                np.abs(new_gauss_points - coordinate)
                            ))
                            old_row = (
                                old_holder_row_start
                                + facet * saved_holder_collocation_count
                                + old_point
                            )
                            new_row = (
                                new_holder_row_start
                                + facet
                                * holder_contact_collocation_points_per_facet
                                + new_point
                            )
                            committed_friction_multipliers[new_row] += (
                                saved_friction_multipliers[old_row]
                            )
                checkpoint_has_contact_multipliers = True
            elif (contact_layout_changed
                    and saved_holder_collocation_count
                    == holder_contact_collocation_points_per_facet
                    and saved_contact_row_count == row_count - 1
                    and saved_holder_position_control is None
                    and holder_position_control_mm is not None
                    and saved_contact_multipliers.shape == (row_count - 1,)):
                committed_contact_multipliers = np.insert(
                    saved_contact_multipliers,
                    symmetry_equality_count, 0.,
                )
                if saved_friction_multipliers.shape == (row_count - 1,):
                    committed_friction_multipliers = np.insert(
                        saved_friction_multipliers,
                        symmetry_equality_count, 0.,
                    )
                checkpoint_has_contact_multipliers = True
            elif (not contact_layout_changed
                    and saved_contact_multipliers.shape == (row_count,)):
                committed_contact_multipliers = saved_contact_multipliers.copy()
                if saved_friction_multipliers.shape == (row_count,):
                    committed_friction_multipliers = (
                        saved_friction_multipliers.copy()
                    )
                checkpoint_has_contact_multipliers = True
            elif (contact_layout_changed
                    and saved_contact_row_count == row_count + 1
                    and saved_holder_collocation_count
                    == holder_contact_collocation_points_per_facet
                    and saved_holder_position_control is not None
                    and holder_position_control_mm is None
                    and saved_contact_multipliers.shape == (row_count + 1,)):
                committed_contact_multipliers = np.delete(
                    saved_contact_multipliers, symmetry_equality_count
                )
                if saved_friction_multipliers.shape == (row_count + 1,):
                    committed_friction_multipliers = np.delete(
                        saved_friction_multipliers, symmetry_equality_count
                    )
                checkpoint_has_contact_multipliers = True
            else:
                committed_contact_multipliers = np.zeros(row_count)
                committed_friction_multipliers = np.zeros(row_count)
            if (checkpoint_holder_position_changed
                    or checkpoint_support_seed_changed):
                # A changed holder actuator setting changes both the contact
                # geometry and the generalized vertical load. A changed
                # support-seed strategy also changes which unilateral rows
                # are mandatory. Rebuild the initial reactions from the
                # restarted state instead of carrying incompatible multipliers.
                committed_contact_multipliers.fill(0.)
                committed_friction_multipliers.fill(0.)
                checkpoint_has_contact_multipliers = False
            elif (checkpoint_has_contact_multipliers
                    and saved_holder_position_control is None
                    and holder_position_control_mm is not None):
                # Converting a force-controlled checkpoint to position control
                # adds one equality multiplier to the holder's generalized
                # force balance. Seed it with the removed actuator load so the
                # first KKT correction starts at the same physical state.
                committed_contact_multipliers[
                    holder_position_equality_row
                ] = holder_force_per_width
            for key in (
                "springback_active",
                "springback_fraction",
                "springback_increment",
                "springback_formed_angle_deg",
                "springback_formed_equivalent_load",
                "springback_formed_displacement",
                "springback_formed_committed",
                "springback_formed_holder_z",
                "springback_formed_friction_multipliers",
                "springback_final_stage_iterations",
                "springback_unload_fractions_json",
                "punch_withdrawal_active",
                "punch_withdrawal_target_stroke_mm",
            ):
                if key in saved.files:
                    checkpoint_extra[key] = np.asarray(saved[key])
            springback_checkpoint_active = bool(
                int(checkpoint_extra.get("springback_active", np.asarray(0)).item())
            )
            if resume_forming_from_springback:
                if not springback_checkpoint_active:
                    raise ValueError(
                        "--resume-forming-from-springback requires a springback checkpoint"
                    )
                required_formed_state = (
                    "springback_formed_displacement",
                    "springback_formed_committed",
                    "springback_formed_holder_z",
                )
                missing_formed_state = [
                    key for key in required_formed_state if key not in saved.files
                ]
                if missing_formed_state:
                    raise ValueError(
                        "springback checkpoint predates formed-state snapshots; "
                        "rerun forming to the target stroke before springback. "
                        f"Missing: {missing_formed_state}"
                    )
                displacement = np.asarray(
                    saved["springback_formed_displacement"], dtype=float
                )
                committed = np.asarray(
                    saved["springback_formed_committed"], dtype=float
                )
                holder_z = float(saved["springback_formed_holder_z"])
                if "springback_formed_friction_multipliers" in saved.files:
                    committed_friction_multipliers = np.asarray(
                        saved["springback_formed_friction_multipliers"], dtype=float
                    )
                else:
                    committed_friction_multipliers.fill(0.)
                for key in (
                    "springback_active",
                    "springback_fraction",
                    "springback_increment",
                    "springback_formed_angle_deg",
                    "springback_formed_equivalent_load",
                    "springback_formed_displacement",
                    "springback_formed_committed",
                    "springback_formed_holder_z",
                    "springback_formed_friction_multipliers",
                    "springback_final_stage_iterations",
                    "springback_unload_fractions_json",
                ):
                    checkpoint_extra.pop(key, None)
                # Resuming the saved formed state starts a new forming and
                # unloading cycle.  Do not carry the prior partial-stroke
                # withdrawal marker forward, or a later full-stroke run with
                # ``--withdraw-punch-after-forming`` will incorrectly think
                # its unloading phase has already completed.
                checkpoint_extra.pop("punch_withdrawal_active", None)
                checkpoint_extra.pop("punch_withdrawal_target_stroke_mm", None)
                # The checkpoint's live multipliers belong to the fully
                # withdrawn state, while displacement and plastic history
                # now belong to the formed state. Rebuild contact reactions
                # from that geometry instead of mixing the two states.
                committed_contact_multipliers.fill(0.)
                committed_friction_multipliers.fill(0.)
                checkpoint_has_contact_multipliers = False

                # Withdrawal cutbacks may leave a tiny suggested increment
                # in the checkpoint. Restart from the last accepted forming
                # increment at the saved formed stroke; subsequent Newton,
                # plastic-increment, and force-prediction controls still cut
                # back as needed.
                formed_stroke = float(checkpoint_extra.get(
                    "punch_withdrawal_target_stroke_mm", accepted_stroke
                ))
                formed_record_index = next((index for index, record in enumerate(records)
                                            if record.get("stroke_mm", -np.inf)
                                            >= formed_stroke - 1e-10), None)
                if formed_record_index is not None and formed_record_index > 0:
                    forming_increment = abs(
                        float(records[formed_record_index]["stroke_mm"])
                        - float(records[formed_record_index - 1]["stroke_mm"])
                    )
                    if forming_increment > 0.:
                        suggested_increment_mm = min(
                            maximum_increment_mm,
                            max(forming_increment_floor_mm, forming_increment),
                        )
            if "previous_displacement" in saved.files:
                saved_previous_displacement = np.asarray(
                    saved["previous_displacement"], dtype=float
                )
                saved_previous_holder_z = float(saved["previous_holder_z"])
                saved_previous_stroke = float(saved["previous_stroke"])
                if (saved_previous_displacement.shape == (basis.N,)
                        and np.all(np.isfinite(saved_previous_displacement))
                        and np.isfinite(saved_previous_holder_z)
                        and np.isfinite(saved_previous_stroke)
                        and saved_previous_stroke < accepted_stroke):
                    previous_displacement = saved_previous_displacement.copy()
                    previous_holder_z = saved_previous_holder_z
                    previous_stroke = saved_previous_stroke
        if (not resume_forming_from_springback
                and springback_checkpoint_active
                and maximum_stroke_mm > accepted_stroke + 1e-10):
            raise ValueError(
                "checkpoint contains a partially unloaded state; use "
                "--resume-forming-from-springback to restore its formed state "
                "before increasing punch stroke"
            )
        expected_state_shape = assembler.initial_state().shape
        if displacement.shape != (basis.N,):
            raise ValueError("checkpoint state arrays do not match the current finite-element model")
        if (not np.isfinite(suggested_increment_mm)
                or suggested_increment_mm <= 0.
                or suggested_increment_mm > maximum_increment_mm * (1. + 1e-12)):
            raise ValueError("checkpoint has an invalid suggested stroke increment")
        # An unresolved reaction jump marks a branch transition.  Repeatedly
        # shrinking toward the previous branch's limit point can trap the
        # continuation at machine-scale stroke increments; resume beyond the
        # same contact transition using the configured force-resolution floor.
        if (suggested_increment_mm < forming_increment_floor_mm
                or any(record.get("force_resolution_unresolved", False)
                       or record.get("holder_jump_unresolved", False)
                       for record in records[-3:])):
            suggested_increment_mm = max(
                suggested_increment_mm, forming_increment_floor_mm
            )
        if committed.shape != expected_state_shape:
            if (expected_state_shape[1] == 1
                    and committed.shape == (
                        expected_state_shape[0], saved_quadrature_count,
                        expected_state_shape[2],
                    )
                    and saved_quadrature_count > 1
                    and np.allclose(
                        committed,
                        np.broadcast_to(committed[:, :1], committed.shape),
                        rtol=1e-12, atol=1e-14,
                    )):
                # A TriP1 strain/stress/history field is elementwise constant;
                # collapsing identical quadrature histories is lossless.
                committed = committed[:, :1].copy()
            else:
                raise ValueError(
                    "checkpoint history shape does not match the current "
                    "finite-element quadrature"
                )
        if (committed_contact_multipliers.shape != (row_count,)
                or not np.all(np.isfinite(committed_contact_multipliers))):
            raise ValueError(
                "checkpoint contact multipliers are incompatible with the "
                f"current contact rows: saved={committed_contact_multipliers.shape}, "
                f"expected={(row_count,)}, "
                f"all_finite={np.all(np.isfinite(committed_contact_multipliers))}"
            )
        if (committed_friction_multipliers.shape != (row_count,)
                or not np.all(np.isfinite(committed_friction_multipliers))):
            raise ValueError(
                "checkpoint friction multipliers are incompatible with the "
                f"current contact rows: saved={committed_friction_multipliers.shape}, "
                f"expected={(row_count,)}, "
                f"all_finite={np.all(np.isfinite(committed_friction_multipliers))}"
            )

    def save_checkpoint():
        if checkpoint is None:
            return
        # A rejected trial may change the active contact layout without
        # changing the accepted displacement/history. Do not relabel that
        # old state with a new row signature or write remapped multipliers
        # until a state has converged with the current layout.
        if resume and not checkpoint_state_current_layout:
            return
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_arrays = {
            "metadata_json": np.asarray(json.dumps(checkpoint_metadata, sort_keys=True)),
            "displacement": displacement,
            "committed": committed,
            "holder_z": np.asarray(holder_z),
            "accepted_stroke": np.asarray(accepted_stroke),
            "suggested_increment_mm": np.asarray(suggested_increment_mm),
            "regularization_floor": np.asarray(regularization_floor[0]),
            "contact_multipliers": committed_contact_multipliers,
            "friction_multipliers": committed_friction_multipliers,
            "records_json": np.asarray(json.dumps(records)),
            "failure_log_json": np.asarray(json.dumps(failure_log)),
        }
        checkpoint_arrays.update(checkpoint_extra)
        if previous_displacement is not None:
            checkpoint_arrays.update({
                "previous_displacement": previous_displacement,
                "previous_holder_z": np.asarray(previous_holder_z),
                "previous_stroke": np.asarray(previous_stroke),
            })
        temporary_fd, temporary_name = tempfile.mkstemp(
            prefix=checkpoint.name + ".", suffix=".tmp", dir=checkpoint.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(temporary_fd, "wb") as stream:
                np.savez_compressed(stream, **checkpoint_arrays)
            os.replace(temporary, checkpoint)
        finally:
            temporary.unlink(missing_ok=True)

    initial_regularization_floor = float(os.environ.get(
        "SCIKIT_FVM_FORMING_INITIAL_REGULARIZATION_FLOOR", "0"
    ))
    if (not np.isfinite(initial_regularization_floor)
            or initial_regularization_floor < 0.):
        raise ValueError(
            "SCIKIT_FVM_FORMING_INITIAL_REGULARIZATION_FLOOR must be finite "
            "and nonnegative"
        )
    regularization_floor = [max(
        saved_regularization_floor, initial_regularization_floor
    )]

    if resume and not checkpoint_has_contact_multipliers:
        # Older checkpoints did not persist the contact reactions.  Recover a
        # bounded least-squares equilibrium estimate on the currently closed
        # rows so the first restarted Newton line search starts from the
        # accepted state instead of treating the old reaction as zero.
        saved_gaps, saved_constraints, _ = tool_state(
            displacement, holder_z, accepted_stroke
        )
        saved_active = np.r_[
            np.ones(equality_count, dtype=bool), saved_gaps <= 1e-6
        ]
        holder_vertical_sensitivity = saved_constraints[holder_rows, -1]
        supportable_holder_rows = holder_rows[
            holder_vertical_sensitivity > 1e-8
        ]
        if supportable_holder_rows.size == 0:
            raise RuntimeError(
                "cannot reconstruct the blank-holder reaction: the current "
                "contact samples have no vertically supporting normal"
            )
        active_holder_supports = supportable_holder_rows[
            saved_active[supportable_holder_rows]
        ]
        if active_holder_supports.size:
            holder_row = int(active_holder_supports[np.argmax(
                saved_constraints[active_holder_supports, -1]
            )])
        else:
            support_gaps = saved_gaps[
                supportable_holder_rows - equality_count
            ]
            holder_row = int(supportable_holder_rows[np.argmin(support_gaps)])
            # Never invent a compressive reaction on an open row just to
            # balance the load-controlled holder. Reconstruct only reactions
            # supported by contact at the saved geometry; Newton must find a
            # real contact transition from an unsupported state.
            if support_gaps.min() <= max(
                    forming_contact_gap_tolerance_mm, 1e-8):
                saved_active[holder_row] = True
        selected_rows = np.flatnonzero(saved_active)
        bounds_lower = np.full(selected_rows.size, -np.inf)
        bounds_upper = np.full(selected_rows.size, np.inf)
        for position, row in enumerate(selected_rows):
            if unilateral[row]:
                bounds_upper[position] = 0.
        saved_evaluation = assembler.evaluate(
            displacement, committed, tangent=False
        )
        saved_gradient = np.r_[
            saved_evaluation.residual, holder_external_force_per_width
        ]
        reaction_fit = lsq_linear(
            saved_constraints[selected_rows].T,
            -saved_gradient,
            bounds=(bounds_lower, bounds_upper),
            tol=1e-12,
            max_iter=500,
        )
        committed_contact_multipliers[selected_rows] = reaction_fit.x

    def solve_contact_linearization(
        evaluation, gradient, constraints, rhs, initial_active, required_active,
        holder_seed_row, tangent=None, support_switch_depth=0,
        holder_support_visited_rows=None, die_support_visited_rows=None,
        die_support_released_rows=None, punch_support_visited_rows=None,
        punch_support_released_rows=None, initial_multipliers=None,
        displacement=None, current_stroke: float = 0.0,
    ):
        soft_die_support_seed = os.environ.get(
            "SCIKIT_FVM_FORMING_SOFT_DIE_SUPPORT_SEED", ""
        ).lower() in {"1", "true", "yes"}
        if holder_support_visited_rows is None:
            holder_support_visited_rows = set()
            if holder_seed_row is not None:
                holder_support_visited_rows.add(int(holder_seed_row))
        if die_support_visited_rows is None:
            die_support_visited_rows = set(
                int(row) for row in die_rows[required_active[die_rows]]
            )
        if die_support_released_rows is None:
            die_support_released_rows = set()
        if punch_support_visited_rows is None:
            punch_support_visited_rows = set(
                int(row) for row in punch_rows[required_active[punch_rows]]
            )
        if punch_support_released_rows is None:
            punch_support_released_rows = set()
        if tangent is None:
            tangent = sparse.block_diag(
                (evaluation.tangent, sparse.csr_matrix((1, 1))), format="csr"
            )
        diagonal_scale = max(
            float(np.max(np.abs(tangent.diagonal()), initial=0.)), 1.
        )
        regularization_factors = (
            0., 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-6,
            1e-4, 1e-2, 1e-1, 1., 10., 100., 1000.
        )
        contact_backend = os.environ.get(
            "SCIKIT_FVM_CONTACT_BACKEND", "auto"
        ).lower()
        if contact_backend not in {"auto", "kktkit", "sparse"}:
            raise ValueError(
                "SCIKIT_FVM_CONTACT_BACKEND must be auto, kktkit, or sparse"
            )
        failures = []
        last_contact_result = None

        def support_prediction_step(contact_result):
            """Match switching to the bounded primal Newton trial."""
            step = np.asarray(contact_result.primal, dtype=float).copy()
            displacement_scale = min(
                1.0,
                displacement_newton_step_limit_mm
                / max(
                    float(np.linalg.norm(step[:-1], ord=np.inf)), 1e-30
                ),
            )
            if displacement_scale < 1.0:
                step *= displacement_scale
            if holder_position_control_mm is None:
                scale = min(
                    1.0,
                    holder_newton_step_limit_mm
                    / max(abs(float(step[-1])), 1e-30),
                )
                if scale < 1.0:
                    step *= scale
            return step

        def switch_holder_support(contact_result):
            """Move the unilateral holder seed when its contact opens or pulls."""
            if contact_semismooth_first:
                # The FB solve treats every holder row as a unilateral
                # complementarity pair, so pinning or transferring one seed
                # would reintroduce an artificial reaction path.
                return None
            if holder_seed_row is None:
                return None
            old_row = int(holder_seed_row)
            if old_row not in holder_rows:
                return None
            predicted_step = support_prediction_step(contact_result)
            predicted_gaps = np.asarray(
                constraints @ predicted_step - rhs, dtype=float
            )
            old_multiplier = float(contact_result.multipliers[old_row])
            old_gap = float(predicted_gaps[old_row])
            holder_sensitivity = constraints[holder_rows, -1]
            # Permit a replacement row with a shallower but still
            # force-opposing normal. Near the holder return the local normal
            # can rotate below |n_z|=0.25 while a neighboring FE edge remains
            # the only available support; the unilateral solve will reject a
            # tensile or ill-conditioned candidate.
            supportable = holder_sensitivity > 0.1
            current_gaps = -rhs
            # A support transfer preserves the holder force path, so its new
            # row must represent physical contact at the current iterate.
            # The configurable search radius is a ceiling for that local
            # search, not permission to pin an open point several sheet
            # thicknesses away: doing so creates an artificial holder force
            # path and can drive the KKT multipliers to enormous values.
            support_gap_limit = min(
                holder_support_search_radius_mm,
                forming_contact_gap_tolerance_mm,
            )
            candidates = holder_rows[
                supportable
                & (holder_rows != old_row)
                & ~np.isin(holder_rows, tuple(holder_support_visited_rows))
                & (current_gaps[holder_rows] <= support_gap_limit)
            ]
            # Keep distinct collocation points on the same FE edge eligible:
            # along a curved tool they can have materially different normals
            # and provide the only nearby replacement support. The sparse KKT
            # solver rank-filters dependent active rows before factorization.
            positive_candidate_gaps = current_gaps[candidates]
            positive_candidate_gaps = positive_candidate_gaps[
                positive_candidate_gaps > forming_contact_gap_tolerance_mm
            ]
            support_spacing = float(np.min(
                positive_candidate_gaps, initial=0.05 * thickness
            ))
            support_switch_gap_tolerance = max(
                forming_contact_gap_tolerance_mm, 0.2 * support_spacing
            )
            support_row_open = (
                old_gap > support_switch_gap_tolerance
            )
            # A tensile reaction makes a required unilateral row infeasible
            # even when curvature linearization predicts a tiny penetration.
            # Transfer that force path to another local sample.
            support_reaction_tensile = (
                old_multiplier > holder_seed_tensile_release_threshold
            )
            if os.environ.get("SCIKIT_FVM_FORMING_TRACE", "") == "contacts":
                print(
                    "[forming-support-switch-check] "
                    f"depth={support_switch_depth} seed_row={old_row} "
                    f"lambda={old_multiplier:.6e} "
                    f"predicted_gap={old_gap:.6e} mm "
                    f"active={bool(contact_result.active_rows[old_row])} "
                    f"contact_gap_tol={forming_contact_gap_tolerance_mm:.3e} mm "
                    f"support_switch_gap_tol="
                    f"{support_switch_gap_tolerance:.3e} mm "
                    f"support_spacing={support_spacing:.3e} mm",
                    flush=True,
                )
            if support_switch_depth >= 8:
                return None
            if not (support_row_open or support_reaction_tensile):
                return None
            # Nearly duplicate collocation rows can carry huge opposing
            # reactions even though the applied holder load is modest. Keep
            # such singular pivots out of both the active and geometric
            # transfer choices; the full unilateral solve still sees them.
            reaction_limit = max(20. * holder_force_per_width, 100.)
            plausible_candidates = candidates[
                contact_result.multipliers[candidates] >= -reaction_limit
            ]
            active_compressive = plausible_candidates[
                contact_result.active_rows[plausible_candidates]
                & (contact_result.multipliers[plausible_candidates] < -1e-2)
            ]
            if active_compressive.size:
                new_row = int(active_compressive[
                    np.argmin(current_gaps[active_compressive])
                ])
            else:
                # A newly needed support is usually inactive in the old
                # linearization, so requiring an already-compressive reaction
                # here can release the only force path.  Once that happens the
                # holder's force-controlled DOF has no restoring stiffness and
                # Newton can drift far from every contact row.  Transfer the
                # required support to the closest geometrically admissible row
                # instead; its linearized gap equation will close the gap.
                geometric_candidates = plausible_candidates
                if geometric_candidates.size:
                    new_row = int(geometric_candidates[
                        np.argmin(current_gaps[geometric_candidates])
                    ])
                    nearest = geometric_candidates[np.argsort(
                        current_gaps[geometric_candidates]
                    )[:8]]
                    candidate_diagnostics = [
                        (
                            int(row),
                            float(current_gaps[row]),
                            float(predicted_gaps[row]),
                            float(contact_result.multipliers[row]),
                            bool(contact_result.active_rows[row]),
                            float(constraints[row, -1]),
                        )
                        for row in nearest
                    ]
                    if (os.environ.get("SCIKIT_FVM_FORMING_TRACE", "")
                            == "contacts"):
                        print(
                            "[forming-support-transfer-candidates] "
                            f"old_row={old_row} selected_row={new_row} "
                            f"selected_gap={current_gaps[new_row]:.6e} mm "
                            f"(row,current_gap,predicted_gap,lambda,active,sensitivity)="
                            f"{candidate_diagnostics}",
                            flush=True,
                        )
                else:
                    # No nearby supportable unilateral row exists in this
                    # contact layout. Release the current seed and let the
                    # caller's regularized path move toward the next contact.
                    if (os.environ.get("SCIKIT_FVM_FORMING_TRACE", "")
                            == "contacts"):
                        nearest = holder_rows[
                            supportable
                            & (holder_rows != old_row)
                            & ~np.isin(holder_rows, tuple(holder_support_visited_rows))
                        ]
                        nearest = nearest[np.argsort(
                            current_gaps[nearest]
                        )[:8]]
                        print(
                            "[forming-support-no-local-candidate] "
                            f"old_row={old_row} "
                            "(row,current_gap,predicted_gap,lambda,active,sensitivity)="
                            f"{[(int(row), float(current_gaps[row]), float(predicted_gaps[row]), float(contact_result.multipliers[row]), bool(contact_result.active_rows[row]), float(constraints[row, -1])) for row in nearest]}",
                            flush=True,
                        )
                    initial_active[old_row] = False
                    required_active[old_row] = False
                    print(
                        "[forming-contact] release unsupported holder seed "
                        f"row={old_row} (lambda={old_multiplier:.3e}, "
                        f"predicted_gap={old_gap:.3e} mm)",
                        flush=True,
                    )
                    return -1
            if new_row == old_row:
                return None
            initial_active[old_row] = False
            initial_active[new_row] = True
            required_active[old_row] = False
            required_active[new_row] = not soft_holder_support_seed
            holder_support_visited_rows.add(int(new_row))
            print(
                "[forming-contact] transfer holder support "
                f"row={old_row} (lambda={old_multiplier:.3e}, gap={old_gap:.3e} mm) "
                f"-> row={new_row} "
                f"(lambda={contact_result.multipliers[new_row]:.3e})",
                flush=True,
            )
            return new_row

        def switch_die_support(contact_result):
            """Transfer a tensile or open-gap required die reaction."""
            if contact_semismooth_first:
                # The semismooth complementarity solve updates all die rows
                # together.  A required-row transfer would turn its unilateral
                # equations back into a pinned active-set support path.
                return None
            required_die_rows = die_rows[required_active[die_rows]]
            if (required_die_rows.size == 0 or support_switch_depth >= 8):
                return None
            old_row = int(required_die_rows[0])
            old_multiplier = float(contact_result.multipliers[old_row])
            current_gaps = -rhs
            old_gap = float(current_gaps[old_row])
            tensile_seed = old_multiplier > 1e-2
            open_compressive_seed = (
                old_gap > forming_contact_gap_tolerance_mm
                and old_multiplier < -1e-2
            )
            if not (tensile_seed or open_compressive_seed):
                return None
            die_vertical_sensitivity = np.asarray(
                constraints[die_rows][:, dof_z].sum(axis=1)
            ).reshape(-1)
            supportable = die_vertical_sensitivity > 0.25
            # A replacement row must already be geometrically close to the
            # die.  A broad thickness-scaled neighborhood can promote a
            # compressive multiplier on an open row; the next active-set
            # iteration then releases that seed and transfers it back again.
            # Keep support migration inside the same gap scale used by the
            # unilateral feasibility check.
            gap_limit = min(
                die_support_gap_limit_mm,
                forming_contact_gap_tolerance_mm,
            )
            candidates = die_rows[
                supportable
                & (die_rows != old_row)
                & ~np.isin(die_rows, tuple(die_support_visited_rows))
                & (current_gaps[die_rows] <= gap_limit)
            ]
            active_compressive = candidates[
                contact_result.active_rows[candidates]
                & (contact_result.multipliers[candidates] < -1e-2)
            ]
            # A die seed only preserves a reaction path while another nearby
            # row is already carrying a compressive contact force.  Promoting
            # an open row merely because its geometric gap is the smallest
            # among many candidates can pin the die to a distant point.  The
            # resulting multiplier is unbounded for a nonsymmetric tangent
            # and repeated transfers walk across the entire die profile.
            # With no active compressive replacement, release this seed and
            # let the unilateral solve choose contact from the actual gaps.
            replacement = (
                int(active_compressive[
                    np.argmin(current_gaps[active_compressive])
                ])
                if active_compressive.size else -1
            )
            initial_active[old_row] = False
            required_active[old_row] = False
            if replacement >= 0:
                initial_active[replacement] = True
                required_active[replacement] = not soft_die_support_seed
                die_support_visited_rows.add(int(replacement))
                print(
                    "[forming-contact] transfer die support "
                    f"row={old_row} (gap={old_gap:.3e} mm, "
                    f"lambda={old_multiplier:.3e}) "
                    f"-> row={replacement} "
                    f"(gap={current_gaps[replacement]:.3e} mm, "
                    f"lambda={contact_result.multipliers[replacement]:.3e})",
                    flush=True,
                )
            else:
                # Keep an unsupported seed out of the force-path reseeding
                # performed at the next Newton iterate.  Otherwise the
                # outer loop selects this same nearby row again and the
                # active-set solve repeats the release indefinitely.
                die_support_released_rows.add(old_row)
                print(
                    "[forming-contact] release unsupported die seed "
                    f"row={old_row} (gap={old_gap:.3e} mm, "
                    f"lambda={old_multiplier:.3e}); "
                    f"no supportable row within {gap_limit:.3e} mm",
                    flush=True,
                )
            return replacement

        def switch_punch_support(
            contact_result, current_displacement, current_stroke,
        ):
            """Transfer a tensile punch seed to a nearby closed punch sample."""
            required_punch_rows = punch_rows[required_active[punch_rows]]
            if required_punch_rows.size == 0 or support_switch_depth >= 8:
                return None
            if current_displacement is None:
                raise ValueError(
                    "punch support transfer requires the current displacement"
                )
            old_row = int(required_punch_rows[0])
            old_multiplier = float(contact_result.multipliers[old_row])
            if old_multiplier <= 1e-2:
                return None
            current_gaps = -rhs
            old_sample = int(np.flatnonzero(punch_rows == old_row)[0])
            if contact_method == "mortar-p0":
                punch_eval = punch_mortar.evaluate(
                    # The punch row geometry is independent of the holder
                    # scalar. The mortar operator still expects the full
                    # mixed variable vector, so append a neutral value.
                    np.r_[current_displacement, 0.],
                    tools["punch"], translation=[0., -float(current_stroke)],
                    curvature_blend_length=(
                        punch_contact_curvature_blend_length_mm
                    ),
                )
                punch_points = punch_eval.facet_points
            else:
                punch_points = sample_coordinates(punch, current_displacement)
            if punch_points.shape[0] != punch_rows.size:
                raise RuntimeError(
                    "punch support geometry does not match the active contact "
                    f"rows: points={punch_points.shape[0]}, rows={punch_rows.size}"
                )
            point_distances = np.linalg.norm(
                punch_points - punch_points[old_sample], axis=1
            )
            point_distances[old_sample] = np.inf
            nearest_sample_spacing = float(np.min(point_distances))
            local_distance_limit = max(
                3.0 * nearest_sample_spacing,
                10. * forming_contact_gap_tolerance_mm,
            )
            local_gap_limit = max(
                0.25 * thickness,
                10. * forming_contact_gap_tolerance_mm,
            )
            local_neighbor_mask = (
                point_distances <= local_distance_limit
            )
            local_candidates = punch_rows[
                (punch_rows != old_row)
                & ~np.isin(punch_rows, tuple(punch_support_released_rows))
                & local_neighbor_mask
                & (current_gaps[punch_rows] <= local_gap_limit)
            ]
            # A row visited earlier in this support-transfer chain can still
            # be the only valid force path after the active-set solve.  Prefer
            # an actual compressive reaction even when that row was visited;
            # ``visited`` only prevents geometric reseeding loops.
            active_compressive = local_candidates[
                contact_result.active_rows[local_candidates]
                & (contact_result.multipliers[local_candidates] < -1e-2)
            ]
            if active_compressive.size:
                replacement = int(active_compressive[
                    np.argmin(current_gaps[active_compressive])
                ])
                replacement_mode = "compressive"
            else:
                # At a support transition the nearest geometrically closed
                # punch row is often inactive in this linearization. Requiring
                # an already-compressive replacement can therefore release
                # the only punch reaction path for one Newton iteration; the
                # blank then jumps until contact is found again, producing a
                # force spike and a succession of microscopic stroke steps.
                # Keep the seed on the nearest local row and let the
                # unilateral KKT solve determine whether it carries reaction.
                geometric_candidates = local_candidates[
                    ~np.isin(
                        local_candidates,
                        tuple(punch_support_visited_rows),
                    )
                ]
                plausible_candidates = geometric_candidates[
                    contact_result.multipliers[geometric_candidates]
                    >= -max(20. * holder_force_per_width, 100.)
                ]
                # Mortar P0 has one unilateral row per subfacet.  When the
                # active minimum-gap location moves across a subfacet
                # boundary, the neighboring row can still be slightly open
                # at the current Newton iterate even though it is the right
                # local support candidate.  Restricting this handoff to 0.05
                # thickness drops the only punch force path and lets the
                # blank jump away before the unilateral solve can close the
                # neighboring row.  Let the KKT solve decide whether the
                # candidate can carry compression; its gap allowance remains
                # bounded by the local search band.
                geometric_gap_limit = (
                    local_gap_limit if contact_method == "mortar-p0" else max(
                        0.05 * thickness,
                        10. * forming_contact_gap_tolerance_mm,
                    )
                )
                plausible_candidates = plausible_candidates[
                    current_gaps[plausible_candidates]
                    <= geometric_gap_limit
                ]
                replacement = (
                    int(plausible_candidates[
                        np.argmin(current_gaps[plausible_candidates])
                    ])
                    if plausible_candidates.size else -1
                )
                replacement_mode = "geometric"
            initial_active[old_row] = False
            required_active[old_row] = False
            punch_support_visited_rows.add(old_row)
            if replacement >= 0:
                initial_active[replacement] = True
                required_active[replacement] = not soft_punch_support_seed
                punch_support_visited_rows.add(replacement)
                print(
                    "[forming-contact] transfer punch seed "
                    f"row={old_row} (lambda={old_multiplier:.3e}, "
                    f"gap={current_gaps[old_row]:.3e} mm) "
                    f"-> row={replacement} "
                    f"(gap={current_gaps[replacement]:.3e} mm, "
                    f"lambda={contact_result.multipliers[replacement]:.3e}, "
                    f"mode={replacement_mode})",
                    flush=True,
                )
            else:
                # Do not seed the same unsupported punch sample again on the
                # next Newton iterate of this prescribed stroke increment.
                # The visited set below is local to one linearization so that
                # viable rows can be reconsidered after the blank moves; a
                # row with no nearby compressive replacement is different: it
                # has already failed as a reaction path for this increment.
                punch_support_released_rows.add(old_row)
                nearest_probe_rows = np.argsort(point_distances)[:min(
                    5, point_distances.size
                )]
                nearest_probe = [
                    (
                        int(punch_rows[index]),
                        float(point_distances[index]),
                        float(current_gaps[punch_rows[index]]),
                        float(contact_result.multipliers[punch_rows[index]]),
                        bool(contact_result.active_rows[punch_rows[index]]),
                        bool(punch_rows[index] in punch_support_visited_rows),
                        bool(punch_rows[index] in punch_support_released_rows),
                    )
                    for index in nearest_probe_rows
                ]
                print(
                    "[forming-contact] release unsupported punch seed "
                    f"row={old_row} (lambda={old_multiplier:.3e}, "
                    f"gap={current_gaps[old_row]:.3e} mm); "
                    f"no local punch sample within {local_distance_limit:.3e} mm "
                    "carried a plausible compression; nearest "
                    "(row,distance,gap,lambda,active,visited,released)="
                    f"{nearest_probe}",
                    flush=True,
                )
            return replacement

        def retry_with_holder_support(new_row, solver_label, previous_result):
            if new_row < 0:
                regularization_floor[0] = max(regularization_floor[0], 1e-6)
                next_seed_row = None
            else:
                next_seed_row = new_row
            contact_result, next_label, downstream_row = solve_contact_linearization(
                evaluation, gradient, constraints, rhs, initial_active,
                required_active, next_seed_row, tangent=tangent,
                support_switch_depth=support_switch_depth + 1,
                holder_support_visited_rows=holder_support_visited_rows,
                die_support_visited_rows=die_support_visited_rows,
                die_support_released_rows=die_support_released_rows,
                punch_support_visited_rows=punch_support_visited_rows,
                punch_support_released_rows=punch_support_released_rows,
                initial_multipliers=previous_result.multipliers,
                displacement=displacement,
                current_stroke=current_stroke,
            )
            return (
                contact_result,
                f"{next_label}; support-transfer-from-{solver_label}",
                downstream_row if downstream_row is not None else new_row,
            )

        def retry_with_die_support(new_row, solver_label, previous_result):
            contact_result, next_label, downstream_row = solve_contact_linearization(
                evaluation, gradient, constraints, rhs, initial_active,
                required_active, holder_seed_row, tangent=tangent,
                support_switch_depth=support_switch_depth + 1,
                holder_support_visited_rows=holder_support_visited_rows,
                die_support_visited_rows=die_support_visited_rows,
                die_support_released_rows=die_support_released_rows,
                punch_support_visited_rows=punch_support_visited_rows,
                punch_support_released_rows=punch_support_released_rows,
                initial_multipliers=previous_result.multipliers,
                displacement=displacement,
                current_stroke=current_stroke,
            )
            return (
                contact_result,
                f"{next_label}; die-support-transfer-from-{solver_label}",
                downstream_row,
            )

        def retry_with_punch_support(new_row, solver_label, previous_result):
            contact_result, next_label, downstream_row = solve_contact_linearization(
                evaluation, gradient, constraints, rhs, initial_active,
                required_active, holder_seed_row, tangent=tangent,
                support_switch_depth=support_switch_depth + 1,
                holder_support_visited_rows=holder_support_visited_rows,
                die_support_visited_rows=die_support_visited_rows,
                die_support_released_rows=die_support_released_rows,
                punch_support_visited_rows=punch_support_visited_rows,
                punch_support_released_rows=punch_support_released_rows,
                initial_multipliers=previous_result.multipliers,
                displacement=displacement,
                current_stroke=current_stroke,
            )
            return (
                contact_result,
                f"{next_label}; punch-support-transfer-from-{solver_label}",
                new_row,
            )

        for regularization_factor in regularization_factors:
            if regularization_factor < regularization_floor[0]:
                continue
            if regularization_factor == 0.:
                hessian = tangent
            else:
                # Regularize the deformable blank DOFs. Keep a force-controlled
                # holder free while a contact row supports its translation;
                # if support switching has temporarily left every holder row
                # inactive, regularize that scalar for this Newton correction
                # so the singular KKT system can move the tool toward contact.
                # The physical residual, gaps, and complementarity checks below
                # still decide acceptance, and the holder diagonal disappears
                # as soon as an active or required support row is available.
                holder_has_linearized_support = bool(np.any(
                    initial_active[holder_rows]
                    | required_active[holder_rows]
                ))
                regularize_holder_translation = (
                    holder_position_control_mm is None
                    and not holder_has_linearized_support
                )
                regularization_mask = np.r_[
                    np.ones(basis.N),
                    float(regularize_holder_translation),
                ]
                regularization = sparse.diags(
                    regularization_mask, format="csr"
                )
                hessian = tangent + (
                    regularization_factor * diagonal_scale * regularization
                )
            if finite_strain_j2:
                # The finite-strain J2 tangent is the Jacobian of the
                # nonlinear residual. Keep this path on the direct active-set
                # KKT solver even when a particular iterate happens to be
                # numerically symmetric: a convex-QP fallback can silently
                # shift that Jacobian and return a direction for a different
                # problem. KKTKit's convex contact optimizer is appropriate
                # for symmetric energy Hessians, not this Newton correction.
                contact_result = solve_sparse_contact_step(
                    hessian, gradient, constraints, rhs, unilateral,
                    initial_active_rows=initial_active,
                    required_active_rows=required_active,
                    tolerance=linear_contact_tolerance,
                    dual_tolerance=contact_dual_tolerance,
                    complementarity_tolerance=(
                        linear_contact_complementarity_tolerance
                    ),
                    max_iterations=contact_solver_max_iterations,
                    rank_tolerance=contact_rank_tolerance,
                    allow_nonsymmetric=True,
                    initial_multipliers=initial_multipliers,
                    semismooth_first=contact_semismooth_first,
                )
                replacement_punch = switch_punch_support(
                    contact_result, displacement, current_stroke
                )
                if replacement_punch is not None:
                    return retry_with_punch_support(
                        replacement_punch,
                        "sparse-nonsymmetric-active-set", contact_result,
                    )
                replacement = switch_holder_support(contact_result)
                if replacement is not None:
                    return retry_with_holder_support(
                        replacement, "sparse-nonsymmetric-active-set",
                        contact_result,
                    )
                replacement_die = switch_die_support(contact_result)
                if replacement_die is not None:
                    return retry_with_die_support(
                        replacement_die, "sparse-nonsymmetric-active-set",
                        contact_result,
                    )
                required_tension = float(np.max(
                    contact_result.multipliers[required_active], initial=0.
                ))
                if ((contact_result.converged or contact_result.inexact)
                        and required_tension <= 1e-2):
                    label = (
                        "sparse-nonsymmetric-semismooth-inexact"
                        if contact_result.inexact and not contact_result.converged
                        else "sparse-nonsymmetric-active-set"
                    )
                    if regularization_factor:
                        label += f"-regularized-{regularization_factor:.0e}"
                    return contact_result, label, None
                last_contact_result = contact_result
                failures.append(
                    f"sparse nonsymmetric(reg={regularization_factor:.0e}): "
                    f"{contact_result.message}"
                )
                continue
            # KKTKit's reduced contact solver stores dense primal and constraint
            # matrices.  Use it only for genuinely small projections; full-order
            # forming systems with a few hundred DOFs still need the sparse,
            # rank-filtered active-set path because they carry many face rows.
            if (contact_backend == "kktkit"
                    or (contact_backend == "auto" and variable_count <= 256)):
                try:
                    # KKTKit's dense active-set LU can emit a warning and
                    # return an unusable factorization for a singular contact
                    # pattern.  Promote it so this regularization loop retries
                    # the same linearization instead of spending Newton steps
                    # with a bogus contact correction.
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", LinAlgWarning)
                        dense_result = solve_kktkit_contact_step(
                            hessian, gradient, constraints, rhs, unilateral,
                            initial_active_rows=initial_active,
                            rank_tolerance=1e-8,
                            required_active_rows=required_active,
                            tolerance=1e-7, max_iterations=100,
                        )
                except (
                    LinAlgWarning, RuntimeError, ValueError,
                    np.linalg.LinAlgError,
                ) as error:
                    failures.append(
                        f"KKTKit(reg={regularization_factor:.0e}): "
                        f"{type(error).__name__}: {error}"
                    )
                else:
                    replacement_punch = switch_punch_support(
                        dense_result, displacement, current_stroke
                    )
                    if replacement_punch is not None:
                        return retry_with_punch_support(
                            replacement_punch, "KKTKit-dense", dense_result
                        )
                    replacement = switch_holder_support(dense_result)
                    if replacement is not None:
                        return retry_with_holder_support(
                            replacement, "KKTKit-dense", dense_result
                        )
                    replacement_die = switch_die_support(dense_result)
                    if replacement_die is not None:
                        return retry_with_die_support(
                            replacement_die, "KKTKit-dense", dense_result
                        )
                    required_tension = float(np.max(
                        dense_result.multipliers[required_active], initial=0.
                    ))
                    if dense_result.converged and required_tension <= 1e-2:
                        label = "KKTKit-dense"
                        if regularization_factor:
                            label += f"-regularized-{regularization_factor:.0e}"
                        return dense_result, label, None
                    last_contact_result = dense_result
                    failures.append(
                        f"KKTKit(reg={regularization_factor:.0e}): "
                        f"{dense_result.message}"
                    )
                continue
            sparse_result = solve_sparse_contact_step(
                hessian, gradient, constraints, rhs, unilateral,
                initial_active_rows=initial_active,
                required_active_rows=required_active,
                tolerance=linear_contact_tolerance,
                dual_tolerance=contact_dual_tolerance,
                complementarity_tolerance=(
                    linear_contact_complementarity_tolerance
                ),
                max_iterations=contact_solver_max_iterations,
                rank_tolerance=contact_rank_tolerance,
                initial_multipliers=initial_multipliers,
                semismooth_first=contact_semismooth_first,
            )
            replacement_punch = switch_punch_support(
                sparse_result, displacement, current_stroke
            )
            if replacement_punch is not None:
                return retry_with_punch_support(
                    replacement_punch, "sparse-active-set", sparse_result
                )
            replacement = switch_holder_support(sparse_result)
            if replacement is not None:
                return retry_with_holder_support(
                    replacement, "sparse-active-set", sparse_result
                )
            replacement_die = switch_die_support(sparse_result)
            if replacement_die is not None:
                return retry_with_die_support(
                    replacement_die, "sparse-active-set", sparse_result
                )
            required_tension = float(np.max(
                sparse_result.multipliers[required_active], initial=0.
            ))
            if ((sparse_result.converged or sparse_result.inexact)
                    and required_tension <= 1e-2):
                label = (
                    "sparse-semismooth-inexact"
                    if sparse_result.inexact and not sparse_result.converged
                    else "sparse-active-set"
                )
                if regularization_factor:
                    label += f"-regularized-{regularization_factor:.0e}"
                return sparse_result, label, None
            last_contact_result = sparse_result
            failures.append(
                f"sparse(reg={regularization_factor:.0e}): {sparse_result.message}"
            )
        if last_contact_result is not None:
            replacement = switch_holder_support(last_contact_result)
            if replacement is not None:
                return retry_with_holder_support(
                    replacement, "failed-contact-linearization",
                    last_contact_result,
                )
            replacement_die = switch_die_support(last_contact_result)
            if replacement_die is not None:
                return retry_with_die_support(
                    replacement_die, "failed-contact-linearization",
                    last_contact_result,
                )
        raise RuntimeError(
            "contact active set failed after sparse/KKTKit regularization attempts: "
            + " | ".join(failures[-4:])
        )

    # A failed contact configuration is cut back by stroke subdivision.  Keep
    # an individual trial bounded so its line searches do not dominate the
    # complete continuation run.
    # Contact branch changes can make the merit line search damp otherwise
    # useful Newton corrections.  Keep iterating toward equilibrium before
    # the stroke controller subdivides an increment whose complementarity is
    # already small.
    contact_newton_iteration_limit = int(os.environ.get(
        "SCIKIT_FVM_FORMING_NEWTON_ITERATION_LIMIT", "280"
    ))
    if contact_newton_iteration_limit < 1:
        raise ValueError(
            "SCIKIT_FVM_FORMING_NEWTON_ITERATION_LIMIT must be positive"
        )
    force_stagnation_iteration_limit = int(os.environ.get(
        "SCIKIT_FVM_FORMING_FORCE_STAGNATION_LIMIT", "8"
    ))
    if force_stagnation_iteration_limit < 1:
        raise ValueError(
            "SCIKIT_FVM_FORMING_FORCE_STAGNATION_LIMIT must be positive"
        )
    forming_force_change_tolerance = float(os.environ.get(
        "SCIKIT_FVM_FORMING_FORCE_CHANGE_TOLERANCE", "0.02"
    ))
    if (not np.isfinite(forming_force_change_tolerance)
            or forming_force_change_tolerance <= 0.):
        raise ValueError(
            "SCIKIT_FVM_FORMING_FORCE_CHANGE_TOLERANCE must be finite and "
            "positive"
        )
    for record in records:
        relative_change = record.get(
            "punch_force_prediction_error_relative",
            record.get("punch_force_relative_change"),
        )
        if relative_change is not None:
            record["force_resolution_unresolved_for_requested_tolerance"] = (
                float(relative_change) > forming_force_change_tolerance
            )
    slow_progress_iteration_limit = int(
        os.environ.get("SCIKIT_FVM_FORMING_SLOW_PROGRESS_LIMIT", "8")
        if forming_slow_progress_limit is None
        else forming_slow_progress_limit
    )
    if slow_progress_iteration_limit < 1:
        raise ValueError(
            "SCIKIT_FVM_FORMING_SLOW_PROGRESS_LIMIT must be positive"
        )
    maximum_plastic_increment_per_step = float(os.environ.get(
        "SCIKIT_FVM_FORMING_MAX_PLASTIC_INCREMENT_PER_STEP", "6e-3"
    ))
    if (not np.isfinite(maximum_plastic_increment_per_step)
            or maximum_plastic_increment_per_step <= 0.):
        raise ValueError(
            "SCIKIT_FVM_FORMING_MAX_PLASTIC_INCREMENT_PER_STEP must be finite "
            "and positive"
        )
    def advance_to(stroke: float, subdivision_depth: int = 0):
        nonlocal checkpoint_state_current_layout
        nonlocal displacement, holder_z, committed, committed_contact_multipliers
        nonlocal committed_friction_multipliers
        nonlocal previous_displacement, previous_holder_z, previous_stroke
        nonlocal accepted_stroke
        regularization_floor[0] = max(
            regularization_floor[0], initial_regularization_floor
        )
        step = len(records)
        # When the punch is fully detached, its prescribed motion does not
        # change the blank equilibrium. Carry the accepted die/holder state
        # through the open gap without repeatedly solving the same KKT system;
        # resume Newton as soon as the requested stroke closes the punch gap.
        if (records and abs(stroke - accepted_stroke) > 1e-12
                and abs(records[-1].get("punch_force_N_half_model", 0.)) <= 1e-12
                and np.all(np.abs(committed_contact_multipliers[punch_rows]) <= 1e-10)
                and np.all(np.abs(committed_friction_multipliers[punch_rows]) <= 1e-10)):
            trial_gaps, _, _ = tool_state(displacement, holder_z, float(stroke))
            punch_gaps = trial_gaps[:punch_rows.size]
            if np.min(punch_gaps, initial=np.inf) >= -1e-7:
                previous_displacement = displacement.copy()
                previous_holder_z = float(holder_z)
                previous_stroke = float(accepted_stroke)
                accepted_stroke = float(stroke)
                if contact_method == "mortar-p0":
                    current_punch_points = punch_mortar.evaluate(
                        np.r_[displacement, 0.], tools["punch"],
                        translation=[0., -float(stroke)],
                        curvature_blend_length=(
                            punch_contact_curvature_blend_length_mm
                        ),
                    ).facet_points
                else:
                    current_punch_points = sample_coordinates(
                        punch, displacement
                    )
                if current_punch_points.shape[0] != punch_gaps.size:
                    raise RuntimeError(
                        "inactive punch gap samples do not match their points: "
                        f"gaps={punch_gaps.size}, "
                        f"points={current_punch_points.shape[0]}"
                    )
                near_punch = punch_gaps <= 1e-6
                record = dict(records[-1])
                record.update({
                    "step": len(records),
                    "stroke_mm": accepted_stroke,
                    "newton_iterations": 0,
                    "newton_step_length": 1.0,
                    "contact_solver": "inactive-punch-continuation",
                    "active_punch_points": 0,
                    "active_punch_points_xz_mm": [],
                    "active_punch_profile_segments": [],
                    "minimum_punch_gap_mm": float(np.min(punch_gaps)),
                    "maximum_contact_penetration_mm": max(
                        -float(np.min(trial_gaps, initial=0.)), 0.
                    ),
                    "near_contact_punch_points_xz_mm": (
                        current_punch_points[near_punch].tolist()
                    ),
                    "punch_force_N_half_model": 0.0,
                    "normal_punch_force_N_half_model": 0.0,
                    "friction_punch_force_N_half_model": 0.0,
                    "force_resolution_unresolved": False,
                    "force_resolution_floor_mm": 0.0,
                })
                records.append(record)
                checkpoint_state_current_layout = True
                save_checkpoint()
                print(
                    "accepted inactive-punch continuation step "
                    f"{len(records) - 1}: stroke={accepted_stroke:.6f} mm, "
                    f"minimum_gap={record['minimum_punch_gap_mm']:.6g} mm",
                    flush=True,
                )
                return
        friction_base_displacement = displacement.copy()
        friction_base_holder_z = float(holder_z)
        friction_base_stroke = float(accepted_stroke)
        variables = np.r_[displacement, holder_z]
        if (previous_displacement is not None
                and previous_stroke is not None
                and (accepted_stroke - previous_stroke)
                * (stroke - accepted_stroke) > 0.):
            history_increment = accepted_stroke - previous_stroke
            requested_increment = stroke - accepted_stroke
            prediction_ratio = requested_increment / history_increment
            previous_branch_jump_unresolved = bool(
                records
                and (
                    records[-1].get("force_resolution_unresolved", False)
                    or records[-1].get("holder_jump_unresolved", False)
                )
            )
            if (prediction_ratio > 0.
                    and os.environ.get(
                        "SCIKIT_FVM_FORMING_DISABLE_EXTRAPOLATION", ""
                    ) not in {"1", "true", "yes"}
                    and not previous_branch_jump_unresolved):
                previous_variables = np.r_[previous_displacement, previous_holder_z]
                # Continue the last accepted displacement trend across a
                # contact jump, but cap extrapolation when the adaptive floor
                # is much larger than the preceding cutback increment.
                variables += min(prediction_ratio, 2.) * (
                    variables - previous_variables
                )
                variables[symmetry_dofs] = 0.
        initial_gaps, initial_constraints, initial_geometry = tool_state(
            variables[:-1], float(variables[-1]), float(stroke)
        )
        initial_contact_active = np.r_[
            np.ones(equality_count, dtype=bool), initial_gaps <= 1e-8
        ]
        initially_active_holder = holder_rows[
            initial_contact_active[holder_rows]
        ]
        holder_vertical_sensitivity = np.abs(
            initial_constraints[holder_rows, -1]
        )
        holder_force_support_rows = holder_rows[
            initial_constraints[holder_rows, -1] > 1e-8
        ]
        active_support_rows = np.intersect1d(
            initially_active_holder, holder_force_support_rows
        )
        compressive_support_rows = active_support_rows[
            committed_contact_multipliers[active_support_rows] < -1e-2
        ]
        if compressive_support_rows.size:
            required_holder_row = int(compressive_support_rows[
                np.argmin(committed_contact_multipliers[compressive_support_rows])
            ])
        elif active_support_rows.size:
            required_holder_row = int(active_support_rows[
                np.argmax(np.abs(initial_constraints[active_support_rows, -1]))
            ])
        elif holder_force_support_rows.size:
            # A load-controlled holder needs a contact normal that can oppose
            # the applied force. Do not constrain a distant open row merely
            # to regularize the first KKT solve: that creates an artificial
            # support path, and the resulting multiplier can grow without
            # bound while Newton tries to close a multi-millimeter gap. When
            # no row is already closed, let the holder translation be
            # regularized until the actual contact set reaches this tolerance.
            closest_holder_support = int(holder_force_support_rows[
                np.argmin(initial_gaps[
                    holder_force_support_rows - equality_count
                ])
            ])
            closest_holder_gap = float(
                initial_gaps[closest_holder_support - equality_count]
            )
            required_holder_row = (
                closest_holder_support
                if closest_holder_gap <= max(
                    forming_contact_gap_tolerance_mm, 1e-8
                )
                else None
            )
        else:
            # Preserve a deterministic fallback for profiles whose normals
            # have no force-opposing vertical component.
            required_holder_row = int(
                holder_rows[np.argmax(holder_vertical_sensitivity)]
            )
        if os.environ.get("SCIKIT_FVM_FORMING_TRACE", "") == "contacts":
            seed_gap = (
                float(initial_gaps[required_holder_row - equality_count])
                if required_holder_row is not None else float("nan")
            )
            if contact_method == "point-collocation":
                holder_debug_points = sample_coordinates(holder, displacement)
                holder_debug_projection = tools["blank_holder"].translated(
                    [0., holder_z]
                ).project(
                    holder_debug_points,
                    curvature_blend_length=HOLDER_CONTACT_CURVATURE_BLEND_LENGTH_MM,
                )
                closest_debug_rows = holder_rows[np.argsort(
                    initial_gaps[holder_rows - equality_count]
                )[:6]]
                print(
                    "[forming-holder-candidate-geometry] "
                    "(row,nodes,x,z,gap,nx,nz,segment,vertical_sensitivity)="
                    f"{[(int(row), holder['nodes'][row - holder_rows[0]].tolist(), *holder_debug_points[row - holder_rows[0]].tolist(), float(initial_gaps[row - equality_count]), *holder_debug_projection.normals[row - holder_rows[0]].tolist(), int(holder_debug_projection.segment_indices[row - holder_rows[0]]), float(initial_constraints[row, -1])) for row in closest_debug_rows]}",
                    flush=True,
                )
            print(
                "[forming-support-seed] "
                f"stroke={stroke:.9f} row={required_holder_row} "
                f"gap={seed_gap:.6e} mm "
                f"layout=(equalities:{equality_count}, total:{row_count}, "
                f"punch:{punch_rows.size}, die:{die_rows.size}, "
                f"holder:{holder_rows.size}) "
                f"active_support_rows={active_support_rows.tolist()} "
                f"extrapolation_disabled="
                f"{os.environ.get('SCIKIT_FVM_FORMING_DISABLE_EXTRAPOLATION', '')}",
                flush=True,
            )
            if (isinstance(initial_geometry, tuple)
                    and initial_geometry[0] == "mortar-p0"):
                holder_eval = initial_geometry[1][2]
                nearest = np.argsort(holder_eval.gaps)[:5]
                print(
                    "[forming-mortar-holder-gap] "
                    f"stroke={stroke:.9f} "
                    f"(row,average_gap,min_sample_gap,facet_point,samples)="
                    f"{[(int(holder_rows[i]), float(holder_eval.gaps[i]), float(holder_eval.minimum_sample_gaps[i]), holder_eval.facet_points[i].tolist(), holder_eval.sample_points.reshape(len(holder_eval.gaps), holder_eval.sample_gaps.size // len(holder_eval.gaps), 2)[i].tolist()) for i in nearest]}",
                    flush=True,
                )
        converged = False
        contact = None
        contact_solver_used = "sparse-active-set"
        multipliers = committed_contact_multipliers.copy()
        small_alpha_iterations = 0
        stagnant_iterations = 0
        slow_progress_iterations = 0
        force_stagnation_iterations = 0
        regularization_retry_count = 0
        regularization_slow_progress_iterations = 0
        regularization_backoff_hold_iterations = 0
        contact_restoration_equilibrium_limit = None
        contact_restoration_holder_residual_limit = None
        holder_support_released = False
        # The finite-strain J2 map is only directionally differentiable when
        # a quadrature point lies on yield. Seed its generalized tangent with
        # the last accepted continuation direction instead of always taking
        # the loading-side derivative at the start of every stroke increment.
        material_tangent_direction = None
        if finite_strain_j2 and previous_displacement is not None:
            continuation_direction = displacement - previous_displacement
            if np.linalg.norm(continuation_direction, ord=np.inf) > 1e-14:
                material_tangent_direction = continuation_direction
        # Remember support samples already tried during this prescribed
        # stroke increment.  Otherwise a tensile seed transferred by the
        # active-set solve can be selected again from the old multiplier
        # vector on the next Newton iteration, creating a two-row support
        # cycle.
        holder_support_visited_rows = (
            {int(required_holder_row)}
            if required_holder_row is not None else set()
        )
        required_die_support_row = None
        die_support_released_rows = set()
        punch_support_released_rows = set()
        try:
            for iteration in range(1, contact_newton_iteration_limit + 1):
                # Prevent transfer cycles within one contact linearization,
                # but allow a support row tried on an earlier Newton iterate
                # to be reconsidered after the blank has moved. Persisting
                # this set for the whole prescribed stroke can exhaust every
                # nearby holder sample and release the only load path even
                # when a previously rejected row has become admissible.
                holder_support_visited_rows = (
                    {int(required_holder_row)}
                    if required_holder_row is not None else set()
                )
                u = variables[:-1]
                h = float(variables[-1])
                approached_holder_z, holder_approached, holder_gap_after = (
                    approach_holder_support(u, h, float(stroke))
                )
                if holder_approached:
                    variables[-1] = approached_holder_z
                    h = approached_holder_z
                    if holder_gap_after > -min(
                        1e-8, forming_contact_gap_tolerance_mm
                    ):
                        # There still is no reaction path.  Continue the
                        # geometry-only approach without evaluating an
                        # unbalanced material Newton state; once a support is
                        # reached, the ordinary unilateral KKT solve takes
                        # over on the next iteration.
                        if os.environ.get(
                                "SCIKIT_FVM_FORMING_TRACE", ""
                        ) == "contacts":
                            print(
                                "[forming-holder-approach] "
                                f"stroke={stroke:.9f} iteration={iteration} "
                                f"row_gap={holder_gap_after:.6e} mm "
                                f"holder_z={h:.9f} mm",
                                flush=True,
                            )
                        continue
                if os.environ.get("SCIKIT_FVM_FORMING_TRACE", "").lower() in {
                    "1", "true", "yes"
                }:
                    print(
                        f"[forming-iteration-start] stroke={stroke:.9f} "
                        f"iteration={iteration} variables={variables.size}",
                        flush=True,
                    )
                if finite_strain_j2:
                    evaluation = assembler.evaluate(
                        u, committed,
                        tangent_direction=material_tangent_direction,
                    )
                else:
                    evaluation = assembler.evaluate(u, committed)
                gaps, constraints, gap_hessians = tool_state(
                    u, h, float(stroke)
                )
                current_minimum_gap = minimum_tool_gap(gaps, gap_hessians)
                current_holder_support_rows = holder_rows[
                    constraints[holder_rows, -1] > 1e-8
                ]
                current_holder_support_gap = float(np.min(
                    gaps[current_holder_support_rows - equality_count],
                    initial=np.inf,
                ))
                equality_residual = u[symmetry_dofs]
                if holder_position_equality_row is not None:
                    equality_residual = np.r_[
                        equality_residual, h - holder_position_control_mm
                    ]
                rhs = np.r_[-equality_residual, -gaps]
                initial_active = np.r_[
                    np.ones(equality_count, dtype=bool), gaps <= 1e-8
                ]
                # Retain a compressive contact through a small positive gap
                # only while its complementarity product is within the same
                # tolerance used for convergence.  Carrying the multiplier
                # across a large opening gap pins a stale support point and
                # can make the holder Newton step alternate indefinitely.
                old_contact_multipliers = multipliers[equality_count:]
                retained_gap_limit = retained_contact_gap_limits()
                initial_active[equality_count:] |= (
                    (old_contact_multipliers < -1e-12)
                    & (gaps <= retained_gap_limit)
                )
                # Seed the active set with the nearest holder support row to
                # regularize the first factorization. Keep it unilateral so a
                # tensile reaction can trigger a support transfer.
                required_active = np.zeros(row_count, dtype=bool)
                punch_support_visited_rows = set()
                # During a displacement-controlled punch step, retain one
                # already-compressive punch sample as a temporary reaction
                # path.  Without this seed, two neighboring samples on the
                # curved punch can alternately enter and leave the active set
                # when the consistent tangent is indefinite.  This does not
                # prescribe a punch force: the row remains unilateral and the
                # committed Newton state is still checked against every gap
                # and multiplier.
                compressive_punch_rows = punch_rows[
                    initial_active[punch_rows]
                    & (multipliers[punch_rows] < -1e-2)
                    & (gaps[punch_rows - equality_count]
                       <= 10. * forming_contact_gap_tolerance_mm)
                    & ~np.isin(
                        punch_rows, tuple(punch_support_released_rows)
                    )
                ]
                if (finite_strain_j2 and compressive_punch_rows.size
                        and not contact_semismooth_first):
                    punch_seed_row = int(compressive_punch_rows[
                        np.argmin(gaps[
                            compressive_punch_rows - equality_count
                        ])
                    ])
                    required_active[punch_seed_row] = (
                        not soft_punch_support_seed
                    )
                    punch_support_visited_rows.add(punch_seed_row)
                active_holder_supports = holder_rows[
                    initial_active[holder_rows]
                    & (constraints[holder_rows, -1] > 1e-8)
                ]
                active_seed_rows = active_holder_supports[
                    multipliers[active_holder_supports] < -1e-2
                ]
                if not holder_support_released:
                    if active_holder_supports.size:
                        if active_seed_rows.size:
                            # Keep the current force path on the same physical
                            # contact sample while it remains compressive.
                            # Selecting the most-compressive sample afresh at
                            # every Newton iteration can alternate between
                            # nearly coincident collocation rows.
                            if required_holder_row not in active_seed_rows:
                                untried_seed_rows = active_seed_rows[~np.isin(
                                    active_seed_rows,
                                    tuple(holder_support_visited_rows),
                                )]
                                if untried_seed_rows.size:
                                    required_holder_row = int(
                                        untried_seed_rows[np.argmin(
                                            multipliers[untried_seed_rows]
                                        )]
                                    )
                                else:
                                    holder_support_released = True
                        elif required_holder_row not in active_holder_supports:
                            untried_active_supports = active_holder_supports[
                                ~np.isin(
                                    active_holder_supports,
                                    tuple(holder_support_visited_rows),
                                )
                            ]
                            if untried_active_supports.size:
                                required_holder_row = int(
                                    untried_active_supports[np.argmin(gaps[
                                        untried_active_supports - equality_count
                                    ])]
                                )
                            else:
                                holder_support_released = True
                    else:
                        supportable_holder_rows = holder_rows[
                            (constraints[holder_rows, -1] > 1e-8)
                            & (gaps[holder_rows - equality_count]
                               <= max(forming_contact_gap_tolerance_mm, 1e-8))
                        ]
                        supportable_holder_rows = supportable_holder_rows[
                            ~np.isin(
                                supportable_holder_rows,
                                tuple(holder_support_visited_rows),
                            )
                        ]
                        if supportable_holder_rows.size:
                            required_holder_row = int(supportable_holder_rows[
                                np.argmin(gaps[
                                    supportable_holder_rows - equality_count
                                ])
                            ])
                        else:
                            # No support is closed.  The geometry-only holder
                            # approach at the start of this Newton iteration
                            # handles this state without imposing a distant
                            # open row as an artificial contact constraint.
                            required_holder_row = None
                    if (not holder_support_released
                            and required_holder_row is not None):
                        initial_active[required_holder_row] = True
                        holder_seed_gap = gaps[
                            required_holder_row - equality_count
                        ]
                        # Keep a currently compressive, closed force-path
                        # point in the linearized KKT system until the bounded
                        # Newton/line-search step actually opens it.  Testing
                        # an unbounded predictor here can release the seed
                        # even when the accepted 0.1 mm holder step leaves it
                        # within the local contact neighborhood.
                        required_active[required_holder_row] = (
                            not soft_holder_support_seed
                            and not contact_semismooth_first
                        )
                        holder_support_visited_rows.add(
                            int(required_holder_row)
                        )
                # A load-controlled blank holder applies a net force to the
                # blank. When the nearby die contact is dropped from the
                # linearized active set, the holder reaction can remain
                # balanced while the blank itself has no opposing tool
                # reaction. Newton then crawls along an almost rigid
                # translation with a nonzero global force imbalance. Retain
                # the nearest die row whose normal can oppose the holder force
                # while it remains in a local contact neighborhood. The row
                # stays unilateral: the active-set solve still checks that its
                # reaction is compressive, and an infeasible seed cuts back
                # the stroke increment.
                if (holder_position_control_mm is None
                        and holder_force_per_width > 0.):
                    die_vertical_sensitivity = np.asarray(
                        constraints[die_rows][:, dof_z].sum(axis=1)
                    ).reshape(-1)
                    die_supportable_rows = die_rows[
                        die_vertical_sensitivity > 0.25
                    ]
                    # Use the same local gap band as support migration below.
                    # A broad seed neighborhood reintroduces an open support
                    # row after switch_die_support() releases it, so the next
                    # Newton iteration repeats the same transfer/release loop.
                    die_support_gap_limit = min(
                        die_support_gap_limit_mm,
                        forming_contact_gap_tolerance_mm,
                    )
                    nearby_die_rows = die_supportable_rows[
                        (gaps[die_supportable_rows - equality_count]
                         <= die_support_gap_limit)
                        & ~np.isin(
                            die_supportable_rows,
                            tuple(die_support_released_rows),
                        )
                    ]
                    active_die_supports = nearby_die_rows[
                        initial_active[nearby_die_rows]
                        & (multipliers[nearby_die_rows] < -1e-2)
                    ]
                    transfer_tensile_die_seed = (
                        required_die_support_row is not None
                        and multipliers[required_die_support_row] > 1e-2
                        and active_die_supports.size > 0
                    )
                    if nearby_die_rows.size:
                        if (required_die_support_row is None
                                or required_die_support_row not in nearby_die_rows
                                or transfer_tensile_die_seed):
                            pool = (
                                active_die_supports
                                if active_die_supports.size
                                else nearby_die_rows
                            )
                            required_die_support_row = int(pool[
                                np.argmin(gaps[pool - equality_count])
                            ])
                        initial_active[required_die_support_row] = True
                        required_active[required_die_support_row] = (
                            not (os.environ.get(
                                "SCIKIT_FVM_FORMING_SOFT_DIE_SUPPORT_SEED", ""
                            ).lower() in {"1", "true", "yes"})
                            and not contact_semismooth_first
                        )
                        if os.environ.get(
                            "SCIKIT_FVM_FORMING_TRACE", ""
                        ) == "contacts":
                            die_local_row = (
                                required_die_support_row - int(die_rows[0])
                            )
                            print(
                                "[forming-die-support-seed] "
                                f"stroke={stroke:.9f} "
                                f"row={required_die_support_row} "
                                f"gap={gaps[required_die_support_row - equality_count]:.6e} mm "
                                f"vertical_sensitivity="
                                f"{die_vertical_sensitivity[die_local_row]:.6e}",
                                flush=True,
                            )
                # The Newton iterate can retain a compressive multiplier
                # after its collocation point has opened.  The active-set
                # target correctly drops that row, but interpolating the old
                # reaction toward zero can make the merit line search take
                # vanishing steps while the old force remains unphysical.
                # Project only non-required rows that are outside the same
                # geometric closure neighborhood used above; keep required
                # support seeds intact so a load-controlled rigid tool never
                # loses its force path during this projection.
                project_to_open = (
                    ~required_active[equality_count:]
                    & (
                        (gaps > retained_gap_limit)
                        | (multipliers[equality_count:] > 1e-2)
                    )
                )
                protected_seed_rows = []
                if (not holder_support_released
                        and required_holder_row is not None
                        and required_active[required_holder_row]):
                    protected_seed_rows.append(int(required_holder_row))
                if (required_die_support_row is not None
                        and required_active[required_die_support_row]):
                    protected_seed_rows.append(int(required_die_support_row))
                if protected_seed_rows:
                    project_to_open[np.asarray(protected_seed_rows) - equality_count] = False
                projected_rows = equality_count + np.flatnonzero(
                    project_to_open
                )
                if projected_rows.size:
                    if os.environ.get(
                        "SCIKIT_FVM_FORMING_TRACE", ""
                    ) == "contacts":
                        print(
                            "[forming-contact] project open/tensile reaction "
                            f"rows={projected_rows.tolist()} "
                            f"max_gap={float(np.max(gaps[project_to_open])):.6e} mm",
                            flush=True,
                        )
                    multipliers[projected_rows] = 0.
                    committed_friction_multipliers[projected_rows] = 0.
                friction_force, friction_tangent, _ = friction_response(
                    u, h, stroke, friction_base_displacement,
                    friction_base_holder_z, friction_base_stroke, multipliers,
                )
                gradient = (
                    np.r_[evaluation.residual, holder_external_force_per_width]
                    + friction_force
                )
                tangent = sparse.block_diag(
                    (evaluation.tangent, sparse.csr_matrix((1, 1))),
                    format="csr",
                ) + friction_tangent
                if os.environ.get(
                    "SCIKIT_FVM_FORMING_CONTACT_GEOMETRIC_TANGENT", "1"
                ).lower() not in {"0", "false", "no"}:
                    tangent = tangent + contact_geometric_tangent(
                        gap_hessians, multipliers
                    )
                (contact, contact_solver_used,
                 transferred_holder_row) = solve_contact_linearization(
                    evaluation, gradient, constraints, rhs, initial_active,
                    required_active,
                    None if holder_support_released else required_holder_row,
                    tangent=tangent,
                    holder_support_visited_rows=holder_support_visited_rows,
                    die_support_released_rows=die_support_released_rows,
                    punch_support_visited_rows=punch_support_visited_rows,
                    punch_support_released_rows=punch_support_released_rows,
                    initial_multipliers=multipliers,
                    displacement=u,
                    current_stroke=float(stroke),
                )
                # At yield, the J2 Jacobian depends on whether the Newton
                # correction loads or unloads each quadrature point. Rebuild
                # the tangent along the correction and resolve the KKT system
                # until that branch choice is self-consistent. Without this
                # inner semismooth update, a tangent selected from the prior
                # load increment can point almost opposite to the actual
                # residual derivative and force the outer line search to take
                # vanishing steps.
                if finite_strain_j2 and directional_tangent_refinements:
                    directional_holder_seed = (
                        transferred_holder_row
                        if transferred_holder_row is not None
                        else required_holder_row
                    )
                    directional_holder_released = (
                        holder_support_released
                        or transferred_holder_row == -1
                    )
                    for _ in range(directional_tangent_refinements):
                        current_direction = np.asarray(
                            contact.primal[:-1], dtype=float
                        )
                        directional_evaluation = assembler.evaluate(
                            u, committed, tangent_direction=current_direction
                        )
                        tangent_change = (
                            directional_evaluation.tangent - evaluation.tangent
                        ).tocsr()
                        tangent_scale = max(
                            float(np.max(np.abs(evaluation.tangent.data), initial=0.)),
                            1.,
                        )
                        tangent_change_scale = float(np.max(
                            np.abs(tangent_change.data), initial=0.
                        ))
                        if tangent_change_scale <= 1e-10 * tangent_scale:
                            break
                        evaluation = directional_evaluation
                        gradient = (
                            np.r_[evaluation.residual,
                                  holder_external_force_per_width]
                            + friction_force
                        )
                        tangent = sparse.block_diag(
                            (evaluation.tangent,
                             sparse.csr_matrix((1, 1))), format="csr"
                        ) + friction_tangent
                        if os.environ.get(
                            "SCIKIT_FVM_FORMING_CONTACT_GEOMETRIC_TANGENT", "1"
                        ).lower() not in {"0", "false", "no"}:
                            tangent = tangent + contact_geometric_tangent(
                                gap_hessians, multipliers
                            )
                        previous_direction = current_direction
                        (contact, contact_solver_used,
                         next_holder_row) = solve_contact_linearization(
                            evaluation, gradient, constraints, rhs,
                            initial_active, required_active,
                            None if directional_holder_released
                            else directional_holder_seed,
                            tangent=tangent,
                            holder_support_visited_rows=(
                                holder_support_visited_rows
                            ),
                            die_support_released_rows=(
                                die_support_released_rows
                            ),
                            punch_support_visited_rows=(
                                punch_support_visited_rows
                            ),
                            punch_support_released_rows=(
                                punch_support_released_rows
                            ),
                            initial_multipliers=multipliers,
                            displacement=u,
                            current_stroke=float(stroke),
                        )
                        if next_holder_row is not None:
                            transferred_holder_row = next_holder_row
                            if next_holder_row == -1:
                                directional_holder_released = True
                                directional_holder_seed = None
                            else:
                                directional_holder_seed = next_holder_row
                        direction_change = np.linalg.norm(
                            contact.primal[:-1] - previous_direction,
                            ord=np.inf,
                        ) / max(
                            np.linalg.norm(previous_direction, ord=np.inf), 1.
                        )
                        if direction_change <= 1e-6:
                            break
                required_die_seed_rows = die_rows[required_active[die_rows]]
                required_die_support_row = (
                    int(required_die_seed_rows[0])
                    if required_die_seed_rows.size else None
                )
                # ``contact.active_rows`` describes the linearized *target*
                # state at the end of this Newton correction.  Keep the
                # current iterate's reactions intact here: the line search
                # below homotopies from these multipliers to the target ones
                # together with the primal correction. Clearing target-inactive
                # rows before evaluating the current stationarity would change
                # the base point (and can erase a valid load-controlled holder
                # reaction before the Newton step has reached the next contact
                # branch).
                contact_primal = np.asarray(contact.primal, dtype=float).copy()
                contact_target_multipliers = np.asarray(
                    contact.multipliers, dtype=float
                ).copy()
                # Some sparse active-set exits retain the warm-start value on
                # rows that ended inactive. The endpoint of the line search
                # must nevertheless have zero traction there; keep the current
                # iterate's multipliers for its residual and interpolate them
                # toward this dual-feasible target below.
                contact_target_multipliers[
                    ~np.asarray(contact.active_rows, dtype=bool)
                ] = 0.
                if ((iteration == 1 or iteration % 10 == 0)
                        and finite_strain_j2
                        and os.environ.get(
                            "SCIKIT_FVM_FORMING_TRACE", ""
                        ).lower() == "directional"):
                    kkt_linearized_stationarity = (
                        tangent @ contact_primal
                        + gradient
                        + constraints.T @ contact_target_multipliers
                    )
                    kkt_linearized_stationarity[symmetry_dofs] = 0.
                    linearized_gap = np.asarray(
                        constraints @ contact_primal - rhs, dtype=float
                    )
                    active_gap_error = float(np.max(np.abs(
                        linearized_gap[contact.active_rows]
                    ), initial=0.))
                    print(
                        "[forming-contact-kkt-direction] "
                        f"stroke={stroke:.9f} "
                        f"solver={contact_solver_used} "
                        f"solver_iterations={contact.iterations} "
                        f"solver_converged={contact.converged} "
                        f"stationarity_inf="
                        f"{np.linalg.norm(kkt_linearized_stationarity, ord=np.inf):.6e} "
                        f"active_gap_inf={active_gap_error:.6e} "
                        f"step_inf={np.linalg.norm(contact_primal, ord=np.inf):.6e} "
                        f"multiplier_inf="
                        f"{np.linalg.norm(contact_target_multipliers, ord=np.inf):.6e}",
                        flush=True,
                    )
                holder_step_scale = 1.0
                newton_step_was_capped = False
                if holder_position_control_mm is None:
                    # The load-controlled holder is a zero-stiffness DOF;
                    # bound its Newton correction so one linearization does
                    # not jump across several changing contact normals.
                    holder_step_scale = min(
                        1.0,
                        holder_newton_step_limit_mm
                        / max(abs(float(contact_primal[-1])), 1e-30),
                    )
                    if holder_step_scale < 1.0:
                        newton_step_was_capped = True
                        contact_primal *= holder_step_scale
                        contact_target_multipliers = multipliers + (
                            holder_step_scale
                            * (contact_target_multipliers - multipliers)
                        )
                displacement_step_scale = min(
                    1.0,
                    displacement_newton_step_limit_mm
                    / max(
                        float(np.linalg.norm(contact_primal[:-1], ord=np.inf)),
                        1e-30,
                    ),
                )
                if displacement_step_scale < 1.0:
                    newton_step_was_capped = True
                    contact_primal *= displacement_step_scale
                    contact_target_multipliers = multipliers + (
                        displacement_step_scale
                        * (contact_target_multipliers - multipliers)
                    )
                if ((iteration == 1 or iteration % 10 == 0)
                        and finite_strain_j2
                        and os.environ.get(
                            "SCIKIT_FVM_FORMING_TRACE", ""
                        ).lower() == "directional"):
                    direction = np.asarray(contact.primal[:-1], dtype=float)
                    direction_scale = float(np.linalg.norm(direction, ord=np.inf))
                    if direction_scale > 0.:
                        epsilon = 1e-6 / direction_scale
                        perturbed = assembler.evaluate(
                            u + epsilon * direction, committed, tangent=False
                        ).residual
                        perturbed_minus = assembler.evaluate(
                            u - epsilon * direction, committed, tangent=False
                        ).residual
                        finite_difference = (perturbed - evaluation.residual) / epsilon
                        centered_difference = (
                            perturbed - perturbed_minus
                        ) / (2. * epsilon)
                        tangent_direction = evaluation.tangent @ direction
                        mismatch = finite_difference - tangent_direction
                        centered_mismatch = centered_difference - tangent_direction
                        denominator = max(
                            float(np.linalg.norm(finite_difference, ord=np.inf)),
                            float(np.linalg.norm(tangent_direction, ord=np.inf)),
                            1e-14,
                        )
                        centered_denominator = max(
                            float(np.linalg.norm(
                                centered_difference, ord=np.inf
                            )),
                            float(np.linalg.norm(
                                tangent_direction, ord=np.inf
                            )),
                            1e-14,
                        )
                        worst = int(np.argmax(np.abs(mismatch)))
                        element_mismatches = []
                        for element in np.flatnonzero(np.any(
                                assembler._dofs == worst, axis=(1, 2))):
                            element_dofs = assembler._dofs[element]
                            gradients = assembler._gradients[element, 0]
                            local_u = u[element_dofs]
                            local_direction = direction[element_dofs]
                            f2 = np.eye(2) + local_u.T @ gradients
                            df2 = local_direction.T @ gradients
                            f3 = np.eye(3)
                            f3[:2, :2] = f2
                            df3 = np.zeros((3, 3))
                            df3[:2, :2] = df2
                            state = committed[element, 0]
                            material_response = material.evaluate(
                                f3, state, tangent=True,
                                tangent_components=(0, 1),
                            )
                            dp_analytic = np.einsum(
                                "ijkl,kl->ij", material_response.tangent, df3
                            )
                            p_plus = material.evaluate(
                                f3 + epsilon * df3, state, tangent=False
                            )
                            p_minus = material.evaluate(
                                f3 - epsilon * df3, state, tangent=False
                            )
                            dp_fd = (
                                p_plus.first_piola_stress
                                - p_minus.first_piola_stress
                            ) / (2. * epsilon)
                            local_an = (
                                assembler._weights[element, 0]
                                * (dp_analytic[:2, :2] @ gradients.T).T
                            ).reshape(-1)
                            local_fd = (
                                assembler._weights[element, 0]
                                * (dp_fd[:2, :2] @ gradients.T).T
                            ).reshape(-1)
                            local_index = np.flatnonzero(
                                element_dofs.reshape(-1) == worst
                            )
                            for index in local_index:
                                alpha = float(state[9])
                                updated_alpha = alpha + float(
                                    material_response.plastic_multiplier
                                )
                                curve = material._hardening_curve
                                knot_distance = (
                                    float(np.min(np.abs(
                                        curve[:, 0] - updated_alpha
                                    ))) if curve is not None else np.inf
                                )
                                element_mismatches.append((
                                    int(element), float(local_fd[index]),
                                    float(local_an[index]), alpha,
                                    updated_alpha, knot_distance,
                                    float(material_response.equivalent_stress)
                                    - material._yield(alpha),
                                    bool(p_plus.plastic_points),
                                    bool(p_minus.plastic_points),
                                ))
                        element_mismatches.sort(
                            key=lambda item: abs(item[1] - item[2]),
                            reverse=True,
                        )
                        print(
                            "[forming-tangent-direction] "
                            f"stroke={stroke:.9f} epsilon={epsilon:.3e} "
                            f"relative_inf_error="
                            f"{np.linalg.norm(mismatch, ord=np.inf) / denominator:.4e} "
                            f"fd_at_worst={finite_difference[worst]:.6e} "
                            f"tangent_at_worst={tangent_direction[worst]:.6e} "
                            f"dof={worst} "
                            f"centered_relative_inf_error="
                            f"{np.linalg.norm(centered_mismatch, ord=np.inf) / centered_denominator:.4e} "
                            f"centered_fd_at_worst={centered_difference[worst]:.6e} "
                            f"element_terms={element_mismatches[:8]}",
                            flush=True,
                        )
                        bounded_direction = np.asarray(
                            contact_primal, dtype=float
                        )
                        contact_epsilon = 1e-6 / max(
                            float(np.linalg.norm(
                                bounded_direction, ord=np.inf
                            )),
                            1e-30,
                        )
                        contact_trial = variables + (
                            contact_epsilon * bounded_direction
                        )
                        contact_trial_eval = assembler.evaluate(
                            contact_trial[:-1], committed, tangent=False
                        )
                        _, contact_trial_constraints, _ = tool_state(
                            contact_trial[:-1], float(contact_trial[-1]),
                            float(stroke),
                        )
                        contact_trial_multipliers = multipliers + (
                            contact_epsilon
                            * (contact_target_multipliers - multipliers)
                        )
                        contact_trial_friction, _, _ = friction_response(
                            contact_trial[:-1], float(contact_trial[-1]),
                            float(stroke), friction_base_displacement,
                            friction_base_holder_z, friction_base_stroke,
                            contact_trial_multipliers, tangent=False,
                        )
                        base_stationarity = gradient + (
                            constraints.T @ multipliers
                        )
                        trial_stationarity = np.r_[
                            contact_trial_eval.residual,
                            holder_external_force_per_width,
                        ] + contact_trial_friction + (
                            contact_trial_constraints.T
                            @ contact_trial_multipliers
                        )
                        base_stationarity[symmetry_dofs] = 0.
                        trial_stationarity[symmetry_dofs] = 0.
                        actual_stationarity_direction = (
                            trial_stationarity - base_stationarity
                        ) / contact_epsilon
                        dual_direction = (
                            contact_target_multipliers - multipliers
                        )
                        actual_contact_force_direction = (
                            contact_trial_constraints.T
                            @ contact_trial_multipliers
                            - constraints.T @ multipliers
                        ) / contact_epsilon
                        contact_geometry_matrix = contact_geometric_tangent(
                            gap_hessians, multipliers
                        )
                        predicted_contact_force_direction = (
                            constraints.T @ dual_direction
                            + contact_geometry_matrix @ bounded_direction
                        )
                        predicted_stationarity_direction = (
                            tangent @ bounded_direction
                            + constraints.T @ dual_direction
                        )
                        linearization_error = (
                            actual_stationarity_direction
                            - predicted_stationarity_direction
                        )
                        direction_scale = max(
                            float(np.linalg.norm(
                                actual_stationarity_direction, ord=np.inf
                            )),
                            float(np.linalg.norm(
                                predicted_stationarity_direction, ord=np.inf
                            )),
                            1e-14,
                        )
                        regularization_factor = 0.
                        if "regularized-" in contact_solver_used:
                            try:
                                regularization_factor = float(
                                    contact_solver_used.rsplit(
                                        "regularized-", 1
                                    )[1].split(";", 1)[0]
                                )
                            except ValueError:
                                regularization_factor = 0.
                        regularized_direction = (
                            predicted_stationarity_direction.copy()
                        )
                        if regularization_factor:
                            direction_diagonal_scale = max(
                                float(np.max(np.abs(tangent.diagonal()),
                                             initial=0.)),
                                1.,
                            )
                            regularized_direction[:-1] += (
                                regularization_factor
                                * direction_diagonal_scale
                                * bounded_direction[:-1]
                            )
                        current_stationarity_directional_derivative = (
                            float(np.dot(
                                base_stationarity,
                                actual_stationarity_direction,
                            ))
                        )
                        worst_directional_dof = int(np.argmax(np.abs(
                            linearization_error
                        )))
                        contact_force_error = (
                            actual_contact_force_direction
                            - predicted_contact_force_direction
                        )
                        worst_contact_force_dof = int(np.argmax(np.abs(
                            contact_force_error
                        )))
                        print(
                            "[forming-contact-directional-derivative] "
                            f"stroke={stroke:.9f} "
                            f"epsilon={contact_epsilon:.3e} "
                            f"regularization={regularization_factor:.1e} "
                            f"jacobian_relative_inf_error="
                            f"{np.linalg.norm(linearization_error, ord=np.inf) / direction_scale:.4e} "
                            f"jacobian_error_dof={worst_directional_dof} "
                            f"actual_at_error_dof="
                            f"{actual_stationarity_direction[worst_directional_dof]:.6e} "
                            f"predicted_at_error_dof="
                            f"{predicted_stationarity_direction[worst_directional_dof]:.6e} "
                            f"actual_direction_norm="
                            f"{np.linalg.norm(actual_stationarity_direction, ord=np.inf):.6e} "
                            f"predicted_direction_norm="
                            f"{np.linalg.norm(predicted_stationarity_direction, ord=np.inf):.6e} "
                            f"residual_dot_direction="
                            f"{current_stationarity_directional_derivative:.6e} "
                            f"actual_dot_direction="
                            f"{np.dot(actual_stationarity_direction, bounded_direction):.6e} "
                            f"predicted_dot_direction="
                            f"{np.dot(predicted_stationarity_direction, bounded_direction):.6e} "
                            f"regularized_dot_direction="
                            f"{np.dot(regularized_direction, bounded_direction):.6e} "
                            f"contact_force_jacobian_error="
                            f"{np.linalg.norm(contact_force_error, ord=np.inf):.6e} "
                            f"contact_force_error_dof={worst_contact_force_dof} "
                            f"contact_force_actual_at_dof="
                            f"{actual_contact_force_direction[worst_contact_force_dof]:.6e} "
                            f"contact_force_predicted_at_dof="
                            f"{predicted_contact_force_direction[worst_contact_force_dof]:.6e}",
                            flush=True,
                        )
                if os.environ.get("SCIKIT_FVM_FORMING_TRACE", "") == "contacts":
                    active_holder = holder_rows[contact.active_rows[holder_rows]]
                    active_contact_rows = np.flatnonzero(
                        contact.active_rows[equality_count:]
                    ) + equality_count
                    active_contact_constraint_rows = []
                    for row in active_contact_rows:
                        nonzero_columns = np.flatnonzero(
                            np.abs(constraints[int(row)]) > 1e-14
                        )
                        active_contact_constraint_rows.append((
                            int(row),
                            float(gaps[int(row) - equality_count]),
                            float(multipliers[int(row)]),
                            [
                                (int(column), float(constraints[int(row), column]))
                                for column in nonzero_columns
                            ],
                        ))
                    contact_step_dof = int(np.argmax(np.abs(contact_primal)))
                    if contact_step_dof == basis.N:
                        contact_step_location = "holder"
                    else:
                        contact_step_node = contact_step_dof // 2
                        contact_step_component = (
                            "x" if contact_step_dof % 2 == 0 else "z"
                        )
                        contact_step_location = (
                            f"{contact_step_component}@"
                            f"({node_x[contact_step_node]:.6g},"
                            f"{node_z[contact_step_node]:.6g})"
                        )
                    diagnostic_holder_rows = np.unique(np.r_[
                        active_holder,
                        ([required_holder_row]
                         if required_holder_row is not None else []),
                    ]).astype(np.int64)
                    holder_supportable = holder_rows[
                        constraints[holder_rows, -1] > 1e-8
                    ]
                    nearest_holder_support = holder_supportable[np.argsort(
                        gaps[holder_supportable - equality_count]
                    )[:6]]
                    nearest_holder_state = [
                        (int(row), float(gaps[row - equality_count]),
                         float(multipliers[row]),
                         bool(contact.active_rows[row]))
                        for row in nearest_holder_support
                    ]
                    if (isinstance(gap_hessians, tuple)
                            and gap_hessians[0] == "mortar-p0"):
                        current_holder_points = gap_hessians[1][2].facet_points
                        _, _, predicted_geometry = tool_state(
                            u + contact_primal[:-1],
                            h + float(contact_primal[-1]), float(stroke),
                        )
                        predicted_holder_points = predicted_geometry[1][
                            2
                        ].facet_points
                    else:
                        current_holder_points = sample_coordinates(holder, u)
                        predicted_holder_points = sample_coordinates(
                            holder, u + contact_primal[:-1]
                        )
                    current_projection = tools["blank_holder"].translated(
                        [0., h]
                    ).project(current_holder_points[diagnostic_holder_rows - holder_rows[0]])
                    predicted_projection = tools["blank_holder"].translated(
                        [0., h + contact_primal[-1]]
                    ).project(predicted_holder_points[
                        diagnostic_holder_rows - holder_rows[0]
                    ])
                    print(
                        "[forming-contact-state] "
                        f"stroke={stroke:.9f} iteration={iteration} "
                        f"active_contact_rows={active_contact_rows.tolist()} "
                        f"active_contact_multipliers="
                        f"{contact_target_multipliers[active_contact_rows].tolist()} "
                        f"active_contact_constraint_rows="
                        f"{active_contact_constraint_rows} "
                        f"max_step_dof={contact_step_dof}:"
                        f"{contact_step_location} "
                        f"max_step={contact_primal[contact_step_dof]:.6e} "
                        f"holder_active={active_holder.tolist()} "
                        f"holder_multipliers="
                        f"{contact_target_multipliers[active_holder].tolist()} "
                        f"holder_constraint_coefficients="
                        f"{np.asarray(constraints[active_holder, -1]).ravel().tolist()} "
                        f"nearest_holder_support(row,gap,lambda,active)="
                        f"{nearest_holder_state} "
                        f"delta_holder={contact_primal[-1]:.6e} "
                        f"holder_step_scale={holder_step_scale:.3e} "
                        f"diagnostic_rows={diagnostic_holder_rows.tolist()} "
                        f"contact_geometry="
                        f"{[(int(r), int(i), int(j), float(g0), float(g1), n0.tolist(), n1.tolist()) for r, i, j, g0, g1, n0, n1 in zip(diagnostic_holder_rows, current_projection.segment_indices, predicted_projection.segment_indices, current_projection.gaps, predicted_projection.gaps, current_projection.normals, predicted_projection.normals, strict=True)]}",
                        flush=True,
                    )
                if transferred_holder_row == -1:
                    holder_support_released = True
                elif transferred_holder_row is not None:
                    required_holder_row = transferred_holder_row
                    holder_support_released = False
                # ``protected_seed_rows`` used above belongs to the current
                # linearization. The contact solve may change the die seed
                # (or transfer the holder seed); build the line-search
                # projection from the resulting support state so the accepted
                # candidate is evaluated with the reactions the next Newton
                # iteration will retain.
                protected_seed_rows = []
                if (not holder_support_released
                        and required_holder_row is not None
                        and required_active[required_holder_row]):
                    protected_seed_rows.append(int(required_holder_row))
                if (required_die_support_row is not None
                        and required_active[required_die_support_row]):
                    protected_seed_rows.append(int(required_die_support_row))
                current_stationarity = gradient + constraints.T @ multipliers
                current_stationarity[symmetry_dofs] = 0.
                current_equilibrium = float(np.linalg.norm(
                    current_stationarity, ord=np.inf
                ))
                current_equilibrium_rms = float(
                    np.linalg.norm(current_stationarity)
                    / np.sqrt(current_stationarity.size)
                )
                current_equilibrium_tolerance = equilibrium_tolerance_for(
                    multipliers, constraints
                )
                if contact_restoration_equilibrium_limit is None:
                    # Contact restoration is a feasibility phase: a useful
                    # gap correction can temporarily raise the force residual
                    # before the next equilibrium Newton step lowers it. Bound
                    # that uncommitted excursion by a configurable multiple
                    # of the equilibrium tolerance, capped by its absolute
                    # floor. The load-scaled tolerance can inflate with
                    # contact multipliers during restoration and otherwise
                    # permit the force residual to ratchet upward. Final
                    # increment acceptance still uses the ordinary test.
                    contact_restoration_equilibrium_limit = (
                        current_equilibrium
                        + contact_restoration_equilibrium_tolerance_multiplier
                        * min(
                            current_equilibrium_tolerance,
                            forming_equilibrium_tolerance,
                        )
                    )
                    contact_restoration_holder_residual_limit = (
                        abs(float(current_stationarity[-1]))
                        + contact_restoration_holder_tolerance_multiplier
                        * current_equilibrium_tolerance
                    )
                equilibrium_dof = int(np.argmax(np.abs(current_stationarity)))
                if (finite_strain_j2
                        and os.environ.get("SCIKIT_FVM_FORMING_TRACE", "")
                        .lower() == "directional"
                        and (iteration == 1 or iteration % 10 == 0)):
                    row_terms = []
                    for row in np.flatnonzero(
                        np.abs(constraints[:, equilibrium_dof]) > 1e-12
                    ):
                        if row < equality_count:
                            group = "equality"
                        elif row in punch_rows:
                            group = "punch"
                        elif row in die_rows:
                            group = "die"
                        else:
                            group = "holder"
                        row_terms.append((
                            int(row), group,
                            float(gaps[row - equality_count])
                            if row >= equality_count else 0.,
                            float(multipliers[row]),
                            float(constraints[row, equilibrium_dof]),
                            bool(contact.active_rows[row]),
                        ))
                    material_residual = (
                        float(evaluation.residual[equilibrium_dof])
                        if equilibrium_dof < basis.N else 0.0
                    )
                    print(
                        "[forming-equilibrium-decomposition] "
                        f"stroke={stroke:.9f} dof={equilibrium_dof} "
                        f"material={material_residual:.9e} "
                        f"contact={float((constraints.T @ multipliers)[equilibrium_dof]):.9e} "
                        f"friction={friction_force[equilibrium_dof]:.9e} "
                        f"stationarity={current_stationarity[equilibrium_dof]:.9e} "
                        f"rows={row_terms}", flush=True,
                    )
                if equilibrium_dof == basis.N:
                    equilibrium_location = "holder"
                else:
                    equilibrium_node = equilibrium_dof // 2
                    equilibrium_component = "x" if equilibrium_dof % 2 == 0 else "z"
                    equilibrium_location = (
                        f"{equilibrium_component}@"
                        f"({node_x[equilibrium_node]:.6g},"
                        f"{node_z[equilibrium_node]:.6g})"
                    )
                minimum_gap_local_row = int(np.argmin(gaps)) if gaps.size else -1
                minimum_gap_row = (
                    equality_count + minimum_gap_local_row
                    if minimum_gap_local_row >= 0 else -1
                )
                minimum_gap_group = (
                    "punch" if minimum_gap_row in punch_rows else
                    "die" if minimum_gap_row in die_rows else
                    "holder" if minimum_gap_row in holder_rows else "none"
                )
                # Scale penetration against the applied tool force over a
                # small geometric tolerance. The largest tangent diagonal is
                # far too stiff a penalty here and can make the line search
                # accept loss of the holder reaction just to remove a tiny gap.
                penetration_scale = max(holder_force_per_width, 1.) / max(
                    1e-3 * thickness, 1e-6
                )
                complementarity_length = max(thickness, 1e-12)
                current_complementarity = float(np.max(np.abs(
                    gaps * multipliers[equality_count:]
                ), initial=0.))
                current_tension = float(np.max(
                    multipliers[equality_count:], initial=0.
                ))
                current_required_active_gap = float(np.max(
                    gaps[required_active[equality_count:]], initial=0.
                ))
                current_contact_feasible = (
                    current_minimum_gap
                    >= -forming_contact_gap_tolerance_mm
                    and current_required_active_gap
                    <= forming_contact_gap_tolerance_mm
                    and np.max(
                        multipliers[equality_count:][
                            ~required_active[equality_count:]
                        ], initial=0.
                    ) <= 1e-2
                    and current_complementarity
                    <= forming_complementarity_tolerance
                )
                current_contact_violation = max(
                    max(-current_minimum_gap, 0.)
                    / max(forming_contact_gap_tolerance_mm, 1e-12),
                    current_required_active_gap
                    / max(forming_contact_gap_tolerance_mm, 1e-12),
                    current_complementarity
                    / max(forming_complementarity_tolerance, 1e-12),
                    current_tension / 1e-2,
                    float(np.max(
                        multipliers[equality_count:][
                            ~required_active[equality_count:]
                        ], initial=0.
                    )) / 1e-2,
                )
                current_merit = (
                    float(np.hypot(
                        current_equilibrium, 2.0 * current_stationarity[-1]
                    ))
                    + penetration_scale * max(-current_minimum_gap, 0.)
                    + current_complementarity / complementarity_length
                    + current_tension
                )
                if os.environ.get("SCIKIT_FVM_FORMING_TRACE", "") not in ("", "0"):
                    print(
                        "[forming-newton] "
                        f"stroke={stroke:.9f} iteration={iteration} "
                        f"equilibrium={current_equilibrium:.4e} "
                        f"equilibrium_tolerance="
                        f"{current_equilibrium_tolerance:.4e} "
                        f"equilibrium_dof={equilibrium_dof}:"
                        f"{equilibrium_location} "
                        f"residual={current_stationarity[equilibrium_dof]:.4e} "
                        f"holder_residual={current_stationarity[-1]:.4e} "
                        f"holder_contact_force="
                        f"{float(constraints[:, -1] @ multipliers):.4e} "
                        f"minimum_gap={current_minimum_gap:.4e} "
                        f"minimum_gap_group={minimum_gap_group} "
                        f"minimum_gap_row={minimum_gap_row} "
                        f"minimum_gap_multiplier="
                        f"{multipliers[minimum_gap_row] if minimum_gap_row >= 0 else 0.0:.4e} "
                        f"minimum_gap_active="
                        f"{bool(contact.active_rows[minimum_gap_row]) if minimum_gap_row >= 0 else False} "
                        f"complementarity={current_complementarity:.4e} "
                        f"tension={current_tension:.4e} merit={current_merit:.4e} "
                        f"regularization_floor={regularization_floor[0]:.1e}",
                        flush=True,
                    )
                # A contact branch change can leave an iterate that already
                # satisfies equilibrium, gap, dual-feasibility, and
                # complementarity tolerances even though the next KKT
                # correction is not a descent direction for the merit
                # function.  In that case the line search below used to
                # reject the increment before the post-step convergence test
                # could see it.  Accept the current (still uncommitted)
                # constitutive trial when it meets the same convergence
                # criteria as a line-search candidate.
                current_is_converged = (
                    current_equilibrium <= current_equilibrium_tolerance
                    # Do not let the early-convergence path accumulate the
                    # full allowed penetration while the prescribed tool
                    # advances.  That can accept many increments with an
                    # unchanged blank state, then present Newton with a large
                    # contact correction at the tolerance boundary.  Keep the
                    # configured gap tolerance for final acceptance, but
                    # resume Newton once one quarter of it is consumed.
                    and current_minimum_gap
                    >= -0.25 * forming_contact_gap_tolerance_mm
                    and current_required_active_gap
                    <= forming_contact_gap_tolerance_mm
                    and np.max(
                        multipliers[equality_count:][
                            ~required_active[equality_count:]
                        ], initial=0.
                    ) <= 1e-2
                    and current_complementarity
                    <= forming_complementarity_tolerance
                )
                candidate_eval = evaluation if current_is_converged else None
                candidate = variables.copy() if current_is_converged else None
                candidate_gaps = gaps if current_is_converged else None
                candidate_constraints = constraints if current_is_converged else None
                candidate_multipliers = (
                    multipliers.copy() if current_is_converged else None
                )
                candidate_is_converged = current_is_converged
                equilibrium = current_equilibrium
                candidate_minimum_gap = current_minimum_gap
                candidate_holder_support_gap = current_holder_support_gap
                stationarity = (
                    current_stationarity.copy() if current_is_converged else None
                )
                candidate_complementarity = current_complementarity
                candidate_tension = current_tension
                candidate_merit = current_merit
                candidate_contact_violation = current_contact_violation
                contact_restoration_accept = False
                contact_gap_restoration_accept = False
                holder_support_restoration_accept = False
                best_line_search = None
                full_step_diagnostics = None
                newton_step_length = 0. if current_is_converged else 1.
                # Near a contact-branch transition the active-set KKT solve
                # can return very large multiplier corrections.  Dampen well
                # below 1/512, but stop at 2**-20: smaller steps are already
                # classified as stagnant below and only repeat effectively
                # identical contact evaluations.  The regularization retry
                # then changes the tangent, and continuation can cut back the
                # punch stroke from the last committed material state.
                for backtrack in range(0 if current_is_converged else 21):
                    newton_step_length = 0.5 ** backtrack
                    candidate = variables + newton_step_length * contact_primal
                    candidate[:basis.N][symmetry_dofs] = 0.
                    candidate_u = candidate[:-1]
                    candidate_h = float(candidate[-1])
                    try:
                        candidate_eval = evaluate_candidate(candidate_u, committed)
                    except (ValueError, np.linalg.LinAlgError):
                        candidate_eval = None
                        continue
                    candidate_gaps, candidate_constraints, candidate_geometry = tool_state(
                        candidate_u, candidate_h, float(stroke)
                    )
                    candidate_minimum_gap = minimum_tool_gap(
                        candidate_gaps, candidate_geometry
                    )
                    candidate_holder_support_rows = holder_rows[
                        candidate_constraints[holder_rows, -1] > 1e-8
                    ]
                    candidate_holder_support_gap = float(np.min(
                        candidate_gaps[
                            candidate_holder_support_rows - equality_count
                        ],
                        initial=np.inf,
                    ))
                    candidate_multipliers = multipliers + newton_step_length * (
                        contact_target_multipliers - multipliers
                    )
                    # Match the contact-reaction projection applied at the
                    # start of the next Newton iteration.  Without this,
                    # the line search evaluates equilibrium with a stale
                    # reaction on a point that has opened; the next iteration
                    # drops that reaction before assembling its equilibrium
                    # residual, so the accepted state can appear to jump by
                    # O(1e2) force units.
                    candidate_retained_gap_limit = retained_contact_gap_limits()
                    candidate_project_to_open = (
                        ~required_active[equality_count:]
                        & (
                            (candidate_gaps > candidate_retained_gap_limit)
                            | (candidate_multipliers[equality_count:] > 1e-2)
                        )
                    )
                    if protected_seed_rows:
                        candidate_project_to_open[
                            np.asarray(protected_seed_rows, dtype=np.int64)
                            - equality_count
                        ] = False
                    candidate_projected_rows = equality_count + np.flatnonzero(
                        candidate_project_to_open
                    )
                    candidate_multipliers[candidate_projected_rows] = 0.
                    candidate_friction_force, _, candidate_friction_multipliers = (
                        friction_response(
                            candidate_u, candidate_h, stroke,
                            friction_base_displacement, friction_base_holder_z,
                            friction_base_stroke, candidate_multipliers,
                            tangent=False,
                        )
                    )
                    stationarity = np.r_[
                        candidate_eval.residual, holder_external_force_per_width
                    ]
                    stationarity += (
                        candidate_constraints.T @ candidate_multipliers
                        + candidate_friction_force
                    )
                    stationarity[symmetry_dofs] = 0.
                    if (backtrack == 0
                            and os.environ.get(
                                "SCIKIT_FVM_FORMING_TRACE", ""
                            ).lower() == "directional"):
                        primal_step = candidate - variables
                        multiplier_step = candidate_multipliers - multipliers
                        predicted_stationarity = (
                            current_stationarity
                            + tangent @ primal_step
                            + constraints.T @ multiplier_step
                        )
                        predicted_stationarity = np.asarray(
                            predicted_stationarity, dtype=float
                        ).reshape(-1)
                        predicted_stationarity[symmetry_dofs] = 0.
                        line_search_linearization_error = (
                            stationarity - predicted_stationarity
                        )
                        worst_linearization_dof = int(np.argmax(np.abs(
                            line_search_linearization_error
                        )))
                        constraint_delta = sparse.csr_matrix(
                            candidate_constraints - constraints
                        )
                        constraint_row_change = np.sqrt(np.asarray(
                            constraint_delta.multiply(constraint_delta)
                            .sum(axis=1)
                        ).reshape(-1))
                        changed_rows = np.argsort(constraint_row_change)[-5:][::-1]
                        changed_row_info = [
                            (
                                int(row),
                                float(constraint_row_change[row]),
                                float(candidate_multipliers[row]),
                                (float(gaps[row - equality_count])
                                 if row >= equality_count else 0.0),
                                [(int(col), float(constraints[row, col]))
                                 for col in np.flatnonzero(constraints[row])],
                                [(int(col), float(candidate_constraints[row, col]))
                                 for col in np.flatnonzero(candidate_constraints[row])],
                            )
                            for row in changed_rows
                        ]
                        print(
                            "[forming-line-search-linearization] "
                            f"stroke={stroke:.9f} iteration={iteration} "
                            f"alpha={newton_step_length:.3e} "
                            f"actual_inf={np.linalg.norm(stationarity, ord=np.inf):.6e} "
                            f"predicted_inf="
                            f"{np.linalg.norm(predicted_stationarity, ord=np.inf):.6e} "
                            f"error_inf="
                            f"{np.linalg.norm(line_search_linearization_error, ord=np.inf):.6e} "
                            f"error_dof={worst_linearization_dof} "
                            f"actual_at_error_dof="
                            f"{stationarity[worst_linearization_dof]:.6e} "
                            f"predicted_at_error_dof="
                            f"{predicted_stationarity[worst_linearization_dof]:.6e} "
                            f"largest_constraint_rows={changed_row_info}",
                            flush=True,
                        )
                    equilibrium = float(np.linalg.norm(stationarity, ord=np.inf))
                    equilibrium_rms = float(
                        np.linalg.norm(stationarity) / np.sqrt(stationarity.size)
                    )
                    penetration = max(
                        -candidate_minimum_gap, 0.
                    )
                    candidate_complementarity = float(np.max(np.abs(
                        candidate_gaps * candidate_multipliers[equality_count:]
                    ), initial=0.))
                    candidate_tension = float(np.max(
                        candidate_multipliers[equality_count:], initial=0.
                    ))
                    candidate_required_active_gap = float(np.max(
                        candidate_gaps[required_active[equality_count:]],
                        initial=0.,
                    ))
                    candidate_contact_violation = max(
                        penetration / max(forming_contact_gap_tolerance_mm, 1e-12),
                        candidate_required_active_gap
                        / max(forming_contact_gap_tolerance_mm, 1e-12),
                        candidate_complementarity
                        / max(forming_complementarity_tolerance, 1e-12),
                        candidate_tension / 1e-2,
                        float(np.max(
                            candidate_multipliers[equality_count:][
                                ~required_active[equality_count:]
                            ], initial=0.
                        )) / 1e-2,
                    )
                    candidate_merit = (
                        float(np.hypot(equilibrium, 2.0 * stationarity[-1]))
                        + penetration_scale * penetration
                        + candidate_complementarity / complementarity_length
                        + candidate_tension
                    )
                    if backtrack == 0:
                        full_step_diagnostics = (
                            equilibrium,
                            candidate_minimum_gap,
                            candidate_complementarity,
                            candidate_merit,
                        )
                    if (best_line_search is None
                            or candidate_merit < best_line_search["merit"]):
                        worst_comp_local = int(np.argmax(np.abs(
                            candidate_gaps * candidate_multipliers[equality_count:]
                        )))
                        worst_comp_row = equality_count + worst_comp_local
                        worst_comp_group = (
                            "punch" if worst_comp_row in punch_rows else
                            "die" if worst_comp_row in die_rows else "holder"
                        )
                        best_line_search = {
                            "alpha": newton_step_length,
                            "merit": candidate_merit,
                            "equilibrium": equilibrium,
                            "equilibrium_rms": equilibrium_rms,
                            "penetration": penetration,
                            "complementarity": candidate_complementarity,
                            "complementarity_contact_group": worst_comp_group,
                            "complementarity_contact_row": worst_comp_row,
                            "complementarity_gap_mm": float(
                                candidate_gaps[worst_comp_local]
                            ),
                            "complementarity_multiplier_N_per_mm": float(
                                candidate_multipliers[worst_comp_row]
                            ),
                            "max_positive_multiplier": float(np.max(
                                candidate_multipliers[equality_count:][
                                    ~required_active[equality_count:]
                                ], initial=0.
                            )),
                        }
                    candidate_is_converged = (
                        equilibrium <= equilibrium_tolerance_for(
                            candidate_multipliers, candidate_constraints
                        )
                        and candidate_minimum_gap
                        >= -forming_contact_gap_tolerance_mm
                        and candidate_required_active_gap
                        <= forming_contact_gap_tolerance_mm
                        and np.max(
                            candidate_multipliers[equality_count:][
                                ~required_active[equality_count:]
                            ], initial=0.
                        ) <= 1e-2
                        and candidate_complementarity <= forming_complementarity_tolerance
                    )
                    # Once contact feasibility is already within its own
                    # tolerances, let Newton reduce force equilibrium even
                    # when the combined merit rises slightly. Near a holder
                    # support transition, accept a Pareto step when it clearly
                    # reduces complementarity without exceeding the force
                    # residual allowance below.
                    candidate_contact_feasible = (
                        candidate_minimum_gap
                        >= -forming_contact_gap_tolerance_mm
                        and candidate_required_active_gap
                        <= forming_contact_gap_tolerance_mm
                        and np.max(
                            candidate_multipliers[equality_count:][
                                ~required_active[equality_count:]
                            ], initial=0.
                        ) <= 1e-2
                        and candidate_complementarity
                        <= forming_complementarity_tolerance
                    )
                    candidate_contact_near_feasible = (
                        candidate_minimum_gap
                        >= -forming_contact_gap_tolerance_mm
                        and candidate_required_active_gap
                        <= forming_contact_gap_tolerance_mm
                        and np.max(
                            candidate_multipliers[equality_count:][
                                ~required_active[equality_count:]
                            ], initial=0.
                        ) <= 1e-2
                        and candidate_complementarity
                        <= 5.0 * forming_complementarity_tolerance
                    )
                    equilibrium_progress = (
                        equilibrium
                        < current_equilibrium
                        - max(
                            1e-8,
                            1e-4 * max(
                                current_equilibrium,
                                current_equilibrium_tolerance,
                            ),
                        )
                    )
                    equilibrium_rms_progress = (
                        equilibrium_rms
                        < current_equilibrium_rms
                        - max(
                            1e-10,
                            1e-4 * max(
                                current_equilibrium_rms,
                                current_equilibrium_tolerance,
                            ),
                        )
                    )
                    equilibrium_rms_guard = (
                        equilibrium
                        <= max(
                            current_equilibrium
                            + 0.1 * current_equilibrium_tolerance,
                            1.05 * current_equilibrium,
                        )
                    )
                    contact_violation_nonworsening = (
                        candidate_contact_violation
                        <= current_contact_violation
                        + max(1e-4, 1e-3 * current_contact_violation)
                    )
                    if current_contact_violation <= 5.0:
                        # Use the same five-times contact band on the current
                        # and trial states. A 1.05 boundary makes the filter
                        # switch modes after one acceptable Newton step and
                        # can alternate forever at a changing support row.
                        # The physical convergence test still enforces the
                        # full complementarity tolerance. Within this wider
                        # band, force progress may not trade away contact
                        # feasibility: recent traces showed complementarity
                        # growing from about 1x to 4x tolerance while Newton
                        # made negligible force progress.
                        contact_feasible_merit_accept = (
                            candidate_contact_near_feasible
                            and equilibrium
                            <= current_equilibrium
                            + 0.1 * current_equilibrium_tolerance
                            and candidate_complementarity
                            < current_complementarity
                            - max(1e-10, 1e-3 * forming_complementarity_tolerance)
                        )
                        equilibrium_filter_accept = (
                            candidate_contact_near_feasible
                            and contact_violation_nonworsening
                            and (
                                equilibrium_progress
                                or (
                                    equilibrium_rms_progress
                                    and equilibrium_rms_guard
                                )
                                or contact_feasible_merit_accept
                            )
                        )
                    else:
                        # While contact is infeasible, do not trade a large
                        # increase in complementarity for a negligible force
                        # reduction. Keep the filter Pareto-like while allowing
                        # a small merit increase from contact-row switching.
                        equilibrium_filter_accept = (
                            equilibrium_progress
                            and contact_violation_nonworsening
                            and candidate_merit
                            <= current_merit
                            + 5e-3 * max(1., current_merit)
                        )
                    # In the restoration phase, prioritize a strict decrease
                    # in normalized contact infeasibility even if the trial
                    # temporarily raises force residual.  Near-grazing tool
                    # contact can require a finite correction before the
                    # equilibrium Newton direction becomes useful again.
                    contact_restoration_accept = (
                        not current_contact_feasible
                        and candidate_contact_violation
                        < current_contact_violation
                        - max(1e-3, 1e-2 * newton_step_length)
                        and equilibrium <= contact_restoration_equilibrium_limit
                        and candidate_merit
                        <= current_merit
                        + 5e-3 * max(1., current_merit)
                    )
                    # The holder reaction is an independent equilibrium
                    # equation. Any filter can otherwise trade its loss for
                    # better tool complementarity. Bound this row separately.
                    holder_residual_guard = (
                        abs(float(stationarity[-1]))
                        <= contact_restoration_holder_residual_limit
                        or (
                            current_contact_feasible
                            and candidate_contact_feasible
                            and equilibrium <= current_equilibrium
                            and (
                                allow_feasible_holder_residual_tradeoff
                                or candidate_merit
                                <= current_merit - max(
                                    1e-8, 1e-4 * max(current_merit, 1.)
                                )
                            )
                        )
                    )
                    # A force-controlled rigid holder can have no closed
                    # contact row while its applied force remains in the
                    # equilibrium equations. In that state equilibrium cannot
                    # improve until the holder reaches the sheet. Permit a
                    # non-committing Newton restoration step that materially
                    # closes the nearest unilateral support gap, provided all
                    # tool gaps remain feasible and neither the force residual
                    # nor contact merit grows appreciably. Plastic history is
                    # still evaluated from ``committed`` and is not stored
                    # until a later iterate satisfies the full equilibrium
                    # and complementarity tests.
                    holder_support_gap_reduction = (
                        current_holder_support_gap
                        - candidate_holder_support_gap
                    )
                    holder_support_restoration_accept = (
                        holder_position_control_mm is None
                        and holder_force_per_width > 0.
                        and current_holder_support_gap
                        > forming_contact_gap_tolerance_mm
                        and holder_support_gap_reduction
                        > max(
                            1e-6,
                            1e-3 * current_holder_support_gap,
                        )
                        and candidate_contact_feasible
                        and equilibrium
                        <= max(
                            current_equilibrium
                            + 0.1 * current_equilibrium_tolerance,
                            1.05 * current_equilibrium,
                        )
                        and abs(float(stationarity[-1]))
                        <= abs(float(current_stationarity[-1]))
                        + max(1e-8, 1e-3 * abs(float(current_stationarity[-1])))
                        and candidate_merit
                        <= current_merit
                        + 5e-3 * max(1., current_merit)
                    )
                    holder_residual_guard = (
                        holder_residual_guard
                        or holder_support_restoration_accept
                    )
                    # A coupled contact Newton direction can close a large
                    # geometric gap while the normal reaction is still
                    # building.  The product gap*force may briefly rise even
                    # as penetration falls sharply; the max-based violation
                    # filter then rejects the only useful correction. Permit
                    # that restoration step when gap closure is material,
                    # complementarity growth is bounded to four times its
                    # current value (with a small absolute floor), and the
                    # separate holder/equilibrium guards pass. The increment still
                    # cannot commit until the strict physical tolerances below
                    # are satisfied.
                    current_penetration = max(-current_minimum_gap, 0.)
                    inactive_multiplier_max = float(np.max(
                        candidate_multipliers[equality_count:][
                            ~required_active[equality_count:]
                        ], initial=0.
                    ))
                    contact_gap_restoration_accept = (
                        not current_contact_feasible
                        and current_penetration
                        > forming_contact_gap_tolerance_mm
                        and candidate_minimum_gap - current_minimum_gap
                        > max(1e-8, 1e-3 * current_penetration)
                        and candidate_complementarity
                        <= max(
                            4. * current_complementarity,
                            10. * forming_complementarity_tolerance,
                        )
                        and candidate_required_active_gap
                        <= forming_contact_gap_tolerance_mm
                        and candidate_tension <= 1e-2
                        and inactive_multiplier_max <= 1e-2
                        and equilibrium <= contact_restoration_equilibrium_limit
                        and holder_residual_guard
                    )
                    # Once both the current and trial states satisfy the
                    # physical gap and complementarity tolerances, permit a
                    # Newton step that makes substantial force-balance
                    # progress even if the normalized contact merit rises.
                    # Near a moving support point that merit can be dominated
                    # by a sub-tolerance gap; insisting that it never grow
                    # shrinks a useful coupled Newton step to machine scale.
                    # The ordinary acceptance checks still reject any trial
                    # outside the contact tolerances or the holder-force
                    # guard, and the increment is committed only after the
                    # full equilibrium test passes.
                    feasible_equilibrium_tradeoff_accept = (
                        current_contact_feasible
                        and candidate_contact_feasible
                        and holder_residual_guard
                        and equilibrium <= current_equilibrium - max(
                            1e-8,
                            feasible_contact_equilibrium_reduction_fraction
                            * current_equilibrium,
                        )
                        and candidate_merit <= current_merit - max(
                            1e-12, 1e-3 * max(current_merit, 1e-12)
                        )
                    )
                    equilibrium_filter_accept = (
                        equilibrium_filter_accept
                        or feasible_equilibrium_tradeoff_accept
                    )
                    # Use a restoration phase for contact before tightening
                    # force balance.  Requiring equilibrium to stay below its
                    # tolerance while the current state penetrates can reject
                    # the full contact correction and accept only microscopic
                    # steps.  The merit still has to decrease, and the guard
                    # limits the temporary force residual; once contact is
                    # feasible, Newton must remain inside the equilibrium
                    # tolerance while reducing its residual.
                    # Once contact is feasible, a state that is still outside
                    # the force-balance tolerance must make strict progress in
                    # equilibrium.  The combined merit can otherwise accept a
                    # step that slightly raises the holder residual merely
                    # because complementarity improves; subsequent Newton
                    # directions then lose descent and collapse to machine
                    # precision at a support transition.
                    force_descent_required = (
                        current_contact_feasible
                        and current_equilibrium
                        > current_equilibrium_tolerance
                    )
                    equilibrium_guard = (
                        current_contact_feasible
                        and not force_descent_required
                        and (
                            equilibrium <= current_equilibrium_tolerance
                            if (current_contact_feasible
                                and current_equilibrium
                                <= current_equilibrium_tolerance)
                            else equilibrium <= current_equilibrium
                            + 5. * current_equilibrium_tolerance
                        )
                    )
                    trace_candidate_from = int(os.environ.get(
                        "SCIKIT_FVM_FORMING_TRACE_CANDIDATES_FROM_ITERATION",
                        "1",
                    ))
                    if (iteration >= trace_candidate_from
                            and os.environ.get(
                                "SCIKIT_FVM_FORMING_TRACE", ""
                            ).lower() == "candidates"):
                        print(
                            "[forming-line-search-candidate] "
                            f"stroke={stroke:.9f} iteration={iteration} "
                            f"backtrack={backtrack} alpha={newton_step_length:.3e} "
                            f"equilibrium={equilibrium:.6e} "
                            f"equilibrium_tolerance="
                            f"{equilibrium_tolerance_for(candidate_multipliers, candidate_constraints):.6e} "
                            f"holder_residual={abs(float(stationarity[-1])):.6e} "
                            f"minimum_gap={candidate_minimum_gap:.6e} "
                            f"current_contact_feasible={current_contact_feasible} "
                            f"current_required_active_gap={current_required_active_gap:.6e} "
                            f"current_complementarity={current_complementarity:.6e} "
                            f"complementarity={candidate_complementarity:.6e} "
                            f"candidate_required_active_gap="
                            f"{candidate_required_active_gap:.6e} "
                            f"candidate_tension={candidate_tension:.6e} "
                            f"candidate_inactive_multiplier_max="
                            f"{float(np.max(candidate_multipliers[equality_count:][~required_active[equality_count:]], initial=0.)):.6e} "
                            f"contact_violation={candidate_contact_violation:.6e} "
                            f"contact_feasible={candidate_contact_feasible} "
                            f"contact_near_feasible={candidate_contact_near_feasible} "
                            f"equilibrium_progress={equilibrium_progress} "
                            f"restoration_accept={contact_restoration_accept} "
                            f"gap_restoration_accept="
                            f"{contact_gap_restoration_accept} "
                            f"holder_support_restoration_accept="
                            f"{holder_support_restoration_accept} "
                            f"equilibrium_filter_accept={equilibrium_filter_accept} "
                            f"feasible_equilibrium_tradeoff_accept="
                            f"{feasible_equilibrium_tradeoff_accept} "
                            f"holder_guard={holder_residual_guard} "
                            f"equilibrium_guard={equilibrium_guard}",
                            flush=True,
                        )
                    if (candidate_is_converged or (
                            holder_residual_guard
                            and (
                                contact_restoration_accept
                                or contact_gap_restoration_accept
                                or holder_support_restoration_accept
                                or equilibrium_filter_accept
                                or (equilibrium_guard and (
                                    candidate_merit < current_merit
                                    - 1e-12 * max(1., current_merit)
                                ))
                            )
                    )):
                        break
                    candidate_eval = None
                if (best_line_search is not None
                        and os.environ.get("SCIKIT_FVM_FORMING_TRACE", "")
                        not in ("", "0")):
                    print(
                        "[forming-line-search] "
                        f"stroke={stroke:.9f} iteration={iteration} "
                        f"accepted_alpha={newton_step_length:.3e} "
                        f"accepted_equilibrium={equilibrium:.4e} "
                        f"accepted_contact_violation="
                        f"{candidate_contact_violation:.4e} "
                        f"best_alpha={best_line_search['alpha']:.3e} "
                        f"best_equilibrium={best_line_search['equilibrium']:.4e} "
                        f"best_merit={best_line_search['merit']:.4e} "
                        f"best_penetration={best_line_search['penetration']:.4e} "
                        f"best_complementarity="
                        f"{best_line_search['complementarity']:.4e} "
                        f"worst_group="
                        f"{best_line_search['complementarity_contact_group']} "
                        f"worst_row="
                        f"{best_line_search['complementarity_contact_row']} "
                        f"worst_gap="
                        f"{best_line_search['complementarity_gap_mm']:.4e} "
                        f"worst_multiplier="
                        f"{best_line_search['complementarity_multiplier_N_per_mm']:.4e} "
                        f"candidate_converged={candidate_is_converged} "
                        f"restoration_accept={contact_restoration_accept} "
                        f"equilibrium_filter_accept={equilibrium_filter_accept} "
                        f"equilibrium_guard={equilibrium_guard} "
                        f"holder_residual_guard={holder_residual_guard} "
                        f"equilibrium_progress={equilibrium_progress} "
                        f"equilibrium_rms={equilibrium_rms:.4e} "
                        f"current_equilibrium_rms={current_equilibrium_rms:.4e} "
                        f"equilibrium_rms_progress={equilibrium_rms_progress} "
                        f"equilibrium_rms_guard={equilibrium_rms_guard} "
                        f"contact_violation_nonworsening="
                        f"{contact_violation_nonworsening} "
                        f"current_contact_violation="
                        f"{current_contact_violation:.4e} "
                        f"restoration_equilibrium_limit="
                        f"{contact_restoration_equilibrium_limit:.4e} "
                        f"contact_violation={candidate_contact_violation:.4e} "
                        f"candidate_holder_residual={stationarity[-1]:.4e} "
                        f"candidate_holder_contact_force="
                        f"{float(candidate_constraints[:, -1] @ candidate_multipliers + candidate_friction_force[-1]):.4e} "
                        f"full_step="
                        f"{full_step_diagnostics if full_step_diagnostics is not None else None} "
                        f"current_merit={current_merit:.4e} "
                        f"contact_step_inf={np.linalg.norm(contact_primal, ord=np.inf):.4e} "
                        f"newton_step_capped={newton_step_was_capped}",
                        flush=True,
                    )
                if candidate_eval is not None and not candidate_is_converged:
                    # A microscopic Armijo step can satisfy the relative
                    # decrease test through roundoff while leaving the
                    # equilibrium iterate effectively unchanged. Let the
                    # regularized tangent retry a few times, then reject this
                    # *stroke increment* so the continuation driver can cut it
                    # back from the last committed material/contact state.
                    merit_decrease = current_merit - candidate_merit
                    merit_floor = 1e-8 * max(1., abs(current_merit))
                    if (newton_step_length < 1e-6
                            or merit_decrease <= merit_floor):
                        stagnant_iterations += 1
                    else:
                        stagnant_iterations = 0
                    if stagnant_iterations >= 12:
                        raise RuntimeError(
                            "contact Newton stagnated for 12 iterations at "
                            f"stroke={stroke:.8g} mm: "
                            f"equilibrium={equilibrium:.3e}, "
                            f"alpha={newton_step_length:.3e}, "
                            f"merit decrease={merit_decrease:.3e}; "
                            "cut back the stroke increment"
                        )
                    relative_merit_decrease = merit_decrease / max(
                        1., abs(current_merit)
                    )
                    if (newton_step_length <= 1. / 256.
                            and relative_merit_decrease < 1e-3):
                        slow_progress_iterations += 1
                    else:
                        slow_progress_iterations = 0
                    # Allow several stronger tangent regularizations to act
                    # before treating small, steady merit reductions as a
                    # reason to cut back the prescribed stroke.  Near contact
                    # transitions the unregularized direction may need the
                    # regularization that follows this check to regain a
                    # useful Newton step.
                    if (slow_progress_iterations >= slow_progress_iteration_limit
                            and regularization_floor[0] >= 1e-8):
                        raise RuntimeError(
                            "contact Newton made only tiny merit reductions "
                            f"for {slow_progress_iterations} iterations at "
                            f"stroke={stroke:.8g} mm: "
                            f"equilibrium={equilibrium:.3e}, "
                            f"alpha={newton_step_length:.3e}, "
                            f"relative merit decrease="
                            f"{relative_merit_decrease:.3e}; "
                            "cut back the stroke increment"
                        )
                    # Once the unilateral constraints are feasible, the only
                    # remaining Newton objective is force equilibrium.  A
                    # merit line search can otherwise alternate contact rows
                    # and accept finite but immaterial steps indefinitely
                    # while the force residual stays flat. Reject this trial
                    # so continuation retries from the committed state with
                    # a smaller stroke; history is still transactional here.
                    if (current_contact_feasible
                            and current_equilibrium
                            > current_equilibrium_tolerance):
                        relative_equilibrium_reduction = (
                            current_equilibrium - equilibrium
                        ) / max(
                            current_equilibrium,
                            current_equilibrium_tolerance,
                        )
                        if relative_equilibrium_reduction < 1e-4:
                            force_stagnation_iterations += 1
                        else:
                            force_stagnation_iterations = 0
                        if (force_stagnation_iterations
                                >= force_stagnation_iteration_limit):
                            raise RuntimeError(
                                "contact Newton made no meaningful force-balance "
                                f"progress for {force_stagnation_iterations} "
                                f"iterations at stroke={stroke:.8g} mm: "
                                f"equilibrium={equilibrium:.3e}, "
                                f"previous={current_equilibrium:.3e}; "
                                "cut back the stroke increment"
                            )
                else:
                    stagnant_iterations = 0
                    slow_progress_iterations = 0
                    force_stagnation_iterations = 0
                if (candidate_eval is not None and not candidate_is_converged
                        and newton_step_length <= 1.25e-1):
                    small_alpha_iterations += 1
                else:
                    small_alpha_iterations = 0
                # Repeatedly damped Armijo steps signal a poor local tangent or
                # a contact branch transition. Raise regularization and retry
                # the same iterate before spending the full Newton budget on
                # nearly unchanged steps.
                if (candidate_eval is not None
                        and not candidate_is_converged
                        and (newton_step_length < 1e-6
                             or (small_alpha_iterations >= 3
                                 and regularization_backoff_hold_iterations <= 0))):
                    regularization_retry_count += 1
                    if regularization_retry_count > 10:
                        raise RuntimeError(
                            "contact Newton exhausted 10 regularization retries "
                            f"at stroke={stroke:.8g} mm without accepting a "
                            f"new iterate: equilibrium={equilibrium:.3e}, "
                            f"alpha={newton_step_length:.3e}; "
                            "cut back the stroke increment"
                        )
                    if regularization_floor[0] >= 1e3:
                        raise RuntimeError(
                            "contact Newton reached the maximum tangent "
                            f"regularization at stroke={stroke:.8g} mm "
                            f"without convergence: equilibrium={equilibrium:.3e}, "
                            f"alpha={newton_step_length:.3e}; "
                            "cut back the stroke increment"
                        )
                    regularization_floor[0] = min(
                        1e3, max(1e-7, 10. * regularization_floor[0])
                    )
                    small_alpha_iterations = 0
                    continue
                if candidate_eval is None:
                    regularization_retry_count += 1
                    if regularization_retry_count > 10:
                        raise RuntimeError(
                            "contact Newton exhausted 10 regularization retries "
                            f"at stroke={stroke:.8g} mm without reducing its "
                            f"merit: current={current_merit:.3e}, "
                            f"best={best_line_search}; "
                            "cut back the stroke increment"
                        )
                    # If no trial satisfies the equilibrium/contact filter,
                    # the current tangent has no usable descent step.  Retry
                    # at the same floor used for repeated microscopic steps;
                    # climbing from 1e-12 by decades can spend many expensive
                    # contact solves before reaching a stabilizing scale.
                    regularization_floor[0] = min(
                        1e3, max(1e-7, 10. * regularization_floor[0])
                    )
                    if regularization_floor[0] < 1e3:
                        # Retry from the same equilibrium iterate with a more
                        # strongly regularized tangent before subdividing the
                        # prescribed stroke increment.  The trial material
                        # state is still based on ``committed`` and remains
                        # uncommitted until the full contact equilibrium passes.
                        continue
                    raise RuntimeError(
                        "contact Newton line search could not reduce equilibrium "
                        "and penetration merit; "
                        f"current={current_merit:.3e}, "
                        f"next_regularization_floor={regularization_floor[0]:.0e}, "
                        f"best={best_line_search}"
                    )
                complementarity = float(np.max(np.abs(
                    candidate_gaps * candidate_multipliers[equality_count:]
                ), initial=0.))
                regularization_floor_lower_bound = max(
                    initial_regularization_floor, saved_regularization_floor,
                    1e-8,
                )
                equilibrium_reduction_fraction = (
                    (current_equilibrium - equilibrium)
                    / max(current_equilibrium, current_equilibrium_tolerance,
                          1e-14)
                )
                if (
                    not candidate_is_converged
                    and "regularized-" in contact_solver_used
                    and newton_step_length >= 1. / 256.
                    and equilibrium_reduction_fraction < 1e-2
                    and regularization_floor[0]
                    > 1.01 * regularization_floor_lower_bound
                ):
                    regularization_slow_progress_iterations += 1
                else:
                    regularization_slow_progress_iterations = 0
                if regularization_slow_progress_iterations >= 4:
                    previous_floor = regularization_floor[0]
                    # The sparse contact fallback tries a discrete sequence
                    # of regularizers (1e-8, then 1e-6, then 1e-4). A single
                    # decade reduction can leave the solver on the same
                    # matrix for several retries, so step down across the
                    # gap between supported levels when physical progress is
                    # persistently small. The line search and contact checks
                    # still decide whether the less-regularized direction is
                    # acceptable.
                    regularization_floor[0] = max(
                        regularization_floor_lower_bound,
                        1e-3 * regularization_floor[0],
                    )
                    regularization_backoff_hold_iterations = 6
                    regularization_slow_progress_iterations = 0
                    print(
                        "[forming-regularization-backoff] "
                        f"stroke={stroke:.9f} iteration={iteration} "
                        f"previous_floor={previous_floor:.1e} "
                        f"next_floor={regularization_floor[0]:.1e} "
                        f"accepted_alpha={newton_step_length:.3e} "
                        f"equilibrium_reduction_fraction="
                        f"{equilibrium_reduction_fraction:.3e}",
                        flush=True,
                    )
                if finite_strain_j2:
                    # At a yield-surface point the consistent material map
                    # has distinct loading and unloading directional
                    # derivatives.  Reuse the accepted Newton direction to
                    # select that generalized derivative on the next
                    # linearization; rejected line-search directions do not
                    # alter this branch hint.
                    material_tangent_direction = contact_primal[:-1].copy()
                variables = candidate
                multipliers = candidate_multipliers
                regularization_backoff_hold_iterations = max(
                    0, regularization_backoff_hold_iterations - 1
                )
                # Once line search has moved the Newton iterate, a later
                # regularization retry starts a new local convergence phase.
                regularization_retry_count = 0
                if (equilibrium <= equilibrium_tolerance_for(
                        multipliers, candidate_constraints
                )
                        and candidate_minimum_gap
                        >= -forming_contact_gap_tolerance_mm
                        and np.max(
                            candidate_gaps[required_active[equality_count:]],
                            initial=0.,
                        ) <= forming_contact_gap_tolerance_mm
                        and np.max(
                            multipliers[equality_count:][
                                ~required_active[equality_count:]
                            ], initial=0.
                        ) <= 1e-2
                        and complementarity <= forming_complementarity_tolerance):
                    converged = True
                    # Keep a stabilizing tangent regularization throughout
                    # this equilibrium solve. A damped or capped correction
                    # can otherwise appear as a full line-search step and
                    # clear the floor, recreating the oversized direction.
                    # The next load increment starts from the configured or
                    # checkpoint floor once this equilibrium is accepted.
                    regularization_floor[0] = max(
                        initial_regularization_floor, saved_regularization_floor
                    )
                    break
            if not converged or contact is None:
                candidate_contact_multipliers = candidate_multipliers[equality_count:]
                candidate_inactive_tension = float(np.max(
                    candidate_contact_multipliers[
                        ~required_active[equality_count:]
                    ], initial=0.
                ))
                candidate_complementarity = float(np.max(np.abs(
                    candidate_gaps * candidate_contact_multipliers
                ), initial=0.))
                worst_comp_local = int(np.argmax(np.abs(
                    candidate_gaps * candidate_contact_multipliers
                )))
                worst_comp_row = equality_count + worst_comp_local
                worst_comp_group = (
                    "punch" if worst_comp_row in punch_rows else
                    "die" if worst_comp_row in die_rows else "holder"
                )
                raise RuntimeError(
                    f"contact Newton failed after {contact_newton_iteration_limit} "
                    f"iterations at stroke={stroke:.6g} mm: "
                    f"equilibrium={equilibrium:.3e}, "
                    f"minimum_gap={candidate_minimum_gap:.3e}, "
                    f"complementarity={candidate_complementarity:.3e}, "
                    f"max_inactive_tension={candidate_inactive_tension:.3e}, "
                    f"worst_complementarity_contact={worst_comp_group}"
                    f"[{worst_comp_row}] gap="
                    f"{candidate_gaps[worst_comp_local]:.3e} mm, multiplier="
                    f"{candidate_contact_multipliers[worst_comp_local]:.3e} N/mm"
                )
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
            failure_log.append({
                "requested_stroke_mm": float(stroke),
                "last_accepted_stroke_mm": float(accepted_stroke),
                "subdivision_depth": int(subdivision_depth),
                "reason": str(error),
            })
            if len(failure_log) > 200:
                del failure_log[:-200]
            if (subdivision_depth >= 12
                    or abs(stroke - accepted_stroke) <= 1e-12):
                save_checkpoint()
                raise RuntimeError(
                    f"contact increment to {stroke:.8g} mm failed after "
                    f"{subdivision_depth} subdivisions: {error}"
                ) from error
            regularization_floor[0] = 0.
            # Persist the unchanged accepted solution together with this
            # rejected trial before recursively subdividing. Long contact
            # branch searches otherwise provide no checkpoint progress until
            # an intermediate stroke is accepted.
            save_checkpoint()
            midpoint = 0.5 * (accepted_stroke + stroke)
            advance_to(midpoint, subdivision_depth + 1)
            # Return after reaching an intermediate equilibrium.  The outer
            # adaptive controller can then carry this smaller step forward
            # instead of repeatedly rediscovering the same bisection.
            return
        trial_displacement = variables[:-1].copy()
        trial_holder_z = float(variables[-1])
        final_gaps, final_constraints, final_geometry = tool_state(
            trial_displacement, trial_holder_z, float(stroke)
        )
        final_minimum_gap = minimum_tool_gap(final_gaps, final_geometry)
        punch_vertical_jacobian = np.asarray(
            final_constraints[punch_rows][:, dof_z].sum(axis=1)
        ).reshape(-1)
        punch_tangent_vertical_jacobian = np.asarray(
            final_constraints[punch_rows][:, dof_x].sum(axis=1)
        ).reshape(-1)
        _, _, trial_friction_multipliers = friction_response(
            trial_displacement, trial_holder_z, float(stroke),
            friction_base_displacement, friction_base_holder_z,
            friction_base_stroke, multipliers, tangent=False,
        )
        normal_punch_force = float(np.sum(
            multipliers[punch_rows] * punch_vertical_jacobian
        ) * 5.0)
        friction_punch_force = float(np.sum(
            trial_friction_multipliers[punch_rows]
            * punch_tangent_vertical_jacobian
        ) * 5.0)
        # The punch reaction includes both the normal and tangential contact
        # tractions. For t=(-n_z, n_x), the vertical tangential contribution
        # is tau*n_x, represented by the x-gradient of the normal gap above.
        punch_force = normal_punch_force + friction_punch_force
        force_resolution_unresolved = False
        force_resolution_unresolved_for_requested_tolerance = False
        holder_jump_unresolved = False
        punch_detachment_event = False
        relative_force_change = 0.0
        relative_force_prediction_error = 0.0
        holder_jump = abs(trial_holder_z - holder_z)
        holder_jump_exceeds_tolerance = (
            holder_position_control_mm is None
            and holder_jump > holder_jump_tolerance_mm
        )
        if records:
            previous_punch_force = float(
                records[-1]["punch_force_N_half_model"]
            )
            relative_force_change = abs(punch_force - previous_punch_force) / max(
                abs(punch_force), abs(previous_punch_force),
                0.05 * punch_force_change_reference_N, 1.0,
            )
            relative_force_prediction_error = relative_force_change
            if len(records) >= 2:
                previous_record = records[-2]
                previous_interval = (
                    float(records[-1]["stroke_mm"])
                    - float(previous_record["stroke_mm"])
                )
                if previous_interval > 1e-14:
                    previous_force_slope = (
                        previous_punch_force
                        - float(previous_record["punch_force_N_half_model"])
                    ) / previous_interval
                    predicted_punch_force = (
                        previous_punch_force + previous_force_slope
                        * (stroke - float(records[-1]["stroke_mm"]))
                    )
                    relative_force_prediction_error = abs(
                        punch_force - predicted_punch_force
                    ) / max(
                        abs(punch_force), abs(predicted_punch_force),
                        0.05 * punch_force_change_reference_N, 1.0,
                    )
            if (relative_force_prediction_error
                    > forming_force_change_tolerance
                    or holder_jump_exceeds_tolerance):
                reasons = []
                if (relative_force_prediction_error
                        > forming_force_change_tolerance):
                    reasons.append(
                        "punch-force continuation cutback: secant-prediction "
                        f"error {relative_force_prediction_error:.3e} exceeds "
                        f"{forming_force_change_tolerance:.3e}"
                    )
                if holder_jump_exceeds_tolerance:
                    reasons.append(
                        "holder branch continuation cutback: position jump "
                        f"{holder_jump:.6g} mm exceeds "
                        f"{holder_jump_tolerance_mm:.6g} mm"
                    )
                reason = "; ".join(reasons) + (
                    f" over {stroke - accepted_stroke:.6g} mm"
                )
                failure_log.append({
                    "requested_stroke_mm": float(stroke),
                    "last_accepted_stroke_mm": float(accepted_stroke),
                    "subdivision_depth": int(subdivision_depth),
                    "reason": reason,
                    "punch_force_relative_change": float(relative_force_change),
                    "punch_force_prediction_error_relative": float(
                        relative_force_prediction_error
                    ),
                    "holder_jump_mm": float(holder_jump),
                })
                if len(failure_log) > 200:
                    del failure_log[:-200]
                if (subdivision_depth < forming_force_change_max_subdivisions
                        and abs(stroke - accepted_stroke)
                        > force_resolution_floor_mm):
                    regularization_floor[0] = 0.
                    midpoint = 0.5 * (accepted_stroke + stroke)
                    advance_to(midpoint, subdivision_depth + 1)
                    return
                punch_gap = float(np.min(final_gaps[:punch_rows.size]))
                if (previous_punch_force > punch_force_change_reference_N
                        and punch_force <= 0.05 * previous_punch_force
                        and punch_gap > 0.1 * thickness
                        and holder_jump > 0.25 * thickness):
                    # A detached state is admissible for frictionless
                    # unilateral contact.  The equilibrium checks above
                    # establish that this is a valid contact branch; keep the
                    # state and expose the snap/detachment so the load--stroke
                    # result can be reviewed instead of stalling continuation.
                    punch_detachment_event = True
                    failure_log[-1]["action"] = (
                        "accepted_equilibrated_punch_detachment_with_holder_shift"
                    )
                # Point-collocation changes and physical contact loss both
                # appear as force jumps.  Once the strict equilibrium, gap,
                # and complementarity checks above pass, retain the branch
                # event and record whether it includes full punch detachment.
                force_resolution_unresolved = (
                    relative_force_prediction_error
                    > forming_force_change_tolerance
                )
                force_resolution_unresolved_for_requested_tolerance = (
                    force_resolution_unresolved
                )
                holder_jump_unresolved = holder_jump_exceeds_tolerance
                if not punch_detachment_event:
                    if force_resolution_unresolved:
                        failure_log[-1]["action"] = (
                            "accepted_converged_equilibrium_with_unresolved_"
                            "force_jump"
                        )
                if holder_jump_unresolved:
                    failure_log[-1]["holder_jump_action"] = (
                        "accepted_equilibrated_holder_branch_jump_at_"
                        "resolution_floor"
                    )

        previous_displacement = displacement.copy()
        previous_holder_z = float(holder_z)
        previous_stroke = float(accepted_stroke)
        displacement = trial_displacement
        holder_z = trial_holder_z
        committed = assembler.commit(candidate_eval)
        committed_contact_multipliers = multipliers.copy()
        _, _, committed_friction_multipliers = friction_response(
            displacement, holder_z, float(stroke), friction_base_displacement,
            friction_base_holder_z, friction_base_stroke, multipliers,
            tangent=False,
        )
        active_friction_capacity = friction_coefficient * np.maximum(
            -multipliers, 0.
        )
        friction_utilization = np.divide(
            np.abs(committed_friction_multipliers), active_friction_capacity,
            out=np.zeros_like(active_friction_capacity),
            where=active_friction_capacity > 1e-12,
        )
        accepted_stroke = float(stroke)
        # ``contact.active_rows`` is the active set for the target
        # linearization, while ``multipliers`` and ``candidate_gaps`` belong
        # to the accepted (possibly damped) Newton state.  During support
        # transfers these differ for an accepted increment: a compressive
        # reaction may be retained until its gap opens far enough even after
        # the target set has released that sample.  Derive reported physical
        # contact from the accepted state that produced the recorded force,
        # and keep target counts separately for solver diagnostics.
        accepted_contact_active = (
            (multipliers[equality_count:] < -1e-2)
            & (candidate_gaps <= forming_contact_gap_tolerance_mm)
        )
        accepted_punch_active = accepted_contact_active[
            punch_rows - equality_count
        ]
        accepted_die_active = accepted_contact_active[
            die_rows - equality_count
        ]
        accepted_holder_active = accepted_contact_active[
            holder_rows - equality_count
        ]
        active_punch = accepted_punch_active
        if (isinstance(final_geometry, tuple) and len(final_geometry) == 2
                and final_geometry[0] == "mortar-p0"):
            punch_eval, die_eval, holder_eval = final_geometry[1]
            current_punch_points = punch_eval.facet_points
            current_die_points = die_eval.facet_points
            current_holder_points = holder_eval.facet_points
        else:
            current_punch_points = sample_coordinates(punch, displacement)
            current_die_points = sample_coordinates(die, displacement)
            current_holder_points = sample_coordinates(holder, displacement)
        punch_projection = tools["punch"].translated(
            [0., -float(stroke)]
        ).project(current_punch_points)
        die_projection = tools["die"].project(current_die_points)
        holder_projection = tools["blank_holder"].translated(
            [0., holder_z]
        ).project(current_holder_points)
        active_die = accepted_die_active
        active_holder = accepted_holder_active
        die_vertical_jacobian = np.asarray(
            final_constraints[die_rows][:, dof_z].sum(axis=1)
        ).reshape(-1)
        holder_vertical_jacobian = np.asarray(
            final_constraints[holder_rows][:, dof_z].sum(axis=1)
        ).reshape(-1)
        punch_gaps = candidate_gaps[:punch_rows.size]
        near_punch = punch_gaps <= 1e-6
        records.append({
            "step": len(records), "stroke_mm": float(stroke), "punch_force_N_half_model": punch_force,
            "normal_punch_force_N_half_model": normal_punch_force,
            "friction_punch_force_N_half_model": friction_punch_force,
            "holder_position_mm": holder_z, "newton_iterations": iteration,
            "newton_step_length": newton_step_length,
            "contact_solver": contact_solver_used,
            "active_punch_points": int(np.count_nonzero(accepted_punch_active)),
            "linearized_active_punch_points": int(np.count_nonzero(
                contact.active_rows[punch_rows]
            )),
            "zero_reaction_contact_seeded": bool(
                records
                and records[-1].get("zero_reaction_contact_seeded", False)
                and abs(punch_force) <= 1e-12
            ),
            "active_punch_points_xz_mm": current_punch_points[active_punch].tolist(),
            "active_punch_contact_row_indices": np.flatnonzero(
                active_punch
            ).tolist(),
            "active_punch_contact_multipliers_N_per_mm": multipliers[
                punch_rows
            ][active_punch].tolist(),
            "active_punch_contact_gaps_mm": punch_gaps[active_punch].tolist(),
            "active_punch_contact_normals": punch_projection.normals[
                active_punch
            ].tolist(),
            "active_punch_profile_segments": punch_projection.segment_indices[
                active_punch
            ].tolist(),
            "normal_die_force_N_half_model": float(np.sum(
                multipliers[die_rows] * die_vertical_jacobian
            ) * 5.0),
            "active_die_points_xz_mm": current_die_points[active_die].tolist(),
            "active_die_contact_row_indices": np.flatnonzero(
                active_die
            ).tolist(),
            "active_die_contact_multipliers_N_per_mm": multipliers[
                die_rows
            ][active_die].tolist(),
            "active_die_contact_gaps_mm": candidate_gaps[
                die_rows - equality_count
            ][active_die].tolist(),
            "active_die_profile_segments": die_projection.segment_indices[
                active_die
            ].tolist(),
            "minimum_punch_gap_mm": float(np.min(punch_gaps, initial=np.inf)),
            "near_contact_punch_points_xz_mm": current_punch_points[near_punch].tolist(),
            "active_die_points": int(np.count_nonzero(accepted_die_active)),
            "linearized_active_die_points": int(np.count_nonzero(
                contact.active_rows[die_rows]
            )),
            "active_holder_points": int(np.count_nonzero(accepted_holder_active)),
            "linearized_active_holder_points": int(np.count_nonzero(
                contact.active_rows[holder_rows]
            )),
            "normal_holder_force_N_half_model": float(np.sum(
                multipliers[holder_rows] * holder_vertical_jacobian
            ) * 5.0),
            "holder_control_constraint_reaction_N_half_model": (
                None if holder_position_equality_row is None else float(
                    multipliers[holder_position_equality_row] * modeled_width
                )
            ),
            "active_holder_points_xz_mm": current_holder_points[
                active_holder
            ].tolist(),
            # Persist row identity as well as geometric feature identity.
            # This distinguishes one material contact sample crossing a
            # line/arc join from the active support transferring to a
            # different collocation row during holder-load branch changes.
            "active_holder_contact_row_indices": np.flatnonzero(
                active_holder
            ).tolist(),
            "active_holder_contact_multipliers_N_per_mm": multipliers[
                holder_rows
            ][active_holder].tolist(),
            "active_holder_contact_normals": holder_projection.normals[
                active_holder
            ].tolist(),
            "active_holder_contact_gaps_mm": final_gaps[
                holder_rows[active_holder] - equality_count
            ].tolist(),
            "active_holder_profile_segments": holder_projection.segment_indices[
                active_holder
            ].tolist(),
            "maximum_contact_penetration_mm": max(
                -final_minimum_gap, 0.
            ),
            "contact_complementarity_N_mm": float(np.max(np.abs(
                candidate_gaps * multipliers[equality_count:]
            ), initial=0.)),
            "holder_force_balance_residual_N": float(stationarity[-1]),
            "active_friction_points": int(np.count_nonzero(
                np.abs(committed_friction_multipliers) > 1e-10
            )),
            "maximum_tangential_contact_force_N": float(
                np.max(np.abs(committed_friction_multipliers), initial=0.)
            ),
            "maximum_coulomb_utilization": float(
                np.max(friction_utilization, initial=0.)
            ),
            "equilibrium_residual_inf_N_per_mm": equilibrium,
            "equilibrium_tolerance_inf_N_per_mm": equilibrium_tolerance_for(
                multipliers, final_constraints
            ),
            "contact_gap_tolerance_mm": forming_contact_gap_tolerance_mm,
            "contact_complementarity_tolerance_N_mm": (
                forming_complementarity_tolerance
            ),
            "punch_force_increment_relative_tolerance": (
                forming_force_change_tolerance
            ),
            "punch_force_increment_scale_floor_N": (
                0.05 * punch_force_change_reference_N
            ),
            "force_resolution_unresolved": force_resolution_unresolved,
            "force_resolution_unresolved_for_requested_tolerance": (
                force_resolution_unresolved_for_requested_tolerance
            ),
            "holder_jump_unresolved": holder_jump_unresolved,
            "punch_detachment_event": punch_detachment_event,
            "punch_force_relative_change": relative_force_change,
            "punch_force_prediction_error_relative": (
                relative_force_prediction_error
            ),
            "holder_jump_mm": holder_jump,
            "holder_jump_tolerance_mm": holder_jump_tolerance_mm,
            "force_resolution_floor_mm": (
                force_resolution_floor_mm if force_resolution_unresolved else 0.0
            ),
            "holder_jump_resolution_floor_mm": (
                force_resolution_floor_mm if holder_jump_unresolved else 0.0
            ),
            "max_equivalent_plastic_strain": float(
                np.max(committed[..., equivalent_plastic_strain_index])
            ),
        })
        checkpoint_state_current_layout = True
        save_checkpoint()
        print(
            "accepted forming step "
            f"{len(records) - 1}: stroke={accepted_stroke:.6f} mm, "
            f"punch_force={punch_force:.3f} N/half-model, "
            f"newton={iteration}, equilibrium={equilibrium:.3e} N/mm, "
            f"backend={contact_solver_used}, "
            f"unresolved_force_jump={force_resolution_unresolved}",
            flush=True,
        )

    if not records:
        advance_to(0.0)
    if maximum_stroke_mm < accepted_stroke - 1e-10:
        raise ValueError(
            f"checkpoint already reached {accepted_stroke:g} mm, beyond requested "
            f"maximum stroke {maximum_stroke_mm:g} mm"
        )
    # Permit continuation to follow very small but converged increments near
    # contact transitions.  Clamping these accepted cutbacks to a coarse floor
    # repeatedly reattempts a known-failing stroke and can stall the path.
    minimum_increment_mm = min(maximum_increment_mm / 4096., 1e-8)
    punch_withdrawal_active = bool(int(checkpoint_extra.get(
        "punch_withdrawal_active", np.asarray(0)
    ).item()))
    punch_withdrawal_target_stroke = float(checkpoint_extra.get(
        "punch_withdrawal_target_stroke_mm", np.asarray(0.)
    ).item())
    punch_withdrawal_completed = (
        not punch_withdrawal_active
        and "punch_withdrawal_target_stroke_mm" in checkpoint_extra
    )

    if resume and (re_equilibrate_checkpoint or (
            holder_position_control_mm is not None
            and abs(maximum_stroke_mm - accepted_stroke) <= 1e-10)):
        if springback_checkpoint_active:
            raise ValueError(
                "checkpoint re-equilibration requires a formed state before springback"
            )
        # A transferred mesh may add unconverged FE modes even though the
        # displacement and material history were prolonged consistently.
        # Solve at the saved stroke first so the next punch increment starts
        # from equilibrium on the new mesh.  Position-control diagnostics
        # also use this path when the requested stroke is unchanged.
        advance_to(accepted_stroke)

    def minimum_closing_punch_gap(stroke: float) -> float:
        """Return the smallest punch gap whose normal closes under downstroke.

        A blank point sliding along the punch's vertical flank can remain at
        zero gap while the punch moves tangentially.  Treating that as a new
        normal-contact event repeatedly seeds tiny Newton increments, even
        though the stroke does not close the gap.  The signed gap derivative
        with respect to downward punch stroke is the projected normal's Z
        component, so only negative components are closing.
        """
        if contact_method == "mortar-p0":
            evaluation = punch_mortar.evaluate(
                np.r_[displacement, holder_z], tools["punch"],
                translation=[0., -stroke],
                curvature_blend_length=punch_contact_curvature_blend_length_mm,
            )
            closing = evaluation.sample_normals[:, 1] < -1e-3
            return float(np.min(evaluation.sample_gaps[closing], initial=np.inf))
        points = sample_coordinates(punch, displacement)
        projection = tools["punch"].translated([0., -stroke]).project(points)
        closing = projection.normals[:, 1] < -1e-3
        return float(np.min(projection.gaps[closing], initial=np.inf))

    while (not punch_withdrawal_active
            and accepted_stroke < maximum_stroke_mm - 1e-10):
        start_stroke = accepted_stroke
        first_new_record = len(records)
        target_stroke = min(
            maximum_stroke_mm, start_stroke + suggested_increment_mm
        )
        punch_reaction_free = (
            records
            and abs(records[-1].get("punch_force_N_half_model", 0.)) <= 1e-12
            and not records[-1].get("zero_reaction_contact_seeded", False)
            and np.all(np.abs(committed_contact_multipliers[punch_rows]) <= 1e-10)
            and np.all(np.abs(committed_friction_multipliers[punch_rows]) <= 1e-10)
        )
        seeded_zero_reaction_contact = False
        if punch_reaction_free and target_stroke > start_stroke:
            current_closing_gap = minimum_closing_punch_gap(start_stroke)
            if abs(current_closing_gap) <= forming_contact_gap_tolerance_mm:
                # Seed contact only inside the geometric deadband.  Bisection
                # is unreliable when the current state is already within the
                # admissible gap tolerance: it can repeatedly return the same
                # stroke and prevent continuation from advancing.
                contact_seed_increment = max(
                    minimum_increment_mm, maximum_increment_mm / 128.
                )
                target_stroke = min(
                    maximum_stroke_mm, start_stroke + contact_seed_increment
                )
                seeded_zero_reaction_contact = True
            elif current_closing_gap > forming_contact_gap_tolerance_mm:
                target_closing_gap = minimum_closing_punch_gap(target_stroke)
                if target_closing_gap <= 1e-6:
                    # Locate first punch contact in stroke space so Newton sees
                    # an active row at nearly zero gap, rather than a large
                    # penetration caused by a coarse continuation increment.
                    lower, upper = float(start_stroke), float(target_stroke)
                    for _ in range(40):
                        middle = 0.5 * (lower + upper)
                        if minimum_closing_punch_gap(middle) > 1e-8:
                            lower = middle
                        else:
                            upper = middle
                    target_stroke = upper
        try:
            advance_to(target_stroke)
            if (seeded_zero_reaction_contact and records
                    and abs(records[-1].get("punch_force_N_half_model", 0.))
                    <= 1e-12):
                records[-1]["zero_reaction_contact_seeded"] = True
        except RuntimeError:
            regularization_floor[0] = 0.
            if accepted_stroke > start_stroke + 1e-12 and len(records) > first_new_record:
                previous_record_stroke = (
                    records[-2]["stroke_mm"] if len(records) >= 2 else start_stroke
                )
                last_accepted_increment = accepted_stroke - previous_record_stroke
                suggested_increment_mm = min(
                    0.5 * suggested_increment_mm,
                    1.5 * last_accepted_increment,
                )
            else:
                suggested_increment_mm *= 0.5
            if suggested_increment_mm < minimum_increment_mm:
                save_checkpoint()
                raise
            save_checkpoint()
            continue

        newly_accepted = records[first_new_record:]
        if accepted_stroke < target_stroke - 1e-10:
            previous_record_stroke = (
                records[-2]["stroke_mm"] if len(records) >= 2 else start_stroke
            )
            last_accepted_increment = accepted_stroke - previous_record_stroke
            suggested_increment_mm = min(
                maximum_increment_mm,
                max(minimum_increment_mm, 1.5 * last_accepted_increment),
            )
        elif len(newly_accepted) > 1:
            last_accepted_increment = (
                newly_accepted[-1]["stroke_mm"]
                - newly_accepted[-2]["stroke_mm"]
            )
            suggested_increment_mm = min(
                maximum_increment_mm,
                max(minimum_increment_mm, 1.5 * last_accepted_increment),
            )
        elif newly_accepted:
            previous_record_stroke = (
                records[first_new_record - 1]["stroke_mm"]
                if first_new_record else start_stroke
            )
            last_accepted_increment = (
                accepted_stroke - previous_record_stroke
            )
            previous_maximum_plastic_strain = (
                records[first_new_record - 1]["max_equivalent_plastic_strain"]
                if first_new_record else 0.
            )
            plastic_increment = (
                newly_accepted[-1]["max_equivalent_plastic_strain"]
                - previous_maximum_plastic_strain
            )
            last_record = newly_accepted[-1]
            # Semismooth contact active sets can require more than 16 Newton
            # iterations at a geometry transition even when the accepted
            # equilibrium is accurate.  Shrinking on that threshold alone
            # trapped this run at the minimum stroke increment.  Keep a wider
            # hysteresis band; actual convergence, gap, and complementarity
            # checks still decide whether an increment is accepted.
            if (last_record["newton_iterations"] >= 48
                    or plastic_increment >= maximum_plastic_increment_per_step):
                suggested_increment_mm = max(
                    minimum_increment_mm, 0.75 * suggested_increment_mm
                )
            if (last_record["newton_iterations"] <= 40
                    and plastic_increment <= 0.5 * maximum_plastic_increment_per_step
                    and not last_record.get(
                        "force_resolution_unresolved", False
                    )
                    and not last_record.get("holder_jump_unresolved", False)):
                # Recover after an accepted low-effort increment, but base
                # the ramp on the increment that actually converged.  The
                # stored suggestion can be much larger after an external
                # maximum-stroke clamp; ramping that stale value immediately
                # after a tiny accepted step recreates the contact jump that
                # forced the cutback.
                suggested_increment_mm = min(
                    maximum_increment_mm,
                    max(
                        minimum_increment_mm,
                        forming_increment_recovery_factor * min(
                            suggested_increment_mm,
                            last_accepted_increment,
                        ),
                    ),
                )
            if (abs(last_record["punch_force_N_half_model"]) <= 1e-12
                    and np.all(np.abs(
                        committed_contact_multipliers[punch_rows]
                    ) <= 1e-10)
                    and np.all(np.abs(
                        committed_friction_multipliers[punch_rows]
                    ) <= 1e-10)):
                suggested_increment_mm = maximum_increment_mm
        if (suggested_increment_mm < forming_increment_floor_mm
                or any(record.get("force_resolution_unresolved", False)
                       or record.get("holder_jump_unresolved", False)
                       for record in records[-3:])):
            suggested_increment_mm = max(
                suggested_increment_mm, forming_increment_floor_mm
            )
        save_checkpoint()

    if (withdraw_punch_after_forming and not punch_withdrawal_active
            and not punch_withdrawal_completed
            and accepted_stroke >= maximum_stroke_mm - 1e-10):
        punch_withdrawal_target_stroke = float(accepted_stroke)
        punch_withdrawal_active = True
        checkpoint_extra.update({
            "punch_withdrawal_active": np.asarray(1, dtype=np.int8),
            "punch_withdrawal_target_stroke_mm": np.asarray(
                punch_withdrawal_target_stroke
            ),
        })
        save_checkpoint()
    if punch_withdrawal_active:
        while accepted_stroke > 1e-10:
            withdrawal_increment = min(
                maximum_increment_mm,
                max(minimum_increment_mm, suggested_increment_mm),
            )
            target_stroke = max(0., accepted_stroke - withdrawal_increment)
            first_withdrawal_record = len(records)
            try:
                advance_to(target_stroke)
            except RuntimeError:
                suggested_increment_mm = max(
                    minimum_increment_mm, 0.5 * withdrawal_increment
                )
                save_checkpoint()
                if suggested_increment_mm <= minimum_increment_mm:
                    raise
                continue

            # A failed withdrawal increment is cut back above. Recover the
            # step size after easy, resolved unloading increments so one
            # difficult contact transition does not pin the entire return
            # path to its smallest accepted step.
            new_withdrawal_records = records[first_withdrawal_record:]
            if new_withdrawal_records:
                last_record = new_withdrawal_records[-1]
                previous_record = records[-2] if len(records) >= 2 else None
                if previous_record is not None:
                    last_accepted_increment = abs(
                        float(last_record["stroke_mm"])
                        - float(previous_record["stroke_mm"])
                    )
                    plastic_increment = abs(
                        float(last_record.get(
                            "max_equivalent_plastic_strain", 0.
                        ))
                        - float(previous_record.get(
                            "max_equivalent_plastic_strain", 0.
                        ))
                    )
                    unresolved = (
                        last_record.get(
                            "force_resolution_unresolved_for_requested_tolerance",
                            last_record.get("force_resolution_unresolved", False),
                        )
                        or last_record.get("holder_jump_unresolved", False)
                    )
                    if len(new_withdrawal_records) > 1:
                        suggested_increment_mm = min(
                            maximum_increment_mm,
                            max(
                                minimum_increment_mm,
                                forming_increment_recovery_factor
                                * last_accepted_increment,
                            ),
                        )
                    elif (last_record.get("newton_iterations", 0) >= 48
                          or plastic_increment
                          >= maximum_plastic_increment_per_step):
                        suggested_increment_mm = max(
                            minimum_increment_mm,
                            0.75 * withdrawal_increment,
                        )
                    elif (last_record.get("newton_iterations", 0) <= 40
                          and plastic_increment
                          <= 0.5 * maximum_plastic_increment_per_step
                          and not unresolved):
                        suggested_increment_mm = min(
                            maximum_increment_mm,
                            max(
                                minimum_increment_mm,
                                forming_increment_recovery_factor
                                * min(
                                    suggested_increment_mm,
                                    last_accepted_increment,
                                ),
                            ),
                        )
                    save_checkpoint()
        # ``accepted_stroke`` describes the maximum formed stroke for the
        # result/checkpoint API. The accepted records retain the descending
        # tool-withdrawal path down to zero.
        accepted_stroke = punch_withdrawal_target_stroke
        previous_displacement = displacement.copy()
        previous_holder_z = float(holder_z)
        previous_stroke = accepted_stroke
        punch_withdrawal_active = False
        punch_withdrawal_completed = True
        checkpoint_extra["punch_withdrawal_active"] = np.asarray(0, dtype=np.int8)
        save_checkpoint()

    def flange_angle(u):
        midplane = np.flatnonzero(np.isclose(node_z, 0.))
        ordered = midplane[np.argsort(node_x[midplane])]
        flange_nodes = ordered[-min(3, ordered.size):]
        x = node_x[flange_nodes] + u[dof_x[flange_nodes]]
        z = node_z[flange_nodes] + u[dof_z[flange_nodes]]
        slope = float(np.polyfit(x, z, 1)[0]) if flange_nodes.size >= 2 else float("nan")
        return float(np.degrees(np.arctan(slope)))

    def thickness_profile_for(u):
        profile = []
        for x_value in x_nodes:
            column = np.flatnonzero(np.isclose(node_x, x_value))
            top_node = column[np.argmax(node_z[column])]
            bottom_node = column[np.argmin(node_z[column])]
            top_position = np.array((
                node_x[top_node] + u[dof_x[top_node]],
                node_z[top_node] + u[dof_z[top_node]],
            ))
            bottom_position = np.array((
                node_x[bottom_node] + u[dof_x[bottom_node]],
                node_z[bottom_node] + u[dof_z[bottom_node]],
            ))
            profile.append((float(x_value), float(np.linalg.norm(
                top_position - bottom_position
            ))))
        return profile

    def centerline_thickness(profile):
        return min(profile, key=lambda item: abs(item[0]))[1]

    if forming_only:
        last_record = records[-1]
        formed_thickness_profile = thickness_profile_for(displacement)
        formed_minimum_thickness_x, formed_minimum_thickness = min(
            formed_thickness_profile, key=lambda item: item[1]
        )
        return {
            "problem": "NUMISHEET '93 profile contact, forming-only run",
            "kinematics": "finite Hencky J2" if finite_strain_j2 else "small strain",
            "material_model": "isotropic J2" if finite_strain_j2 else "Hill48",
            "hardening_reference": (
                "Puso UCRL-MA-140587 isotropic law"
                if puso_isotropic_hardening else "official Abaqus deck"
            ),
            "friction_coefficient": float(friction_coefficient),
            "holder_position_control_mm": holder_position_control_mm,
            "maximum_stroke_mm": maximum_stroke_mm,
            "maximum_increment_mm": maximum_increment_mm,
            "forming_increment_continuation_floor_mm": (
                forming_increment_floor_mm
            ),
            "holder_jump_tolerance_mm": holder_jump_tolerance_mm,
            "unresolved_holder_jump_count": sum(
                bool(record.get("holder_jump_unresolved", False))
                for record in records
            ),
            "unresolved_force_jump_count": sum(
                bool(record.get(
                    "force_resolution_unresolved_for_requested_tolerance",
                    record.get("force_resolution_unresolved", False),
                ))
                for record in records
            ),
            "forming_force_stroke_curve_resolved": not any(
                record.get(
                    "force_resolution_unresolved_for_requested_tolerance",
                    record.get("force_resolution_unresolved", False),
                )
                or record.get("holder_jump_unresolved", False)
                for record in records
            ),
            "forming_tolerance_history_complete": all(
                "contact_complementarity_tolerance_N_mm" in record
                and "contact_gap_tolerance_mm" in record
                and "punch_force_increment_relative_tolerance" in record
                for record in records
            ),
            "forming_force_resolution_floor_mm": (
                force_resolution_floor_mm
            ),
            "forming_equilibrium_relative_tolerance": (
                forming_equilibrium_relative_tolerance
            ),
            "requested_benchmark_stroke_mm": float(
                reference["process"]["punch_stroke"]
            ),
            "accepted_stroke_mm": float(accepted_stroke),
            "forming_steps": len(records),
            "punch_withdrawal_completed": punch_withdrawal_completed,
            "punch_force_by_stroke": [
                [r["stroke_mm"], r["punch_force_N_half_model"]] for r in records
            ],
            "normal_punch_force_by_stroke": [
                [
                    r["stroke_mm"],
                    r.get(
                        "normal_punch_force_N_half_model",
                        r["punch_force_N_half_model"],
                    ),
                ]
                for r in records
            ],
            "friction_punch_force_by_stroke": [
                [r["stroke_mm"], r.get("friction_punch_force_N_half_model", 0.0)]
                for r in records
            ],
            "normal_holder_force_by_stroke": [
                [r["stroke_mm"], r.get("normal_holder_force_N_half_model", 0.0)]
                for r in records
            ],
            "holder_control_constraint_reaction_by_stroke": [
                [
                    r["stroke_mm"],
                    r.get("holder_control_constraint_reaction_N_half_model"),
                ]
                for r in records
            ],
            "maximum_punch_force_N_half_model": max(
                r["punch_force_N_half_model"] for r in records
            ),
            "maximum_equivalent_plastic_strain": float(
                np.max(committed[..., equivalent_plastic_strain_index])
            ),
            "formed_flange_angle_deg": flange_angle(displacement),
            "initial_blank_thickness_mm": float(thickness),
            "formed_centerline_thickness_mm": float(
                centerline_thickness(formed_thickness_profile)
            ),
            "formed_centerline_thickness_change_mm": float(
                centerline_thickness(formed_thickness_profile) - thickness
            ),
            "formed_centerline_thickness_change_percent": float(
                100. * (centerline_thickness(formed_thickness_profile)
                        / thickness - 1.)
            ),
            "formed_minimum_column_thickness_mm": float(
                formed_minimum_thickness
            ),
            "formed_minimum_thickness_change_mm": float(
                formed_minimum_thickness - thickness
            ),
            "formed_minimum_thickness_change_percent": float(
                100. * (formed_minimum_thickness / thickness - 1.)
            ),
            "formed_minimum_thickness_x_mm": float(
                formed_minimum_thickness_x
            ),
            "formed_thickness_profile_x_mm_and_thickness_mm": (
                formed_thickness_profile
            ),
            "forming_equilibrium_residual_inf_N_per_mm": float(
                last_record["equilibrium_residual_inf_N_per_mm"]
            ),
            "forming_equilibrium_tolerance_inf_N_per_mm": float(
                last_record["equilibrium_tolerance_inf_N_per_mm"]
            ),
            "maximum_contact_penetration_mm": float(
                last_record["maximum_contact_penetration_mm"]
            ),
            "contact_complementarity_N_mm": float(
                last_record["contact_complementarity_N_mm"]
            ),
            "holder_force_balance_residual_N": float(
                last_record["holder_force_balance_residual_N"]
            ),
            "springback_pending": True,
            "springback_comparison_valid": False,
        }

    springback_resume = bool(
        int(checkpoint_extra.get("springback_active", np.asarray(0)).item())
    )
    formed_angle = (
        float(checkpoint_extra["springback_formed_angle_deg"].item())
        if springback_resume else flange_angle(displacement)
    )
    if not springback_resume:
        # Preserve the last tool-constrained state separately from the
        # changing springback iterate.  A springback checkpoint can then be
        # audited or branched back into forming without mistaking a partially
        # unloaded displacement for the formed geometry.
        checkpoint_extra.update({
            "springback_formed_displacement": displacement.copy(),
            "springback_formed_committed": committed.copy(),
            "springback_formed_holder_z": np.asarray(holder_z),
            "springback_formed_friction_multipliers": (
                committed_friction_multipliers.copy()
            ),
        })
    # Remove punch, die, and holder, then equilibrate the residual plastic state.
    # The vertical correction at the symmetry center is fixed only to remove
    # the free body's rigid-translation null mode; its formed position is kept.
    center_node = int(np.argmin(node_x**2 + node_z**2))
    springback_dofs = np.r_[symmetry_dofs, dof_z[center_node]]
    springback_constraints = np.zeros((springback_dofs.size, basis.N))
    springback_constraints[np.arange(symmetry_dofs.size), symmetry_dofs] = 1.
    springback_constraints[-1, dof_z[center_node]] = 1.
    springback_free = np.setdiff1d(np.arange(basis.N), springback_dofs)
    # Use the formed-state free residual as the equivalent tool reaction, then
    # release that load in stages.  A direct jump to zero load makes the
    # elastoplastic Newton solve unnecessarily sensitive to the initial step.
    formed_equivalent_load = (
        np.asarray(checkpoint_extra["springback_formed_equivalent_load"], dtype=float)
        if springback_resume else assembler.evaluate(
            displacement, committed, tangent=False
        ).residual
    )
    # Use an absolute floor plus a tight relative tolerance against the
    # characteristic forming reaction. This keeps Newton from spending many
    # iterations below the residual scale relevant to the released blank.
    springback_tolerance = max(
        2e-5,
        2.0e-6 * float(np.linalg.norm(
            formed_equivalent_load[springback_free], ord=np.inf
        )),
    )
    springback_residual = np.inf
    unload_fractions = (
        json.loads(str(checkpoint_extra["springback_unload_fractions_json"].item()))
        if springback_resume and "springback_unload_fractions_json" in checkpoint_extra
        else []
    )
    unload_fraction = (
        float(checkpoint_extra["springback_fraction"].item())
        if springback_resume else 0.
    )
    unload_increment = (
        min(
            float(checkpoint_extra["springback_increment"].item()),
            maximum_unload_increment,
        )
        if springback_resume and "springback_increment" in checkpoint_extra
        else initial_unload_increment
    )
    springback_iteration = (
        int(checkpoint_extra["springback_final_stage_iterations"].item())
        if springback_resume
        and "springback_final_stage_iterations" in checkpoint_extra
        else 0
    )
    if not np.isfinite(minimum_unload_increment) or minimum_unload_increment <= 0.:
        raise ValueError("minimum_unload_increment must be finite and positive")
    if (not np.isfinite(initial_unload_increment)
            or initial_unload_increment < minimum_unload_increment
            or initial_unload_increment > maximum_unload_increment):
        raise ValueError(
            "initial_unload_increment must be finite and between "
            "minimum_unload_increment and maximum_unload_increment"
        )
    if (not np.isfinite(maximum_unload_increment)
            or maximum_unload_increment < minimum_unload_increment
            or maximum_unload_increment > 1.):
        raise ValueError(
            "maximum_unload_increment must be finite and between "
            "minimum_unload_increment and 1"
        )
    while unload_fraction < 1. - 1e-12:
        target_unload_fraction = min(1., unload_fraction + unload_increment)
        unload_trace = os.environ.get("SCIKIT_FVM_UNLOAD_TRACE", "") not in ("", "0")
        if unload_trace:
            print(
                f"[springback] begin fraction={target_unload_fraction:.6f} "
                f"increment={unload_increment:.3e}",
                flush=True,
            )
        stage_start_displacement = displacement.copy()
        stage_start_committed = committed.copy()
        remaining_load = 1. - target_unload_fraction
        stage_converged = False
        stage_failure_reason = ""
        springback_iteration_limit = 100
        for springback_iteration in range(1, springback_iteration_limit + 1):
            springback_eval = assembler.evaluate(displacement, committed)
            adjusted_residual = (
                springback_eval.residual - remaining_load * formed_equivalent_load
            )
            current_springback_residual = float(np.linalg.norm(
                adjusted_residual[springback_free], ord=np.inf
            ))
            current_springback_merit = float(np.linalg.norm(
                adjusted_residual[springback_free]
            ))
            if unload_trace:
                print(
                    f"[springback] fraction={target_unload_fraction:.6f} "
                    f"iteration={springback_iteration} "
                    f"residual={current_springback_residual:.4e} "
                    f"l2={current_springback_merit:.4e}",
                    flush=True,
                )
            if current_springback_residual <= springback_tolerance:
                springback_residual = current_springback_residual
                committed = assembler.commit(springback_eval)
                stage_converged = True
                break
            springback_rhs = np.r_[-displacement[symmetry_dofs], 0.]
            accepted_springback_step = False
            best_springback_merit = np.inf
            best_springback_alpha = 0.
            diagonal_scale = max(
                float(np.max(np.abs(springback_eval.tangent.diagonal()), initial=0.)), 1.
            )
            regularization = 0.
            springback_step = None
            minimum_line_search_alpha = 1e-3
            for regularization_attempt in range(12):
                kkt_hessian = springback_eval.tangent
                if regularization > 0.:
                    kkt_hessian = kkt_hessian + regularization * sparse.eye(
                        basis.N, format="csr"
                    )
                equality_backend = os.environ.get(
                    "SCIKIT_FVM_UNLOAD_LINEAR_SOLVER", "kktkit"
                ).lower()
                if equality_backend not in {"kktkit", "sparse"}:
                    raise ValueError(
                        "SCIKIT_FVM_UNLOAD_LINEAR_SOLVER must be kktkit or sparse"
                    )
                if equality_backend == "sparse":
                    springback_step = solve_sparse_equality_step(
                        kkt_hessian, adjusted_residual,
                        springback_constraints, -springback_rhs,
                        allow_nonsymmetric=finite_strain_j2,
                    )
                else:
                    try:
                        springback_step = solve_kktkit_equality_step(
                            kkt_hessian, adjusted_residual,
                            springback_constraints, -springback_rhs,
                        )
                    except ImportError:
                        if os.environ.get(
                                "SCIKIT_FVM_CONTACT_BACKEND", "auto").lower() == "kktkit":
                            raise
                        springback_step = solve_sparse_equality_step(
                            kkt_hessian, adjusted_residual,
                            springback_constraints, -springback_rhs,
                        )
                for backtrack in range(20):
                    alpha = 0.5 ** backtrack
                    candidate = displacement + alpha * springback_step.primal
                    candidate[symmetry_dofs] = 0.
                    candidate[dof_z[center_node]] = displacement[dof_z[center_node]]
                    try:
                        candidate_eval = evaluate_candidate(candidate, committed)
                    except (ValueError, np.linalg.LinAlgError):
                        continue
                    candidate_residual = (
                        candidate_eval.residual - remaining_load * formed_equivalent_load
                    )
                    candidate_merit = float(np.linalg.norm(
                        candidate_residual[springback_free]
                    ))
                    candidate_residual_inf = float(np.linalg.norm(
                        candidate_residual[springback_free], ord=np.inf
                    ))
                    if candidate_merit < best_springback_merit:
                        best_springback_merit = candidate_merit
                        best_springback_alpha = alpha
                    if (candidate_merit < current_springback_merit
                            and candidate_residual_inf <= (
                                current_springback_residual
                                * (1. + 1e-12)
                            )
                            and (alpha >= minimum_line_search_alpha
                                 or candidate_residual_inf <= springback_tolerance)):
                        accepted_springback_step = True
                        break
                if accepted_springback_step:
                    break
                regularization = (
                    1e-8 * diagonal_scale if regularization == 0.
                    else 10. * regularization
                )
            if not accepted_springback_step:
                stage_failure_reason = (
                    "springback regularized Newton line search failed: "
                    f"unload_fraction={target_unload_fraction:.6f}, "
                    f"residual={current_springback_residual:.16e}, "
                    f"best_l2={best_springback_merit:.16e} "
                    f"(current_l2={current_springback_merit:.16e}) "
                    f"at alpha={best_springback_alpha:.3e}; "
                    f"last_step_inf={np.linalg.norm(springback_step.primal, ord=np.inf):.3e}"
                )
                break
            # Convergence is defined by equilibrium on the unconstrained DOFs.
            # The equality multipliers returned by the KKT solve are Newton
            # corrections, not the support reactions at the candidate state;
            # adding them to the candidate residual can therefore report a
            # false residual floor (especially after a damped line-search step).
            springback_residual = float(np.linalg.norm(
                candidate_residual[springback_free], ord=np.inf
            ))
            displacement = candidate
            if unload_trace:
                print(
                    f"[springback] fraction={target_unload_fraction:.6f} "
                    f"accepted_alpha={best_springback_alpha:.3e} "
                    f"residual={springback_residual:.4e}",
                    flush=True,
                )
            if springback_residual <= springback_tolerance:
                committed = assembler.commit(candidate_eval)
                stage_converged = True
                break
        if not stage_converged:
            displacement = stage_start_displacement
            committed = stage_start_committed
            unload_increment *= 0.5
            if unload_increment < minimum_unload_increment:
                raise RuntimeError(
                    f"springback cutback reached minimum unload increment "
                    f"{minimum_unload_increment:.1e} at fraction "
                    f"{target_unload_fraction:.6f}: "
                    f"{stage_failure_reason or f'Newton residual={springback_residual:.3e}'}"
                )
            continue
        unload_fraction = target_unload_fraction
        unload_fractions.append(float(unload_fraction))
        # The finite-strain algorithmic tangent is numerically differentiated,
        # so useful accepted unload increments can take more than ten Newton
        # iterations even while the equilibrium residual decreases steadily.
        # Allow controlled growth through twenty iterations; this is important for
        # a practical full springback path.
        if springback_iteration <= 20:
            unload_increment = min(
                maximum_unload_increment, 1.25 * unload_increment
            )
        elif springback_iteration >= 35:
            unload_increment = max(
                minimum_unload_increment, 0.75 * unload_increment
            )
        checkpoint_extra.update({
            "springback_active": np.asarray(1, dtype=np.int8),
            "springback_fraction": np.asarray(unload_fraction),
            "springback_increment": np.asarray(unload_increment),
            "springback_formed_angle_deg": np.asarray(formed_angle),
            "springback_formed_equivalent_load": formed_equivalent_load,
            "springback_final_stage_iterations": np.asarray(
                springback_iteration, dtype=np.int32
            ),
            "springback_unload_fractions_json": np.asarray(
                json.dumps(unload_fractions)
            ),
        })
        save_checkpoint()
    final_springback_eval = assembler.evaluate(displacement, committed, tangent=False)
    springback_residual = float(np.linalg.norm(
        final_springback_eval.residual[springback_free], ord=np.inf
    ))
    springback_angle = flange_angle(displacement)
    springback_angle_magnitude = abs(springback_angle)
    thickness_profile = thickness_profile_for(displacement)
    final_center_thickness = centerline_thickness(thickness_profile)
    minimum_thickness_x, minimum_thickness = min(
        thickness_profile, key=lambda item: item[1]
    )
    formed_displacement_for_metrics = np.asarray(
        checkpoint_extra.get("springback_formed_displacement", displacement),
        dtype=float,
    )
    formed_thickness_profile = thickness_profile_for(
        formed_displacement_for_metrics
    )
    formed_minimum_thickness_x, formed_minimum_thickness = min(
        formed_thickness_profile, key=lambda item: item[1]
    )
    reference_stroke = float(reference["process"]["punch_stroke"])
    reference_stroke_coverage = float(min(
        1., accepted_stroke / max(reference_stroke, 1e-30)
    ))
    reference_stroke_reached = bool(
        accepted_stroke >= reference_stroke
        - max(1e-10, 1e-10 * reference_stroke)
    )
    reference_friction_matches = bool(np.isclose(
        friction_coefficient,
        float(reference["process"]["friction_coefficient"]),
        rtol=1e-10, atol=1e-12,
    ))
    reference_material_matches = bool(
        not finite_strain_j2 and not puso_isotropic_hardening
    )
    reference_holder_control_matches = holder_position_control_mm is None
    # The local solver uses a plane-strain solid mesh and point/mortar contact;
    # the cited benchmark uses shell elements with thickness-aware contact.
    reference_formulation_matches = False
    reference_condition_checks = {
        "full_reference_stroke_reached": reference_stroke_reached,
        "friction_coefficient_matches": reference_friction_matches,
        "material_law_matches": reference_material_matches,
        "load_control_matches": reference_holder_control_matches,
        "shell_formulation_and_thickness_contact_match": (
            reference_formulation_matches
        ),
    }
    return {
        "problem": (
            "NUMISHEET '93 profile contact, finite-strain plane-strain isotropic J2 proof"
            if finite_strain_j2 else
            "NUMISHEET '93 profile contact, small-strain plane-strain Hill48 proof"
        ),
        "kinematics": "finite Hencky J2" if finite_strain_j2 else "small strain",
        "material_model": "isotropic J2" if finite_strain_j2 else "Hill48",
        "hardening_reference": (
            "Puso UCRL-MA-140587 isotropic law"
            if puso_isotropic_hardening else "official Abaqus deck"
        ),
        "friction_coefficient": float(friction_coefficient),
        "scope": (
            "source blank/tools/material; partial stroke only; frictional contact "
            "uses a regularized Coulomb law; no shell contact thickness"
            if friction_coefficient > 0. else
            "source blank/tools/material; partial stroke only; frictionless contact; "
            "no shell contact thickness"
        ),
        "mesh": {
            "base_nx": nx,
            "base_nz": nz,
            "refine_tool_contacts": refine_tool_contacts,
            "uniform_refinement_level": mesh_refinement_level,
            "effective_nx": nx * (2 ** mesh_refinement_level),
            "effective_nz": nz * (2 ** mesh_refinement_level),
            "triangles": int(mesh.nelements),
                 "nodes": node_count, "dofs": int(basis.N)},
        "element_quadrature": {
            "rule": "triangle centroid; exact for affine TriP1 internal residual/tangent",
            "points_per_element": int(basis.dx.shape[1]),
        },
        "x_mesh_regions_mm_and_cells": [
            [float(left), float(right), int(cells)]
            for left, right, cells in regions
        ],
        "tool_profiles": {name: profile.segment_count for name, profile in tools.items()},
        "blank_holder_corner_smoothing_radius_mm": HOLDER_CORNER_SMOOTHING_MM,
        "holder_contact_curvature_blend_length_mm": (
            HOLDER_CONTACT_CURVATURE_BLEND_LENGTH_MM
        ),
        "punch_contact_curvature_blend_length_mm": (
            punch_contact_curvature_blend_length_mm
        ),
        "contact_method": contact_method,
        "contact_sampling": (
            f"{mortar_subfacets_per_facet} P0 subfacets per selected boundary "
            f"facet with {contact_collocation_points_per_facet} Gauss points "
            f"per subfacet; {mortar_gap_reduction} gap reduction"
            + (f" ({mortar_gap_smoothing_temperature_mm:g} mm temperature)"
               if mortar_gap_reduction == "softmin" else "")
            if contact_method == "mortar-p0" else
            f"{contact_collocation_points_per_facet} Gauss-point collocation "
            "points per selected boundary facet"
        ),
        "contact_solver_policy": (
            (
                "Fischer-Burmeister semismooth-first (default for finite-strain "
                "J2; disable with SCIKIT_FVM_FORMING_CONTACT_SEMISMOOTH_FIRST=0), "
                "then direct nonsymmetric sparse active-set fallback"
                if contact_semismooth_first else
                "direct nonsymmetric sparse active-set for finite-strain J2"
            )
            if finite_strain_j2 else
            "KKTKit reduced unilateral active set when primal variables <= 256; "
            "rank-filtered sparse active set for larger systems"
        ),
        "contact_rank_tolerance": contact_rank_tolerance,
        "contact_solver_max_iterations": contact_solver_max_iterations,
        "contact_semismooth_first": contact_semismooth_first,
        "directional_tangent_refinements": directional_tangent_refinements,
        "soft_holder_support_seed": soft_holder_support_seed,
        "soft_punch_support_seed": soft_punch_support_seed,
        "soft_die_support_seed": os.environ.get(
            "SCIKIT_FVM_FORMING_SOFT_DIE_SUPPORT_SEED", ""
        ).lower() in {"1", "true", "yes"},
        "maximum_stroke_mm": maximum_stroke_mm,
        "punch_withdrawal_completed": punch_withdrawal_completed,
        "maximum_increment_mm": maximum_increment_mm,
        "forming_increment_continuation_floor_mm": (
            forming_increment_floor_mm
        ),
        "forming_force_resolution_floor_mm": (
            force_resolution_floor_mm
        ),
        "holder_jump_tolerance_mm": holder_jump_tolerance_mm,
        "unresolved_holder_jump_count": sum(
            bool(record.get("holder_jump_unresolved", False))
            for record in records
        ),
        "unresolved_force_jump_count": sum(
            bool(record.get(
                "force_resolution_unresolved_for_requested_tolerance",
                record.get("force_resolution_unresolved", False),
            ))
            for record in records
        ),
        "forming_force_stroke_curve_resolved": not any(
            record.get(
                "force_resolution_unresolved_for_requested_tolerance",
                record.get("force_resolution_unresolved", False),
            )
            or record.get("holder_jump_unresolved", False)
            for record in records
        ),
        "forming_tolerance_history_complete": all(
            "contact_complementarity_tolerance_N_mm" in record
            and "contact_gap_tolerance_mm" in record
            and "punch_force_increment_relative_tolerance" in record
            for record in records
        ),
        "requested_benchmark_stroke_mm": reference["process"]["punch_stroke"],
        "benchmark_comparison": {
            "requested_reference_stroke_mm": reference_stroke,
            "accepted_model_stroke_mm": float(accepted_stroke),
            "reference_stroke_coverage_fraction": reference_stroke_coverage,
            "full_reference_stroke_reached": reference_stroke_reached,
            "reference_condition_checks": reference_condition_checks,
            "same_condition_benchmark_comparison_possible": bool(
                all(reference_condition_checks.values())
            ),
            "punch_force_stroke": {
                "model_curve_available": True,
                "reference_curve_available": bool(reference.get(
                    "reference_run", {}
                ).get("independent_force_curve_available_locally", False)),
                "direct_comparison_valid": False,
                "reason": (
                    "The reference input requests punch RF/U history, but its "
                    "output database and an independent force-stroke curve are "
                    "not included locally. The model curve is exported in "
                    "punch_force_by_stroke."
                ),
            },
            "thickness": {
                "model_profile_available": True,
                "reference_measurements_available": False,
                "direct_comparison_valid": False,
                "reason": (
                    "The local benchmark data provide the initial 0.78 mm "
                    "blank thickness but no measured post-forming thickness "
                    "profile. Model profiles are reported separately for the "
                    "formed and springback states."
                ),
            },
            "springback_angle": {
                "model_angle_magnitude_deg": springback_angle_magnitude,
                "experimental_mean_deg": reference["springback_reference"][
                    "experimental_mean_after_unloading"
                ],
                "experimental_range_deg": reference["springback_reference"][
                    "experimental_range_after_unloading"
                ],
                "error_from_experimental_mean_deg": (
                    springback_angle_magnitude
                    - reference["springback_reference"][
                        "experimental_mean_after_unloading"
                    ]
                ),
                "direct_comparison_valid": False,
                "reason": (
                    "The experimental angle is a useful target, but this run "
                    "uses a plane-strain solid model with "
                    + ("isotropic finite-strain J2" if finite_strain_j2
                       else "small-strain Hill48")
                    + ", friction coefficient "
                    + f"{friction_coefficient:g}, and no shell-thickness "
                    "contact. The benchmark uses a shell formulation, "
                    "friction coefficient 0.144, and tabular Hill48 response. "
                    "The published 17.20 degree "
                    "isotropic result uses a fitted power-law hardening curve, "
                    "not the local deck material curve."
                ),
            },
        },
        "forming_equilibrium_tolerance_inf_N_per_mm": forming_equilibrium_tolerance,
        "forming_equilibrium_relative_tolerance": (
            forming_equilibrium_relative_tolerance
        ),
        "forming_contact_gap_tolerance_mm": forming_contact_gap_tolerance_mm,
        "forming_die_contact_feature_blend_temperature_mm": (
            die_contact_feature_blend_temperature_mm
        ),
        "forming_complementarity_tolerance": forming_complementarity_tolerance,
        "forming_contact_restoration_equilibrium_tolerance_multiplier": (
            contact_restoration_equilibrium_tolerance_multiplier
        ),
        "forming_contact_restoration_holder_tolerance_multiplier": (
            contact_restoration_holder_tolerance_multiplier
        ),
        "forming_punch_force_increment_relative_tolerance": (
            forming_force_change_tolerance
        ),
        "forming_punch_force_increment_max_subdivisions": (
            forming_force_change_max_subdivisions
        ),
        "forming_punch_force_increment_reference_N": (
            punch_force_change_reference_N
        ),
        "forming_slow_progress_iteration_limit": (
            slow_progress_iteration_limit
        ),
        "forming_force_stagnation_iteration_limit": (
            force_stagnation_iteration_limit
        ),
        "blank_holder_force_N_half_model": half_model_holder_force,
        "steps": records,
        "increment_rejection_count": len(failure_log),
        "last_increment_rejections": failure_log,
        "punch_force_by_stroke": [
            [r["stroke_mm"], r["punch_force_N_half_model"]] for r in records
        ],
        "normal_punch_force_by_stroke": [
            [
                r["stroke_mm"],
                r.get(
                    "normal_punch_force_N_half_model",
                    r["punch_force_N_half_model"],
                ),
            ]
            for r in records
        ],
        "friction_punch_force_by_stroke": [
            [r["stroke_mm"], r.get("friction_punch_force_N_half_model", 0.0)]
            for r in records
        ],
        "maximum_punch_force_N_half_model": max(r["punch_force_N_half_model"] for r in records),
        "maximum_equivalent_plastic_strain": float(
            np.max(committed[..., equivalent_plastic_strain_index])
        ),
        "formed_flange_angle_deg": formed_angle,
        "initial_blank_thickness_mm": float(thickness),
        "formed_centerline_thickness_mm": float(
            centerline_thickness(formed_thickness_profile)
        ),
        "formed_centerline_thickness_change_mm": float(
            centerline_thickness(formed_thickness_profile) - thickness
        ),
        "formed_centerline_thickness_change_percent": float(
            100. * (centerline_thickness(formed_thickness_profile)
                    / thickness - 1.)
        ),
        "formed_minimum_column_thickness_mm": float(
            formed_minimum_thickness
        ),
        "formed_minimum_thickness_change_mm": float(
            formed_minimum_thickness - thickness
        ),
        "formed_minimum_thickness_change_percent": float(
            100. * (formed_minimum_thickness / thickness - 1.)
        ),
        "formed_minimum_thickness_x_mm": float(
            formed_minimum_thickness_x
        ),
        "formed_thickness_profile_x_mm_and_thickness_mm": (
            formed_thickness_profile
        ),
        "springback_flange_angle_deg": springback_angle,
        "springback_flange_angle_magnitude_deg": springback_angle_magnitude,
        "springback_centerline_thickness_mm": float(final_center_thickness),
        "springback_centerline_thickness_change_mm": float(
            final_center_thickness - thickness
        ),
        "springback_centerline_thickness_change_percent": float(
            100. * (final_center_thickness / thickness - 1.)
        ),
        "springback_minimum_column_thickness_mm": minimum_thickness,
        "springback_minimum_thickness_change_mm": float(
            minimum_thickness - thickness
        ),
        "springback_minimum_thickness_change_percent": float(
            100. * (minimum_thickness / thickness - 1.)
        ),
        "springback_minimum_thickness_x_mm": minimum_thickness_x,
        "springback_thickness_profile_x_mm_and_thickness_mm": thickness_profile,
        "springback_unload_method": (
            (
                "adaptive-stage linear release of the punch-withdrawn state's "
                "equivalent free-DOF reaction load; rigid-tool contact is inactive "
                "during this continuation"
                if punch_withdrawal_completed
                else
                "adaptive-stage linear release of the formed-state equivalent "
                "free-DOF reaction load; rigid-tool contact is inactive during "
                "this continuation"
            )
        ),
        "springback_unload_fractions": unload_fractions,
        "springback_equilibrium_residual_inf_N_per_mm": float(springback_residual),
        "springback_newton_iterations_final_stage": int(springback_iteration),
        "springback_newton_tolerance_inf_N_per_mm": springback_tolerance,
        "springback_minimum_unload_increment": minimum_unload_increment,
        "springback_initial_unload_increment": initial_unload_increment,
        "springback_maximum_unload_increment": maximum_unload_increment,
        "springback_vertical_anchor_reaction_N_per_mm": float(
            -final_springback_eval.residual[dof_z[center_node]]
        ),
        "springback_reference_angle_deg": reference["springback_reference"][
            "experimental_mean_after_unloading"
        ],
        "springback_reference_angle_error_deg": (
            springback_angle_magnitude - reference["springback_reference"][
                "experimental_mean_after_unloading"
            ]
        ),
        "springback_comparison_valid": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=70)
    parser.add_argument("--nz", type=int, default=4)
    parser.add_argument(
        "--mesh-refinement-level", type=int, default=0,
        help=("uniformly bisect every triangle this many times; nested mesh "
              "checkpoints can preserve each parent's plastic history"),
    )
    parser.add_argument("--maximum-stroke-mm", type=float, default=70.0)
    parser.add_argument("--maximum-increment-mm", type=float, default=0.1)
    parser.add_argument("--refine-punch-arc", action="store_true",
                        help="redistribute x cells toward the punch corner radius")
    parser.add_argument("--refine-holder-corner", action="store_true",
                        help="concentrate x cells around the rounded blank-holder corner")
    parser.add_argument("--refine-tool-contacts", action="store_true",
                        help="refine both the holder fillet and die-entry contact path")
    parser.add_argument(
        "--refine-holder-return", action="store_true",
        help=("concentrate cells along the blank-holder return where the "
              "forming path previously rotated a long surface facet"),
    )
    parser.add_argument("--finite-strain-j2", action="store_true",
                        help="use multiplicative finite-strain J2 with the official deck hardening curve")
    parser.add_argument(
        "--puso-isotropic-hardening", action="store_true",
        help=("with --finite-strain-j2, use the isotropic hardening law from "
              "Puso UCRL-MA-140587 instead of the official Abaqus deck curve"),
    )
    parser.add_argument("--checkpoint", type=Path,
                        help="write a restart checkpoint after each accepted forming increment")
    parser.add_argument("--resume", action="store_true",
                        help="resume from the checkpoint passed with --checkpoint")
    parser.add_argument(
        "--re-equilibrate-checkpoint", action="store_true",
        help=("solve the resumed state at its saved stroke before continuing; "
              "use after transferring to a refined mesh"),
    )
    parser.add_argument("--resume-forming-from-springback", action="store_true",
                        help="restore the saved formed state from a springback checkpoint")
    parser.add_argument("--forming-only", action="store_true",
                        help="return forming force/stroke results without springback")
    parser.add_argument(
        "--withdraw-punch-after-forming", action="store_true",
        help=("continue the frictionless contact equilibrium path while "
              "retracting the punch from the formed stroke to zero"),
    )
    parser.add_argument(
        "--holder-position-control-mm", type=float, default=None,
        help=("diagnostic mode: fix the blank-holder vertical position in mm; "
              "with --resume, a force-controlled checkpoint can seed a "
              "separate position-controlled continuation"),
    )
    parser.add_argument("--minimum-unload-increment", type=float, default=1e-5,
                        help="minimum load-release fraction increment for springback cutbacks")
    parser.add_argument("--initial-unload-increment", type=float, default=0.1,
                        help="initial load-release fraction increment for springback")
    parser.add_argument("--maximum-unload-increment", type=float, default=0.2,
                        help="maximum load-release fraction increment for springback")
    parser.add_argument("--friction-coefficient", type=float, default=0.0,
                        help="Coulomb coefficient for punch, die, and holder contact")
    parser.add_argument("--forming-equilibrium-tolerance", type=float, default=2.0e-1,
                        help="absolute floor for forming equilibrium residual in N/mm")
    parser.add_argument(
        "--forming-equilibrium-relative-tolerance", type=float, default=5.0e-3,
        help="relative residual tolerance scaled by punch force per modeled width",
    )
    parser.add_argument("--forming-complementarity-tolerance", type=float, default=1.0e-2,
                        help="accepted forming contact complementarity residual")
    parser.add_argument(
        "--forming-slow-progress-limit", type=int, default=None,
        help=("iterations of small merit decrease before cutting back a stroke "
              "increment (overrides SCIKIT_FVM_FORMING_SLOW_PROGRESS_LIMIT)"),
    )
    args = parser.parse_args()
    print(json.dumps(run(args.nx, args.nz, args.maximum_stroke_mm,
                         args.maximum_increment_mm, args.refine_punch_arc,
                         args.refine_holder_corner,
                         args.refine_tool_contacts,
                         args.finite_strain_j2, args.checkpoint, args.resume,
                         args.minimum_unload_increment,
                         args.resume_forming_from_springback,
                         args.forming_only,
                         args.friction_coefficient,
                         args.forming_equilibrium_tolerance,
                         args.forming_complementarity_tolerance,
                         args.forming_equilibrium_relative_tolerance,
                         args.forming_slow_progress_limit,
                         args.holder_position_control_mm,
                         args.refine_holder_return,
                         args.mesh_refinement_level,
                         args.re_equilibrate_checkpoint,
                         args.puso_isotropic_hardening,
                         args.initial_unload_increment,
                         args.maximum_unload_increment,
                         args.withdraw_punch_after_forming), indent=2))


if __name__ == "__main__":
    main()
