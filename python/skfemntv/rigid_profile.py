"""Projection utilities for 2D analytical rigid profiles made of lines/arcs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RigidProfileProjection:
    """Closest points, signed gaps, normals, and signed-gap Hessians."""

    closest_points: np.ndarray
    normals: np.ndarray
    gaps: np.ndarray
    segment_indices: np.ndarray
    gap_hessians: np.ndarray


@dataclass(frozen=True)
class _Line:
    start: np.ndarray
    end: np.ndarray


@dataclass(frozen=True)
class _Arc:
    start: np.ndarray
    end: np.ndarray
    center: np.ndarray
    sweep: float


class AnalyticalRigidProfile2D:
    """Oriented, open rigid curve consisting of line and circular-arc pieces.

    Segment order defines the material-facing side: normals point to the left
    of the directed profile. ``from_keyword_profile`` reads the START/LINE/
    CIRCL data used by Abaqus analytical TYPE=CYLINDER surfaces and maps its
    local profile plane into global X-Z coordinates.
    """

    def __init__(self, segments):
        if not segments:
            raise ValueError("a rigid profile must contain at least one segment")
        self._segments = tuple(segments)

    @property
    def segment_count(self) -> int:
        """Number of directed line and arc pieces in the profile."""
        return len(self._segments)

    @classmethod
    def from_keyword_profile(cls, profile, *, scale: float = 1.0):
        """Build from raw keyword rows: ``[START,x,y]``, ``[LINE,x,y]``,
        or ``[CIRCL,x,y,cx,cy]``. The returned profile is in local coordinates.
        """
        if not np.isfinite(scale) or scale <= 0.:
            raise ValueError("scale must be finite and positive")
        start = None
        current = None
        segments = []
        for row in profile:
            if not row:
                raise ValueError("profile rows cannot be empty")
            kind = str(row[0]).upper()
            values = np.asarray(row[1:], dtype=float) * scale
            if not np.all(np.isfinite(values)):
                raise ValueError("profile coordinates must be finite")
            if kind == "START" and values.shape == (2,) and current is None:
                start = current = values
            elif kind == "LINE" and values.shape == (2,) and current is not None:
                end = values
                if np.linalg.norm(end - current) <= np.finfo(float).tiny:
                    raise ValueError("line profile segment has zero length")
                segments.append(_Line(current.copy(), end.copy()))
                current = end
            elif kind == "CIRCL" and values.shape == (4,) and current is not None:
                end, center = values[:2], values[2:]
                r0 = float(np.linalg.norm(current - center))
                r1 = float(np.linalg.norm(end - center))
                if r0 <= np.finfo(float).tiny or not np.isclose(r0, r1, rtol=1e-8, atol=1e-12):
                    raise ValueError("circular arc endpoints must share a nonzero radius")
                a0 = float(np.arctan2(current[1] - center[1], current[0] - center[0]))
                a1 = float(np.arctan2(end[1] - center[1], end[0] - center[0]))
                sweep = float((a1 - a0 + np.pi) % (2. * np.pi) - np.pi)
                if abs(sweep) <= 1e-12 or abs(sweep) >= np.deg2rad(179.74) + 1e-10:
                    raise ValueError("circular arc sweep must be nonzero and less than 179.74 degrees")
                segments.append(_Arc(current.copy(), end.copy(), center.copy(), sweep))
                current = end
            else:
                raise ValueError(f"invalid {kind!r} profile row or segment ordering")
        if start is None or not segments:
            raise ValueError("profile needs one START row and at least one LINE/CIRCL row")
        return cls(segments)

    @staticmethod
    def keyword_local_to_global_xz(points, axis_line, generator):
        """Map TYPE=CYLINDER local profile coordinates to global X-Z.

        ``axis_line`` contains global points a and b defining local x. The
        generator point defines the sweep direction; local y is chosen as
        ``cross(local_x, generator_direction)``.
        """
        ab = np.asarray(axis_line, dtype=float)
        gen = np.asarray(generator, dtype=float)
        pts = np.asarray(points, dtype=float)
        if ab.shape != (2, 3) or gen.shape != (3,) or pts.shape[-1:] != (2,):
            raise ValueError("axis_line must be (2,3), generator (3,), points (...,2)")
        if not np.all(np.isfinite(ab)) or not np.all(np.isfinite(gen)) or not np.all(np.isfinite(pts)):
            raise ValueError("profile transform data must be finite")
        matrix, offset = AnalyticalRigidProfile2D.keyword_local_transform_xz(ab, gen)
        return pts @ matrix.T + offset

    @staticmethod
    def keyword_local_transform_xz(axis_line, generator):
        """Return the orthonormal local-to-global X-Z map for a cylinder."""
        ab = np.asarray(axis_line, dtype=float)
        gen = np.asarray(generator, dtype=float)
        if ab.shape != (2, 3) or gen.shape != (3,):
            raise ValueError("axis_line must be (2,3) and generator must be (3,)")
        ex = ab[1] - ab[0]
        gen_direction = gen - ab[0]
        if (np.linalg.norm(ex) <= np.finfo(float).tiny
                or np.linalg.norm(gen_direction) <= np.finfo(float).tiny):
            raise ValueError("cylinder axis and generator must have nonzero length")
        ex /= np.linalg.norm(ex)
        gen_direction /= np.linalg.norm(gen_direction)
        ey = np.cross(ex, gen_direction)
        norm_ey = np.linalg.norm(ey)
        if norm_ey <= 1e-12:
            raise ValueError("cylinder generator must not be parallel to the local x axis")
        ey /= norm_ey
        matrix = np.column_stack((ex[[0, 2]], ey[[0, 2]]))
        offset = ab[0, [0, 2]].copy()
        return matrix, offset

    def transformed(self, matrix, offset):
        """Map the profile by a planar rigid transform or uniform scaling."""
        transform = np.asarray(matrix, dtype=float)
        shift = np.asarray(offset, dtype=float)
        if transform.shape != (2, 2) or shift.shape != (2,):
            raise ValueError("matrix and offset must have shapes (2,2) and (2,)")
        if not np.all(np.isfinite(transform)) or not np.all(np.isfinite(shift)):
            raise ValueError("profile transform must be finite")
        gram = transform.T @ transform
        scale2 = 0.5 * float(np.trace(gram))
        if scale2 <= 0. or not np.allclose(gram, scale2 * np.eye(2), rtol=1e-10, atol=1e-12):
            raise ValueError("profile transform must preserve circles (rotation/reflection and uniform scale)")
        orientation = 1. if np.linalg.det(transform) > 0. else -1.

        def apply(point):
            return transform @ point + shift

        moved = []
        for segment in self._segments:
            if isinstance(segment, _Line):
                moved.append(_Line(apply(segment.start), apply(segment.end)))
            else:
                moved.append(_Arc(
                    apply(segment.start), apply(segment.end), apply(segment.center),
                    orientation * segment.sweep,
                ))
        return AnalyticalRigidProfile2D(moved)

    def translated(self, translation):
        """Return a rigidly translated profile in its 2D coordinate plane."""
        shift = np.asarray(translation, dtype=float)
        if shift.shape != (2,) or not np.all(np.isfinite(shift)):
            raise ValueError("translation must be a finite 2-vector")
        moved = []
        for segment in self._segments:
            if isinstance(segment, _Line):
                moved.append(_Line(segment.start + shift, segment.end + shift))
            else:
                moved.append(_Arc(
                    segment.start + shift, segment.end + shift,
                    segment.center + shift, segment.sweep,
                ))
        return AnalyticalRigidProfile2D(moved)

    def with_rounded_corners(self, radius: float):
        """Return a copy with circular fillets at line-line vertices.

        Line-arc joins are retained as supplied. A finite fillet removes the
        undefined contact normal at a sharp vertex while keeping the two
        adjacent faces tangent to the replacement arc.
        """
        if not np.isfinite(radius) or radius < 0.:
            raise ValueError("radius must be finite and nonnegative")
        if radius == 0.:
            return AnalyticalRigidProfile2D(self._segments)
        joints: list[tuple[np.ndarray, np.ndarray, np.ndarray, float] | None] = []
        for first, second in zip(self._segments[:-1], self._segments[1:], strict=True):
            if (not isinstance(first, _Line) or not isinstance(second, _Line)
                    or np.linalg.norm(first.end - second.start) > 1e-10):
                joints.append(None)
                continue
            tangent_in = (first.end - first.start) / np.linalg.norm(
                first.end - first.start
            )
            tangent_out = (second.end - second.start) / np.linalg.norm(
                second.end - second.start
            )
            cross = float(tangent_in[0] * tangent_out[1]
                          - tangent_in[1] * tangent_out[0])
            dot = float(np.clip(tangent_in @ tangent_out, -1., 1.))
            turn = float(np.arctan2(cross, dot))
            if abs(turn) <= 1e-8 or abs(abs(turn) - np.pi) <= 1e-8:
                joints.append(None)
                continue
            offset = radius * np.tan(0.5 * abs(turn))
            if (offset >= 0.45 * np.linalg.norm(first.end - first.start)
                    or offset >= 0.45 * np.linalg.norm(second.end - second.start)):
                raise ValueError("rounded-corner radius is too large for adjacent lines")
            vertex = first.end
            q_in = vertex - offset * tangent_in
            q_out = vertex + offset * tangent_out
            normal_in = np.array([-tangent_in[1], tangent_in[0]])
            center = q_in + np.sign(turn) * radius * normal_in
            angle_in = float(np.arctan2(q_in[1] - center[1], q_in[0] - center[0]))
            angle_out = float(np.arctan2(q_out[1] - center[1], q_out[0] - center[0]))
            sweep = float((angle_out - angle_in + np.pi) % (2. * np.pi) - np.pi)
            if np.sign(sweep) != np.sign(turn):
                sweep += np.sign(turn) * 2. * np.pi
            joints.append((q_in, q_out, center, sweep))

        rounded = []
        for index, segment in enumerate(self._segments):
            start = (joints[index - 1][1]
                     if index > 0 and joints[index - 1] is not None
                     else segment.start)
            end = (joints[index][0]
                   if index < len(joints) and joints[index] is not None
                   else segment.end)
            if isinstance(segment, _Line):
                if np.linalg.norm(end - start) > 1e-12:
                    rounded.append(_Line(start.copy(), end.copy()))
            else:
                rounded.append(segment)
            if index < len(joints) and joints[index] is not None:
                q_in, q_out, center, sweep = joints[index]
                rounded.append(_Arc(q_in.copy(), q_out.copy(), center.copy(), sweep))
        return AnalyticalRigidProfile2D(rounded)

    def project(
        self, points, *, curvature_blend_length: float = 0.0,
        feature_blend_temperature: float = 0.0,
    ) -> RigidProfileProjection:
        """Project points onto the nearest segment and return signed gaps.

        Positive gaps lie on the left-normal side of the directed profile.
        By default, ``gap_hessians`` are the exact signed-distance Hessians.
        A positive ``curvature_blend_length`` smoothly tapers only the Hessian
        contribution near tangent line/arc joins. Closest points, normals,
        and gaps remain exact; the tapered Hessian is a quasi-Newton
        regularization for contact solvers crossing a curvature discontinuity.
        A positive ``feature_blend_temperature`` applies a consistent
        log-sum-exp blend to the two nearest adjacent profile features when
        their signed gaps are within 30 temperatures. This regularizes the
        normal jump at a sharp feature switch; the gap bias is bounded by
        ``temperature * log(2)``.
        """
        if (not np.isfinite(curvature_blend_length)
                or curvature_blend_length < 0.):
            raise ValueError("curvature_blend_length must be finite and nonnegative")
        if (not np.isfinite(feature_blend_temperature)
                or feature_blend_temperature < 0.):
            raise ValueError(
                "feature_blend_temperature must be finite and nonnegative"
            )
        query = np.asarray(points, dtype=float)
        if query.ndim == 1:
            query = query[None, :]
        if query.ndim != 2 or query.shape[1] != 2 or not np.all(np.isfinite(query)):
            raise ValueError("points must be a finite array of shape (n,2)")
        best_distance2 = np.full(query.shape[0], np.inf)
        best_point = np.zeros_like(query)
        best_normal = np.zeros_like(query)
        best_hessian = np.zeros((query.shape[0], 2, 2), dtype=float)
        best_index = np.full(query.shape[0], -1, dtype=np.int64)
        second_distance2 = np.full(query.shape[0], np.inf)
        second_point = np.zeros_like(query)
        second_normal = np.zeros_like(query)
        second_hessian = np.zeros((query.shape[0], 2, 2), dtype=float)
        second_index = np.full(query.shape[0], -1, dtype=np.int64)
        for index, segment in enumerate(self._segments):
            if isinstance(segment, _Line):
                vector = segment.end - segment.start
                length = np.linalg.norm(vector)
                tangent = vector / length
                raw_parameter = (query - segment.start) @ tangent
                parameter = np.clip(raw_parameter, 0., length)
                endpoint_projection = (raw_parameter < 0.) | (raw_parameter > length)
                closest = segment.start + parameter[:, None] * tangent
                normal = np.broadcast_to(
                    np.array([-tangent[1], tangent[0]]), query.shape
                ).copy()
                normal_jacobian = np.zeros((query.shape[0], 2, 2), dtype=float)
            else:
                radius = np.linalg.norm(segment.start - segment.center)
                radial_vector = query - segment.center
                radial_distance = np.linalg.norm(radial_vector, axis=1)
                radial_unit = np.zeros_like(radial_vector)
                nonzero_radius = radial_distance > np.finfo(float).tiny
                radial_unit[nonzero_radius] = (
                    radial_vector[nonzero_radius]
                    / radial_distance[nonzero_radius, None]
                )
                a0 = np.arctan2(segment.start[1] - segment.center[1],
                                segment.start[0] - segment.center[0])
                delta = np.arctan2(query[:, 1] - segment.center[1],
                                   query[:, 0] - segment.center[0]) - a0
                if segment.sweep > 0.:
                    travel = np.mod(delta, 2. * np.pi)
                    within = travel <= segment.sweep
                    angle = a0 + np.clip(travel, 0., segment.sweep)
                else:
                    travel = np.mod(-delta, 2. * np.pi)
                    within = travel <= -segment.sweep
                    angle = a0 - np.clip(travel, 0., -segment.sweep)
                projected_arc = segment.center + radius * np.column_stack((
                    np.cos(angle), np.sin(angle)
                ))
                endpoint_distance_start = np.sum((query - segment.start)**2, axis=1)
                endpoint_distance_end = np.sum((query - segment.end)**2, axis=1)
                use_start = endpoint_distance_start < endpoint_distance_end
                closest = np.where(within[:, None], projected_arc,
                                   np.where(use_start[:, None], segment.start, segment.end))
                endpoint_projection = ~within
                orientation = np.sign(segment.sweep)
                tangent_at = orientation * np.column_stack((-np.sin(angle), np.cos(angle)))
                normal = np.column_stack((-tangent_at[:, 1], tangent_at[:, 0]))
                normal_jacobian = np.zeros((query.shape[0], 2, 2), dtype=float)
                arc_curvature = -orientation / np.maximum(
                    radial_distance, np.finfo(float).tiny
                )
                identity = np.eye(2)
                curved_projection = within & nonzero_radius
                normal_jacobian[curved_projection] = (
                    arc_curvature[curved_projection, None, None]
                    * (identity - radial_unit[curved_projection, :, None]
                       * radial_unit[curved_projection, None, :])
                )
            endpoint_vector = query - closest
            endpoint_distance = np.linalg.norm(endpoint_vector, axis=1)
            endpoint_side = np.where(
                np.einsum("ij,ij->i", endpoint_vector, normal) >= 0., 1., -1.
            )
            replace_normal = endpoint_projection & (endpoint_distance > np.finfo(float).tiny)
            normal[replace_normal] = (
                endpoint_side[replace_normal, None]
                * endpoint_vector[replace_normal]
                / endpoint_distance[replace_normal, None]
            )
            endpoint_normals = normal[replace_normal]
            endpoint_identity = np.eye(2)
            normal_jacobian[replace_normal] = (
                endpoint_side[replace_normal, None, None]
                * (endpoint_identity
                   - endpoint_normals[:, :, None] * endpoint_normals[:, None, :])
                / endpoint_distance[replace_normal, None, None]
            )
            distance2 = np.sum((query - closest)**2, axis=1)
            take = distance2 < best_distance2
            promote = take & (best_distance2 < second_distance2)
            second_distance2[promote] = best_distance2[promote]
            second_point[promote] = best_point[promote]
            second_normal[promote] = best_normal[promote]
            second_hessian[promote] = best_hessian[promote]
            second_index[promote] = best_index[promote]
            alternate = (~take) & (distance2 < second_distance2)
            second_distance2[alternate] = distance2[alternate]
            second_point[alternate] = closest[alternate]
            second_normal[alternate] = normal[alternate]
            second_hessian[alternate] = normal_jacobian[alternate]
            second_index[alternate] = index
            best_distance2[take] = distance2[take]
            best_point[take] = closest[take]
            best_normal[take] = normal[take]
            best_hessian[take] = normal_jacobian[take]
            best_index[take] = index
        if curvature_blend_length > 0.:
            for index, segment in enumerate(self._segments):
                if not isinstance(segment, _Arc):
                    continue
                selected = best_index == index
                if not np.any(selected):
                    continue
                weight = np.ones(np.count_nonzero(selected), dtype=float)
                if index > 0 and isinstance(self._segments[index - 1], _Line):
                    weight = np.minimum(
                        weight,
                        np.linalg.norm(
                            best_point[selected] - segment.start, axis=1
                        ) / curvature_blend_length,
                    )
                if (index + 1 < len(self._segments)
                        and isinstance(self._segments[index + 1], _Line)):
                    weight = np.minimum(
                        weight,
                        np.linalg.norm(
                            best_point[selected] - segment.end, axis=1
                        ) / curvature_blend_length,
                    )
                best_hessian[selected] *= np.clip(weight, 0., 1.)[:, None, None]
        gaps = np.einsum("ij,ij->i", query - best_point, best_normal)
        if feature_blend_temperature > 0. and len(self._segments) > 1:
            second_gaps = np.einsum(
                "ij,ij->i", query - second_point, second_normal
            )
            adjacent = np.abs(best_index - second_index) == 1
            same_side = gaps * second_gaps >= 0.
            near_switch = (
                np.abs(gaps - second_gaps)
                <= 30. * feature_blend_temperature
            )
            blend = adjacent & same_side & near_switch
            if np.any(blend):
                tau = feature_blend_temperature
                first = gaps[blend]
                second = second_gaps[blend]
                minimum = np.minimum(first, second)
                first_weight = np.exp(-(first - minimum) / tau)
                second_weight = np.exp(-(second - minimum) / tau)
                weight_sum = first_weight + second_weight
                first_weight /= weight_sum
                second_weight /= weight_sum
                gaps[blend] = minimum - tau * np.log(weight_sum)
                normal_difference = (
                    best_normal[blend] - second_normal[blend]
                )
                normals = (
                    first_weight[:, None] * best_normal[blend]
                    + second_weight[:, None] * second_normal[blend]
                )
                best_normal[blend] = normals
                best_hessian[blend] = (
                    first_weight[:, None, None] * best_hessian[blend]
                    + second_weight[:, None, None] * second_hessian[blend]
                    - (first_weight * second_weight / tau)[:, None, None]
                    * np.einsum(
                        "ni,nj->nij", normal_difference, normal_difference
                    )
                )
                best_point[blend] = (
                    first_weight[:, None] * best_point[blend]
                    + second_weight[:, None] * second_point[blend]
                )
        return RigidProfileProjection(
            best_point, best_normal, gaps, best_index, best_hessian,
        )


__all__ = ["AnalyticalRigidProfile2D", "RigidProfileProjection"]
