"""Transient forward-depth evidence for conservative doorway candidates.

Depth is optical-axis distance in metres, not Euclidean range. Optical coordinates
are right/down/forward; robot coordinates are forward/left/up. Only the candidate
and small scalar diagnostics leave ``detect``: points and rays are never retained.

The first version assumes a roughly frontal, vertical opening. Its crossing-plane
normal follows the horizontal camera heading; strongly oblique walls, partial
doorways, glass, and openings beyond valid sensor range are deliberately missed.
An observed wide/narrow/wide ray fan is geometric evidence, not a room label.
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

DEFAULT_GEOMETRY_CONFIG = {
    "robot_width": 0.60,
    "safety_margin": 0.10,
    "portal_min_width": 0.75,
    "portal_max_width": 2.20,
    "depth_min": 0.20,
    "depth_max": 5.0,
    "roi_top": 0.25,
    "roi_bottom": 0.80,
    "sample_stride": 2,
    "max_sample_columns": 160,
    "max_sample_rows": 100,
    "height_min": 0.12,
    "height_max": 1.60,
    "jamb_min_height": 0.20,
    "jamb_min_samples": 3,
    "jamb_support_fraction": 0.55,
    "jamb_depth_tolerance": 0.25,
    "depth_jump": 0.55,
    "clearance_depth": 0.40,
    "free_fraction": 0.65,
    "max_blocked_fraction": 0.15,
    "profile_step": 0.25,
    "profile_clearance": 0.05,
    "profile_before_distance": 0.40,
    "profile_after_distance": 0.75,
    "profile_widening_ratio": 1.15,
}


@dataclass(frozen=True)
class CameraModel:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    camera_to_robot: np.ndarray


@dataclass(frozen=True)
class PortalCandidate:
    x: float
    y: float
    normal_yaw: float
    width: float
    confidence: float
    kind: str


@dataclass
class GeometryResult:
    candidate: Optional[PortalCandidate]
    diagnostics: dict


class DepthPortalDetector:
    """Single-frame detector; callers own temporal confirmation and room memory."""

    def __init__(self, config: dict):
        if set(config) - set(DEFAULT_GEOMETRY_CONFIG):
            raise ValueError("unknown depth geometry config key")
        self.config = dict(DEFAULT_GEOMETRY_CONFIG)
        self.config.update(config)
        c = self.config
        for key, value in c.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError("geometry config must be finite and nonnegative: " + key)
            if key != "roi_top" and value == 0:
                raise ValueError("geometry config must be positive: " + key)
        if (
            not c["height_min"] < c["height_max"]
            or not c["portal_min_width"] < c["portal_max_width"]
        ):
            raise ValueError("invalid geometry height/portal bounds")
        for key in ("jamb_support_fraction", "free_fraction", "max_blocked_fraction"):
            if c[key] > 1:
                raise ValueError(key + " must not exceed 1")
        if not 0 <= c["roi_top"] < c["roi_bottom"] <= 1:
            raise ValueError("depth ROI must satisfy 0 <= top < bottom <= 1")
        for key in ("sample_stride", "max_sample_columns", "max_sample_rows", "jamb_min_samples"):
            if int(c[key]) < 1:
                raise ValueError(key + " must be at least 1")
        if not 0 < c["depth_min"] < c["depth_max"]:
            raise ValueError("depth limits must be positive and increasing")
        if c["profile_step"] <= 0:
            raise ValueError("profile_step must be positive")

    def detect(self, depth: np.ndarray, camera: CameraModel, pose) -> GeometryResult:
        c = self.config
        depth = np.asarray(depth)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if depth.shape != (camera.height, camera.width):
            raise ValueError("depth shape does not match camera resolution")
        transform = np.asarray(camera.camera_to_robot, dtype=float)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError("camera_to_robot must be a finite 4x4 transform")
        if not (camera.fx > 0 and camera.fy > 0):
            raise ValueError("camera focal lengths must be positive")
        top = int(camera.height * c["roi_top"])
        bottom = int(math.ceil(camera.height * c["roi_bottom"]))
        stride = max(
            int(c["sample_stride"]),
            math.ceil(camera.width / c["max_sample_columns"]),
            math.ceil((bottom - top) / c["max_sample_rows"]),
        )
        columns = np.arange(0, camera.width, stride)
        rows = np.arange(top, bottom, stride)
        samples = np.asarray(depth[np.ix_(rows, columns)], dtype=float)
        valid = np.isfinite(samples) & (samples >= c["depth_min"]) & (samples <= c["depth_max"])
        diagnostics = {
            "valid_fraction": float(valid.mean()) if valid.size else 0.0,
            "profile": [],
            "wide_narrow_wide": False,
            "reason": "insufficient_valid_depth",
            "sample_stride": stride,
        }
        if not valid.any() or len(columns) < 3 or len(rows) < 3:
            return GeometryResult(None, diagnostics)

        uu, vv = np.meshgrid(columns, rows)
        optical_rays = np.stack(
            ((uu - camera.cx) / camera.fx, (vv - camera.cy) / camera.fy, np.ones_like(uu)), axis=-1
        )
        rays = optical_rays @ transform[:3, :3].T
        origin = transform[:3, 3]
        forward = transform[:2, 2].copy()
        norm = np.linalg.norm(forward)
        if norm < 1e-6:
            diagnostics["reason"] = "camera_has_no_horizontal_forward_axis"
            return GeometryResult(None, diagnostics)
        forward /= norm
        left = np.array([-forward[1], forward[0]])
        # Sanitizing before multiplication avoids NaN/Inf poisoning geometric bins.
        points = rays * np.where(valid, samples, 0.0)[..., None] + origin
        ray_forward = rays[..., :2] @ forward
        ray_left = rays[..., :2] @ left
        advances = (points[..., :2] - origin[:2]) @ forward
        heights = points[..., 2]
        slopes = np.divide(
            ray_left, ray_forward, out=np.zeros_like(ray_left), where=ray_forward > 1e-6
        )
        vertical_slopes = np.divide(
            rays[..., 2], ray_forward, out=np.zeros_like(ray_forward), where=ray_forward > 1e-6
        )
        forward_valid = valid & (ray_forward > 1e-6)

        def slice_evidence(distance, clearance):
            at_height = origin[2] + distance * vertical_slopes
            eligible = (
                (ray_forward > 1e-6)
                & (at_height >= c["height_min"])
                & (at_height <= c["height_max"])
            )
            totals = eligible.sum(axis=0)
            observed_free = forward_valid & eligible & (advances >= distance + clearance)
            fractions = observed_free.sum(axis=0) / np.maximum(totals, 1)
            supported = (totals >= c["jamb_min_samples"]) & (fractions >= c["free_fraction"])
            blocked = forward_valid & eligible & (advances <= distance + c["profile_clearance"])
            blocked_fraction = blocked.sum(axis=0) / np.maximum(totals, 1)
            lateral = np.median(slopes, axis=0) * distance
            return supported, fractions, blocked_fraction, lateral

        profile = []
        for distance in np.arange(c["profile_step"], c["depth_max"], c["profile_step"]):
            supported, _, _, lateral = slice_evidence(distance, c["profile_clearance"])
            runs = self._runs(supported)
            width = max((abs(lateral[end] - lateral[start]) for start, end in runs), default=0.0)
            profile.append(
                {
                    "distance": float(distance),
                    "free_width": float(width),
                    "observed_columns": int(supported.sum()),
                }
            )
        diagnostics["profile"] = profile

        structural = forward_valid & (heights >= c["height_min"]) & (heights <= c["height_max"])
        column_depth = np.full(len(columns), np.nan)
        for j in range(len(columns)):
            values = advances[:, j][structural[:, j]]
            if len(values) >= c["jamb_min_samples"]:
                column_depth[j] = np.median(values)

        opening_edges, closing_edges = [], []
        for j in range(len(columns) - 1):
            jump = column_depth[j + 1] - column_depth[j]
            if not np.isfinite(jump) or abs(jump) < c["depth_jump"]:
                continue
            near_col, far_col = (j, j + 1) if jump > 0 else (j + 1, j)
            support = (
                structural[:, near_col]
                & forward_valid[:, far_col]
                & (advances[:, far_col] - advances[:, near_col] >= c["depth_jump"])
                & (abs(advances[:, near_col] - column_depth[near_col]) <= c["jamb_depth_tolerance"])
            )
            count = int(support.sum())
            possible = int(structural[:, near_col].sum())
            if (
                count < c["jamb_min_samples"]
                or count / max(possible, 1) < c["jamb_support_fraction"]
            ):
                continue
            extent = float(np.ptp(heights[:, near_col][support]))
            if extent < c["jamb_min_height"]:
                continue
            edge = (j, float(np.median(advances[:, near_col][support])), extent)
            (opening_edges if jump > 0 else closing_edges).append(edge)

        diagnostics.update(
            reason="no_bilateral_jambs",
            opening_edges=len(opening_edges),
            closing_edges=len(closing_edges),
        )
        best = None
        for first in opening_edges:
            for second in closing_edges:
                if second[0] <= first[0]:
                    continue
                if abs(first[1] - second[1]) > c["jamb_depth_tolerance"]:
                    continue
                plane = (first[1] + second[1]) / 2
                # Pixel-edge midpoints estimate the aperture rather than wall centers.
                edge_slopes = [
                    float(np.median((slopes[:, edge[0]] + slopes[:, edge[0] + 1]) / 2))
                    for edge in (first, second)
                ]
                lateral_edges = np.asarray(edge_slopes) * plane
                width = float(abs(lateral_edges[1] - lateral_edges[0]))
                minimum = max(c["portal_min_width"], c["robot_width"] + c["safety_margin"])
                if not minimum < width < c["portal_max_width"]:
                    diagnostics["reason"] = "opening_width_out_of_bounds"
                    continue
                start, stop = first[0] + 1, second[0] + 1
                supported, fractions, blocked, _ = slice_evidence(plane, c["clearance_depth"])
                # Every sampled aperture column must have observed free support.
                # Unknown returns reduce fractions, and an observed center obstruction
                # cannot be erased by a median of the otherwise open doorway.
                if (
                    not supported[start:stop].all()
                    or (blocked[start:stop] > c["max_blocked_fraction"]).any()
                ):
                    diagnostics["reason"] = "opening_not_observed_free"
                    continue
                wide_narrow_wide, waist_width = self._waist_evidence(profile, plane, width)
                midpoint = float(lateral_edges.mean())
                robot_xy = origin[:2] + forward * plane + left * midpoint
                cosine, sine = math.cos(pose.yaw), math.sin(pose.yaw)
                world_x = pose.x + cosine * robot_xy[0] - sine * robot_xy[1]
                world_y = pose.y + sine * robot_xy[0] + cosine * robot_xy[1]
                yaw = pose.yaw + math.atan2(forward[1], forward[0])
                yaw = math.atan2(math.sin(yaw), math.cos(yaw))
                confidence = min(
                    1.0,
                    0.55
                    + 0.25 * float(fractions[start:stop].mean())
                    + 0.15 * float(wide_narrow_wide),
                )
                candidate = PortalCandidate(
                    float(world_x),
                    float(world_y),
                    yaw,
                    width,
                    confidence,
                    "bottleneck" if wide_narrow_wide else "doorway_like",
                )
                if best is None or candidate.confidence > best[0].confidence:
                    best = (candidate, plane, wide_narrow_wide, waist_width)

        if best is None:
            return GeometryResult(None, diagnostics)
        candidate, plane, wide_narrow_wide, waist_width = best
        diagnostics.update(
            reason="candidate",
            crossing_distance=float(plane),
            wide_narrow_wide=wide_narrow_wide,
            waist_width=waist_width,
        )
        return GeometryResult(candidate, diagnostics)

    @staticmethod
    def _runs(mask):
        """Inclusive bounds of contiguous observed-free columns."""
        transitions = np.diff(np.pad(mask.astype(np.int8), (1, 1)))
        return zip(np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1) - 1)

    def _waist_evidence(self, profile, plane, opening_width):
        c = self.config
        before = [
            item["free_width"]
            for item in profile
            if plane - c["profile_before_distance"] - c["profile_step"]
            <= item["distance"]
            < plane - c["profile_clearance"]
        ]
        waist = [
            item["free_width"]
            for item in profile
            if plane <= item["distance"] <= plane + c["profile_step"]
        ]
        after = [
            item["free_width"]
            for item in profile
            if plane + c["profile_step"] < item["distance"] <= plane + c["profile_after_distance"]
        ]
        if not before or not waist or not after:
            return False, 0.0
        waist_width = min(waist)
        # The observed waist must agree with the paired jambs. Profile extrema from
        # an unrelated side opening cannot promote this candidate to a bottleneck.
        agreement = abs(waist_width - opening_width) <= max(0.20, opening_width * 0.25)
        ratio = c["profile_widening_ratio"]
        supported = (
            waist_width > 0
            and agreement
            and max(before) >= waist_width * ratio
            and max(after) >= waist_width * ratio
        )
        return bool(supported), float(waist_width)
