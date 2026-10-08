"""Metric verification of RGB doorway proposals; temporary ROI geometry only.

A small local wall strip (even a single visible side) estimates the opening
plane. Observed lower-box depth supplies a threshold fallback. Neither paired
jambs nor wide/narrow/wide geometry is required. Unknown depth is never free.
"""

import math
from dataclasses import dataclass

import numpy as np

from .geometry import PortalCandidate

DEFAULT_DEPTH_CONFIG = dict(
    depth_min=0.2,
    depth_max=7.0,
    robot_width=0.60,
    safety_margin=0.10,
    portal_min_width=0.75,
    portal_max_width=2.8,
    min_visual_confidence=0.25,
    clearance_depth=0.35,
    valid_fraction=0.65,
    free_fraction=0.65,
    sample_columns=25,
    sample_rows=25,
    height_min=0.12,
    height_max=1.6,
    wall_band_fraction=0.10,
    plane_tolerance=0.08,
    plane_min_span=0.12,
    floor_tolerance=0.35,
    max_plane_angle_deg=80.0,
    roi_inset=0.05,
)


@dataclass(frozen=True)
class VisualPortalObservation:
    candidate: PortalCandidate
    bbox_xyxy: tuple
    visual_confidence: float
    depth_confidence: float
    geometry_supported: bool = False


class DoorwayDepthVerifier:
    def __init__(self, config):
        if set(config) - set(DEFAULT_DEPTH_CONFIG):
            raise ValueError("unknown visual doorway depth config key")
        self.config = {**DEFAULT_DEPTH_CONFIG, **config}
        for key, value in self.config.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError("invalid visual depth config: " + key)
        c = self.config
        for key in ("valid_fraction", "free_fraction", "min_visual_confidence"):
            if c[key] > 1:
                raise ValueError("invalid fraction: " + key)
        if not (
            c["depth_min"] < c["depth_max"]
            and c["portal_min_width"] < c["portal_max_width"]
            and c["height_min"] < c["height_max"]
            and c["roi_inset"] < 0.4
            and c["wall_band_fraction"] < 0.5
            and c["max_plane_angle_deg"] < 90
        ):
            raise ValueError("invalid visual depth bounds")
        if any(
            not isinstance(c[k], int) or not 5 <= c[k] <= 100
            for k in ("sample_rows", "sample_columns")
        ):
            raise ValueError("visual depth sampling must be 5..100 integers")

    @staticmethod
    def _rays(camera, u, v):
        optical = np.stack(
            ((u - camera.cx) / camera.fx, (v - camera.cy) / camera.fy, np.ones_like(u)), axis=-1
        )
        transform = np.asarray(camera.camera_to_robot, dtype=float)
        return optical @ transform[:3, :3].T, transform[:3, 3]

    def _plane(self, frame, box):
        c = self.config
        camera = frame.camera
        x1, y1, x2, y2 = box
        # Floor-contact fallback is a metric estimate, not an invented wall depth.
        ray, origin = self._rays(camera, np.array((x1 + x2) / 2), np.array(y2))
        if ray[2] >= -1e-4:
            return None
        floor = origin - ray * (origin[2] / ray[2])
        # Use the observed threshold first. A Habitat navmesh origin can be
        # above the scanned floor; assuming ground is exactly robot z=0 would
        # move distant doorway planes metres toward the camera.
        us = np.linspace(x1, x2, 25)
        vs = np.full_like(us, y2)
        rays, _ = self._rays(camera, us, vs)
        depths = frame.depth[np.rint(vs).astype(int), np.rint(us).astype(int)]
        valid = np.isfinite(depths) & (depths >= c["depth_min"]) & (depths <= c["depth_max"])
        anchor = floor[:2]
        threshold_observed = valid.sum() >= len(us) * c["valid_fraction"]
        if threshold_observed:
            anchor = np.median((origin + rays[valid] * depths[valid, None])[:, :2], axis=0)
        wall_points = []
        pad = max(3, (x2 - x1) * c["wall_band_fraction"])
        for low, high in ((max(0, x1 - pad), x1), (x2, min(camera.width - 1, x2 + pad))):
            if high - low < 2:
                continue
            us, vs = np.meshgrid(np.linspace(low, high, 7), np.linspace(max(0, y1), y2, 20))
            rays, origin = self._rays(camera, us, vs)
            depths = frame.depth[np.rint(vs).astype(int), np.rint(us).astype(int)]
            points = origin + rays * np.where(np.isfinite(depths), depths, 0)[..., None]
            valid = (
                np.isfinite(depths)
                & (depths >= c["depth_min"])
                & (depths <= c["depth_max"])
                & (points[..., 2] >= c["height_min"])
                & (points[..., 2] <= c["height_max"])
            )
            wall_points.extend(points[valid, :2])
        points = np.asarray(wall_points)
        best = None
        if len(points) >= 6:
            # Bounded deterministic hypotheses, no map or retained point cloud.
            sample = points[np.linspace(0, len(points) - 1, min(24, len(points))).astype(int)]
            for a in range(len(sample)):
                for b in range(a + 1, len(sample)):
                    tangent = sample[b] - sample[a]
                    span = np.linalg.norm(tangent)
                    if span < c["plane_min_span"]:
                        continue
                    normal = np.array([-tangent[1], tangent[0]]) / span
                    if np.dot(normal, anchor - origin[:2]) < 0:
                        normal = -normal
                    if np.dot(normal, ray[:2]) / np.linalg.norm(ray[:2]) < math.cos(
                        math.radians(c["max_plane_angle_deg"])
                    ):
                        continue
                    offset = float(np.dot(normal, sample[a]))
                    if offset - np.dot(normal, origin[:2]) < c["depth_min"]:
                        continue
                    if abs(np.dot(normal, anchor) - offset) > c["floor_tolerance"]:
                        continue
                    errors = abs(points @ normal - offset)
                    inliers = errors < c["plane_tolerance"]
                    support = int(inliers.sum())
                    if support < 6:
                        continue
                    quality = support - 0.1 * abs(np.dot(normal, anchor) - offset)
                    if best is None or quality > best[0]:
                        best = (quality, normal, offset)
        if best is not None:
            return best[1], best[2], "wall_strip"
        if not threshold_observed:
            # RGB plus an assumed ground height cannot prove that a closed panel
            # is behind a doorway. Require a measured wall or threshold plane.
            return None
        normal = ray[:2] / np.linalg.norm(ray[:2])
        return (
            normal,
            float(np.dot(normal, anchor)),
            "depth_threshold",
        )

    def verify(self, frame, detection):
        c = self.config
        camera = frame.camera
        diag = {"reason": "low_visual_confidence", "visual_confidence": float(detection.confidence)}
        if detection.confidence < c["min_visual_confidence"]:
            return None, diag
        box = np.array(detection.bbox_xyxy, dtype=float)
        if box.shape != (4,) or not np.isfinite(box).all():
            raise ValueError("invalid doorway RGB box")
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= camera.width and 0 <= y1 < y2 <= camera.height):
            diag["reason"] = "invalid_or_outside_rgb_box"
            return None, diag
        if y2 >= camera.height - 1:
            diag["reason"] = "threshold_outside_view"
            return None, diag
        # Detector boxes use continuous pixel boundaries; array indices do not.
        box[2] = x2 = min(x2, camera.width - 1)
        if np.asarray(frame.depth).shape != (camera.height, camera.width):
            raise ValueError("depth and RGB camera mismatch")
        plane = self._plane(frame, box)
        if plane is None:
            diag["reason"] = "no_observed_threshold_or_plane"
            return None, diag
        normal, offset, method = plane
        # Project left/right box rays onto the measured plane, allowing oblique doors.
        rays, origin = self._rays(camera, np.array([x1, x2]), np.full(2, camera.cy))
        denom = rays[:, :2] @ normal
        if (denom <= 1e-4).any():
            diag["reason"] = "plane_outside_forward_view"
            return None, diag
        along = (offset - origin[:2] @ normal) / denom
        ends = origin[:2] + rays[:, :2] * along[:, None]
        width = float(np.linalg.norm(ends[1] - ends[0]))
        diag.update(width=width, plane_method=method)
        if (
            not max(c["robot_width"] + c["safety_margin"], c["portal_min_width"])
            < width
            < c["portal_max_width"]
        ):
            diag["reason"] = "opening_width_out_of_bounds"
            return None, diag
        inset = c["roi_inset"]
        us, vs = np.meshgrid(
            np.linspace(x1 + (x2 - x1) * inset, x2 - (x2 - x1) * inset, c["sample_columns"]),
            np.linspace(y1, y2, c["sample_rows"]),
        )
        rays, origin = self._rays(camera, us, vs)
        denom = rays[..., :2] @ normal
        t = np.divide(
            offset - origin[:2] @ normal, denom, out=np.zeros_like(denom), where=denom > 1e-4
        )
        heights = origin[2] + rays[..., 2] * t
        eligible = (
            (denom > 1e-4)
            & (t >= c["depth_min"])
            & (t <= c["depth_max"])
            & (heights >= c["height_min"])
            & (heights <= c["height_max"])
        )
        depths = np.asarray(frame.depth)[np.rint(vs).astype(int), np.rint(us).astype(int)]
        valid = (
            np.isfinite(depths) & (depths >= c["depth_min"]) & (depths <= c["depth_max"]) & eligible
        )
        points = origin + np.where(valid, depths, 0)[..., None] * rays
        beyond = points[..., :2] @ normal - offset
        totals = eligible.sum(axis=0)
        observed = valid.sum(axis=0) / np.maximum(totals, 1)
        free = (valid & (beyond >= c["clearance_depth"])).sum(axis=0) / np.maximum(totals, 1)
        passable = (totals >= 3) & (observed >= c["valid_fraction"]) & (free >= c["free_fraction"])
        changes = np.diff(np.pad(passable.astype(int), (1, 1)))
        spans = list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1) - 1))
        # Require a robot-width contiguous strip, not just some deep pixels.
        center_rays, _ = self._rays(camera, us[0], np.full(c["sample_columns"], camera.cy))
        den = center_rays[:, :2] @ normal
        xy = origin[:2] + center_rays[:, :2] * ((offset - origin[:2] @ normal) / den)[:, None]
        clear_width = max((float(np.linalg.norm(xy[b] - xy[a])) for a, b in spans), default=0.0)
        diag.update(
            clear_width=clear_width,
            valid_fraction=float(valid.sum() / max(eligible.sum(), 1)),
            free_fraction=float(
                (valid & (beyond >= c["clearance_depth"])).sum() / max(eligible.sum(), 1)
            ),
        )
        if clear_width < c["robot_width"] + c["safety_margin"]:
            diag["reason"] = "opening_not_observed_free"
            return None, diag
        center = (ends[0] + ends[1]) / 2
        cs, sn = math.cos(frame.pose.yaw), math.sin(frame.pose.yaw)
        world = np.array([frame.pose.x, frame.pose.y]) + np.array([[cs, -sn], [sn, cs]]) @ center
        yaw = frame.pose.yaw + math.atan2(normal[1], normal[0])
        yaw = math.atan2(math.sin(yaw), math.cos(yaw))
        depth_conf = diag["free_fraction"]
        candidate = PortalCandidate(
            float(world[0]),
            float(world[1]),
            yaw,
            width,
            min(float(detection.confidence), depth_conf),
            "visual_opening",
        )
        diag["reason"] = "accepted_visual_opening"
        return VisualPortalObservation(
            candidate, tuple(box), float(detection.confidence), depth_conf
        ), diag
