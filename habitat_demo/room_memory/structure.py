"""Bounded, single-level structural evidence from ephemeral RGB-D projections.

Coverage means a horizontal location has been seen at some height, NOT that a
robot can drive there. Only repeated tall-surface evidence is retained as walls;
there is no persistent furniture occupancy or raw cloud. Tall cabinets remain a
known ambiguity without a wall segmentation model.
"""

import math
from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image, ImageDraw


@dataclass
class RegionConfig:
    enabled: bool = True
    cell_size: float = 0.05
    grid_size: int = 1024
    update_interval: float = 0.75
    depth_min: float = 0.2
    depth_max: float = 6.0
    sample_columns: int = 160
    sample_rows: int = 64
    height_min: float = 0.10
    height_max: float = 2.4
    wall_min_span: float = 0.65
    wall_min_top: float = 1.2
    wall_bin_size: float = 0.12
    wall_min_length: float = 0.35
    wall_confirm_frames: int = 2
    wall_padding: float = 0.10
    portal_padding: float = 0.12
    seed_radius: float = 0.3
    floor_tolerance: float = 0.35
    min_region_area: float = 0.02

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError("regions.enabled must be boolean")
        for k, v in asdict(self).items():
            if k == "enabled":
                continue
            if (
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(v)
                or v <= 0
            ):
                raise ValueError("invalid region configuration: " + k)
        for k in ("grid_size", "sample_columns", "sample_rows", "wall_confirm_frames"):
            if type(getattr(self, k)) is not int:
                raise ValueError("regions integer required: " + k)
        if not (32 <= self.grid_size <= 2048 and 0.02 <= self.cell_size <= 0.5):
            raise ValueError("region grid requires 32..2048 cells, 0.02..0.5 m resolution")
        if not (8 <= self.sample_columns <= 320 and 8 <= self.sample_rows <= 160):
            raise ValueError("region sampling exceeds bounded limits")
        if not (self.depth_min < self.depth_max <= 20 and self.height_min < self.height_max):
            raise ValueError("invalid region depth/height bounds")
        if self.wall_confirm_frames < 2 or self.wall_confirm_frames > 255:
            raise ValueError("wall_confirm_frames must be 2..255")
        if (
            not 0.025 <= self.wall_bin_size <= 0.5
            or math.ceil(self.wall_min_length / self.wall_bin_size / 2) > 8
            or math.ceil(self.wall_padding / self.cell_size) > 8
            or math.ceil(self.seed_radius / self.cell_size) > 32
            or self.portal_padding > 0.5
        ):
            raise ValueError("region neighbourhood exceeds bounded work limits")


class StructuralMapBuilder:
    def __init__(self, config=None):
        try:
            self.config = (
                config if isinstance(config, RegionConfig) else RegionConfig(**(config or {}))
            )
        except TypeError as exc:
            raise ValueError("unknown regions configuration key") from exc
        n = self.config.grid_size
        self.coverage = np.zeros((n, n), dtype=np.uint8)
        self.wall_hits = np.zeros((n, n), dtype=np.uint8)
        self.origin = None
        self.floor_height = None
        self.last_update = -float("inf")
        self.revision = 0
        self.status = "waiting" if self.config.enabled else "disabled"
        self.clipped = False

    @property
    def walls(self):
        return self.wall_hits >= self.config.wall_confirm_frames

    def cell(self, x, y):
        return (
            int(math.floor((x - self.origin[0]) / self.config.cell_size + 1e-8)),
            int(math.floor((y - self.origin[1]) / self.config.cell_size + 1e-8)),
        )

    def world(self, x, y):
        return {
            "x": round(self.origin[0] + x * self.config.cell_size, 5),
            "y": round(self.origin[1] + y * self.config.cell_size, 5),
        }

    def observe(self, frame):
        c = self.config
        if not c.enabled or frame.depth is None:
            return False
        elevation = getattr(frame, "elevation", None)
        if elevation is not None:
            if not math.isfinite(elevation):
                raise ValueError("nonfinite region elevation")
            if self.floor_height is None:
                self.floor_height = elevation
            if abs(elevation - self.floor_height) > c.floor_tolerance:
                self.status = "floor_mismatch"
                return False
        if frame.timestamp - self.last_update < c.update_interval:
            return False
        depth, camera = np.asarray(frame.depth), frame.camera
        transform = np.asarray(camera.camera_to_robot, dtype=float)
        if (
            depth.shape != (camera.height, camera.width)
            or transform.shape != (4, 4)
            or not np.isfinite(transform).all()
            or not all(math.isfinite(v) for v in (camera.fx, camera.fy, camera.cx, camera.cy))
            or camera.fx <= 0
            or camera.fy <= 0
        ):
            raise ValueError("invalid structural depth/camera calibration")
        if not all(
            math.isfinite(v) for v in (frame.timestamp, frame.pose.x, frame.pose.y, frame.pose.yaw)
        ):
            raise ValueError("invalid structural pose/timestamp")
        n = c.grid_size
        if self.origin is None:
            self.origin = (
                math.floor(frame.pose.x / c.cell_size) * c.cell_size - n * c.cell_size / 2,
                math.floor(frame.pose.y / c.cell_size) * c.cell_size - n * c.cell_size / 2,
            )
        ix, iy = self.cell(frame.pose.x, frame.pose.y)
        if not (0 <= ix < n and 0 <= iy < n):
            self.clipped = True
            self.status = "capacity_reached"
            return False
        self.last_update = frame.timestamp
        _, points, valid, rotation, translation, origin_xy, endpoints, good = self.project_columns(frame)
        if not valid.any():
            self.status = "insufficient_depth"
            return False
        mask = Image.new("L", (n, n))
        draw = ImageDraw.Draw(mask)
        start = self.cell(*origin_xy)
        for i in range(len(endpoints) - 1):
            if not (good[i] and good[i + 1]):
                continue
            a, b = endpoints[i] - origin_xy, endpoints[i + 1] - origin_xy
            ra, rb = np.linalg.norm(a), np.linalg.norm(b)
            radius = min(ra, rb)
            if radius <= 1e-6:
                continue
            ends = [
                self.cell(*(origin_xy + a * radius / ra)),
                self.cell(*(origin_xy + b * radius / rb)),
            ]
            self.clipped |= any(not (0 <= x < n and 0 <= y < n) for x, y in ends)
            draw.polygon([start, *ends], fill=1)
        self.coverage |= np.asarray(mask)
        self._walls(points[valid], rotation, translation)
        self.revision += 1
        self.status = "capacity_reached" if self.clipped else "ready"
        return True

    def project_columns(self, frame):
        """Pure per-frame farthest rays shared by structure and semantic projection.

        Uses structural depth/height validity and the identical row/column samples.
        Does not update coverage, timestamps, walls, or room ownership.
        """
        c, camera = self.config, frame.camera
        depth = np.asarray(frame.depth)
        transform = np.asarray(camera.camera_to_robot, dtype=float)
        rows = np.linspace(0, camera.height - 1, min(c.sample_rows, camera.height), dtype=int)
        cols = np.linspace(0, camera.width - 1, min(c.sample_columns, camera.width), dtype=int)
        u, v = np.meshgrid(cols, rows)
        z = depth[np.ix_(rows, cols)].astype(float)
        valid = np.isfinite(z) & (z >= c.depth_min) & (z < c.depth_max)
        z = np.where(valid, z, 0)
        rays = np.stack(
            ((u - camera.cx) / camera.fx, (v - camera.cy) / camera.fy, np.ones_like(u)), -1
        )
        points = (rays * z[..., None]) @ transform[:3, :3].T + transform[:3, 3]
        valid &= (points[..., 2] >= c.height_min) & (points[..., 2] <= c.height_max)
        co, si = math.cos(frame.pose.yaw), math.sin(frame.pose.yaw)
        rotation = np.array([[co, -si], [si, co]])
        translation = np.array([frame.pose.x, frame.pose.y])
        origin_xy = transform[:2, 3] @ rotation.T + translation
        xy = points[..., :2] @ rotation.T + translation
        # Farthest actually measured ray per image column covers horizontal
        # membership above low furniture. It never clears beyond that ray's hit.
        ranges = np.linalg.norm(xy - origin_xy, axis=-1)
        choices = np.argmax(np.where(valid, ranges, -1), axis=0)
        endpoints = xy[choices, np.arange(len(cols))]
        good = valid.any(axis=0)
        return cols, points, valid, rotation, translation, origin_xy, endpoints, good

    def _walls(self, points, rotation, translation):
        c = self.config
        bins = np.floor(points[:, :2] / c.wall_bin_size).astype(int)
        unique, inverse = np.unique(bins, axis=0, return_inverse=True)
        lo, hi = np.full(len(unique), np.inf), np.full(len(unique), -np.inf)
        np.minimum.at(lo, inverse, points[:, 2])
        np.maximum.at(hi, inverse, points[:, 2])
        supported = (hi - lo >= c.wall_min_span) & (hi >= c.wall_min_top)
        # Require neighbouring tall bins, not a single narrow post/chair leg.
        candidates = unique[supported]
        if not len(candidates):
            return
        occupied = {tuple(p) for p in candidates}
        radius = max(1, int(math.ceil(c.wall_min_length / c.wall_bin_size / 2)))
        keep = [
            sum(
                (x + dx, y + dy) in occupied
                for dx in range(-radius, radius + 1)
                for dy in range(-radius, radius + 1)
            )
            >= max(2, radius + 1)
            for x, y in candidates
        ]
        chosen = {tuple(p) for p in candidates[np.asarray(keep)]}
        # Use the measured endpoints, not bin centres, to avoid shifting a wall.
        select = np.fromiter((tuple(p) in chosen for p in bins), bool, len(bins))
        xy = points[select, :2] @ rotation.T + translation
        cells = {self.cell(x, y) for x, y in xy}
        n = c.grid_size
        for x, y in cells:
            if 0 <= x < n and 0 <= y < n:
                self.wall_hits[y, x] = min(255, int(self.wall_hits[y, x]) + 1)

    def diagnostics(self):
        return dict(
            enabled=self.config.enabled,
            status=self.status,
            revision=self.revision,
            cell_size=self.config.cell_size,
            grid_size=self.config.grid_size,
            observed_area_m2=round(
                float(np.count_nonzero(self.coverage)) * self.config.cell_size**2, 3
            ),
            wall_cells=int(self.walls.sum()),
            floor_height=self.floor_height,
            origin=self.origin,
            capacity_reached=self.clipped,
        )

    def save(self, path):
        """Compact structural evidence only; no RGB-D, clouds or object layer."""
        if self.origin is not None:
            with open(path, "wb") as out:
                np.savez_compressed(
                    out,
                    coverage=self.coverage,
                    wall_hits=self.wall_hits,
                    origin=self.origin,
                    cell_size=self.config.cell_size,
                    revision=self.revision,
                )
