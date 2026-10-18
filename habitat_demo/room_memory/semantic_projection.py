"""Frame-local RGB-D evidence projected into a fixed world grid, never Room IDs.

Inspired by ApexNav/VLFM's visibility masks and FOV weighting. This independently
implemented layer uses structural farthest-depth rays and accumulated multiclass means;
CLIP scores are relative scene evidence, not per-pixel semantic segmentation.
"""
import base64
import io
import math
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
from PIL import Image, ImageOps

from .semantics import SemanticFrame

ROOM_COLORS = {
    'bedroom': '#a78bfa', 'living room': '#f59e0b', 'bathroom': '#38bdf8',
    'kitchen': '#ef4444', 'dining room': '#f472b6', 'office room': '#6366f1',
    'gym': '#84cc16',
    'corridor': '#e2c765',
}


@dataclass
class ProjectionConfig:
    enabled: bool = True
    view_mode: str = 'full'  # Optional retained split mode: left_right.
    interval: float = 1.5  # wall seconds; pending frames are replaceable
    max_range: Optional[float] = None  # Legacy option accepted but no longer limits projection.
    filter_low_confidence: bool = True
    low_top1_threshold: float = .35
    top3_spread_threshold: float = .10
    # Legacy geometry keys remain parseable; regions config owns visibility now.
    depth_min: float = .2
    depth_max: float = 15.0
    height_min: float = .2
    height_max: float = 1.6
    endpoint_margin: float = .10
    sample_columns: int = 160
    sample_rows: int = 32
    max_weight: float = 20.0

    def __post_init__(self):
        if self.view_mode not in ('full', 'left_right'):
            raise ValueError('semantic projection view_mode must be full or left_right')
        if not isinstance(self.enabled, bool) or not isinstance(self.filter_low_confidence, bool):
            raise ValueError('semantic projection enabled/filter flags must be boolean')
        if self.max_range is not None and (isinstance(self.max_range, bool)
                or not isinstance(self.max_range, (int, float))
                or not math.isfinite(self.max_range) or self.max_range < 0):
            raise ValueError('invalid legacy max_range')
        for key, value in asdict(self).items():
            if key not in ('enabled', 'filter_low_confidence', 'max_range', 'view_mode') and (isinstance(value, bool) or not isinstance(value, (int, float))
                                     or not math.isfinite(value) or value <= 0):
                raise ValueError('invalid semantic projection option: ' + key)
        if not (self.depth_min < self.depth_max <= 30
                and self.height_min < self.height_max <= 3 and self.max_weight <= 100
                and self.endpoint_margin < self.depth_max and self.interval >= .1
                and self.low_top1_threshold <= 1 and self.top3_spread_threshold <= 1):
            raise ValueError('invalid semantic projection bounds')
        for name, upper in [('sample_columns', 320), ('sample_rows', 100)]:
            if type(getattr(self, name)) is not int or not 8 <= getattr(self, name) <= upper:
                raise ValueError('invalid semantic projection samples')


@dataclass(frozen=True)
class ViewPatch:
    frame: SemanticFrame
    indices: np.ndarray
    weights: np.ndarray
    crop: tuple


class SemanticEvidenceMap:
    def __init__(self, structure, labels, config=None):
        self.structure = structure
        self.config = config if isinstance(config, ProjectionConfig) else ProjectionConfig(**(config or {}))
        self.labels = tuple(labels)
        if not 2 <= len(self.labels) <= 16 or len(set(self.labels)) != len(self.labels):
            raise ValueError('semantic map requires 2..16 unique room labels')
        n = structure.config.grid_size
        self.values = np.zeros((n, n, len(labels)), np.float32)
        self.weights = np.zeros((n, n), np.float32)
        self.revision = 0
        self.timestamp = None
        self.filter_timestamp = None
        self.accepted_views = self.rejected_views = 0
        self.source_pose = None
        self.last_views = []
        self.palette = [ROOM_COLORS.get(label, '#94a3b8') for label in labels]
        self._snapshot = None

    def project(self, frame):
        """Return frozen world cell patches and square RGB crops from ONE RGB-D frame."""
        c, s, cam = self.config, self.structure, frame.camera
        if not c.enabled or s.origin is None or not s.config.enabled:
            return []
        if frame.elevation is not None and s.floor_height is not None:
            if abs(frame.elevation - s.floor_height) > s.config.floor_tolerance:
                return []
        depth = np.asarray(frame.depth)
        transform = np.asarray(cam.camera_to_robot, dtype=float)
        if (depth.shape != (cam.height, cam.width) or transform.shape != (4, 4)
                or not np.isfinite(transform).all() or not np.isfinite([cam.fx, cam.fy, cam.cx, cam.cy,
                frame.timestamp, frame.pose.x, frame.pose.y, frame.pose.yaw]).all()
                or min(cam.fx, cam.fy) <= 0):
            raise ValueError('invalid synchronized semantic RGB-D/calibration/pose')
        image = Image.open(io.BytesIO(frame.rgb)).convert('RGB')
        if image.size != (cam.width, cam.height):
            raise ValueError('semantic RGB and depth must be pixel aligned')
        columns, _, _, _, _, origin, far_endpoints, far_valid = s.project_columns(frame)
        walls = s.walls
        n = s.config.grid_size
        patches = []
        # Default full-frame recognition; retain the previous two-view design as
        # an explicit option. Letterboxing preserves the entire source FOV.
        view_count = 2 if c.view_mode == 'left_right' else 1
        for view in range(view_count):
            left = view * cam.width // view_count
            right = (view + 1) * cam.width // view_count
            top, bottom = 0, cam.height
            crop = (left, top, right, bottom)
            selected = (columns >= left) & (columns < right)
            cols = columns[selected]
            good = far_valid[selected]
            endpoints = far_endpoints[selected] - origin
            cells = {}
            center_angle = math.atan(((left + right) / 2 - cam.cx) / cam.fx)
            half_angle = max(abs(math.atan((left - cam.cx) / cam.fx) - center_angle),
                             abs(math.atan((right - cam.cx) / cam.fx) - center_angle))
            distances = np.linalg.norm(endpoints, axis=-1)
            for j, col in enumerate(cols):
                if not good[j]:
                    continue
                direction = endpoints[j] / max(distances[j], 1e-8)
                samples = [(direction, distances[j], float(col))]
                # Cover the angular interval between measured neighbouring columns
                # with the SHORTER depth. Never bridge an invalid-depth column.
                if j + 1 < len(cols) and good[j + 1]:
                    next_direction = endpoints[j + 1] / max(distances[j + 1], 1e-8)
                    short = min(distances[j], distances[j + 1])
                    gap = float(np.linalg.norm(direction - next_direction)) * short
                    divisions = max(1, int(math.ceil(gap / (s.config.cell_size / 2))))
                    for ratio in np.arange(1, divisions) / divisions:
                        between = (1 - ratio) * direction + ratio * next_direction
                        between /= max(np.linalg.norm(between), 1e-8)
                        samples.append((between, short, (1 - ratio) * col + ratio * cols[j + 1]))
                for direction, distance, pixel in samples:
                    limit = distance
                    if limit <= 0:
                        continue
                    angle = math.atan((pixel - cam.cx) / cam.fx) - center_angle
                    weight = math.cos(min(abs(angle) / half_angle, 1) * math.pi / 2) ** 2
                    if weight <= 1e-6:
                        continue
                    travel = np.append(np.arange(0, limit, s.config.cell_size / 2), limit)
                    positions = origin + travel[:, None] * direction
                    indices = np.floor((positions - s.origin) / s.config.cell_size + 1e-8).astype(int)
                    ix, iy = indices.T
                    inside = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
                    obstructed = ~inside
                    obstructed[inside] |= walls[iy[inside], ix[inside]]
                    stops = np.flatnonzero(obstructed)
                    end = int(stops[0]) if len(stops) else len(travel)
                    ix, iy, travel = ix[:end], iy[:end], travel[:end]
                    observed = s.coverage[iy, ix].astype(bool)
                    for index in np.unique(iy[observed] * n + ix[observed]).tolist():
                        cells[index] = max(cells.get(index, 0), weight)
            if not cells:
                continue
            jpeg = io.BytesIO()
            side = max(right - left, bottom - top)
            ImageOps.pad(image.crop(crop), (side, side), color=(114, 114, 114)).save(jpeg, 'JPEG', quality=90)
            patches.append(ViewPatch(SemanticFrame(f'projection-{frame.timestamp}-{view}', jpeg.getvalue()),
                                     np.array(list(cells), np.int64), np.array(list(cells.values()), np.float32), crop))
        return patches

    def fuse(self, timestamp, predictions, pose=None):
        if self.filter_timestamp is not None and timestamp <= self.filter_timestamp:
            return False
        # Aggregate within a frame before temporal fusion, so overlapping rays
        # or contexts cannot give the same camera observation multiple votes.
        evidence, mass = {}, {}
        views = []
        for patch, probabilities in predictions:
            scores = np.array([probabilities.get(label, 0) for label in self.labels], np.float32)
            if not np.isfinite(scores).all() or (scores < 0).any() or scores.sum() <= 0:
                raise ValueError('invalid room probability vector')
            scores /= scores.sum()
            top3 = np.sort(scores)[-3:][::-1]
            rejected = (self.config.filter_low_confidence and len(top3) == 3
                        and (float(top3[0]) < self.config.low_top1_threshold
                             or float(top3[0] - top3[2]) < self.config.top3_spread_threshold))
            views.append(dict(frame_id=patch.frame.frame_id, crop=list(patch.crop),
                              probabilities=dict(zip(self.labels, scores.tolist())),
                              accepted=not rejected, reason='low_or_ambiguous_top3' if rejected else 'accepted',
                              top3_scores=top3.tolist(), top3_spread=float(top3[0] - top3[-1])))
            if rejected:
                continue
            for index, weight in zip(patch.indices.tolist(), patch.weights.tolist()):
                evidence[index] = evidence.get(index, 0) + weight * scores
                mass[index] = mass.get(index, 0) + weight
        self.last_views = views
        self.filter_timestamp = timestamp
        self.accepted_views = sum(v['accepted'] for v in views)
        self.rejected_views = len(views) - self.accepted_views
        self._snapshot = None
        if not mass:
            return False
        indices = np.array(list(mass), np.int64)
        votes = np.stack([evidence[i] / mass[i] for i in indices])
        new_weight = np.minimum(np.array([mass[i] for i in indices]), 1.)
        old = self.weights.ravel()[indices]
        total = old + new_weight
        values = self.values.reshape(-1, len(self.labels))
        values[indices] = (old[:, None] * values[indices] + new_weight[:, None] * votes) / total[:, None]
        self.weights.ravel()[indices] = np.minimum(total, self.config.max_weight)
        self.timestamp, self.source_pose = timestamp, pose
        self.last_views = views
        self.revision += 1
        self._snapshot = None
        return True

    def snapshot(self):
        # Re-mask new walls, even if no new semantic result has arrived.
        key = (self.revision, self.structure.revision)
        if self._snapshot is not None and self._snapshot[0] == key:
            return self._snapshot[1]
        s = self.structure
        result = dict(enabled=True, status=('filtered' if self.rejected_views and not self.accepted_views
                                              else 'ready' if self.revision else 'waiting'), revision=self.revision,
                      labels=list(self.labels), colors=self.palette, view_mode=self.config.view_mode,
                      source_timestamp=self.timestamp,
                      source_pose=self.source_pose, views=self.last_views, cell_size=s.config.cell_size,
                      filter_timestamp=self.filter_timestamp, accepted_views=self.accepted_views,
                      rejected_views=self.rejected_views,
                      filter=dict(enabled=self.config.filter_low_confidence,
                                  low_top1_threshold=self.config.low_top1_threshold,
                                  top3_spread_threshold=self.config.top3_spread_threshold),
                      fusion='bounded_weighted_probability_mean', image=None, bounds=None,
                      evidence_cells=0)
        mask = (self.weights > 0) & s.coverage.astype(bool) & ~s.walls
        ys, xs = np.nonzero(mask)
        if len(xs):
            x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
            probabilities = self.values[y0:y1, x0:x1]
            winner = probabilities.argmax(axis=-1)
            colors = np.array([tuple(bytes.fromhex(color[1:])) for color in self.palette], np.uint8)
            rgba = np.zeros((*winner.shape, 4), np.uint8)
            rgba[..., :3] = colors[winner]
            rgba[..., 3] = np.where(mask[y0:y1, x0:x1], 85 + 140 * probabilities.max(axis=-1), 0).astype(np.uint8)
            out = io.BytesIO()
            Image.fromarray(np.flipud(rgba)).save(out, 'PNG')
            a, b = s.world(x0, y0), s.world(x1, y1)
            result.update(image='data:image/png;base64,' + base64.b64encode(out.getvalue()).decode(),
                          bounds=[a['x'], a['y'], b['x'], b['y']], evidence_cells=len(xs))
        self._snapshot = (key, result)
        return result

    def save(self, path):
        indices = np.flatnonzero(self.weights)
        with open(path, 'wb') as out:
            np.savez_compressed(out, indices=indices, probabilities=self.values.reshape(-1, len(self.labels))[indices],
                                weights=self.weights.ravel()[indices], labels=np.array(self.labels),
                                origin=np.array(self.structure.origin), cell_size=self.structure.config.cell_size,
                                grid_size=self.structure.config.grid_size, timestamp=self.timestamp,
                                revision=self.revision)
