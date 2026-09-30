"""Record observed planar poses; never integrate commands or infer topology."""

from __future__ import annotations

import math
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Pose2D:
    """Metres and counter-clockwise radians in a fixed, named source frame."""

    stamp_ns: int
    x_m: float
    y_m: float
    yaw_rad: float
    frame_id: str
    source: str


class TrajectoryMap:
    """Thread-safe bounded history, with an independently updated current arrow.

    Reset explicitly at episode/clock/frame changes. Source timestamps order samples;
    the monotonic receive clock determines freshness, so simulation time is not mixed
    with wall time. The display origin translates the first position to (0, 0).
    """

    def __init__(self, min_step_m=0.05, jump_distance_m=2.0, max_points=4096, stale_after_s=1.0):
        for value in (min_step_m, jump_distance_m, stale_after_s):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("distances and timeout must be finite and positive")
        if isinstance(max_points, bool) or not isinstance(max_points, int) or max_points < 4:
            raise ValueError("max_points must be an integer >= 4")
        self.min_step_m = float(min_step_m)
        self.jump_distance_m = float(jump_distance_m)
        self.max_points = max_points
        self.stale_after_s = float(stale_after_s)
        self._lock = threading.RLock()
        self.reset()

    def reset(self):
        """Start a new coordinate/time session without retaining any old polyline."""
        with self._lock:
            self._session = uuid.uuid4().hex
            self._revision = 0
            self._current: Optional[Pose2D] = None
            self._origin = None
            self._received_s = None
            self._segments = []
            self._decimated = False
            self._history_full = False

    def update(self, pose: Pose2D, *, received_s=None):
        """Accept a fresh pose, or return False for duplicate/out-of-order stamps.

        Invalid values or a source/frame change raise ValueError without mutation.
        Even after history fills with discontinuities, the current arrow stays live.
        """
        if (
            isinstance(pose.stamp_ns, bool)
            or not isinstance(pose.stamp_ns, int)
            or pose.stamp_ns < 0
        ):
            raise ValueError("stamp_ns must be a nonnegative integer")
        if not all(math.isfinite(v) for v in (pose.x_m, pose.y_m, pose.yaw_rad)):
            raise ValueError("position and heading must be finite")
        if not all(isinstance(v, str) and v.strip() for v in (pose.frame_id, pose.source)):
            raise ValueError("frame_id and source must be nonempty strings")
        received = time.monotonic() if received_s is None else float(received_s)
        if not math.isfinite(received):
            raise ValueError("receive time must be finite")
        with self._lock:
            previous = self._current
            if previous is not None:
                if (pose.source, pose.frame_id) != (previous.source, previous.frame_id):
                    raise ValueError("reset before changing source or frame")
                if pose.stamp_ns <= previous.stamp_ns:
                    return False
                if received < self._received_s:
                    raise ValueError("receive clock must be monotonic")
            if self._origin is None:
                self._origin = (float(pose.x_m), float(pose.y_m))
            point = (pose.x_m - self._origin[0], pose.y_m - self._origin[1])
            if not self._history_full:
                if previous is None:
                    self._segments = [[point]]
                elif (
                    math.hypot(pose.x_m - previous.x_m, pose.y_m - previous.y_m)
                    > self.jump_distance_m
                ):
                    self._segments.append([point])
                elif math.dist(point, self._segments[-1][-1]) >= self.min_step_m:
                    self._segments[-1].append(point)
                self._bound_history()
            self._current = pose
            self._received_s = received
            self._revision += 1
            return True

    def _bound_history(self):
        # Reserve one slot for the live endpoint. Never join discontinuous segments.
        while sum(map(len, self._segments)) > self.max_points - 1:
            compact = [s[:1] + s[2:-1:2] + s[-1:] if len(s) > 2 else s[:] for s in self._segments]
            if sum(map(len, compact)) < sum(map(len, self._segments)):
                self._segments = compact
                self._decimated = True
            else:
                # Too many segment boundaries to simplify honestly: freeze history.
                self._segments[-1].pop()
                if not self._segments[-1]:
                    self._segments.pop()
                self._history_full = True
                break

    def snapshot(self, *, now_s=None):
        """Return detached JSON-compatible data; reading never advances the history."""
        now = time.monotonic() if now_s is None else float(now_s)
        with self._lock:
            segments = [[list(p) for p in segment] for segment in self._segments]
            current = None
            age = None
            if self._current is not None:
                p = self._current
                point = [p.x_m - self._origin[0], p.y_m - self._origin[1]]
                current = {
                    "x": point[0],
                    "y": point[1],
                    "yaw": math.atan2(math.sin(p.yaw_rad), math.cos(p.yaw_rad)),
                    "stamp_ns": p.stamp_ns,
                }
                if not self._history_full and segments[-1][-1] != point:
                    segments[-1].append(point)
                age = max(0.0, now - self._received_s)
            return {
                "schema_version": 1,
                "session_id": self._session,
                "revision": self._revision,
                "source": self._current.source if self._current else None,
                "frame_id": self._current.frame_id if self._current else None,
                "origin": list(self._origin) if self._origin else None,
                "start": [0.0, 0.0] if self._origin else None,
                "segments": segments,
                "current": current,
                "point_count": sum(map(len, segments)),
                "decimated": self._decimated,
                "history_full": self._history_full,
                "age_s": age,
                "stale_after_s": self.stale_after_s,
                "status": "waiting"
                if age is None
                else ("stale" if age > self.stale_after_s else "live"),
            }
