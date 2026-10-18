"""Orientation-aware configuration space, preserving the offset base origin."""
import math
from copy import copy

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.signal import fftconvolve


class CollisionChecker:
    def __init__(self, grid, footprint):
        self.grid, self.config = grid, grid.config
        self.footprint = np.asarray(footprint, dtype=float)
        if self.footprint.shape != (4, 2) or not np.isfinite(self.footprint).all():
            raise ValueError('finite rectangular footprint required')
        self.lo, self.hi = self.footprint.min(axis=0), self.footprint.max(axis=0)
        self.radius = float(np.linalg.norm(self.footprint, axis=1).max())
        self.cache = {}
        self.labels = grid.labels()
        self.allow_unknown = False
        self.clearance = distance_transform_edt(self.labels == 1) * self.config.resolution_m

    def for_planning(self):
        """Separate permissive query view; never change the execution checker/map."""
        view = copy(self)
        view.allow_unknown = True
        view.clearance = distance_transform_edt(self.labels != 2) * self.config.resolution_m
        return view

    def layers(self, yaw):
        c = self.config
        angle_step = 2 * math.pi / c.heading_bins
        key = round(yaw / angle_step) % c.heading_bins
        if key not in self.cache:
            yaw = key * angle_step
            # Include cell-centre quantization and the heading-bin rounding error.
            pad = c.safety_margin_m + c.resolution_m * math.sqrt(2) + self.radius * angle_step / 2
            radius = math.ceil((self.radius + pad) / c.resolution_m)
            yy, xx = np.mgrid[-radius:radius+1, -radius:radius+1] * c.resolution_m
            co, si = math.cos(yaw), math.sin(yaw)
            x, y = co * xx + si * yy, -si * xx + co * yy
            kernel = ((x >= self.lo[0] - pad) & (x <= self.hi[0] + pad)
                      & (y >= self.lo[1] - pad) & (y <= self.hi[1] + pad))
            kernel = kernel[::-1, ::-1]  # Obstacle ⊕ reflected footprint.
            unknown = np.pad(self.labels == 0, radius, constant_values=True)
            self.cache[key] = (
                fftconvolve(self.labels == 2, kernel, mode='same') > .5,
                (fftconvolve(unknown, kernel, mode='same') > .5)[radius:-radius, radius:-radius],
            )
        return self.cache[key]

    def reason(self, pose, *, allow_outside=False):
        if len(pose) != 3 or not all(math.isfinite(v) for v in pose):
            return 'invalid_pose'
        ix, iy = self.grid.cell(*pose[:2])
        unobserved_allowed = self.allow_unknown and allow_outside
        if not (0 <= ix < self.grid.n and 0 <= iy < self.grid.n):
            return 'clear' if unobserved_allowed else 'outside_window'
        if not self.grid.valid[iy, ix] and not unobserved_allowed:
            return 'outside_window'
        occupied, unknown = self.layers(pose[2])
        if occupied[iy, ix]:
            return 'occupied'
        if unknown[iy, ix] and not self.allow_unknown:
            return 'unknown'
        return 'clear'

    def segment(self, a, b, *, allow_outside=False):
        allow_outside = allow_outside and self.allow_unknown
        # Reject before discretization: even finite model coordinates can be huge.
        # A segment with either endpoint outside the window cannot be authorized.
        for p in (a, b):
            reason = self.reason(p, allow_outside=allow_outside)
            if reason != 'clear':
                return reason
        angle = math.atan2(math.sin(b[2] - a[2]), math.cos(b[2] - a[2]))
        lo, hi = 0., 1.
        dx, dy = b[0] - a[0], b[1] - a[1]
        if not all(math.isfinite(v) for v in (dx, dy, angle)):
            return 'invalid_pose'
        if allow_outside:
            # Clip prediction validation to the finite map rectangle. Do not
            # allocate millions of samples for a remote unobserved endpoint;
            # still check every in-map crossing for known inflated obstacles.
            for axis, delta in enumerate((dx, dy)):
                low = self.grid.origin[axis] * self.config.resolution_m
                high = low + self.grid.n * self.config.resolution_m
                if abs(delta) < 1e-12:
                    if not low <= a[axis] <= high:
                        return 'clear'
                else:
                    enter, leave = sorted(((low - a[axis]) / delta, (high - a[axis]) / delta))
                    lo, hi = max(lo, enter), min(hi, leave)
            if hi < lo:
                return 'clear'
        travel = (math.hypot(dx, dy) + self.radius * abs(angle)) * (hi - lo)
        count = max(1, math.ceil(travel / (self.config.resolution_m / 2)))
        for t in np.linspace(lo, hi, count + 1):
            reason = self.reason((a[0] + t * dx, a[1] + t * dy, a[2] + t * angle),
                                 allow_outside=allow_outside)
            if reason != 'clear':
                return reason
        return 'clear'
