"""World-aligned circular memory. Eviction is spatial, never time-based."""
import math

import numpy as np


class RollingGrid:
    def __init__(self, config, height_bins=10):
        self.config = config
        self.n = 2 * math.ceil(config.window_radius_m / config.resolution_m) + 3
        self.origin = None
        self.pose = (0., 0., 0.)
        self.revision = 0
        self.hits = np.zeros((self.n, self.n), dtype=np.uint32)
        self.free = np.zeros((self.n, self.n), dtype=bool)
        self.ground = np.zeros_like(self.free)
        # One-time user-authorized birth prior, distinct from measured evidence.
        self.initial_free = np.zeros_like(self.free)
        # Near lower-frustum priors are also world-fixed, not measured floor.
        self.blind_free = np.zeros_like(self.free)
        self.blind_initialized = np.zeros_like(self.free)
        self.clear_counts = np.zeros((height_bins, self.n, self.n), dtype=np.uint8)
        self.valid = np.zeros_like(self.free)

    def cell(self, x, y):
        r = self.config.resolution_m
        return math.floor(x / r) - self.origin[0], math.floor(y / r) - self.origin[1]

    def world(self, x, y):
        r = self.config.resolution_m
        return (x + self.origin[0] + .5) * r, (y + self.origin[1] + .5) * r

    def recenter(self, pose):
        r = self.config.resolution_m
        half = self.n // 2
        origin = (math.floor(pose[0] / r) - half, math.floor(pose[1] / r) - half)
        if self.origin is not None and origin != self.origin:
            dx, dy = origin[0] - self.origin[0], origin[1] - self.origin[1]
            for array in self._layers():
                old = array.copy()
                array[...] = 0
                if abs(dx) < self.n and abs(dy) < self.n:
                    sx, sy = max(0, dx), max(0, dy)
                    tx, ty = max(0, -dx), max(0, -dy)
                    w, h = self.n - abs(dx), self.n - abs(dy)
                    array[..., ty:ty+h, tx:tx+w] = old[..., sy:sy+h, sx:sx+w]
        self.origin, self.pose = origin, tuple(pose)
        yy, xx = np.indices(self.free.shape)
        wx, wy = self.world(xx, yy)
        self.valid = np.hypot(wx - pose[0], wy - pose[1]) <= self.config.window_radius_m
        for array in self._layers():
            array[..., ~self.valid] = 0
        self.revision += 1

    def labels(self):
        labels = np.zeros_like(self.hits, dtype=np.uint8)  # 0 unknown, 1 free, 2 obstacle
        labels[self.valid & (self.free | self.initial_free | self.blind_free)] = 1
        labels[self.valid & (self.hits != 0)] = 2
        return labels

    def _layers(self):
        return (self.hits, self.free, self.ground, self.initial_free,
                self.blind_free, self.blind_initialized, self.clear_counts)
