"""Temporary RGB-D ray evidence, with height-aware clearing of known obstacles."""
import math

import numpy as np
from PIL import Image, ImageDraw

from .models import pose_tuple


class DepthMapper:
    def __init__(self, config, robot_height):
        self.config, self.height = config, robot_height + config.height_margin_m
        self.bins = math.ceil(self.height / config.height_bin_m)
        if self.bins > 30:
            raise ValueError('robot height exceeds local height-mask capacity')
        self.last_stamp = None

    def update(self, grid, observation):
        c = self.config
        pose = pose_tuple(observation['pose'])
        stamp = (observation['sim_time'], pose)
        if stamp == self.last_stamp:
            return False
        depth = np.asarray(observation['depth'], dtype=float)
        camera = observation['depth_camera']
        h, w = int(camera['height']), int(camera['width'])
        transform = np.asarray(camera['camera_to_robot'], dtype=float)
        intrinsics = np.array([camera[k] for k in ('fx', 'fy', 'cx', 'cy')], dtype=float)
        if (depth.shape != (h, w) or transform.shape != (4, 4)
                or not np.isfinite(transform).all() or not np.isfinite(intrinsics).all()
                or min(intrinsics[:2]) <= 0 or not math.isfinite(float(observation['sim_time']))):
            raise ValueError('invalid calibrated RGB-D observation')
        self.last_stamp = stamp
        grid.recenter(pose)
        rows = np.linspace(0, h-1, min(h, c.sample_rows), dtype=int)
        cols = np.linspace(0, w-1, min(w, c.sample_columns), dtype=int)
        u, v = np.meshgrid(cols, rows)
        z = depth[np.ix_(rows, cols)]
        valid = np.isfinite(z) & (z >= c.depth_min_m) & (z <= c.depth_max_m)
        z = np.where(valid, z, 0.)
        points = np.stack(((u - camera['cx']) * z / camera['fx'],
                           (v - camera['cy']) * z / camera['fy'], z), axis=-1)
        points = points @ transform[:3, :3].T + transform[:3, 3]
        co, si = math.cos(pose[2]), math.sin(pose[2])
        rot = np.array([[co, -si], [si, co]])
        xy = points[..., :2] @ rot.T + pose[:2]
        start = transform[:2, 3] @ rot.T + pose[:2]
        floor = valid & (abs(points[..., 2]) <= c.ground_tolerance_m)
        obstacles = valid & (points[..., 2] > c.ground_tolerance_m) & (points[..., 2] <= self.height)

        # Floor support joins only adjacent floor samples with short metric edges.
        # It never fills an entire frustum across an occluder or depth discontinuity.
        image = Image.new('L', (grid.n, grid.n))
        draw = ImageDraw.Draw(image)
        cells = np.floor(xy[floor] / c.resolution_m).astype(int) - grid.origin
        inside = (cells >= 0).all(axis=1) & (cells < grid.n).all(axis=1)
        xx, yy = cells[inside].T
        grid.ground[yy, xx] = True
        quads = floor[:-1, :-1] & floor[1:, :-1] & floor[:-1, 1:] & floor[1:, 1:]
        yy, xx = np.nonzero(quads)
        polygons = np.stack((xy[yy, xx], xy[yy, xx+1], xy[yy+1, xx+1], xy[yy+1, xx]), axis=1)
        short_edges = np.linalg.norm(polygons - np.roll(polygons, 1, axis=1), axis=2).max(axis=1) <= .25
        polygons = np.floor(polygons[short_edges] / c.resolution_m).astype(int) - grid.origin
        for polygon in polygons.tolist():
            draw.polygon([tuple(p) for p in polygon], fill=1)
        grid.ground |= np.asarray(image).astype(bool) & grid.valid
        grid.ground &= grid.valid

        hit_bits = np.zeros_like(grid.hits)
        if obstacles.any():
            endpoints = xy[obstacles]
            cells = np.floor(endpoints / c.resolution_m).astype(int) - grid.origin
            bins = np.minimum(self.bins-1, (points[..., 2][obstacles] / c.height_bin_m).astype(int))
            inside = (cells >= 0).all(axis=1) & (cells < grid.n).all(axis=1)
            xx, yy = cells[inside].T
            np.bitwise_or.at(hit_bits, (yy, xx), np.left_shift(np.uint32(1), bins[inside].astype(np.uint32)))

        # Traverse sampled rays in batches. Only crossing the same height bin can
        # clear its old hit; high rays cannot erase chair legs underneath them.
        clear_bits = np.zeros_like(grid.hits)
        endpoints = xy[valid]
        heights = points[..., 2][valid]
        if len(endpoints):
            delta = endpoints - start
            ranges = np.linalg.norm(delta, axis=1)
            max_steps = math.ceil(min(float(ranges.max()), 2*c.window_radius_m) / (c.resolution_m / 2))
            for step in range(1, max_steps + 1):
                dist = step * c.resolution_m / 2
                fraction = dist / np.maximum(ranges, 1e-9)
                xyz_height = transform[2, 3] + fraction * (heights - transform[2, 3])
                usable = ((dist < ranges - c.resolution_m) & (xyz_height > c.ground_tolerance_m)
                          & (xyz_height <= self.height))
                positions = start + fraction[usable, None] * delta[usable]
                cells = np.floor(positions / c.resolution_m).astype(int) - grid.origin
                bins = np.minimum(self.bins-1, (xyz_height[usable] / c.height_bin_m).astype(int))
                inside = (cells >= 0).all(axis=1) & (cells < grid.n).all(axis=1)
                xx, yy = cells[inside].T
                np.bitwise_or.at(clear_bits, (yy, xx), np.left_shift(np.uint32(1), bins[inside].astype(np.uint32)))
        clear_bits[~grid.valid] = 0
        hit_bits[~grid.valid] = 0
        for b in range(self.bins):
            bit = np.uint32(1 << b)
            clear = ((clear_bits & bit) != 0) & ((hit_bits & bit) == 0)
            counts = grid.clear_counts[b]
            counts[~clear] = 0
            counts[clear] = np.minimum(counts[clear] + 1, c.clear_confirm_frames)
            grid.hits[counts >= c.clear_confirm_frames] &= ~bit
        grid.hits |= hit_bits
        # Measured floor/obstacles permanently replace the birth assumption.
        # Invalid depth and high rays are not evidence of a clear body volume.
        grid.initial_free[grid.ground | (grid.hits != 0)] = False
        grid.blind_free[grid.ground | (grid.hits != 0)] = False
        # Direct floor support plus no retained occupied-height evidence defines
        # projected free space. Hits always take precedence in grid.labels().
        grid.free |= grid.ground & grid.valid
        return True
