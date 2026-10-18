"""User-authorized free prior, only for unknown near floor BELOW the camera FOV."""
import math

import numpy as np


def initialize_near_blindspot(grid, camera):
    distance = grid.config.near_blind_clear_distance_m
    if distance <= 0:
        return
    # Measurements win before any new prior is considered. Record that a cell
    # has been initialized separately, so it cannot be reseeded after a scan.
    grid.blind_free[grid.ground | (grid.hits != 0)] = False
    candidates = grid.valid & (grid.labels() == 0) & ~grid.blind_initialized
    iy, ix = np.nonzero(candidates)
    if not len(ix):
        return
    wx, wy = grid.world(ix, iy)
    x, y, yaw = grid.pose
    co, si = math.cos(yaw), math.sin(yaw)
    dx, dy = wx - x, wy - y
    local = np.column_stack((co * dx + si * dy, -si * dx + co * dy, np.zeros_like(dx)))
    transform = np.asarray(camera['camera_to_robot'], dtype=float)
    # Row-vector inverse rigid transform: robot floor -> optical (+Z forward,
    # +Y down). Use the same capture calibration as the depth mapper.
    optical = (local - transform[:3, 3]) @ transform[:3, :3]
    front = optical[:, 2] > 1e-6
    z = np.where(front, optical[:, 2], 1.)
    u = camera['fx'] * optical[:, 0] / z + camera['cx']
    v = camera['fy'] * optical[:, 1] / z + camera['cy']
    eligible = (front & (local[:, 0] >= 0) & (np.hypot(local[:, 0], local[:, 1]) <= distance)
                & (u >= -.5) & (u <= camera['width'] - .5)
                & (v > camera['height'] - .5))
    iy, ix = iy[eligible], ix[eligible]
    grid.blind_free[iy, ix] = True
    grid.blind_initialized[iy, ix] = True
