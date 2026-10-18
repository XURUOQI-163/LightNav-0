"""Metric contracts shared by the worker and the passive room observer."""

import math

import numpy as np


def configure_pinhole_camera(camera, hfov):
    """Keep RTX rendering and exported intrinsics on the same pinhole model.

    Isaac 5.0's RTX renderer did not honour our previous tiny .02 focal
    length: a flat floor reconstructed about 1.49 m below its actual height.
    A 2.0 focal length with the same aperture ratio passes the rendered-plane
    check (see scripts/check_camera_calibration.py). Values use Camera API
    stage units; the SDK handles USD's tenths-of-stage-unit conversion.
    """
    if not math.isfinite(hfov) or not 0 < hfov < 180:
        raise ValueError('camera hfov must be between 0 and 180 degrees')
    focal = 2.0
    camera.set_focal_length(focal)
    camera.set_horizontal_aperture(2 * focal * math.tan(math.radians(hfov) / 2))
    camera.set_clipping_range(.05, 100.)


def chase_camera_pose(position, yaw, distance, height, target_height):
    """World camera axes: +X looks forward, +Z up. Follow yaw, ignore chassis roll."""
    eye = np.asarray(position) + [-distance * math.cos(yaw),
                                 -distance * math.sin(yaw), height]
    pitch = math.atan2(height - target_height, distance)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    return eye, np.array([cy * cp, -sy * sp, cy * sp, sy * cp])


class IsaacGTPoseSource:
    def pose_at(self, timestamp, observation):
        from room_memory.models import Pose2D

        if abs(timestamp - observation["sim_time"]) > 1e-6:
            raise ValueError("unsynchronized Isaac pose")
        p = observation["pose"]
        result = Pose2D(p["x"], p["y"], p["yaw"])
        if not all(math.isfinite(v) for v in (result.x, result.y, result.yaw)):
            raise ValueError("nonfinite Isaac pose")
        return result


def camera_extrinsic(translation):
    # Optical right/down/forward -> robot forward/left/up.
    transform = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0],
                          [0, 0, 0, 1]], dtype=float)
    transform[:3, 3] = translation
    return transform


def clean_depth(depth, width, height):
    result = np.array(depth, dtype=np.float32, copy=True)
    if result.shape != (height, width):
        raise ValueError(f"depth shape {result.shape} differs from RGB {(height, width)}")
    result[~np.isfinite(result) | (result <= 0)] = 0
    return result


def wheel_speeds(linear, angular, radius, base):
    if not all(math.isfinite(v) for v in (linear, angular, radius, base)) or min(radius, base) <= 0:
        raise ValueError("invalid wheel geometry or command")
    return np.array([(linear - angular * base / 2) / radius,
                     (linear + angular * base / 2) / radius])


def grounded_camera_pose(position, rotation, offset, floor_z):
    """Return a level, ground-projected GT pose and optical-to-that-frame matrix."""
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    level = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                      [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    origin = np.array([position[0], position[1], floor_z])
    transform = camera_extrinsic((0, 0, 0))
    transform[:3, :3] = level.T @ rotation @ transform[:3, :3]
    transform[:3, 3] = level.T @ (position + rotation @ offset - origin)
    return dict(x=float(position[0]), y=float(position[1]), z=float(floor_z), yaw=yaw), transform
