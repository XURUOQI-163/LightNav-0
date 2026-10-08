"""Metric ray-cast fixtures; no simulator, image processing, or point-cloud store."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
from room_memory.geometry import CameraModel, DepthPortalDetector


def camera_model(height=0.6, translation=(0.0, 0.0)):
    transform = np.array(
        [[0, 0, 1, translation[0]], [-1, 0, 0, translation[1]], [0, -1, 0, height], [0, 0, 0, 1]],
        dtype=float,
    )
    return CameraModel(160, 120, 80.0, 80.0, 79.5, 59.5, transform)


def pose(x=0.0, y=0.0, yaw=0.0):
    return SimpleNamespace(x=x, y=y, yaw=yaw)


def raycast_room(camera, robot_pose, *, width=1.0, center=0.0, floor_only=False,
                 blocked=False, scene_origin=(0.0, 0.0), scene_yaw=0.0):
    """Ray-plane intersections with a wall at x=2, rear wall at 4.8, and ground.

    The front wall has an actual 2 m high aperture; ground intersections provide
    realistic shorter depths in lower image rows. Returned t is optical Z depth.
    """
    uu, vv = np.meshgrid(np.arange(camera.width), np.arange(camera.height))
    optical_rays = np.stack(((uu - camera.cx) / camera.fx,
                             (vv - camera.cy) / camera.fy, np.ones_like(uu)), axis=-1)
    rays = optical_rays @ camera.camera_to_robot[:3, :3].T
    origin = camera.camera_to_robot[:3, 3].copy()
    heading = robot_pose.yaw - scene_yaw
    rotation = np.array([[math.cos(heading), -math.sin(heading), 0],
                         [math.sin(heading), math.cos(heading), 0], [0, 0, 1]])
    rays = rays @ rotation.T
    origin = rotation @ origin
    dx, dy = robot_pose.x - scene_origin[0], robot_pose.y - scene_origin[1]
    origin[:2] += [math.cos(scene_yaw) * dx + math.sin(scene_yaw) * dy,
                   -math.sin(scene_yaw) * dx + math.cos(scene_yaw) * dy]
    depth = np.full(uu.shape, np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        ground = -origin[2] / rays[..., 2]
        depth = np.minimum(depth, np.where(ground > 0, ground, np.inf))
        if not floor_only:
            for distance, doorway in ((2.0, True), (4.8, False)):
                t = (distance - origin[0]) / rays[..., 0]
                y = origin[1] + t * rays[..., 1]
                z = origin[2] + t * rays[..., 2]
                hit = (t > 0) & (z >= 0)
                if doorway:
                    hit &= (np.abs(y - center) >= width / 2) | (z > 2.0)
                depth = np.minimum(depth, np.where(hit, t, np.inf))
            if blocked:
                t = (1.8 - origin[0]) / rays[..., 0]
                y = origin[1] + t * rays[..., 1]
                z = origin[2] + t * rays[..., 2]
                hit = (t > 0) & (np.abs(y - center) < width / 2) & (z >= 0)
                depth = np.minimum(depth, np.where(hit, t, np.inf))
    return depth.astype(np.float32)


def test_metric_doorway_and_observed_wide_narrow_wide_profile():
    camera = camera_model()
    result = DepthPortalDetector({}).detect(raycast_room(camera, pose()), camera, pose())
    assert result.candidate is not None
    assert result.candidate.x == pytest.approx(2.0, abs=0.08)
    assert result.candidate.y == pytest.approx(0.0, abs=0.04)
    assert result.candidate.width == pytest.approx(1.0, abs=0.10)
    assert result.candidate.normal_yaw == pytest.approx(0.0)
    assert 0.0 < result.candidate.confidence <= 1.0
    assert result.diagnostics["wide_narrow_wide"] is True
    assert len(result.diagnostics["profile"]) >= 3


@pytest.mark.parametrize("robot_x", [0.0, 0.35, 0.7])
def test_crossing_plane_stays_at_jambs_when_robot_translates(robot_x):
    camera = camera_model()
    robot_pose = pose(robot_x)
    result = DepthPortalDetector({}).detect(
        raycast_room(camera, robot_pose), camera, robot_pose,
    )
    assert result.candidate is not None
    assert result.candidate.x == pytest.approx(2.0, abs=0.08)
    assert result.candidate.y == pytest.approx(0.0, abs=0.04)


def test_off_center_doorway_world_heading_and_camera_translation():
    camera = camera_model(translation=(0.2, 0.1))
    robot_pose = pose(10.0, -3.0, math.pi / 2)
    depth = raycast_room(camera, robot_pose, center=0.4,
                         scene_origin=(10.0, -3.0), scene_yaw=math.pi / 2)
    candidate = DepthPortalDetector({}).detect(depth, camera, robot_pose).candidate
    assert candidate is not None
    assert candidate.x == pytest.approx(9.6, abs=0.07)
    assert candidate.y == pytest.approx(-1.0, abs=0.08)
    assert candidate.normal_yaw == pytest.approx(math.pi / 2)


def test_obstacle_in_opening_is_not_free_space():
    camera = camera_model()
    result = DepthPortalDetector({}).detect(
        raycast_room(camera, pose(), blocked=True), camera, pose(),
    )
    assert result.candidate is None


@pytest.mark.parametrize("invalid", [0.0, np.nan, np.inf])
def test_unknown_center_depth_is_never_free(invalid):
    camera = camera_model()
    depth = raycast_room(camera, pose())
    depth[:, 72:88] = invalid
    result = DepthPortalDetector({}).detect(depth, camera, pose())
    assert result.candidate is None
    assert 0.0 <= result.diagnostics["valid_fraction"] < 1.0


def test_floor_points_cannot_supply_vertical_jamb_support():
    camera = camera_model()
    result = DepthPortalDetector({}).detect(
        raycast_room(camera, pose(), floor_only=True), camera, pose(),
    )
    assert result.candidate is None


@pytest.mark.parametrize("width", [0.55, 2.6])
def test_opening_width_must_fit_robot_and_portal_limits(width):
    camera = camera_model()
    assert DepthPortalDetector({}).detect(
        raycast_room(camera, pose(), width=width), camera, pose(),
    ).candidate is None


def test_robot_width_overrides_smaller_portal_minimum():
    camera = camera_model()
    assert DepthPortalDetector({"robot_width": 1.0, "safety_margin": 0.15}).detect(
        raycast_room(camera, pose()), camera, pose(),
    ).candidate is None


def test_a_single_visible_jamb_is_insufficient():
    camera = camera_model()
    assert DepthPortalDetector({}).detect(
        raycast_room(camera, pose(), width=2.0, center=1.7), camera, pose(),
    ).candidate is None


def test_empty_and_invalid_depth_return_diagnostics_without_retained_points():
    camera = camera_model()
    detector = DepthPortalDetector({})
    result = detector.detect(np.full((120, 160), np.nan), camera, pose())
    assert result.candidate is None
    assert result.diagnostics["valid_fraction"] == 0.0
    assert result.diagnostics["reason"]
    # Detection is stateless: a subsequent valid observation still works.
    assert detector.detect(raycast_room(camera, pose()), camera, pose()).candidate is not None
