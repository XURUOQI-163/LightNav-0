import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from room_memory.models import Pose2D, RGBDFrame
from test_room_geometry import camera_model, pose, raycast_room


def verifier():
    import importlib.util

    assert importlib.util.find_spec("room_memory.doorway_depth"), "visual depth verifier missing"
    from room_memory.doorway_depth import DoorwayDepthVerifier

    return DoorwayDepthVerifier({})


def observation(width=1.0, blocked=False, yaw=0.0):
    camera = camera_model()
    camera_pose = pose(yaw=yaw)
    depth = raycast_room(camera, camera_pose, width=width, blocked=blocked)
    # Project corners of x=2 doorway into rotated camera; optical right=-robot left.
    pixels = []
    for y in (-width / 2, width / 2):
        for z in (0.0, 2.0):
            forward = 2 * math.cos(yaw) + y * math.sin(yaw)
            left = -2 * math.sin(yaw) + y * math.cos(yaw)
            pixels.append(
                (
                    camera.cx - camera.fx * left / forward,
                    camera.cy - camera.fy * (z - 0.6) / forward,
                )
            )
    xs, ys = zip(*pixels)
    box = (max(0, min(xs)), max(0, min(ys)), min(159, max(xs)), min(119, max(ys)))
    frame = RGBDFrame(0, b"", depth, Pose2D(0, 0, yaw), camera)
    detection = SimpleNamespace(bbox_xyxy=box, confidence=0.8, label="doorway")
    return frame, detection


def test_visual_door_depth_accepts_opening_with_world_location():
    f, d = observation()
    measurement, diag = verifier().verify(f, d)
    assert measurement is not None, diag
    assert measurement.candidate.x == pytest.approx(2, abs=0.18)
    assert abs(measurement.candidate.y) < 0.12
    assert measurement.candidate.width == pytest.approx(1, abs=0.18)


def test_navmesh_height_offset_does_not_override_measured_wall_depth():
    f, d = observation()
    # Robot origin may be above the scanned floor. RGB/depth are still aligned;
    # camera extrinsic is .45 m while its distance above the floor is .60 m.
    transform = f.camera.camera_to_robot.copy()
    transform[2, 3] = 0.45
    f.camera = replace(f.camera, camera_to_robot=transform)
    m, diag = verifier().verify(f, d)
    assert m is not None, diag
    assert m.candidate.x == pytest.approx(2, abs=0.15)
    assert m.candidate.width == pytest.approx(1, abs=0.15)


def test_missing_threshold_and_wall_depth_cannot_invent_plane_before_closed_door():
    f, d = observation(width=2.0, blocked=True)
    x1, _, x2, y2 = d.bbox_xyxy
    f.depth[:, :int(x1) + 1] = np.nan
    f.depth[:, int(x2):] = np.nan
    f.depth[int(y2) - 2:] = np.nan
    transform = f.camera.camera_to_robot.copy()
    transform[2, 3] = .25
    f.camera = replace(f.camera, camera_to_robot=transform)
    m, diag = verifier().verify(f, d)
    assert m is None, diag


def test_single_visible_frame_side_and_oblique_door_do_not_require_bilateral_jambs():
    f, d = observation(yaw=math.radians(25))
    # Remove all depth evidence outside the right side: one wall strip remains.
    f.depth[:, int(d.bbox_xyxy[2]) :] = np.nan
    measurement, diag = verifier().verify(f, d)
    assert measurement is not None, diag
    assert measurement.candidate.x == pytest.approx(2, abs=0.3)
    assert abs(measurement.candidate.y) < 0.35
    assert abs(measurement.candidate.normal_yaw) < math.radians(20)


@pytest.mark.parametrize("kind", ["closed", "invalid", "narrow"])
def test_rgb_proposal_is_rejected_without_metric_clearance(kind):
    f, d = observation(width=0.5 if kind == "narrow" else 1, blocked=kind == "closed")
    if kind == "invalid":
        f.depth[:] = np.nan
    m, diag = verifier().verify(f, d)
    assert m is None, diag
