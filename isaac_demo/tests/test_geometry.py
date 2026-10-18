import math

import numpy as np
import pytest
from isaac_web.geometry import (
    IsaacGTPoseSource,
    camera_extrinsic,
    clean_depth,
    grounded_camera_pose,
    wheel_speeds,
)


def test_isaac_world_pose_keeps_xy_and_rejects_unsynchronized_data():
    source = IsaacGTPoseSource()
    obs = dict(sim_time=2.0, pose=dict(x=3., y=-4., z=0.25, yaw=math.pi / 2))
    p = source.pose_at(2.0, obs)
    assert (p.x, p.y, p.yaw) == (3., -4., math.pi / 2)
    with pytest.raises(ValueError):
        source.pose_at(1.0, obs)


def test_optical_right_down_forward_projects_to_robot_forward_left_up():
    transform = camera_extrinsic((0.4, 0., 0.6))
    assert np.allclose(transform @ [0., 0., 2., 1.], [2.4, 0., 0.6, 1.])
    assert np.allclose(transform @ [1., 1., 2., 1.], [2.4, -1., -0.4, 1.])


def test_depth_keeps_axial_metres_and_masks_invalid_values():
    depth = np.array([[2., np.inf], [np.nan, -1.]], dtype=np.float64)
    result = clean_depth(depth, 2, 2)
    assert result.dtype == np.float32
    assert result.tolist() == [[2., 0.], [0., 0.]]
    assert math.isinf(depth[0, 1])  # source sensor buffer must not be modified
    with pytest.raises(ValueError):
        clean_depth(np.zeros((3, 2)), 2, 2)


def test_wheels_forward_and_left_turn_have_correct_signs():
    assert np.allclose(wheel_speeds(.4, 0., .2, .6), [2., 2.])
    assert np.allclose(wheel_speeds(0., 1., .2, .6), [-1.5, 1.5])
    with pytest.raises(ValueError):
        wheel_speeds(float('nan'), 0., .2, .6)


def test_exported_pose_and_camera_transform_reconstruct_real_camera_height():
    # Base chassis is above the floor and pitched; 2D map pose stays level.
    a = .1
    rotation = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0],
                         [-np.sin(a), 0, np.cos(a)]])
    position, offset = np.array([3., 4., .27]), np.array([.4, 0., .6])
    pose, transform = grounded_camera_pose(position, rotation, offset, floor_z=0.)
    world_from_robot = np.eye(4)
    world_from_robot[:3, 3] = [pose["x"], pose["y"], pose["z"]]
    actual_camera = position + rotation @ offset
    np.testing.assert_allclose((world_from_robot @ transform)[:3, 3], actual_camera)
    assert pose["z"] == 0.


def test_camera_scale_keeps_rendered_fov_consistent_with_exported_intrinsics():
    from isaac_web.geometry import configure_pinhole_camera

    class Camera:
        def set_focal_length(self, value):
            self.focal = value

        def set_horizontal_aperture(self, value):
            self.aperture = value

        def set_clipping_range(self, near, far):
            self.clip = near, far

    camera = Camera()
    configure_pinhole_camera(camera, 90.)
    reported_fx = 640 * camera.focal / camera.aperture
    # Observed on this Isaac 5.0 RTX build: 0.02 behaved as 0.05 in rendering.
    rendered_fx = 640 * max(.05, camera.focal) / camera.aperture
    assert reported_fx == pytest.approx(320.)
    assert rendered_fx == pytest.approx(reported_fx)
