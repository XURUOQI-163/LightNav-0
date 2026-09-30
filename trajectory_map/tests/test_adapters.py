import math
from types import SimpleNamespace as NS

import pytest
from lightnav_trajectory.adapters import pose_from_mujoco, pose_from_odometry


def test_mujoco_uses_simulation_time_and_explicit_source():
    p = pose_from_mujoco({"sim_time": 1.25, "pose": {"x": 1, "y": 2, "yaw": 3}})
    assert (p.stamp_ns, p.x_m, p.y_m, p.source) == (1250000000, 1, 2, "mujoco_ground_truth")


def test_odom_preserves_source_stamp_frame_and_quaternion_yaw():
    q = NS(x=0, y=0, z=math.sqrt(0.5), w=math.sqrt(0.5))
    message = NS(
        header=NS(stamp=NS(sec=12, nanosec=34), frame_id="odom"),
        pose=NS(pose=NS(position=NS(x=3, y=4), orientation=q)),
    )
    p = pose_from_odometry(message)
    assert p.stamp_ns == 12000000034
    assert p.frame_id == "odom"
    assert p.source == "ros_ekf"
    assert p.yaw_rad == pytest.approx(math.pi / 2)
    q.w = q.z = 0
    with pytest.raises(ValueError):
        pose_from_odometry(message)
