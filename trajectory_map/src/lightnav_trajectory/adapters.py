"""Convert source observations without importing either ROS or MuJoCo.

Habitat adapters should supply Pose2D after converting their world axes into a
right-handed planar frame. Never pass a simulator quaternion as ROS x/y/z/w
without an explicit coordinate conversion.
"""

import math

from .model import Pose2D


def pose_from_mujoco(snapshot):
    """Use one atomic Simulation.snapshot(), including its simulation timestamp."""
    pose = snapshot["pose"]
    stamp = float(snapshot["sim_time"])
    if not math.isfinite(stamp) or stamp < 0:
        raise ValueError("invalid simulation time")
    return Pose2D(
        round(stamp * 1_000_000_000),
        float(pose["x"]),
        float(pose["y"]),
        float(pose["yaw"]),
        "mujoco_world",
        "mujoco_ground_truth",
    )


def pose_from_odometry(message, source="ros_ekf"):
    """Consume /odometry/filtered, preserving its header clock and frame.

    Duck-typed nav_msgs/Odometry: this helper can be tested without rclpy.
    No TF is published and no additional filtering is applied.
    """
    stamp = message.header.stamp
    if (
        not isinstance(stamp.sec, int)
        or not isinstance(stamp.nanosec, int)
        or stamp.sec < 0
        or not 0 <= stamp.nanosec < 1_000_000_000
    ):
        raise ValueError("invalid ROS timestamp")
    pose = message.pose.pose
    q = pose.orientation
    values = [float(q.x), float(q.y), float(q.z), float(q.w)]
    norm = math.sqrt(sum(v * v for v in values))
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError("invalid orientation quaternion")
    x, y, z, w = [v / norm for v in values]
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return Pose2D(
        stamp.sec * 1_000_000_000 + stamp.nanosec,
        float(pose.position.x),
        float(pose.position.y),
        yaw,
        message.header.frame_id,
        source,
    )
