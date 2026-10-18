"""Isaac adapters for the repository's official MPC and finite world-frame paths."""

import asyncio
import importlib.util
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from habitat_web.control import command_for_path
from local_navigation.models import transform_path


def official_mpc():
    # Load only the ROS-free controller, without MuJoCo or model dependencies.
    name = '_lightnav_official_mpc'
    if name not in sys.modules:
        path = Path(__file__).resolve().parents[2] / 'mujoco_demo/vln_mujoco/mpc.py'
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[name]
            raise
    return sys.modules[name]


def angle_error(a, b):
    return math.atan2(math.sin(a - b), math.cos(a - b))


class WorldPath:
    """Keep a capture-aligned path fixed as the robot moves; advance monotonically."""

    def __init__(self, points, pose, position_tolerance=.10, yaw_tolerance=math.radians(10)):
        self.points = transform_path(points, pose)
        self.index = 0
        self.position_tolerance = position_tolerance
        self.yaw_tolerance = yaw_tolerance

    def remaining(self, pose):
        # Nearest weighted pose follows the official MPC reference convention.
        costs = [10 * math.dist(p[:2], pose[:2])**2 + angle_error(p[2], pose[2])**2
                 for p in self.points[self.index:]]
        self.index += int(np.argmin(costs))
        return self.points[self.index:]

    def complete(self, pose):
        self.remaining(pose)
        goal = self.points[-1]
        return (self.index >= len(self.points) - 2
                and math.dist(goal[:2], pose[:2]) <= self.position_tolerance
                and abs(angle_error(goal[2], pose[2])) <= self.yaw_tolerance)


class PathTracker:
    """A single solver worker; cancellation drains it before any later command."""

    def __init__(self, use_mpc, max_linear, max_angular, dt):
        self.use_mpc = use_mpc
        self.max_linear, self.max_angular = max_linear, max_angular
        self.dt = .1 if use_mpc else dt
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='isaac-mpc') if use_mpc else None
        self.controller = None
        self.previous = (0., 0.)

    def _solve(self, points, pose):
        m = official_mpc()
        if self.controller is None:
            self.controller = m.MPCController(
                horizon=m.HORIZON, dt_s=m.MPC_DT_S, w_max=self.max_angular,
                a_max_v=m.A_MAX_V, a_max_w=m.A_MAX_W,
                q_weights=m.Q_WEIGHTS, r_weights=m.R_WEIGHTS,
            )
        # Match the official wrapper: weighted pose reference, local solve frame.
        reference = m.build_pose_aligned_reference(np.asarray(points), np.asarray(pose),
                                                   horizon=m.HORIZON, weights=m.Q_WEIGHTS)
        local = m.project_world_to_local(reference, np.asarray(pose))
        command, _ = self.controller.solve(np.zeros(3), local, self.previous, self.max_linear)
        return command

    async def command(self, points, pose, *, finish_segment=False, position_tolerance=.10):
        if self.use_mpc:
            future = asyncio.get_running_loop().run_in_executor(self.pool, self._solve, points, pose)
            try:
                return await asyncio.shield(future)
            except asyncio.CancelledError:
                await future
                raise
        local = transform_path(points, pose, inverse=True)
        if finish_segment and math.hypot(*local[-1][:2]) <= position_tolerance:
            # The old single-step follower has no terminal heading controller.
            return 0., float(np.clip(local[-1][2] / self.dt, -self.max_angular, self.max_angular))
        return command_for_path(local, self.dt, self.max_linear, self.max_angular)

    def close(self):
        if self.pool:
            self.pool.shutdown(wait=True, cancel_futures=True)
