"""Single-thread-owned headless Habitat-Sim; navmesh motion, no robot dynamics."""

import io
import math

import numpy as np
from PIL import Image


class HabitatScene:
    def __init__(
        self,
        scene,
        seed=0,
        width=640,
        height=480,
        camera_height=0.45,
        hfov=90.0,
        *,
        scene_dataset_config=None,
        enable_depth=False,
    ):
        import habitat_sim

        self.hs = habitat_sim
        config = habitat_sim.SimulatorConfiguration()
        if scene_dataset_config is not None:
            # Habitat applies dataset stage axes/materials; movement stays in its
            # canonical Y-up world. Do not rotate the agent or navmesh again.
            config.scene_dataset_config_file = str(scene_dataset_config)
        config.scene_id = str(scene)
        config.enable_physics = False
        sensor = habitat_sim.CameraSensorSpec()
        sensor.uuid = "rgb"
        sensor.sensor_type = habitat_sim.SensorType.COLOR
        sensor.resolution = [height, width]
        sensor.position = [0.0, camera_height, 0.0]
        sensor.hfov = hfov
        agent = habitat_sim.agent.AgentConfiguration()
        agent.sensor_specifications = [sensor]
        self.enable_depth = enable_depth
        if enable_depth:
            depth_sensor = habitat_sim.CameraSensorSpec()
            depth_sensor.uuid = "depth"
            depth_sensor.sensor_type = habitat_sim.SensorType.DEPTH
            depth_sensor.resolution = [height, width]
            depth_sensor.position = [0.0, camera_height, 0.0]
            depth_sensor.hfov = hfov
            agent.sensor_specifications.append(depth_sensor)
            focal = width / (2 * math.tan(math.radians(hfov) / 2))
            self.depth_camera = dict(
                fx=focal,
                fy=focal,
                cx=(width - 1) / 2,
                cy=(height - 1) / 2,
                camera_to_robot=[
                    [0, 0, 1, 0],
                    [-1, 0, 0, 0],
                    [0, -1, 0, camera_height],
                    [0, 0, 0, 1],
                ],
            )
        self.sim = habitat_sim.Simulator(habitat_sim.Configuration(config, [agent]))
        try:
            if not self.sim.pathfinder.is_loaded:
                raise RuntimeError("scene has no navigation mesh; supply a scene with its .navmesh")
            self.sim.pathfinder.seed(seed)
            self.start = np.asarray(self.sim.pathfinder.get_random_navigable_point()).copy()
            if not np.isfinite(self.start).all():
                raise RuntimeError("unable to sample a navigable start")
            self.agent = self.sim.initialize_agent(0)
            self.reset()
        except BaseException:
            self.sim.close()
            raise

    def _set(self):
        state = self.hs.AgentState()
        state.position = self.position
        # Habitat looks along -Z: positive yaw turns left towards -X.
        state.rotation = self.hs.utils.common.quat_from_angle_axis(
            self.yaw, np.array([0.0, 1.0, 0.0])
        )
        self.agent.set_state(state, infer_sensor_states=True)

    def reset(self):
        self.sim.reset()
        self.position = self.start.copy()
        self.yaw = 0.0
        self.sim_time = 0.0
        self._set()
        return self.observe(False)

    def step(self, linear, angular, dt):
        n = max(1, math.ceil(dt / 0.05))
        subdt = dt / n
        collided = False
        for _ in range(n):
            angle = self.yaw + angular * subdt / 2
            target = self.position + np.array(
                [-linear * math.sin(angle) * subdt, 0.0, -linear * math.cos(angle) * subdt]
            )
            filtered = np.asarray(self.sim.pathfinder.try_step(self.position, target))
            if not np.isfinite(filtered).all():
                raise RuntimeError("invalid navmesh position")
            collided |= np.linalg.norm(filtered - target) > 1e-3
            self.position = filtered.copy()
            self.yaw = math.atan2(
                math.sin(self.yaw + angular * subdt), math.cos(self.yaw + angular * subdt)
            )
        self.sim_time += dt
        self._set()
        return self.observe(bool(collided))

    def observe(self, collided=False):
        observations = self.sim.get_sensor_observations()
        rgb = np.asarray(observations["rgb"])[..., :3]
        output = io.BytesIO()
        Image.fromarray(rgb).save(output, format="JPEG", quality=85)
        result = {
            "jpeg": output.getvalue(),
            "pose": dict(zip(("x", "y", "z"), map(float, self.position)), yaw=self.yaw),
            "sim_time": self.sim_time,
            "collided": collided,
        }
        if self.enable_depth:
            # Same render/pose as RGB; no UI annotations and no persistent cloud.
            result["depth"] = np.asarray(observations["depth"], dtype=np.float32).copy()
            result["depth_camera"] = self.depth_camera
        return result

    def close(self):
        self.sim.close()
