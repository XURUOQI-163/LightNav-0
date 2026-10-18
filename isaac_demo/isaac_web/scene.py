"""Carter differential-drive physics and aligned RGB/axial-depth observations.

Only imported by worker_main. All Kit/USD/PhysX calls stay on that process's main thread.
Source USD files are referenced into a fresh anonymous stage and never saved.
"""

import faulthandler
import io
import math
import os
import time
import traceback

import numpy as np
from PIL import Image

from .carter_resize import resize_carter
from .geometry import (
    chase_camera_pose,
    clean_depth,
    configure_pinhole_camera,
    grounded_camera_pose,
    wheel_speeds,
)
from .usd_scene import normalize_mdl_source_paths, repair_static_colliders


class CarterScene:
    def __init__(self, config, *, progress_callback=None, error_callback=None):
        self.progress_callback = progress_callback
        self.cfg = config
        self.app = None
        self.sim_time = 0.
        self.collided = False
        self.cached = None
        self.preview_callback = None
        self._explicit_capture_mode = False
        materials = config.get("materials")
        if materials:
            os.environ["MDL_SYSTEM_PATH"] = materials + (
                os.pathsep + os.environ["MDL_SYSTEM_PATH"] if os.environ.get("MDL_SYSTEM_PATH") else ""
            )
        # No Isaac/omni/pxr import is allowed above SimulationApp construction.
        self._progress("engine", "启动 Isaac 引擎与渲染器")
        from isaacsim import SimulationApp

        self.app = SimulationApp({
            "headless": True, "width": config["width"], "height": config["height"],
            "active_gpu": config["gpu"], "physics_gpu": config["gpu"],
            "multi_gpu": False, "renderer": "RayTracedLighting",
            "sync_loads": True, "disable_viewport_updates": True,
        })
        try:
            self._build()
        except BaseException:
            # Kit fast shutdown can terminate Python. Report BEFORE close(),
            # otherwise worker_main never gets to forward the original error.
            message = traceback.format_exc()
            print(message, flush=True)
            try:
                if error_callback is not None:
                    error_callback(message)
            finally:
                self.close()
            raise

    def _progress(self, phase, message, **details):
        print(f"[isaac-web] {message}", flush=True)
        if self.progress_callback is not None:
            self.progress_callback(dict(phase=phase, message=message, **details))

    def _build(self):
        from isaacsim.core.api import World
        from isaacsim.core.utils.stage import create_new_stage
        from isaacsim.robot.wheeled_robots.robots import WheeledRobot
        from isaacsim.sensors.camera import Camera
        from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics

        cfg = self.cfg
        create_new_stage()
        self.world = World(stage_units_in_meters=1., physics_dt=cfg["physics_dt"],
                           rendering_dt=cfg["physics_dt"], backend="numpy", device="cpu")
        stage = self.world.stage
        self._progress("scene", "加载 GRScenes USD 场景与引用资产")
        source = Usd.Stage.Open(cfg["scene"])
        up = str(UsdGeom.GetStageUpAxis(source))
        units = UsdGeom.GetStageMetersPerUnit(source)
        default = source.GetDefaultPrim()
        if not default:
            raise ValueError("scene USD requires a default prim")
        environment = UsdGeom.Xform.Define(stage, "/World/Environment")
        asset = stage.DefinePrim("/World/Environment/Asset")
        asset.GetReferences().AddReference(cfg["scene"], default.GetPath())
        if up == "Y":
            environment.AddRotateXOp().Set(90.)
        elif up != "Z":
            raise ValueError(f"unsupported scene up axis: {up}")
        environment.AddScaleOp().Set(Gf.Vec3d(units))
        # The source contains a centimetre-based physics scene. Use only our World scene.
        for prim in list(Usd.PrimRange(environment.GetPrim())):
            if prim.IsA(UsdPhysics.Scene):
                prim.SetActive(False)
        if cfg.get("normalize_mdl_paths", True):
            self._progress("materials", "统一本地 MDL 材质路径（保留材质参数和贴图）")
            stats = normalize_mdl_source_paths(environment.GetPrim())
            self._progress("materials", "MDL 路径归一化完成", mdl_paths=stats)
            print(f"[isaac-web] MDL paths: {stats}", flush=True)
        self._progress("collision", "准备静态碰撞网格与 Carter 机器人")
        repaired = repair_static_colliders(environment.GetPrim())
        print(f"[isaac-web] Enabled static collision on {repaired} concrete meshes", flush=True)
        print(f"[isaac-web] Scene referenced: up={up}, metres/unit={units}", flush=True)
        del source
        if cfg.get("ambient_light", 0) > 0:
            light = UsdLux.DomeLight.Define(stage, "/World/ViewerFillLight")
            light.CreateIntensityAttr(float(cfg["ambient_light"]))
        x, y, z, yaw = cfg["spawn"]
        self.start_position = np.array([x, y, z], dtype=float)
        self.start_orientation = np.array([math.cos(yaw / 2), 0., 0., math.sin(yaw / 2)])
        self.robot = self.world.scene.add(WheeledRobot(
            prim_path="/World/Carter", name="carter",
            wheel_dof_names=["left_wheel", "right_wheel"], create_robot=True,
            usd_path=cfg["robot"], position=self.start_position,
            orientation=self.start_orientation,
        ))
        resize_carter(stage.GetPrimAtPath("/World/Carter"), cfg.get("robot_xy_scale", 1.))
        self.robot.set_default_state(self.start_position, self.start_orientation)
        self._progress("physics", "初始化物理与 Carter 落地（可能包含资产加载）")
        faulthandler.dump_traceback_later(180, repeat=True)
        self.world.reset()
        self._settle()
        self._check_pose()
        print(f"[isaac-web] Settled pose: {self.robot.get_world_pose()[0]}", flush=True)
        # A calibrated camera follows the chassis pose; no lidar or built-in sensor graphs.
        self.camera = Camera("/World/NavigationCamera", resolution=(cfg["width"], cfg["height"]))
        configure_pinhole_camera(self.camera, cfg["hfov"])
        self._progress("camera", "初始化 RGB-D 与跟随相机，等待材质加载与编译")
        self.camera.initialize()
        self.camera.add_distance_to_image_plane_to_frame()
        # Human display only; model and calibrated depth continue using NavigationCamera.
        self.third_camera = Camera("/World/ChaseCamera", resolution=(cfg["width"], cfg["height"]))
        configure_pinhole_camera(self.third_camera, 90.)
        self.third_camera.initialize()
        self._progress("first_frame", "等待完整 RGB-D 首帧（材质、纹理加载与渲染）")
        self._capture(report_progress=True)
        faulthandler.cancel_dump_traceback_later()
        print("[isaac-web] Carter physics and RGB-D camera initialized", flush=True)

    def _wheels(self, linear, angular):
        from isaacsim.core.utils.types import ArticulationAction

        speeds = wheel_speeds(linear, angular, self.cfg["wheel_radius"], self.cfg["wheel_base"])
        self.robot.apply_wheel_actions(ArticulationAction(joint_velocities=speeds))

    def _settle(self):
        self._wheels(0., 0.)
        for _ in range(self.cfg["settle_steps"]):
            self.world.step(render=False)
        self.sim_time = 0.
        self.collided = False
        self.cached = None

    def reset(self):
        self.world.reset(soft=True)
        self.robot.set_world_pose(self.start_position, self.start_orientation)
        self.robot.set_linear_velocity(np.zeros(3))
        self.robot.set_angular_velocity(np.zeros(3))
        self._settle()
        return self._capture()

    def step(self, linear, angular, dt):
        if not all(math.isfinite(v) for v in (linear, angular, dt)) or not 0 < dt <= 1:
            raise ValueError("step requires finite velocities and 0 < dt <= 1s")
        before, _ = self.robot.get_world_pose()
        count = max(1, round(dt / self.cfg["physics_dt"]))
        started = time.monotonic()
        initial_time = self.sim_time
        render_ms = 0.
        physics_ms = 0.
        fps = self.cfg.get("display_fps", 15.)
        interval = max(1, math.ceil(1 / (fps * self.cfg["physics_dt"]))) if fps else count
        self._wheels(linear, angular)
        try:
            for index in range(count):
                physics_start = time.monotonic()
                self.world.step(render=False)
                physics_ms += (time.monotonic() - physics_start) * 1000
                self.sim_time = initial_time + (index + 1) * self.cfg["physics_dt"]
                if self.preview_callback and fps and (index + 1) % interval == 0 and index + 1 < count:
                    render_start = time.monotonic()
                    frame = self._capture_preview()
                    render_ms += (time.monotonic() - render_start) * 1000
                    # Never burst all intermediate poses at once on a fast renderer.
                    time.sleep(max(0., started + self.sim_time - initial_time - time.monotonic()))
                    self.preview_callback(frame)
        finally:
            self._wheels(0., 0.)
        after, _ = self.robot.get_world_pose()
        # This is a motion-blocked heuristic, not an exact contact sensor.
        self.collided = bool(abs(linear) * dt > .02 and
                             np.linalg.norm(after[:2] - before[:2]) < abs(linear) * dt * .15)
        self.cached = None
        render_start = time.monotonic()
        observation = self._capture()
        render_ms += (time.monotonic() - render_start) * 1000
        if fps:
            time.sleep(max(0., started + count * self.cfg["physics_dt"] - time.monotonic()))
        observation["timing"] = dict(physics_ms=physics_ms, render_ms=render_ms,
                                     step_wall_ms=(time.monotonic() - started) * 1000)
        return observation

    def observe(self):
        # Reading the browser/model frame must never advance physics or rendering time.
        return self.cached if self.cached is not None else self._capture()

    def _position_cameras(self):
        from isaacsim.core.utils.rotations import quat_to_rot_matrix

        self._check_pose()
        p, q = self.robot.get_world_pose()
        rotation = quat_to_rot_matrix(q)
        camera_position = p + rotation @ np.array(self.cfg["camera_offset"])
        self.camera.set_world_pose(camera_position, q, camera_axes="world")
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        eye, orientation = chase_camera_pose(
            p, yaw, self.cfg.get("chase_distance", 1.8),
            self.cfg.get("chase_height", 1.3), self.cfg.get("chase_target_height", .3),
        )
        self.third_camera.set_world_pose(eye, orientation, camera_axes="world")
        return p, q, rotation

    @staticmethod
    def _jpeg(rgb):
        stream = io.BytesIO()
        Image.fromarray(rgb).save(stream, format="JPEG", quality=90)
        return stream.getvalue()

    def _render_sensor_frame(self):
        """Explicitly schedule annotators at the current pose, without a physics step."""
        import carb
        import omni.replicator.core as rep

        # world.render() alone can return before the SDG annotator buffers exist.
        # Keep the same physics guard used by World.render around Replicator's
        # app updates too. delta_time=0 preserves the simulation timeline.
        settings = carb.settings.get_settings()
        key = "/app/player/playSimulations"
        previous = settings.get(key)
        settings.set(key, False)
        try:
            rep.orchestrator.step(delta_time=0.0, pause_timeline=False, wait_for_render=True)
        finally:
            settings.set(key, previous)

    def _ready_camera_buffers(self, *, require_depth, report_progress=False, force_capture=False):
        """Retry only missing/empty data, never publish partial observations."""
        started = time.monotonic()
        limit = self.cfg.get("camera_ready_max_renders", 60)
        timeout = self.cfg.get("camera_ready_timeout_s", 30.)
        expected = (self.cfg["height"], self.cfg["width"])
        captures = 0
        # Replicator.step leaves the orchestrator in STEPPED. Once used, keep
        # explicitly scheduling every observation, rather than treating an old
        # nonempty annotator buffer as a newly rendered frame.
        need_capture = force_capture or getattr(self, "_explicit_capture_mode", False)
        detail = "fresh sensor capture required"
        while True:
            if need_capture:
                if captures >= limit or time.monotonic() - started >= timeout:
                    raise RuntimeError(
                        f"Camera buffers not ready after {captures} explicit captures "
                        f"({time.monotonic() - started:.1f}s): {detail}; "
                        "inspect render-product / annotator errors in this run's Kit log"
                    )
                if report_progress and (captures == 0 or captures % 10 == 0):
                    self._progress("sensor_wait", f"等待相机数据：{detail}；显式采集 {captures + 1}/{limit}")
                self._explicit_capture_mode = True
                self._render_sensor_frame()
                captures += 1
            buffers = {"front_rgb": self.camera.get_rgb(),
                       "third_rgb": self.third_camera.get_rgb()}
            if require_depth:
                buffers["depth"] = self.camera.get_depth()
            pending = []
            for name, value in buffers.items():
                if value is None or np.size(value) == 0:
                    pending.append(f"{name}={None if value is None else np.shape(value)}")
                    continue
                shape = expected if name == "depth" else (*expected, 3)
                if np.shape(value) != shape:
                    raise RuntimeError(f"{name} shape {np.shape(value)} differs from expected {shape}")
            if not pending:
                # Copy the whole observation from one read round, before another
                # render may overwrite any annotator's underlying host buffer.
                return {name: np.array(value, copy=True) for name, value in buffers.items()}
            detail = ", ".join(pending)
            need_capture = True

    def _capture_preview(self):
        self._position_cameras()
        buffers = self._ready_camera_buffers(require_depth=False, force_capture=True)
        return dict(jpeg=self._jpeg(buffers["front_rgb"]),
                    third_jpeg=self._jpeg(buffers["third_rgb"]), sim_time=self.sim_time)

    def _capture(self, *, report_progress=False):
        p, q, rotation = self._position_cameras()
        if report_progress:
            # Startup-only warmup. Steady observations schedule the annotators
            # directly, avoiding four redundant world renders before readback.
            for index in range(self.cfg["render_frames"]):
                self.world.render()
                self._progress("first_frame", f"首帧预热渲染 {index + 1}/{self.cfg['render_frames']} 已返回")
        if report_progress:
            self._progress("readback", "读取并检查前向 RGB、深度和第三人称画面")
        buffers = self._ready_camera_buffers(require_depth=True, report_progress=report_progress,
                                             force_capture=not report_progress)
        rgb = buffers["front_rgb"]
        depth = clean_depth(buffers["depth"], self.cfg["width"], self.cfg["height"])
        k = self.camera.get_intrinsics_matrix()
        pose, optical = grounded_camera_pose(p, rotation, np.array(self.cfg["camera_offset"]),
                                             self.cfg["floor_z"])
        camera_info = dict(width=self.cfg["width"], height=self.cfg["height"],
                           fx=float(k[0, 0]), fy=float(k[1, 1]),
                           cx=float(k[0, 2]), cy=float(k[1, 2]),
                           camera_to_robot=optical.tolist())
        self.cached = dict(
            jpeg=self._jpeg(rgb), third_jpeg=self._jpeg(buffers["third_rgb"]),
            depth=depth, depth_camera=camera_info,
            pose=pose, pose_frame="ground_projected_base",
            chassis_pose=dict(position=p.tolist(), orientation_wxyz=q.tolist()),
            sim_time=self.sim_time, floor_elevation=self.cfg["floor_z"],
            collided=self.collided, depth_definition="camera_forward_z_meters",
            simulator="Isaac Sim · Carter",
        )
        return self.cached

    def _check_pose(self):
        p, q = self.robot.get_world_pose()
        if (not np.isfinite(p).all() or not np.isfinite(q).all() or
                abs(float(p[2]) - self.cfg["floor_z"]) > 3.):
            raise RuntimeError(f"Carter left the floor at {p}; check spawn and scene collision geometry")

    def close(self):
        if self.app is not None:
            self.app.close()
            self.app = None
