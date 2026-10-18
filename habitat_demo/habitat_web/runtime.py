"""Observation → prediction → bounded simulation step; display never advances time."""

import asyncio
import contextlib
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

from .clients import ModelClient
from .control import command_for_path
from .waypoint_overlay import project_waypoints


class Runtime:
    def __init__(
        self,
        scene_factory,
        urls,
        *,
        scene_name="",
        dt=1 / 3,
        max_linear=0.15,
        max_angular=0.5,
        max_steps=0,
        timeout=30.0,
        client_factory=None,
        room_memory=None,
    ):
        self.scene_factory, self.urls = scene_factory, urls
        self.room_memory = room_memory
        self.dt, self.max_linear, self.max_angular = dt, max_linear, max_angular
        self.max_steps = max_steps
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 0:
            raise ValueError("max_steps must be a nonnegative integer (0 means unlimited)")
        self.client_factory = client_factory or (
            lambda backend, url: ModelClient(backend, url, timeout)
        )
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="habitat-egl")
        self.scene = None
        self.task = None
        self.commands = None
        self.jpeg = None
        self.view_jpeg = None
        self.pointing = None
        self.waypoint_overlay = None
        self.camera_calibration = None
        self.state = dict(
            status="loading",
            stop_reason=None,
            backend="lightnav",
            instruction="",
            error="",
            step=0,
            sim_time=0.0,
            latency_ms=None,
            model_latency_ms=None,
            pose=None,
            waypoints=[],
            frame_id=0,
            collided=False,
            max_steps=max_steps,
            scene=scene_name,
        )

    async def _sim_call(self, fn, *args):
        # Cancelling a Future cannot interrupt EGL/C++; await its completion before
        # acknowledging STOP or RESET so no old step can occur after acknowledgment.
        future = asyncio.get_running_loop().run_in_executor(self.executor, fn, *args)
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            await future
            raise

    async def initialize(self):
        self.commands = asyncio.Lock()
        try:
            if self.room_memory is not None:
                await self.room_memory.start()
            self.scene = await self._sim_call(self.scene_factory)
            self._publish(await self._sim_call(self.scene.reset))
            self.state["status"] = "idle"
        except BaseException:
            if self.scene is not None:
                await self._sim_call(self.scene.close)
            self.executor.shutdown(wait=True)
            if self.room_memory is not None:
                await self.room_memory.close()
            raise

    def _publish(self, observation, preserve_view=False):
        self.jpeg = observation["jpeg"]
        self.camera_calibration = deepcopy(observation.get("depth_camera"))
        if not preserve_view:
            self.view_jpeg = self.jpeg
            self.pointing = None
            self.waypoint_overlay = None
        self.state.update({k: observation[k] for k in ("pose", "sim_time", "collided")})
        self.state["frame_id"] += 1
        if self.room_memory is not None:
            # Pointing pins the displayed RGB to the prediction's input frame.
            # Geometry observes the new pose, but live semantics follows the view.
            self.room_memory.offer(observation, live=not preserve_view)

    async def start(self, backend, instruction):
        async with self.commands:
            if backend not in self.urls:
                raise ValueError("unsupported model")
            if (
                not isinstance(instruction, str)
                or not instruction.strip()
                or len(instruction) > 1024
            ):
                raise ValueError("请输入 1–1024 字符的导航指令")
            if self.task is not None and not self.task.done():
                raise ValueError("任务正在运行，请先停止")
            self.view_jpeg, self.pointing = self.jpeg, None
            self.waypoint_overlay = None
            if self.room_memory is not None:
                self.room_memory.observe_camera(self.view_jpeg, self.state["sim_time"])
            self.state["frame_id"] += 1
            self.state.update(
                status="running",
                stop_reason=None,
                backend=backend,
                instruction=instruction.strip(),
                error="",
                step=0,
                waypoints=[],
                latency_ms=None,
                model_latency_ms=None,
            )
            self.task = asyncio.create_task(self._run(backend, instruction.strip()))

    async def _run(self, backend, instruction):
        try:
            async with self.client_factory(backend, self.urls[backend]) as client:
                while self.max_steps == 0 or self.state["step"] < self.max_steps:
                    captured_jpeg = self.jpeg
                    captured_time = self.state["sim_time"]
                    captured_camera = deepcopy(self.camera_calibration)
                    captured_pose = deepcopy(self.state['pose'])
                    prediction = await client.predict(captured_jpeg, instruction)
                    self.view_jpeg = captured_jpeg
                    if self.room_memory is not None:
                        self.room_memory.observe_camera(captured_jpeg, captured_time)
                    self.pointing = prediction.pointing if backend == "lightnav" else None
                    self.waypoint_overlay = project_waypoints(prediction.waypoints, captured_camera)
                    self.state["frame_id"] += 1
                    self.state.update(
                        waypoints=prediction.waypoints,
                        latency_ms=prediction.latency_ms,
                        model_latency_ms=prediction.model_latency_ms,
                    )
                    if prediction.stop:
                        self.state["stop_reason"] = "model_stop"
                        break
                    if not prediction.waypoints:
                        self.state["stop_reason"] = "empty_path"
                        break
                    path = await self.prepare_path(prediction.waypoints, captured_pose)
                    if not path:
                        if await self.recover_blocked():
                            continue  # Request new waypoints from the new camera pose.
                        self.state['stop_reason'] = self.state.get('navigation_reason', 'blocked_path')
                        break
                    linear, angular = command_for_path(
                        path, self.dt, self.max_linear, self.max_angular
                    )
                    linear, angular = await self.filter_command(linear, angular, self.dt)
                    if linear == 0 and angular == 0:
                        if await self.recover_blocked():
                            continue
                        self.state["stop_reason"] = self.state.get('navigation_reason') or "zero_command"
                        break
                    observation = await self._sim_call(self.scene.step, linear, angular, self.dt)
                    self._publish(observation, preserve_view=(
                        self.pointing is not None or self.waypoint_overlay is not None
                    ))
                    self.state["step"] += 1
                    if not await self.check_motion_result(observation, linear):
                        self.state['stop_reason'] = self.state.get('navigation_reason') or 'motion_blocked'
                        break
                else:
                    self.state["stop_reason"] = "step_limit"
            self.state["status"] = "stopped"
            if self.room_memory is not None:
                await self.room_memory.flush()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.state.update(
                status="error", stop_reason="error", error=f"{type(exc).__name__}: {exc}"
            )

    async def prepare_path(self, waypoints, capture_pose):
        """Optional adapter hook; default navigation remains unchanged."""
        return waypoints

    async def filter_command(self, linear, angular, duration, *, manual=False):
        return linear, angular

    async def recover_blocked(self):
        """Optional simulator recovery; Habitat retains its previous stop behaviour."""
        return False

    async def check_motion_result(self, observation, linear):
        """Optional adapter handling of physical stalls; default keeps existing behaviour."""
        return True

    async def _stop(self):
        if self.state["status"] == "running":
            self.state["stop_reason"] = "user_stop"
        if self.task is not None:
            self.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.task
            self.task = None
            # Cancellation may arrive after an EGL step starts. _sim_call waits
            # for it to finish; publish that final pose/frame before STOP returns.
            observation = await self._sim_call(self.scene.observe)
            if observation["sim_time"] > self.state["sim_time"] + 1e-9:
                self.state["step"] += 1
            self._publish(observation)
        self.state["status"] = "stopped"

    async def stop(self):
        async with self.commands:
            await self._stop()
            if self.room_memory is not None:
                await self.room_memory.flush()

    async def reset(self):
        async with self.commands:
            await self._stop()
            if self.room_memory is not None:
                await self.room_memory.reset()
            self._publish(await self._sim_call(self.scene.reset))
            self.state.update(
                status="idle",
                stop_reason=None,
                step=0,
                waypoints=[],
                error="",
                latency_ms=None,
                model_latency_ms=None,
            )

    async def close(self):
        await self.stop()
        if self.scene is not None:
            await self._sim_call(self.scene.close)
        if self.room_memory is not None:
            await self.room_memory.close()
        self.executor.shutdown(wait=True)
