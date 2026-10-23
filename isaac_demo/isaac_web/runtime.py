"""Isaac lifecycle, mapping and selectable LightNav execution scheduling."""

import asyncio
import contextlib
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from habitat_web.runtime import Runtime


class IsaacRuntime(Runtime):
    sensor_api = True

    def __init__(self, *args, background_startup=False, local_navigation=None,
                 use_mpc=False, realtime_update=True, path_position_tolerance=.10,
                 path_yaw_tolerance=math.radians(10), path_timeout=60., **kwargs):
        super().__init__(*args, **kwargs)
        self.use_mpc, self.realtime_update = use_mpc, realtime_update
        self.path_position_tolerance = path_position_tolerance
        self.path_yaw_tolerance = path_yaw_tolerance
        self.path_timeout = path_timeout
        self.local_navigation = local_navigation
        self.navigation_executor = (ThreadPoolExecutor(max_workers=1, thread_name_prefix='local-navigation')
                                    if local_navigation else None)
        self.local_snapshot = local_navigation.snapshot() if local_navigation else {'enabled': False}
        self.background_startup = background_startup
        self.startup_cancel = threading.Event()
        self.initialization_task = None
        self._startup_loop = None
        self._startup_started = time.monotonic()
        self._phase_started = self._startup_started
        self._startup_finished = None
        self._initialization_cleaned = False
        self._closing = False
        self.state.update(scene_ready=False, startup=dict(
            phase="queued", message="网页服务启动，等待场景初始化"))
        self.observation = None
        self._recovery_turn_used = 0.
        self._recovery_anchor = None
        self.state['recovery'] = dict(active=False)
        self.observation_frame_id = 0
        self.display_frame = None
        self.display_frame_id = 0
        self.state.update(simulator="Isaac Sim · Carter", manual_control=True,
                          live_view=True, motion_phase="idle",
                          max_linear=self.max_linear, max_angular=self.max_angular)
        self.state['tracking_config'] = dict(mpc=use_mpc, realtime_update=realtime_update,
            position_tolerance=path_position_tolerance, yaw_tolerance=path_yaw_tolerance,
            path_timeout=path_timeout)

    async def _run(self, backend, instruction):
        if backend == 'lightnav':
            from .execution import run_lightnav
            await run_lightnav(self, instruction)
        else:
            # OmniVLA retains its original prediction/execution loop.
            self.state.pop('tracking', None)
            await super()._run(backend, instruction)

    async def begin_initialize(self):
        if not self.background_startup:
            await self.initialize()
            return
        # Return to aiohttp immediately, so HTTP is available during Kit startup.
        self.initialization_task = asyncio.create_task(self._initialize_in_background())

    async def _initialize_in_background(self):
        try:
            await self.initialize()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.state.update(status="error", error=f"{type(exc).__name__}: {exc}")
            self.state["startup"].update(phase="error", message="场景加载失败，请检查终端并重启网页服务")

    async def initialize(self):
        self._startup_loop = asyncio.get_running_loop()
        self._startup_started = self._phase_started = time.monotonic()
        try:
            await super().initialize()
        except BaseException:
            # Runtime.initialize owns failure cleanup (worker, memory, executor).
            self._initialization_cleaned = True
            self._startup_finished = time.monotonic()
            raise
        self._startup_finished = time.monotonic()
        self.state["scene_ready"] = True
        self.state["startup"].update(phase="ready", message="Carter 与 RGB-D 已就绪")

    def report_startup_progress(self, progress):
        # Called in the bridge executor thread, never mutate UI state there.
        loop = self._startup_loop
        if loop is not None and not loop.is_closed() and not self._closing:
            loop.call_soon_threadsafe(self._apply_startup_progress, dict(progress))

    def _apply_startup_progress(self, progress):
        if self._closing or self.state["scene_ready"] or self._startup_finished is not None:
            return
        if progress.get("phase") != self.state["startup"].get("phase"):
            self._phase_started = time.monotonic()
        self.state["startup"].update(progress)

    def status_snapshot(self):
        snapshot = dict(self.state)
        end = self._startup_finished or time.monotonic()
        snapshot["startup"] = dict(self.state["startup"],
                                   elapsed_s=round(end - self._startup_started, 1),
                                   phase_elapsed_s=round(end - self._phase_started, 1))
        return snapshot

    def _require_ready(self):
        if not self.state["scene_ready"]:
            raise ValueError("场景尚未就绪；请等待加载完成，加载失败时请重启网页服务")

    async def reset(self):
        self._require_ready()
        await super().reset()
        self.state.pop('tracking', None)
        self._reset_recovery()

    async def stop(self):
        self._require_ready()
        await super().stop()
        if self.local_navigation:
            await self._nav_call(self.local_navigation.clear_prediction)

    def _publish_preview(self, frame):
        # Display-only frames must never enter model history or Room Memory.
        self.display_frame = frame
        self.display_frame_id += 1

    def _publish(self, observation, preserve_view=False):
        self.observation = observation
        self.observation_frame_id += 1
        self._publish_preview(observation)
        self.state.update(observation.get("timing", {}))
        for key in ("applied_command", "body_attitude", "chassis_pose"):
            if key in observation:
                self.state[key] = observation[key]
        super()._publish(observation, preserve_view=preserve_view)

    async def _sim_call(self, fn, *args):
        if self.local_navigation and fn is not self.scene_factory and getattr(fn, '__name__', '') == 'reset':
            await self._nav_call(self.local_navigation.reset)
        result = await self._isaac_call(fn, *args)
        if self.local_navigation and isinstance(result, dict) and 'jpeg' in result:
            await self._nav_call(self.local_navigation.observe, result)
        return result

    async def _nav_call(self, fn, *args, **kwargs):
        future = asyncio.get_running_loop().run_in_executor(self.navigation_executor, partial(fn, *args, **kwargs))
        try:
            result = await asyncio.shield(future)
        except asyncio.CancelledError:
            await future
            raise
        finally:
            # A cancelled HTTP request can still finish its worker mutation.
            # Publish actual mode/map state before propagating cancellation.
            self.local_snapshot = self.local_navigation.snapshot()
            self.state['navigation_reason'] = (self.local_snapshot['reason']
                if self.local_snapshot['status'] in ('blocked', 'waiting_observation') else '')
        return result

    async def prepare_path(self, waypoints, capture_pose):
        if not self.local_navigation:
            return waypoints
        from local_navigation.models import pose_tuple
        return await self._nav_call(self.local_navigation.plan, waypoints, pose_tuple(capture_pose))

    async def replan_prediction(self, waypoints, capture_pose, start_index):
        from local_navigation.models import pose_tuple
        return await self._nav_call(self.local_navigation.plan_remaining,
                                    waypoints, pose_tuple(capture_pose), start_index)

    async def filter_command(self, linear, angular, duration, *, manual=False):
        if not self.local_navigation:
            return linear, angular
        return await self._nav_call(self.local_navigation.guard, (linear, angular), duration,
                                    max_angular=self.max_angular, manual=manual)

    def _reset_recovery(self):
        self._recovery_turn_used = 0.
        self._recovery_anchor = None
        self.state['recovery'] = dict(active=False)

    async def check_motion_result(self, observation, linear):
        if (self.local_snapshot.get('execution_collision_check', False)
                and self.local_navigation and self.local_navigation.config.recovery_max_turn_deg > 0
                and linear > 1e-6 and observation.get('collided')):
            # Isaac's flag means insufficient translation, not an exact contact sensor.
            return await self.recover_blocked()
        return True

    async def recover_blocked(self):
        """User-authorized simulation escape: v=0, no rotation collision guard."""
        nav = self.local_navigation
        if nav is None or self.local_snapshot.get('mode') != 'guarded':
            return False
        cfg = nav.config
        if cfg.recovery_max_turn_deg == 0:
            return False
        from local_navigation.recovery import choose_depth_turn

        pose = self.state['pose']
        xy = (pose['x'], pose['y'])
        if self._recovery_anchor is None or math.dist(xy, self._recovery_anchor) >= cfg.recovery_reset_translation_m:
            self._recovery_anchor, self._recovery_turn_used = xy, 0.
        budget = math.radians(cfg.recovery_max_turn_deg)
        speed = min(cfg.recovery_angular_speed, self.max_angular)
        decision = choose_depth_turn(self.observation, cfg)
        detail = dict(decision, active=True, rotation_assumed_safe=True,
                      turn_used_deg=math.degrees(self._recovery_turn_used), reason='depth_turn')
        self.state['recovery'] = detail

        def finish(reason, success=False):
            detail.update(reason=reason, active=False,
                          turn_used_deg=math.degrees(self._recovery_turn_used))
            self.state['navigation_reason'] = '' if success else reason
            return success

        try:
            if self._recovery_turn_used >= budget - 1e-9:
                return finish('recovery_turn_limit')
            if not decision['direction']:
                return finish('recovery_insufficient_depth')
            if speed <= 0 or self.dt <= 0:
                return finish('recovery_zero_speed')
            # Lock this burst's selected direction; don't oscillate left/right
            # on small score changes while the camera is turning.
            while self._recovery_turn_used < budget - 1e-9:
                if self.max_steps and self.state['step'] >= self.max_steps:
                    return finish('step_limit')
                duration = min(self.dt, 1., (budget - self._recovery_turn_used) / speed)
                self._recovery_turn_used += speed * duration
                detail['turn_used_deg'] = math.degrees(self._recovery_turn_used)
                # Only this pure-rotation recovery command bypasses the guard.
                # The worker applies zero wheel commands at each step's end.
                frame = await self._sim_call(self.scene.step, 0., decision['direction'] * speed, duration)
                self._publish(frame)
                self.state['step'] += 1
                if await self._nav_call(nav.recovery_forward_clear):
                    return finish('recovery_forward_clear', True)
                current = choose_depth_turn(self.observation, cfg)
                detail['latest_depth'] = current
                if not current['direction']:
                    return finish('recovery_insufficient_depth')
            return finish('recovery_turn_limit')
        except asyncio.CancelledError:
            detail['reason'] = 'recovery_cancelled'
            raise
        finally:
            detail['active'] = False

    async def _isaac_call(self, fn, *args):
        if fn is not self.scene_factory:
            if getattr(fn, "__name__", "") == "step":
                loop = asyncio.get_running_loop()
                self.scene.preview_callback = lambda frame: loop.call_soon_threadsafe(
                    self._publish_preview, frame
                )
                self.state["motion_phase"] = "moving"
                try:
                    return await super()._sim_call(fn, *args)
                finally:
                    self.state["motion_phase"] = ('tracking_path' if self.state.get('tracking')
                                                   else 'waiting_model')
            return await super()._sim_call(fn, *args)
        future = asyncio.get_running_loop().run_in_executor(self.executor, fn, *args)
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            # initialize() never assigns a cancelled call's result. Retain it so
            # its exception cleanup can close the child instead of leaking it.
            self.scene = await future
            raise

    async def start(self, backend, instruction):
        self._require_ready()
        if self.local_snapshot.get('mode') == 'observe_only':
            raise ValueError('手动采图模式不允许自动导航；请先在局部地图面板启用轨迹纠正')
        if self.commands.locked():
            raise ValueError("当前操作尚未结束，请稍后重新开始导航")
        await super().start(backend, instruction)
        self._reset_recovery()

    async def navigation_mode(self, mode):
        self._require_ready()
        if not self.local_navigation:
            raise ValueError('未启用 --local-navigation')
        if self.commands.locked() or self.state['status'] in ('running', 'manual'):
            raise ValueError('请先停止导航并等待当前操作结束，再切换采图/轨迹纠正模式')
        async with self.commands:
            await self._nav_call(self.local_navigation.set_mode, mode)

    async def manual(self, linear, angular, duration):
        self._require_ready()
        values = (linear, angular, duration)
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
               for v in values):
            raise ValueError("manual command requires finite numeric velocities/duration")
        if abs(linear) > self.max_linear or abs(angular) > self.max_angular or not 0 < duration <= 1:
            raise ValueError("manual command exceeds speed limits or 1 second duration")
        if self.commands.locked() or self.state["status"] in ("running", "manual", "loading"):
            raise ValueError("请先停止自动导航，并等待当前操作结束")
        async with self.commands:
            self.state.update(status="manual", error="", waypoints=[])
            try:
                linear, angular = await self.filter_command(linear, angular, duration, manual=True)
                if self.local_navigation and linear == 0 and angular == 0:
                    self.state.update(status='stopped', stop_reason=self.state.get('navigation_reason') or 'zero_command')
                    return
                obs = await self._sim_call(self.scene.step, linear, angular, duration)
                self._publish(obs)
                self.state.update(status="stopped", stop_reason="manual_step")
                self.state["step"] += 1
            except asyncio.CancelledError:
                self._publish(await self._sim_call(self.scene.observe))
                self.state.update(status="stopped", stop_reason="manual_step")
                raise
            except Exception as exc:
                self.state.update(status="error", error=f"{type(exc).__name__}: {exc}")
                raise ValueError(self.state["error"]) from exc

    async def close(self):
        try:
            await self._close_isaac()
        finally:
            if self.navigation_executor:
                self.navigation_executor.shutdown(wait=True, cancel_futures=True)

    async def _close_isaac(self):
        if self._closing:
            return
        self._closing = True
        if self.initialization_task is not None and not self.initialization_task.done():
            self.startup_cancel.set()
            self.initialization_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.initialization_task
        if self._initialization_cleaned:
            return
        if not self.state["scene_ready"]:
            self.executor.shutdown(wait=True)
            if self.room_memory is not None:
                await self.room_memory.close()
            return
        # A dead worker must not prevent memory flush or thread cleanup.
        try:
            await self.stop()
        finally:
            try:
                if self.scene is not None:
                    await self._sim_call(self.scene.close)
            finally:
                if self.room_memory is not None:
                    await self.room_memory.close()
                self.executor.shutdown(wait=True)
