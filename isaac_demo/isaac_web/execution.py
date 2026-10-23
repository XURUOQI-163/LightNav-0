"""Isaac LightNav execution; the shared Habitat/OmniVLA loop stays intact."""

import asyncio
from copy import deepcopy

from habitat_web.waypoint_overlay import project_waypoints
from local_navigation.models import NavigationConfig, pose_tuple

from .tracking import PathTracker, WorldPath


def full_plan(runtime, count):
    if runtime.local_navigation is None:
        return True
    plan = runtime.local_snapshot.get('prediction') or {}
    return (plan.get('status') in ('unchanged', 'fallback') or
            (plan.get('status') == 'corrected' and plan.get('goal_index') == count - 1))


async def execute_path(runtime, tracker, waypoints, capture_pose, *, refresh=None):
    """Return True only when another prediction is allowed by the selected mode."""
    if runtime.local_navigation is not None and not runtime.realtime_update:
        from .incremental_execution import execute_incrementally
        return await execute_incrementally(runtime, tracker, waypoints, capture_pose)
    r = runtime
    pose = pose_tuple(r.state['pose'])
    path = await r.prepare_path(waypoints, capture_pose)
    complete_plan = full_plan(r, len(waypoints)) if path else True
    # In batch mode a blocked path is retained in WORLD coordinates during recovery.
    route = WorldPath(path or waypoints, pose if path and r.local_navigation else pose_tuple(capture_pose),
                      r.path_position_tolerance, r.path_yaw_tolerance)
    needs_replan = not path
    began = r.state['sim_time']
    moved = False
    tracking = r.state['tracking']
    cfg = r.local_navigation.config if r.local_navigation else NavigationConfig()
    tracking.setdefault('zero_command_steps', 0)
    tracking.setdefault('zero_command_replans', 0)
    while not r.max_steps or r.state['step'] < r.max_steps:
        if refresh is not None and refresh.ready():
            return True  # Consume/swap the next capture-aligned path between steps.
        pose = pose_tuple(r.state['pose'])
        remaining = route.remaining(pose)
        r.state['tracking'].update(path_index=route.index, path_points=len(route.points),
                                   path_elapsed_s=r.state['sim_time'] - began)
        if not needs_replan and route.complete(pose) and (not r.realtime_update or refresh is not None):
            if not complete_plan and refresh is None:
                r.state['stop_reason'] = 'path_incomplete_safe_prefix'
                return False
            if not moved:
                r.state['stop_reason'] = 'path_already_reached'
                return False  # Avoid repeatedly querying a stationary/zero path.
            r.state['tracking']['completed_paths'] += 1
            return True
        if r.state['sim_time'] - began >= r.path_timeout:
            r.state['stop_reason'] = 'path_execution_timeout'
            return False
        if needs_replan:
            # These are already world points, not points relative to the turned robot.
            repaired = await r.prepare_path(remaining, dict(x=0., y=0., yaw=0.))
            # Without local mapping, convert the retained world route directly.
            if r.local_navigation is None:
                from local_navigation.models import transform_path
                repaired = transform_path(remaining, pose, inverse=True)
            if repaired:
                complete_plan = complete_plan and full_plan(r, len(remaining))
                route = WorldPath(repaired, pose, r.path_position_tolerance, r.path_yaw_tolerance)
                remaining = route.remaining(pose)
                needs_replan = False
            else:
                if await r.recover_blocked():
                    tracker.previous = (0., 0.)
                    if r.realtime_update:
                        return True
                    continue
                r.state['stop_reason'] = r.state.get('navigation_reason') or 'blocked_path'
                return False
        linear, angular = await tracker.command(remaining, pose, finish_segment=not r.realtime_update,
                                                position_tolerance=r.path_position_tolerance)
        linear, angular = await r.filter_command(linear, angular, tracker.dt)
        if refresh is not None and refresh.ready():
            return True  # A STOP/error/new route can arrive while the solver runs.
        zero = abs(linear) < 1e-6 and abs(angular) < 1e-6
        if zero and r.state.get('navigation_reason', '').startswith('blocked_command_'):
            # An explicit guard veto is different from a controller zero crossing.
            if await r.recover_blocked():
                tracker.previous = (0., 0.)
                needs_replan = True
                continue
            r.state['stop_reason'] = r.state['navigation_reason']
            return False
        if zero:
            # Execute the zero crossing before solving again. Re-solving with
            # the old previous control repeats the acceleration-limited zero.
            linear, angular = 0., 0.
        else:
            tracking.update(zero_command_steps=0, zero_command_replans=0)
        if refresh is not None and refresh.ready():
            return True
        r.state['tracking']['command'] = [linear, angular]
        observation = await r._sim_call(r.scene.step, linear, angular, tracker.dt)
        tracker.previous = tuple(observation.get("applied_command", (linear, angular)))
        r._publish(observation, preserve_view=True)
        r.state['step'] += 1
        stationary = all(abs(v) < 1e-6 for v in tracker.previous)
        moved = moved or not stationary
        if not stationary:
            tracking.update(zero_command_steps=0, zero_command_replans=0)
        if zero and stationary:
            tracking['zero_command_steps'] += 1
            tracking['request_reason'] = 'zero_command_hold'
            if r.max_steps and r.state['step'] >= r.max_steps:
                r.state['stop_reason'] = 'step_limit'
                return False
            if tracking['zero_command_steps'] >= cfg.path_failure_confirmations:
                # Preserve explicitly enabled recovery, without turning it on.
                if await r.recover_blocked():
                    tracker.previous = (0., 0.)
                    tracking['zero_command_steps'] = 0
                    needs_replan = True
                    continue
                if tracking['zero_command_replans'] >= cfg.model_replan_limit:
                    r.state['stop_reason'] = 'model_replan_limit'
                    return False
                tracking['zero_command_replans'] += 1
                tracking.update(zero_command_steps=0,
                    request_reason='zero_command_persistent',
                    last_reprediction_reason='zero_command_persistent')
                return True  # Retire this route and await the next prediction.
        if (r.local_snapshot.get('execution_collision_check', False)
                and linear > 1e-6 and observation.get('collided')):
            if not await r.recover_blocked():
                r.state['stop_reason'] = r.state.get('navigation_reason') or 'motion_blocked'
                return False
            tracker.previous = (0., 0.)
            needs_replan = True
        if refresh is not None:
            if not r.max_steps or r.state['step'] < r.max_steps:
                refresh.request()  # At most one pending request, using a fresh observation.
                await asyncio.sleep(0)  # Let immediate model responses/STOP run before next solve.
        elif r.realtime_update:
            return True
    r.state['stop_reason'] = 'step_limit'
    return False


def publish_prediction(r, prediction, jpeg, stamp, camera):
    r.view_jpeg = jpeg
    if r.room_memory is not None:
        r.room_memory.observe_camera(jpeg, stamp)
    r.pointing = prediction.pointing
    r.waypoint_overlay = project_waypoints(prediction.waypoints, camera)
    r.state['frame_id'] += 1
    r.state.update(waypoints=prediction.waypoints, latency_ms=prediction.latency_ms,
                   model_latency_ms=prediction.model_latency_ms)


async def run_lightnav(runtime, instruction):
    r = runtime
    tracker = PathTracker(r.use_mpc, r.max_linear, r.max_angular, r.dt)
    r.state['tracking'] = dict(controller='mpc' if r.use_mpc else 'simple',
        realtime_update=r.realtime_update, control_dt=tracker.dt, requests=0,
        completed_paths=0, path_index=0, path_points=0, path_elapsed_s=0.,
        inference_pending=False, asynchronous_prediction=r.realtime_update)
    try:
        async with r.client_factory('lightnav', r.urls['lightnav']) as client:
            if r.realtime_update:
                from .realtime_prediction import run_realtime
                await run_realtime(r, tracker, client, instruction)
            while not r.realtime_update and (not r.max_steps or r.state['step'] < r.max_steps):
                r.state['motion_phase'] = 'waiting_model'
                jpeg, stamp = r.jpeg, r.state['sim_time']
                camera, pose = deepcopy(r.camera_calibration), deepcopy(r.state['pose'])
                r.state['tracking']['requests'] += 1
                prediction = await client.predict(jpeg, instruction)
                publish_prediction(r, prediction, jpeg, stamp, camera)
                if prediction.stop or not prediction.waypoints:
                    r.state['stop_reason'] = 'model_stop' if prediction.stop else 'empty_path'
                    break
                r.state['motion_phase'] = 'tracking_path'
                if not await execute_path(r, tracker, prediction.waypoints, pose):
                    break
            else:
                if not r.realtime_update:
                    r.state['stop_reason'] = 'step_limit'
        r.state['status'] = 'stopped'
        if r.room_memory is not None:
            await r.room_memory.flush()
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        r.state.update(status='error', stop_reason='error', error=f'{type(exc).__name__}: {exc}')
    finally:
        tracker.close()
        r.state['motion_phase'] = 'idle'
        r.state['tracking']['command'] = [0., 0.]
