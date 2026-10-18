"""Map-driven execution of one immutable prediction with bounded model fallback."""

import math

from local_navigation.models import pose_tuple

from .tracking import WorldPath, angle_error


async def execute_incrementally(r, tracker, waypoints, capture_pose):
    cfg = r.local_navigation.config
    raw = WorldPath(waypoints, pose_tuple(capture_pose),
                    r.path_position_tolerance, r.path_yaw_tolerance)
    tracking = r.state['tracking']
    tracking.setdefault('model_replans', 0)
    tracking.setdefault('failed_prediction_retries', 0)
    tracking.setdefault('retry_anchor', list(pose_tuple(r.state['pose'])[:2]))
    tracking.setdefault('local_replans', 0)
    tracking.update(request_reason='executing_prediction', failure_confirmations=0)
    began = last_progress = r.state['sim_time']
    progress_pose = pose_tuple(r.state['pose'])
    rotation_only = all(math.dist(p[:2], progress_pose[:2]) <= r.path_position_tolerance
                        for p in raw.points)
    route, complete_plan = None, False
    last_plan_time, last_revision = -math.inf, None
    session = r.local_snapshot.get('session_id')
    failures, recovery_attempted = 0, False
    force_plan, moved = True, False

    def stopped(reason):
        r.state['stop_reason'] = reason
        return False

    async def repredict(reason):
        # A failed route is retired, never replayed while the new request is pending.
        tracker.previous = (0., 0.)
        tracking.update(command=[0., 0.], request_reason=reason, last_reprediction_reason=reason)
        await r._nav_call(r.local_navigation.clear_prediction)
        if tracking['failed_prediction_retries'] >= cfg.model_replan_limit:
            return stopped('model_replan_limit')
        tracking['failed_prediction_retries'] += 1
        tracking['model_replans'] += 1
        print(f"[isaac-web] 当前预测无法继续 ({reason})，请求新的 LightNav 路径 "
              f"[{tracking['failed_prediction_retries']}/{cfg.model_replan_limit}]", flush=True)
        return True

    async def refresh_stationary():
        # A fresh RGB-D frame, not another search on the same unchanged snapshot.
        frame = await r._sim_call(r.scene.step, 0., 0., tracker.dt)
        r._publish(frame, preserve_view=True)
        r.state['step'] += 1
        tracker.previous = (0., 0.)

    async def recover_once(reason):
        nonlocal recovery_attempted, last_progress, progress_pose
        if recovery_attempted:
            return False
        recovery_attempted = True
        tracking['request_reason'] = reason + '_recovering'
        recovered = await r.recover_blocked()
        tracker.previous = (0., 0.)
        progress_pose = pose_tuple(r.state['pose'])
        last_progress = r.state['sim_time']
        return recovered

    while not r.max_steps or r.state['step'] < r.max_steps:
        pose = pose_tuple(r.state['pose'])
        now = r.state['sim_time']
        if r.local_snapshot.get('session_id') != session:
            return stopped('navigation_session_changed')
        # Only actual translation replenishes repeated failed-prediction retries.
        if math.dist(pose[:2], tracking['retry_anchor']) >= cfg.recovery_reset_translation_m:
            tracking['failed_prediction_retries'] = 0
            tracking['retry_anchor'] = list(pose[:2])
        heading_progress = (rotation_only and
            abs(angle_error(raw.points[-1][2], progress_pose[2])) -
            abs(angle_error(raw.points[-1][2], pose[2])) >= r.path_yaw_tolerance)
        if math.dist(pose[:2], progress_pose[:2]) >= cfg.path_progress_translation_m or heading_progress:
            last_progress, progress_pose = now, pose
        raw.remaining(pose)
        tracking.update(path_elapsed_s=now - began, raw_path_index=raw.index,
                        failure_confirmations=failures)
        # Completion is decided on the latest accepted full correction, never on a prefix.
        if route is not None and complete_plan and route.complete(pose):
            if not moved:
                return stopped('path_already_reached')
            tracking['completed_paths'] += 1
            tracking.update(request_reason='path_completed', failed_prediction_retries=0)
            return True
        if now - began >= r.path_timeout:
            return await repredict('path_execution_timeout')
        if now - last_progress >= cfg.path_no_progress_s:
            if await recover_once('path_no_progress'):
                route, force_plan, failures = None, True, 0
                continue
            if r.max_steps and r.state['step'] >= r.max_steps:
                return stopped('step_limit')
            return await repredict('path_no_progress')
        revision = r.local_snapshot.get('map_revision')
        prefix_done = route is not None and not complete_plan and route.complete(pose)
        periodic = now - last_plan_time >= cfg.path_replan_interval_s and revision != last_revision
        if force_plan or route is None or prefix_done or periodic:
            r.state['motion_phase'] = 'correcting_path'
            path = await r.replan_prediction(waypoints, capture_pose, raw.index)
            last_plan_time, last_revision = now, revision
            tracking['local_replans'] += 1
            result = r.local_snapshot.get('prediction') or {}
            complete_plan = (result.get('status') in ('unchanged', 'fallback') or
                (result.get('status') == 'corrected' and result.get('goal_index') == len(raw.points) - 1))
            tracking.update(plan_status=result.get('status'), plan_reason=result.get('reason'),
                            request_reason='local_path_update')
            route = (WorldPath(path, pose, r.path_position_tolerance, r.path_yaw_tolerance)
                     if path else None)
            force_plan = False
        r.state['motion_phase'] = 'tracking_path'
        # A newly revealed complete route may already be within goal tolerance.
        if route is not None and complete_plan and route.complete(pose):
            if not moved:
                return stopped('path_already_reached')
            tracking['completed_paths'] += 1
            tracking.update(request_reason='path_completed', failed_prediction_retries=0)
            return True
        linear, angular = 0., 0.
        if route is not None and (complete_plan or not route.complete(pose)):
            remaining = route.remaining(pose)
            tracking.update(path_index=route.index, path_points=len(route.points))
            linear, angular = await tracker.command(remaining, pose, finish_segment=True,
                position_tolerance=r.path_position_tolerance)
            linear, angular = await r.filter_command(linear, angular, tracker.dt)
        if abs(linear) < 1e-6 and abs(angular) < 1e-6:
            failures += 1
            tracking.update(failure_confirmations=failures, command=[0., 0.])
            if failures >= cfg.path_failure_confirmations:
                if await recover_once('path_blocked'):
                    route, force_plan, failures = None, True, 0
                    continue
                if r.max_steps and r.state['step'] >= r.max_steps:
                    return stopped('step_limit')
                return await repredict('path_unexecutable')
            await refresh_stationary()
            force_plan = True
            continue
        tracking['command'] = [linear, angular]
        frame = await r._sim_call(r.scene.step, linear, angular, tracker.dt)
        tracker.previous = (linear, angular)
        r._publish(frame, preserve_view=True)
        r.state['step'] += 1
        moved = True
        if (r.local_snapshot.get('execution_collision_check', False)
                and linear > 1e-6 and frame.get('collided')):
            # Contact/stall invalidates the executable route, not the original prediction.
            failures += 1
            route, force_plan = None, True
            if failures >= cfg.path_failure_confirmations:
                if await recover_once('motion_blocked'):
                    failures = 0
                    continue
                if r.max_steps and r.state['step'] >= r.max_steps:
                    return stopped('step_limit')
                return await repredict('motion_blocked')
        else:
            failures = 0
    return stopped('step_limit')
