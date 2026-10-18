"""Bounded SE(2) search around a model path, never snap through a wall."""
import heapq
import math
import time

import numpy as np


def advance(pose, distance, turn):
    x, y, yaw = pose
    if abs(turn) < 1e-9:
        return (x + distance * math.cos(yaw), y + distance * math.sin(yaw), yaw)
    radius = distance / turn
    return (x + radius * (math.sin(yaw + turn) - math.sin(yaw)),
            y - radius * (math.cos(yaw + turn) - math.cos(yaw)),
            math.atan2(math.sin(yaw + turn), math.cos(yaw + turn)))


def advisory_path(raw, pose, checker, result):
    """Keep correction advisory, including departure from an occupied start.

    Never erase map evidence. The original prefix up to the first clear model
    point is explicitly unvalidated; the existing planner repairs the suffix.
    Failed correction keeps the original prediction, not a claimed safe route.
    """
    if not raw or any(len(p) != 3 or not all(math.isfinite(v) for v in p)
                      for p in [pose] + list(raw)):
        return result
    if result['reason'] == 'start_occupied':
        started = time.monotonic()
        planner = checker.for_planning()
        result['start_overlap_ignored'] = True
        for index, anchor in enumerate(raw):
            if planner.reason(anchor) != 'clear':
                continue
            if index + 1 < len(raw):
                suffix = correct_path(raw[index + 1:], anchor, checker)
                if suffix['corrected']:
                    result.update(corrected=[list(p) for p in [pose] + list(raw[:index + 1])]
                                  + suffix['corrected'][1:],
                                  status=suffix['status'] if suffix['status'] != 'unchanged' else 'fallback',
                                  reason='start_occupied_repaired_suffix',
                                  unvalidated_prefix_points=index + 1,
                                  expanded=suffix['expanded'])
                    if 'goal_index' in suffix:
                        result['goal_index'] = suffix['goal_index'] + index + 1
                    result['planning_ms'] += round((time.monotonic() - started) * 1000, 2)
                    return result
            break
    if not result['corrected']:
        result['correction_failure'] = result['reason']
        result.update(corrected=[list(p) for p in [pose] + list(raw)], status='fallback',
                      reason=result['reason'] + '_continue_raw')
    return result


def correct_path(raw, pose, checker):
    start_time = time.monotonic()
    cfg, grid = checker.config, checker.grid
    raw = [tuple(map(float, p)) for p in raw]
    execution_checker = checker
    point_status = [checker.reason(p) for p in raw]
    checker = checker.for_planning()
    segment_status = []
    for a, b in zip([pose] + raw, raw):
        if time.monotonic() - start_time > cfg.search_timeout_s:
            segment_status.extend(['not_checked'] * (len(raw) - len(segment_status)))
            break
        status = checker.segment(a, b, allow_outside=True)
        # Preserve unknown/outside provenance in the UI. An obstacle between
        # unknown endpoints takes priority over the original strict early exit.
        segment_status.append(execution_checker.segment(a, b) if status == 'clear' else status)
    result = dict(raw=raw, point_status=point_status, segment_status=segment_status,
                  corrected=[], status='blocked', reason='no_reachable_path', expanded=0,
                  planning_policy='known_obstacles_only')

    def finish(path, status, reason):
        result.update(corrected=[list(p) for p in path], status=status, reason=reason,
                      planning_ms=round((time.monotonic() - start_time) * 1000, 2))
        return result

    if not raw:
        return finish([], 'blocked', 'empty_path')
    if checker.reason(pose) != 'clear':
        return finish([], 'blocked', 'start_' + checker.reason(pose))
    if any(s in ('invalid_pose', 'not_checked') for s in segment_status):
        return finish([], 'blocked', 'path_validation_incomplete')
    if 'occupied' not in segment_status:
        reason = 'clear' if all(s == 'clear' for s in segment_status) else 'no_known_obstacle'
        return finish([pose] + raw, 'unchanged', reason)

    prefix = [pose]
    for p, status in zip(raw, segment_status):
        if status not in ('clear', 'unknown'):
            break
        prefix.append(p)

    # Repairs may cross unknown cells, but remain inside the local window and
    # avoid the full inflated occupied volume. Execution remains separately guarded.
    goals = []
    for index in reversed(range(len(raw))):
        p = raw[index]
        if math.hypot(p[0] - pose[0], p[1] - pose[1]) > cfg.window_radius_m:
            continue
        for radius in (0., cfg.max_goal_adjustment_m / 2, cfg.max_goal_adjustment_m):
            for angle in (np.linspace(0, 2 * math.pi, 12, endpoint=False) if radius else [0.]):
                q = (p[0] + radius * math.cos(angle), p[1] + radius * math.sin(angle), p[2])
                if math.hypot(q[0] - pose[0], q[1] - pose[1]) < cfg.primitive_length_m:
                    continue
                if checker.reason(q) == 'clear':
                    goals.append((index, q))
        if goals:
            break
    if not goals:
        result['reason'] = 'no_local_goal'
    # An unreachable endpoint still supplies a search direction. It never
    # authorizes driving there; every expansion requires a clear swept body.
    goal_xy = np.array([p[:2] for _, p in goals] or [raw[-1][:2]])
    refs = np.array([pose] + raw)[:, :2]
    lengths = np.linalg.norm(np.diff(refs, axis=0), axis=1)
    segments = refs[1:] - refs[:-1]
    cumulative = np.concatenate(([0.], np.cumsum(lengths)))
    max_length = float(lengths.sum()) + cfg.max_detour_extra_m
    angle_step = 2 * math.pi / cfg.heading_bins

    def key(p):
        return (*grid.cell(*p[:2]), round(p[2] / angle_step) % cfg.heading_bins)

    def heuristic(p):
        return float(np.linalg.norm(goal_xy - p[:2], axis=1).min())

    def projection(p):
        p = np.asarray(p)
        t = np.clip(np.sum((p[:2] - refs[:-1]) * segments, axis=1) /
                    np.maximum(lengths**2, 1e-12), 0, 1)
        distances = np.linalg.norm(refs[:-1] + t[:, None] * segments - p[:2], axis=1)
        index = int(np.argmin(distances))
        return float(distances[index]), float(cumulative[index] + t[index] * lengths[index])

    def reconstruct(node_id):
        path = []
        while node_id >= 0:
            node, node_id, _ = nodes[node_id]
            path.append(node)
        return list(reversed(path))

    # Nodes retain continuous poses and parent IDs even when a cheaper state replaces a bin.
    nodes = [(tuple(pose), -1, 0.)]
    costs = {key(pose): 0.}
    queue = [(heuristic(pose), 0., 0)]
    best_id = None
    best_score = projection(prefix[-1])[1] if len(prefix) > 1 else 0.
    while queue and result['expanded'] < cfg.max_search_expansions:
        if time.monotonic() - start_time > cfg.search_timeout_s:
            result['reason'] = 'search_timeout'
            break
        _, cost, node_id = heapq.heappop(queue)
        here, _, length = nodes[node_id]
        if cost > costs.get(key(here), math.inf) + 1e-9:
            continue
        result['expanded'] += 1
        if goals and heuristic(here) <= cfg.primitive_length_m:
            # Finish with an already validated reachable state; no unchecked snap to goal.
            path = reconstruct(node_id)
            if len(path) > 1:
                # Consumers executing a whole prediction must distinguish an
                # adjusted final goal from a reachable earlier waypoint.
                result['goal_index'] = max(index for index, goal in goals
                    if math.dist(here[:2], goal[:2]) <= cfg.primitive_length_m)
                return finish(path, 'corrected', 'local_detour')
        for distance, turn in ((cfg.primitive_length_m, 0.),
                               (cfg.primitive_length_m, angle_step),
                               (cfg.primitive_length_m, -angle_step),
                               (0., angle_step), (0., -angle_step)):
            nxt = advance(here, distance, turn)
            if length + distance > max_length:
                continue
            # Arc checked at its midpoint as well as along both swept chords.
            mid = advance(here, distance / 2, turn / 2)
            if (checker.segment(here, mid) != 'clear' or checker.segment(mid, nxt) != 'clear'
                    or checker.segment(here, nxt) != 'clear'):
                continue
            ix, iy = grid.cell(*nxt[:2])
            near = max(0., cfg.soft_clearance_m - checker.clearance[iy, ix])
            deviation, progress = projection(nxt)
            new_cost = cost + distance + .08 * abs(turn) + .08 * deviation + .3 * near
            k = key(nxt)
            if new_cost >= costs.get(k, math.inf):
                continue
            costs[k] = new_cost
            nodes.append((nxt, node_id, length + distance))
            # Preserve a meaningful, verified approach even if the far goal
            # cannot be reached. Pure turns and sideways/backwards drifting do
            # not count as progress. Unknown space is allowed for planning only.
            score = progress - deviation
            if (progress >= cfg.primitive_length_m and deviation <= cfg.max_goal_adjustment_m
                    and math.dist(nxt[:2], pose[:2]) >= cfg.primitive_length_m
                    and score > best_score + 1e-9):
                best_id, best_score = len(nodes) - 1, score
            heapq.heappush(queue, (new_cost + cfg.heuristic_weight * heuristic(nxt), new_cost, len(nodes) - 1))
    if best_id is not None:
        result['search_reason'] = result['reason']
        return finish(reconstruct(best_id), 'safe_prefix', 'reachable_progress')
    return finish(prefix if len(prefix) > 1 else [],
                  'safe_prefix' if len(prefix) > 1 else 'blocked', result['reason'])
