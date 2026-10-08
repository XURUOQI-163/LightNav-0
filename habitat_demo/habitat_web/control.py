"""Protocol units and a shared bounded waypoint follower (not the MuJoCo MPC)."""

import math


def parse_waypoints(value, backend):
    points = []
    width = 4 if backend == "omnivla" else 3

    def collect(item, depth=0):
        if depth > 8 or len(points) > 512:
            raise ValueError("too many or deeply nested waypoints")
        if isinstance(item, dict) and backend == "lightnav":
            if "actions" not in item:
                raise ValueError("missing actions")
            collect(item["actions"], depth + 1)
        elif isinstance(item, list):
            if item and all(isinstance(x, (int, float)) for x in item):
                if len(item) % width:
                    raise ValueError("invalid waypoint dimensions")
                for offset in range(0, len(item), width):
                    row = item[offset : offset + width]
                    if any(isinstance(x, bool) or not math.isfinite(x) for x in row):
                        raise ValueError("waypoints must contain finite numbers")
                    if backend == "omnivla":
                        if math.hypot(row[2], row[3]) < 1e-8:
                            raise ValueError("invalid heading vector")
                        points.append((row[0] * 0.1, row[1] * 0.1, math.atan2(row[3], row[2])))
                    else:
                        points.append(tuple(float(x) for x in row))
            else:
                for nested in item:
                    collect(nested, depth + 1)
        else:
            raise ValueError("waypoints must be an array")

    collect(value)
    if len(points) > 512:
        raise ValueError("too many waypoints")
    return tuple(points)


def command_for_path(points, dt, max_linear, max_angular):
    if not points:
        return 0.0, 0.0
    forward, left, yaw = points[min(4, len(points) - 1)]
    if math.hypot(forward, left) < 1e-8:
        return 0.0, max(-max_angular, min(max_angular, yaw / dt))
    angle = math.atan2(left, forward)
    v = max(0.0, forward / dt)
    w = angle / dt
    scale = min(1.0, max_linear / v if v else 1.0, max_angular / abs(w) if w else 1.0)
    return v * scale, w * scale
