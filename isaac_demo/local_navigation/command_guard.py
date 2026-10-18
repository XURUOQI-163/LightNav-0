"""Check the command's actual unicycle sweep, including a braking extension."""
import math

from .path_corrector import advance


def command_sweep(command, dt, pose, checker):
    v, w = command
    if not all(math.isfinite(x) for x in (v, w, dt)) or not 0 < dt <= 1:
        return 'invalid_command', [pose]
    c = checker.config
    coast = dt + c.reaction_time_s
    brake = max(abs(v) / c.braking_deceleration, abs(w) / c.braking_angular_deceleration)
    step = min(.05, c.resolution_m / (2 * max(.01, abs(v) + checker.radius * abs(w))))
    if (coast + brake) / step > 2048:
        return 'sweep_budget', [pose]
    samples = [pose]
    elapsed = 0.
    while elapsed < coast + brake:
        delta = min(step, coast + brake - elapsed)
        # Conservative simultaneous linear decay: the slower braking axis sets duration.
        scale = 1. if elapsed < coast else max(0., 1 - (elapsed - coast) / max(brake, 1e-9))
        nxt = advance(samples[-1], v * scale * delta, w * scale * delta)
        reason = checker.segment(samples[-1], nxt)
        if reason != 'clear':
            return reason, samples
        samples.append(nxt)
        elapsed += delta
    return 'clear', samples


def guard_command(command, dt, pose, checker, path, *, adjust=True):
    reason, sweep = command_sweep(command, dt, pose, checker)
    if reason == 'clear':
        return tuple(command), dict(reason='clear', sweep=sweep, adjusted=False)
    original_reason = reason
    if adjust and path and checker.reason(pose) == 'clear':
        target = next((p for p in path if math.hypot(p[0] - pose[0], p[1] - pose[1]) > .1), path[-1])
        angle = math.atan2(target[1] - pose[1], target[0] - pose[0]) - pose[2]
        turn = math.copysign(abs(command[1]) or .2, math.atan2(math.sin(angle), math.cos(angle)))
        candidates = [(command[0] * scale, command[1] * scale) for scale in (.5, .25)]
        candidates.append((0., turn))
        for candidate in candidates:
            reason, sweep = command_sweep(candidate, dt, pose, checker)
            if reason == 'clear':
                return candidate, dict(reason='adjusted_' + original_reason, sweep=sweep, adjusted=True)
    return (0., 0.), dict(reason='blocked_command_' + original_reason, sweep=[pose], adjusted=False)
