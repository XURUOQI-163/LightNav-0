"""CPU checks for bounded wheel actuation and real-asset runtime overrides."""
import json
from pathlib import Path

import pytest
from isaac_web.robot_profiles import apply_robot_profile
from isaac_web.stability import DriveLimiter, StabilityConfig

BASE = Path(__file__).parents[1] / 'config/grscenes_carter.json'


def small():
    return apply_robot_profile(json.loads(BASE.read_text()), 'half_xy')['stability']


def test_small_profile_has_lower_limits_and_cli_caps_old_fast_command(tmp_path):
    from isaac_web.__main__ import parse_args
    cfg = json.loads(BASE.read_text())
    asset = tmp_path/'asset.usd'
    asset.touch()
    cfg.update(scene=str(asset), robot=str(asset), materials=str(tmp_path))
    config = tmp_path/'config.json'
    config.write_text(json.dumps(cfg))
    a, resolved = parse_args(['--config', str(config), '--robot-profile', 'half_xy',
                              '--max-linear', '1', '--max-angular', '2'])
    assert (a.max_linear, a.max_angular) == (.35, .8)
    assert resolved['physics_dt'] == pytest.approx(1/120)
    assert resolved['robot_height'] == cfg.get('robot_height', .663)


def test_start_brake_reverse_and_turn_have_bounded_accelerations():
    c = StabilityConfig(**small())
    drive = DriveLimiter(c)
    dt = 1/120
    for target in [(1., 2.)]*240 + [(0., -2.)]*240 + [(-1., 2.)]*240:
        old = drive.command
        v, w = drive.advance(*target, dt)
        assert abs(v) <= c.max_linear + 1e-9
        assert abs(w) <= c.max_angular + 1e-9
        assert abs(v*w) <= c.max_lateral_acceleration + 1e-9
        assert abs(v-old[0]) <= max(c.linear_acceleration, c.linear_deceleration)*dt + 1e-9
        assert abs(w-old[1]) <= c.angular_acceleration*dt + 1e-9
    assert drive.command[0] < 0
    drive.reset()
    assert drive.command == (0., 0.)


def test_command_boundaries_do_not_reset_ramp():
    a, b = DriveLimiter(small()), DriveLimiter(small())
    for _ in range(12):
        a.advance(.35, .8, 1/120)
    previous = a.command
    for _ in range(12):
        a.advance(.35, .8, 1/120)
    for _ in range(24):
        b.advance(.35, .8, 1/120)
    assert a.command == pytest.approx(b.command)
    assert a.command[0] > previous[0]


@pytest.mark.parametrize('options', [dict(wheel_max_torque=float('inf')),
    dict(com_lowering=.3), dict(linear_acceleration=0), dict(solver_velocity_iterations=2.5)])
def test_invalid_dynamics_fail_before_loading_isaac(options):
    with pytest.raises(ValueError):
        StabilityConfig(**options)
