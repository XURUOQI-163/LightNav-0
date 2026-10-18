import math

import numpy as np
import pytest
from local_navigation.collision import CollisionChecker
from local_navigation.command_guard import command_sweep, guard_command
from local_navigation.models import NavigationConfig
from local_navigation.path_corrector import correct_path
from local_navigation.rolling_grid import RollingGrid

SMALL = [[-.2155, -.1575], [.1395, -.1575], [.1395, .1575], [-.2155, .1575]]
LARGE = [[x * 2, y * 2] for x, y in SMALL]


def free_map(radius=5):
    g = RollingGrid(NavigationConfig(window_radius_m=radius))
    g.recenter((0, 0, 0))
    g.free[g.valid] = True
    return g


def obstacle(g, x, y):
    ix, iy = g.cell(x, y)
    g.hits[iy, ix] = 1


def test_spatial_window_drops_outside_without_age_or_wrap():
    g = free_map()
    obstacle(g, -4.8, 0)
    obstacle(g, 1, 0)
    for _ in range(5):
        g.recenter((0, 0, math.pi))
    assert g.hits.sum() == 2
    g.recenter((1, 0, 0))
    assert g.hits.sum() == 1
    g.recenter((0, 0, 0))
    ix, iy = g.cell(-4.8, 0)
    assert not g.free[iy, ix] and not g.hits[iy, ix]
    assert g.hits.sum() == 1
    assert g.n <= 203


def test_unknown_and_boundary_are_not_drivable():
    g = free_map()
    c = CollisionChecker(g, SMALL)
    assert c.reason((0, 0, 0)) == 'clear'
    assert c.reason((4.95, 0, 0)) != 'clear'
    g.free[:] = False
    assert CollisionChecker(g, SMALL).reason((0, 0, 0)) == 'unknown'


def test_far_segment_is_rejected_before_allocating_samples(monkeypatch):
    c = CollisionChecker(free_map(), SMALL)
    def no_samples(*args, **kwargs):
        raise AssertionError('outside-window segment must not allocate samples')
    monkeypatch.setattr(np, 'linspace', no_samples)
    assert c.segment((0, 0, 0), (10000000, 0, 0)) == 'outside_window'


def test_footprint_changes_narrow_door_and_segment_checks_wall():
    g = free_map()
    for y in np.arange(-4, 4, .04):
        if abs(y) > .36:
            obstacle(g, 1, y)
    assert CollisionChecker(g, SMALL).segment((.5, 0, 0), (1.5, 0, 0)) == 'clear'
    assert CollisionChecker(g, LARGE).segment((.5, 0, 0), (1.5, 0, 0)) == 'occupied'
    obstacle(g, 1, 0)
    c = CollisionChecker(g, SMALL)
    assert c.reason((.5, 0, 0)) == c.reason((1.5, 0, 0)) == 'clear'
    assert c.segment((.5, 0, 0), (1.5, 0, 0)) == 'occupied'


def test_corrector_preserves_raw_and_finds_detour():
    g = free_map()
    for y in np.arange(-.25, .26, .04):
        obstacle(g, 1, y)
    raw = [(i * .2, 0., 0.) for i in range(1, 11)]
    before = list(raw)
    c = CollisionChecker(g, SMALL)
    result = correct_path(raw, (0, 0, 0), c)
    assert raw == before
    assert result['status'] == 'corrected', result
    assert max(abs(p[1]) for p in result['corrected']) > .3
    assert all(c.segment(a, b) == 'clear' for a, b in zip(result['corrected'], result['corrected'][1:]))


def test_unreachable_late_goal_retains_search_discovered_safe_progress():
    g = free_map()
    for y in np.arange(-5, 5.01, .04):
        obstacle(g, 1., y)
    checker = CollisionChecker(g, SMALL)
    # The first raw point already collides, so there is no raw safe prefix.
    # A corrected approach to x~0.5 remains possible, though the wall blocks x=2.
    raw = [(.8, 0, 0), (1.2, 0, 0), (2., 0, 0)]
    result = correct_path(raw, (0, 0, 0), checker)
    assert result['status'] == 'safe_prefix', result
    assert result['corrected'][-1][0] >= .3
    assert all(checker.segment(a, b) == 'clear' for a, b in
               zip(result['corrected'], result['corrected'][1:]))


def test_unknown_goals_are_preserved_but_execution_remains_guarded():
    g = free_map()
    yy, xx = np.indices(g.free.shape)
    wx, wy = g.world(xx, yy)
    g.free &= (wx < .8) & (wx > -.8) & (abs(wy) < .8)
    checker = CollisionChecker(g, SMALL)
    # Even the 0.75 m goal-adjustment neighbourhood is unobserved.
    result = correct_path([(2., 0, 0), (3., 0, 0)], (0, 0, 0), checker)
    assert result['status'] == 'unchanged', result
    assert result['reason'] == 'no_known_obstacle'
    assert result['corrected'][-1] == [3., 0., 0.]
    assert checker.segment((0, 0, 0), (2., 0, 0)) == 'unknown'


def test_small_known_patch_does_not_truncate_prediction_or_expand_free_map():
    g = free_map()
    yy, xx = np.indices(g.free.shape)
    wx, wy = g.world(xx, yy)
    g.free &= (wx < .27) & (wx > -.34) & (abs(wy) < .29)
    checker = CollisionChecker(g, SMALL)
    before = g.free.copy()
    result = correct_path([(2., 0, 0)], (0, 0, 0), checker)
    assert result['status'] == 'unchanged', result
    np.testing.assert_array_equal(before, g.free)
    command, _ = guard_command((.3, 0), .33, (0, 0, 0), checker, [], adjust=False)
    assert command == (0., 0.)


def test_no_escape_from_occupied_start_and_guard_stops_unsafe_command():
    g = free_map()
    obstacle(g, 0, 0)
    c = CollisionChecker(g, SMALL)
    result = correct_path([(1, 0, 0)], (0, 0, 0), c)
    assert result['status'] == 'blocked' and result['corrected'] == []
    command, detail = guard_command((.3, 0), .33, (0, 0, 0), c, [])
    assert command == (0., 0.)
    assert detail['reason'] != 'clear'


def test_command_braking_extension_rejects_future_collision_from_clear_start():
    g = free_map()
    obstacle(g, .45, 0)
    checker = CollisionChecker(g, SMALL)
    assert checker.reason((0, 0, 0)) == 'clear'
    accepted, detail = guard_command((.3, 0), .33, (0, 0, 0), checker, [], adjust=False)
    assert accepted == (0., 0.) and detail['reason'] == 'blocked_command_occupied'


def test_unbounded_velocity_guard_fails_closed():
    checker = CollisionChecker(free_map(), SMALL)
    assert command_sweep((0, 100000), 1, (0, 0, 0), checker)[0] == 'sweep_budget'
    assert command_sweep((float('nan'), 0), 1, (0, 0, 0), checker)[0] == 'invalid_command'


@pytest.mark.parametrize('options', [dict(window_radius_m=float('nan')),
                                    dict(window_radius_m=0), dict(resolution_m=.001),
                                    dict(max_search_expansions=True)])
def test_config_rejects_unbounded_or_invalid_work(options):
    with pytest.raises(ValueError):
        NavigationConfig(**options)
