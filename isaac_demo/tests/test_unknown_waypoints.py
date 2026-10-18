"""Unknown is allowed for predictions, but does not authorize a command sweep."""
import numpy as np
from local_navigation.collision import CollisionChecker
from local_navigation.command_guard import guard_command
from local_navigation.path_corrector import correct_path
from test_local_navigation import SMALL, free_map, obstacle


def test_unknown_waypoints_remain_unchanged_without_relabeling_map():
    grid = free_map()
    grid.free[:] = False
    checker = CollisionChecker(grid, SMALL)
    before = grid.labels().copy()
    raw = [(i * .3, 0., 0.) for i in range(1, 11)]
    result = correct_path(raw, (0., 0., 0.), checker)
    assert result['status'] == 'unchanged' and result['expanded'] == 0
    np.testing.assert_allclose(result['corrected'][1:], raw)
    assert result['point_status'] == ['unknown'] * 10
    np.testing.assert_array_equal(grid.labels(), before)
    assert checker.reason((0., 0., 0.)) == 'unknown'
    command, detail = guard_command((.2, 0.), .1, (0., 0., 0.), checker, result['corrected'])
    assert command == (0., 0.) and detail['reason'] == 'blocked_command_unknown'


def test_unknown_endpoints_do_not_hide_wall_between_them():
    grid = free_map()
    grid.free[:] = False
    for y in np.arange(-.25, .26, .04):
        obstacle(grid, 1., y)
    checker = CollisionChecker(grid, SMALL)
    result = correct_path([(2., 0., 0.)], (0., 0., 0.), checker)
    assert result['point_status'] == ['unknown']
    assert result['segment_status'] == ['occupied']
    assert result['status'] == 'corrected', result
    assert max(abs(p[1]) for p in result['corrected']) > .3
    planner = checker.for_planning()
    assert all(planner.segment(a, b) == 'clear' for a, b in
               zip(result['corrected'], result['corrected'][1:]))
    assert not grid.free.any()


def test_waypoint_in_inflated_obstacle_is_corrected_even_without_raw_hit():
    grid = free_map()
    grid.free[:] = False
    obstacle(grid, 1., 0.)
    checker = CollisionChecker(grid, SMALL)
    point = (.8, 0., 0.)
    ix, iy = grid.cell(*point[:2])
    assert grid.labels()[iy, ix] == 0 and checker.reason(point) == 'occupied'
    result = correct_path([point], (0., 0., 0.), checker)
    assert result['status'] == 'corrected', result
    assert result['point_status'] == ['occupied']
    assert checker.for_planning().reason(result['corrected'][-1]) == 'clear'


def test_outside_prediction_is_preserved_but_checks_observed_segment():
    grid = free_map()
    checker = CollisionChecker(grid, SMALL)
    far = (10000000., 0., 0.)
    result = correct_path([far], (0., 0., 0.), checker)
    assert result['status'] == 'unchanged' and result['expanded'] == 0
    assert result['corrected'][-1] == list(far)
    assert checker.segment((0., 0., 0.), far) == 'outside_window'
    obstacle(grid, 1., 0.)
    result = correct_path([far], (0., 0., 0.), CollisionChecker(grid, SMALL))
    assert result['segment_status'] == ['occupied']
    assert result['status'] != 'unchanged'


def test_new_obstacle_triggers_correction_of_previously_unknown_prediction():
    grid = free_map()
    grid.free[:] = False
    raw = [(1., 0., 0.), (2., 0., 0.)]
    initial = correct_path(raw, (0., 0., 0.), CollisionChecker(grid, SMALL))
    assert initial['status'] == 'unchanged'
    obstacle(grid, 1., 0.)
    updated = correct_path(raw, (0., 0., 0.), CollisionChecker(grid, SMALL))
    assert updated['status'] == 'corrected'
    assert updated['raw'] == raw and updated['point_status'][0] == 'occupied'
