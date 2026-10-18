import numpy as np
import pytest
from local_navigation.models import NavigationConfig
from local_navigation.service import LocalNavigation
from test_local_depth import observation
from test_local_navigation import LARGE, SMALL


def blank_frame(x=0., timestamp=0.):
    frame = observation(np.zeros((64, 64), dtype=np.float32))
    frame['pose']['x'] = x
    frame['sim_time'] = timestamp
    return frame


@pytest.mark.parametrize('footprint', [SMALL, LARGE])
def test_initial_prior_remains_separate_in_planning_only_mode(footprint):
    nav = LocalNavigation(NavigationConfig(), footprint, .663, 'test')
    nav.observe(blank_frame())
    assert nav.mode == 'planning_only'
    assert nav.checker.reason((0, 0, 0)) == 'clear'
    assert nav.plan([(1, 0, 0)], (0, 0, 0))
    assert nav.guard((.3, 0), .33) == (.3, 0)
    assert not nav.grid.ground.any()  # Assumption is not measured floor.
    assert nav.snapshot()['initial_clearance']['remaining_cells'] > 0


def test_prior_stays_at_birth_and_never_replenishes_on_motion_or_return():
    nav = LocalNavigation(NavigationConfig(), SMALL, .663, 'half_xy')
    nav.observe(blank_frame())
    for i, x in enumerate(np.arange(.8, 8.1, .8), 1):
        nav.observe(blank_frame(float(x), i))
        ix, iy = nav.grid.cell(x + 2.1, 0)
        assert nav.grid.labels()[iy, ix] == 0
    assert not nav.grid.initial_free.any()
    for i, x in enumerate(np.arange(7.2, -.1, -.8), 20):
        nav.observe(blank_frame(float(x), i))
    assert nav.checker.reason((0, 0, 0)) == 'unknown'
    assert not nav.grid.initial_free.any()


@pytest.mark.parametrize('obstacle_on_first_frame', [True, False])
def test_real_obstacle_overrides_prior_and_invalid_depth_does_not_restore_it(obstacle_on_first_frame):
    nav = LocalNavigation(NavigationConfig(), SMALL, .663, 'half_xy')
    if not obstacle_on_first_frame:
        nav.observe(blank_frame())
    frame = observation(np.full((64, 64), .25, dtype=np.float32))
    frame['sim_time'] = 1.
    nav.observe(frame)
    assert nav.grid.hits.any()
    assert not nav.grid.initial_free[nav.grid.hits != 0].any()
    assert nav.checker.reason((0, 0, 0)) == 'occupied'
    assert nav.guard((.3, 0), .33) == (.3, 0.)  # Evidence retained, no execution veto.
    nav.observe(blank_frame(timestamp=2.))
    assert nav.checker.reason((0, 0, 0)) == 'occupied'


def test_real_floor_replaces_assumption_and_discontinuity_does_not_reseed():
    nav = LocalNavigation(NavigationConfig(), SMALL, .663, 'half_xy')
    nav.observe(blank_frame())
    count = int(nav.grid.initial_free.sum())
    frame = blank_frame(timestamp=1.)
    slope = (np.arange(64) - 31.5) / 32
    frame['depth'][slope > 0] = (.5 / slope[slope > 0])[:, None]
    nav.observe(frame)
    assert nav.grid.initial_free.sum() < count
    assert not (nav.grid.initial_free & nav.grid.ground).any()
    nav.observe(blank_frame(10., 2.))
    assert nav.checker.reason((10., 0, 0)) == 'unknown'
    assert not nav.grid.initial_free.any()
    nav.reset()
    nav.observe(blank_frame(10., 3.))
    assert nav.checker.reason((10., 0, 0)) == 'clear'


def test_prior_can_be_disabled_and_radius_is_validated():
    nav = LocalNavigation(NavigationConfig(initial_clear_radius_m=0), SMALL, .663, 'half_xy')
    nav.observe(blank_frame())
    assert nav.checker.reason((0, 0, 0)) == 'unknown'
    for radius in (-1, float('nan'), True, 20):
        with pytest.raises(ValueError):
            NavigationConfig(initial_clear_radius_m=radius)
