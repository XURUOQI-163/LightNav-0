"""Only unknown floor below the near front camera frustum receives a prior."""
import math

import numpy as np
import pytest
from local_navigation.models import NavigationConfig
from local_navigation.service import LocalNavigation
from test_initial_clearance import blank_frame
from test_local_navigation import SMALL


def make_nav(**options):
    return LocalNavigation(NavigationConfig(initial_clear_radius_m=0,
        near_blind_clear_distance_m=options.pop('distance', 1.5), **options), SMALL, .663, 'half_xy')


def label(nav, x, y):
    ix, iy = nav.grid.cell(x, y)
    return nav.grid.labels()[iy, ix]


def test_only_below_frustum_and_unknown_cells_are_initialized():
    nav = make_nav()
    frame = blank_frame()
    nav.observe(frame)
    assert label(nav, .3, 0) == 1  # Floor projects below image bottom.
    assert label(nav, 1., 0) == 0  # In view, but invalid depth: do not invent free.
    assert label(nav, .3, .8) == 0  # Outside horizontal field of view.
    assert label(nav, -.3, 0) == 0  # Behind robot.
    assert not nav.grid.ground.any() and not nav.grid.free.any()
    assert nav.snapshot()['near_blind_clearance']['remaining_cells'] > 0

    ix, iy = nav.grid.cell(.3, 0)
    nav.grid.hits[iy, ix] = 1
    frame['sim_time'] = 1.
    nav.observe(frame)
    assert label(nav, .3, 0) == 2
    assert not nav.grid.blind_free[iy, ix]
    assert nav.checker.reason((.3, 0, 0)) == 'occupied'


def test_real_depth_overrides_prior_and_it_is_not_reseeded():
    nav = make_nav()
    nav.observe(blank_frame())
    ix, iy = nav.grid.cell(.325, 0)
    assert nav.grid.blind_free[iy, ix]
    frame = blank_frame(timestamp=1.)
    frame['depth'][:] = .32  # A real vertical surface in that world cell.
    nav.observe(frame)
    assert label(nav, .325, 0) == 2 and not nav.grid.blind_free[iy, ix]
    nav.observe(blank_frame(timestamp=2.))
    assert label(nav, .325, 0) == 2
    # Even if later height evidence clears a hit without floor support, this
    # previously initialized cell must not acquire a second optimistic prior.
    nav.grid.hits[iy, ix] = 0
    nav.observe(blank_frame(timestamp=3.))
    assert label(nav, .325, 0) == 0


def test_motion_and_turn_use_capture_pose_and_leave_old_cells_world_fixed():
    nav = make_nav()
    nav.observe(blank_frame())
    assert label(nav, .3, 0) == 1
    frame = blank_frame(.8, 1.)
    frame['pose']['yaw'] = math.pi / 2
    nav.observe(frame)
    assert label(nav, .8, .3) == 1
    assert label(nav, .8, 1.) == 0
    assert label(nav, .8, -.3) == 0
    assert label(nav, .3, 0) == 1
    # Spatial eviction applies to priors and their initialization records.
    nav.grid.recenter((8., 0., 0.))
    assert not nav.grid.blind_free.any() and not nav.grid.blind_initialized.any()


def test_measured_floor_replaces_prior_when_camera_later_sees_it():
    nav = make_nav()
    nav.observe(blank_frame())
    frame = blank_frame(-.5, 1.)
    slope = (np.arange(64) - 31.5) / 32
    frame['depth'][slope > 0] = (.5 / slope[slope > 0])[:, None]
    nav.observe(frame)
    ix, iy = nav.grid.cell(.3, 0)
    assert nav.grid.ground[iy, ix]
    assert label(nav, .3, 0) == 1 and not nav.grid.blind_free[iy, ix]


def test_known_floor_is_not_reclassified_as_assumption():
    nav = make_nav()
    nav.observe(blank_frame())
    ix, iy = nav.grid.cell(.3, 0)
    nav.grid.ground[iy, ix] = nav.grid.free[iy, ix] = True
    nav.observe(blank_frame(timestamp=1.))
    assert label(nav, .3, 0) == 1 and not nav.grid.blind_free[iy, ix]


def test_bound_and_disable_do_not_fill_distant_lower_frustum():
    for distance in (0., .2):
        nav = make_nav(distance=distance)
        nav.observe(blank_frame())
        assert label(nav, .3, 0) == 0
    nav = make_nav()
    frame = blank_frame()
    frame['depth_camera']['camera_to_robot'][2][3] = 2.
    nav.observe(frame)
    assert label(nav, 1.2, 0) == 1
    assert label(nav, 1.8, 0) == 0  # Still below view, but outside near-distance cap.
    for distance in (-1, 4., float('nan'), True):
        with pytest.raises(ValueError):
            NavigationConfig(near_blind_clear_distance_m=distance)


def test_first_scan_obstacle_wins_and_duplicate_frames_do_not_reinitialize():
    nav = make_nav()
    frame = blank_frame()
    frame['depth'][:] = .32
    nav.observe(frame)
    ix, iy = nav.grid.cell(.325, 0)
    assert label(nav, .325, 0) == 2
    assert not nav.grid.blind_free[iy, ix] and not nav.grid.blind_initialized[iy, ix]
    revision = nav.grid.revision
    prior = nav.grid.blind_free.copy()
    nav.observe(frame)
    assert nav.grid.revision == revision
    np.testing.assert_array_equal(prior, nav.grid.blind_free)


def test_camera_offset_and_pitch_control_actual_blind_area():
    nav = make_nav()
    frame = blank_frame()
    transform = np.asarray(frame['depth_camera']['camera_to_robot'], dtype=float)
    transform[0, 3] = .2
    frame['depth_camera']['camera_to_robot'] = transform.tolist()
    nav.observe(frame)
    assert label(nav, .1, 0) == 0  # Behind camera plane, not front lower frustum.
    assert label(nav, .5, 0) == 1
    # Downward looking camera brings that floor into the actual image.
    nav.reset()
    angle = math.pi / 4
    rotation = np.array([[math.cos(angle), 0, math.sin(angle)], [0, 1, 0],
                         [-math.sin(angle), 0, math.cos(angle)]])
    transform[:3, :3] = rotation @ transform[:3, :3]
    frame['depth_camera']['camera_to_robot'] = transform.tolist()
    nav.observe(frame)
    assert label(nav, .5, 0) == 0
