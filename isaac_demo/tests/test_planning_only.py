"""Occupancy advises trajectories; it does not veto simulation commands."""
import asyncio

import numpy as np
import pytest
from isaac_web.runtime import IsaacRuntime
from local_navigation.collision import CollisionChecker
from local_navigation.models import NavigationConfig
from local_navigation.service import LocalNavigation
from test_local_navigation import SMALL, free_map, obstacle
from test_tracking_execution import Client, KinematicScene


def navigation(grid):
    nav = LocalNavigation(NavigationConfig(), SMALL, .663, 'half_xy')
    nav.grid = grid
    nav.checker = CollisionChecker(grid, SMALL)
    return nav


def test_occupied_start_continues_but_still_repairs_obstacle_ahead():
    grid = free_map()
    obstacle(grid, -.2, 0.)
    for y in np.arange(-.25, .26, .04):
        obstacle(grid, 1.3, y)
    nav = navigation(grid)
    before = grid.labels().copy()
    assert nav.checker.reason(grid.pose) == 'occupied'
    raw = [(i * .25, 0., 0.) for i in range(1, 11)]
    path = nav.plan(raw, (0., 0., 0.))
    assert path, nav.snapshot()['prediction']
    result = nav.snapshot()['prediction']
    assert result['start_overlap_ignored'] is True
    assert max(abs(p[1]) for p in path) > .3
    assert result['raw'] == raw
    assert result['goal_index'] == 9
    np.testing.assert_array_equal(grid.labels(), before)
    assert nav.guard((.2, .1), .1) == (.2, .1)
    assert nav.snapshot()['execution_collision_check'] is False


def test_no_clear_exit_preserves_prediction_and_records_uncorrected_fallback():
    grid = free_map()
    for x in np.arange(-.3, 2.5, .04):
        obstacle(grid, x, 0.)
    nav = navigation(grid)
    raw = [(i * .2, 0., 0.) for i in range(1, 11)]
    before = grid.labels().copy()
    np.testing.assert_allclose(nav.plan(raw, (0., 0., 0.)), raw)
    result = nav.snapshot()['prediction']
    assert result['status'] == 'fallback'
    assert result['reason'] == 'start_occupied_continue_raw'
    np.testing.assert_array_equal(grid.labels(), before)


@pytest.mark.parametrize('map_kind', ['missing', 'unknown', 'occupied'])
def test_execution_does_not_filter_command_using_map(map_kind):
    nav = LocalNavigation(NavigationConfig(), SMALL, .663, 'half_xy')
    if map_kind != 'missing':
        grid = free_map()
        grid.free[:] = False
        if map_kind == 'occupied':
            obstacle(grid, 0., 0.)
        nav = navigation(grid)
    assert nav.guard((.3, .4), .1) == (.3, .4)
    assert nav.snapshot()['mode'] == 'planning_only'
    assert nav.snapshot()['command']['sweep'] == []


@pytest.mark.parametrize('mpc', [False, True])
@pytest.mark.parametrize('realtime', [False, True])
def test_occupied_start_and_collision_flag_do_not_stop_execution(mpc, realtime):
    class FlaggedScene(KinematicScene):
        def observe(self):
            return dict(super().observe(), collided=True)

    class MapNavigation(LocalNavigation):
        def observe(self, frame):
            # Fixed obstacle behind initial robot, no fabricated depth/camera.
            self.grid.recenter(tuple(frame['pose'][k] for k in ('x', 'y', 'yaw')))
            self.grid.free[self.grid.valid] = True
            obstacle(self.grid, -.2, 0.)
            self.checker = CollisionChecker(self.grid, SMALL)
            self._snapshot = self._build_snapshot()

    async def scenario():
        scene = FlaggedScene()
        client = Client(scene, [(i / 10, 0., 0.) for i in range(1, 11)])
        nav = MapNavigation(NavigationConfig(), SMALL, .663, 'half_xy')
        r = IsaacRuntime(lambda: scene, {'lightnav': 'unused'}, local_navigation=nav,
            realtime_update=realtime, use_mpc=mpc, client_factory=lambda *args: client)
        await r.initialize()
        try:
            assert r.local_snapshot['start_check'] == 'occupied'
            await r.start('lightnav', 'go')
            await asyncio.wait_for(r.task, 30)
            assert r.state['stop_reason'] == 'model_stop', r.state
            assert scene.x > 0 and len(client.calls) == 2
            if not realtime:
                assert scene.x >= .9 - 1e-6
                assert r.state['tracking']['completed_paths'] == 1
        finally:
            await r.close()
    asyncio.run(scenario())
