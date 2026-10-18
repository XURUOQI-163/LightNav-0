import asyncio
import threading

import numpy as np
import pytest
from aiohttp.test_utils import TestClient, TestServer
from habitat_web.server import create_app
from isaac_web.runtime import IsaacRuntime
from local_navigation.models import NavigationConfig
from local_navigation.service import LocalNavigation
from test_local_depth import observation
from test_local_navigation import SMALL


def test_legacy_guarded_mode_still_requires_explicit_manual_collection():
    depth = np.zeros((64, 64), dtype=np.float32)
    slope = (np.arange(64) - 31.5) / 32
    depth[slope > 0] = (.5 / slope[slope > 0])[:, None]
    nav = LocalNavigation(NavigationConfig(initial_clear_radius_m=0), SMALL, .663, 'half_xy')
    nav.set_mode('guarded')  # Explicit legacy mode; default is planning_only.
    nav.observe(observation(depth))
    assert nav.checker.reason((0, 0, 0)) == 'unknown'
    assert nav.plan([(1, 0, 0)], (0, 0, 0)) == [[1., 0., 0.]]
    assert nav.snapshot()['prediction']['status'] == 'unchanged'
    assert nav.guard((.1, 0), .33, manual=True) == (0., 0.)
    nav.set_mode('observe_only')
    before = nav.grid.free.copy()
    assert nav.guard((.1, 0), .33, manual=True) == (.1, 0)
    assert nav.snapshot()['command']['reason'] == 'manual_unprotected'
    assert nav.plan([(1, 0, 0)], (0, 0, 0)) == []
    assert nav.guard((.1, 0), .33) == (0., 0.)
    np.testing.assert_array_equal(nav.grid.free, before)  # No invented free disk.
    nav.set_mode('guarded')
    assert nav.guard((.1, 0), .33, manual=True) == (0., 0.)
    with pytest.raises(ValueError):
        nav.set_mode('arbitrary')
    nav.reset()
    assert nav.snapshot()['mode'] == 'planning_only'


def test_service_no_time_expiry_reset_and_raw_path_snapshot():
    nav = LocalNavigation(NavigationConfig(initial_clear_radius_m=0), SMALL, .663, 'half_xy')
    nav.set_mode('guarded')  # Check strict corrector invariants independently.
    frame = observation(np.ones((32, 32), dtype=np.float32))
    nav.observe(frame)
    snapshot = nav.snapshot()
    assert snapshot['radius'] == 5
    assert snapshot['enabled']
    assert snapshot['occupied_count'] > 0
    raw = [(i / 5, 0, 0) for i in range(1, 11)]
    nav.plan(raw, (0, 0, 0))
    snapshot = nav.snapshot()
    assert len(snapshot['prediction']['raw']) == 10
    assert snapshot['prediction']['status'] != 'unchanged'
    path = snapshot['prediction']['corrected']
    assert all(nav.checker.for_planning().segment(a, b) == 'clear' for a, b in zip(path, path[1:]))
    frame['depth'][:] = 0
    frame['sim_time'] = 10000.
    nav.observe(frame)
    assert nav.snapshot()['occupied_count'] == snapshot['occupied_count']
    nav.reset()
    assert nav.snapshot()['prediction'] is None
    assert nav.snapshot()['occupied_count'] == 0


def test_replanning_suffix_retains_full_model_points_and_global_goal_index(monkeypatch):
    from local_navigation.models import transform_path

    nav = LocalNavigation(NavigationConfig(), SMALL, .663, 'half_xy')
    nav.observe(observation(np.zeros((32, 32), dtype=np.float32)))
    points = [(i / 10, 0., 0.) for i in range(1, 11)]
    capture = (1., 2., .4)
    expected = transform_path(points, capture)

    def correct(raw, pose, checker):
        assert raw == expected[4:]
        return dict(raw=raw, corrected=[pose, raw[-1]], status='corrected', reason='local_detour',
                    point_status=['clear'] * 6, segment_status=['clear'] * 6,
                    goal_index=5, planning_ms=0.)

    monkeypatch.setattr('local_navigation.service.correct_path', correct)
    nav.plan_remaining(points, capture, 4)
    result = nav.snapshot()['prediction']
    np.testing.assert_allclose(result['raw'], expected)
    assert result['goal_index'] == 9
    assert result['remaining_start_index'] == 4
    assert result['capture_pose'] == list(capture)
    assert result['point_status'] == ['passed'] * 4 + ['clear'] * 6


def test_runtime_guarded_http_is_optional_and_reads_do_not_step():
    class Scene:
        def reset(self):
            self.moves = []
            return self.observe()

        def observe(self):
            return dict(observation(np.ones((32, 32), dtype=np.float32)), jpeg=b'jpeg', collided=False)

        def step(self, *args):
            self.moves.append(args)
            return self.observe()

        def close(self):
            pass

    async def scenario():
        nav = LocalNavigation(NavigationConfig(initial_clear_radius_m=0), SMALL, .663, 'half_xy')
        runtime = IsaacRuntime(Scene, {'lightnav': 'unused'}, local_navigation=nav)
        async with TestClient(TestServer(create_app(runtime))) as client:
            response = await client.get('/api/local-navigation')
            assert response.status == 200
            assert (await response.json())['radius'] == 5
            response = await client.post('/api/manual', json=dict(linear=.1, angular=0, duration=.33))
            assert response.status == 200
            assert len(runtime.scene.moves) == 1
            assert runtime.state['stop_reason'] == 'manual_step'
            for _ in range(3):
                await client.get('/api/local-navigation')
            assert len(runtime.scene.moves) == 1  # GETs did not move the robot.
            response = await client.post('/api/local-navigation-mode', json={'mode':'observe_only'})
            assert response.status == 200
            response = await client.post('/api/start', json={'backend':'lightnav', 'instruction':'forward'})
            assert response.status == 400  # Manual collection must not bypass auto safety.
            response = await client.post('/api/manual', json=dict(linear=.1, angular=0, duration=.33))
            assert response.status == 200 and len(runtime.scene.moves) == 2
            response = await client.post('/api/local-navigation-mode', json={'mode':'guarded'})
            assert response.status == 200
            await client.post('/api/manual', json=dict(linear=.1, angular=0, duration=.33))
            assert len(runtime.scene.moves) == 2  # Explicit legacy guard still blocks.
        runtime = IsaacRuntime(Scene, {'lightnav': 'unused'})
        async with TestClient(TestServer(create_app(runtime))) as client:
            data = await (await client.get('/api/local-navigation')).json()
            assert data == {'enabled': False}
    asyncio.run(scenario())


def test_runtime_preserves_model_points_and_executes_only_guarded_result():
    from habitat_web.clients import Prediction
    from local_navigation.collision import CollisionChecker
    from test_local_navigation import obstacle

    raw = [(i / 5, 0, 0) for i in range(1, 11)]

    class Scene:
        def reset(self):
            self.moves, self.time = [], 0.
            return self.observe()

        def observe(self):
            return dict(observation(np.zeros((32, 32), dtype=np.float32)), jpeg=b'original_rgb',
                        sim_time=self.time, collided=False)

        def step(self, *args):
            self.moves.append(args)
            self.time += args[2]
            return self.observe()

        def close(self):
            pass

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def predict(self, jpeg, instruction):
            assert jpeg == b'original_rgb'
            return Prediction(raw, False, 1)

    async def scenario():
        nav = LocalNavigation(NavigationConfig(), SMALL, .663, 'half_xy')
        runtime = IsaacRuntime(Scene, {'lightnav': 'unused'}, local_navigation=nav,
                               max_steps=1, client_factory=lambda *args: Client())
        await runtime.initialize()
        try:
            # Synthetic known-free map; no GPU/simulator truth is used in production.
            nav.grid.free[nav.grid.valid] = True
            nav.grid.ground[nav.grid.valid] = True
            for y in np.arange(-.25, .26, .04):
                obstacle(nav.grid, 1, y)
            nav.checker = CollisionChecker(nav.grid, SMALL)
            await runtime.start('lightnav', 'forward')
            await runtime.task
            assert runtime.state['status'] == 'stopped', runtime.state
            assert runtime.state['waypoints'] == raw
            assert runtime.scene.moves
            assert nav.prediction['status'] == 'corrected'
            assert nav.prediction['raw'] == raw
            assert runtime.view_jpeg == b'original_rgb'
            assert abs(runtime.scene.moves[0][0]) <= runtime.max_linear
            assert abs(runtime.scene.moves[0][1]) <= runtime.max_angular
        finally:
            await runtime.close()
    asyncio.run(scenario())


def test_cancelled_plan_cannot_execute_after_stop():
    from habitat_web.clients import Prediction
    from test_isaac_runtime import Scene
    entered, release = threading.Event(), threading.Event()

    class SlowNavigation(LocalNavigation):
        def plan(self, *args):
            entered.set()
            release.wait(2)
            return [(1, 0, 0)]

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def predict(self, *args):
            return Prediction([(1, 0, 0)], False, 1)

    async def scenario():
        runtime = IsaacRuntime(Scene, {'lightnav':'unused'}, local_navigation=SlowNavigation(
            NavigationConfig(), SMALL, .663, 'half_xy'), client_factory=lambda *args: Client())
        await runtime.initialize()
        try:
            await runtime.start('lightnav','go')
            while not entered.is_set():
                await asyncio.sleep(.001)
            stopped = asyncio.create_task(runtime.stop())
            await asyncio.sleep(.01)
            release.set()
            await stopped
            assert runtime.scene.time == 0
            assert runtime.local_snapshot['prediction'] is None
        finally:
            release.set()
            await runtime.close()
    asyncio.run(scenario())


def test_cancelled_mode_change_still_publishes_actual_protection_state():
    from test_isaac_runtime import Scene
    entered, release = threading.Event(), threading.Event()

    class SlowNavigation(LocalNavigation):
        def set_mode(self, mode):
            entered.set()
            release.wait(2)
            super().set_mode(mode)

    async def scenario():
        nav = SlowNavigation(NavigationConfig(), SMALL, .663, 'half_xy')
        runtime = IsaacRuntime(Scene, {'lightnav':'unused'}, local_navigation=nav)
        await runtime.initialize()
        try:
            change = asyncio.create_task(runtime.navigation_mode('observe_only'))
            while not entered.is_set():
                await asyncio.sleep(.001)
            change.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await change
            assert nav.mode == runtime.local_snapshot['mode'] == 'observe_only'
            with pytest.raises(ValueError, match='手动采图'):
                await runtime.start('lightnav', 'forward')
            await runtime.reset()
            assert nav.mode == runtime.local_snapshot['mode'] == 'planning_only'
        finally:
            release.set()
            await runtime.close()
    asyncio.run(scenario())
