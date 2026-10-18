import asyncio
import threading

import pytest
from habitat_web.clients import Prediction
from isaac_web.runtime import IsaacRuntime
from local_navigation.models import NavigationConfig, transform_path
from test_tracking_execution import Client, KinematicScene


class Navigation:
    def __init__(self, **options):
        self.config = NavigationConfig(**options)
        self.reset()

    def reset(self):
        self.prediction = None
        self.revision = 0
        self.plans = []
        self.commands = []
        self.session = 'one'

    def observe(self, frame):
        self.pose = tuple(frame['pose'][k] for k in ('x', 'y', 'yaw'))
        self.time = frame['sim_time']
        self.revision += 1

    def snapshot(self):
        return dict(mode='guarded', status='ready', reason='', prediction=self.prediction,
                    map_revision=self.revision, session_id=self.session, start_check='clear')

    def plan_remaining(self, points, capture_pose, start_index):
        raw = transform_path(points, capture_pose)
        self.plans.append((raw, start_index, self.revision))
        self.prediction = dict(status='unchanged', raw=raw)
        return transform_path(raw[start_index:], self.pose, inverse=True)

    def guard(self, command, *args, **kwargs):
        self.commands.append(command)
        return command

    def clear_prediction(self):
        self.prediction = None


class Runtime(IsaacRuntime):
    recoveries = 0

    async def recover_blocked(self):
        self.recoveries += 1
        return False


def make_runtime(nav, scene=None, client_type=Client, **options):
    scene = scene or KinematicScene()
    client = client_type(scene, [(i / 10, 0., 0.) for i in range(1, 11)])
    runtime = Runtime(lambda: scene, {'lightnav': 'unused'}, local_navigation=nav,
        realtime_update=False, client_factory=lambda *args: client, **options)
    return runtime, scene, client


@pytest.mark.parametrize('mpc', [False, True])
def test_new_maps_replan_original_world_points_periodically(mpc):
    async def scenario():
        nav = Navigation()
        r, scene, client = make_runtime(nav, use_mpc=mpc)
        await r.initialize()
        try:
            await r.start('lightnav', 'go')
            await asyncio.wait_for(r.task, 10)
            assert r.state['stop_reason'] == 'model_stop', r.state
            assert scene.x >= .9 - 1e-6 and len(client.calls) == 2
            assert len(nav.plans) > 2
            assert all(raw == nav.plans[0][0] for raw, _, _ in nav.plans)
            assert len({revision for _, _, revision in nav.plans}) == len(nav.plans)
            assert nav.plans[-1][1] > nav.plans[0][1]
            assert r.state['tracking']['model_replans'] == 0
        finally:
            await r.close()
    asyncio.run(scenario())


def test_transient_search_failures_wait_for_new_depth_without_model_request():
    class Transient(Navigation):
        def plan_remaining(self, *args):
            path = super().plan_remaining(*args)
            if len(self.plans) < 3:
                self.prediction['status'] = 'blocked'
                return []
            return path

    async def scenario():
        nav = Transient()
        r, scene, client = make_runtime(nav)
        await r.initialize()
        try:
            await r.start('lightnav', 'go')
            await r.task
            assert r.state['stop_reason'] == 'model_stop', r.state
            assert len(client.calls) == 2 and client.calls[-1]['pose']['x'] >= .9 - 1e-6
            assert all(v == w == 0 for v, w, _ in scene.moves[:2])
            assert r.recoveries == 0
        finally:
            await r.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('blocked_guard', [False, True])
def test_repeated_failure_then_recovery_requests_fresh_prediction(blocked_guard):
    class Blocked(Navigation):
        def plan_remaining(self, *args):
            path = super().plan_remaining(*args)
            if not blocked_guard:
                self.prediction['status'] = 'blocked'
                return []
            return path

        def guard(self, command, *args, **kwargs):
            return (0., 0.) if blocked_guard else command

    async def scenario():
        nav = Blocked()
        r, scene, client = make_runtime(nav)
        await r.initialize()
        try:
            await r.start('lightnav', 'go')
            await r.task
            assert r.state['stop_reason'] == 'model_stop', r.state
            assert r.recoveries == 1 and len(client.calls) == 2
            assert len(nav.plans) == nav.config.path_failure_confirmations
            assert client.calls[1]['jpeg'] != client.calls[0]['jpeg']
            assert all(v == w == 0 for v, w, _ in scene.moves)
            assert r.state['tracking']['completed_paths'] == 0
            assert r.state['tracking']['model_replans'] == 1
            assert nav.prediction is None  # Old route retired before next inference.
        finally:
            await r.close()
    asyncio.run(scenario())


def test_identical_unexecutable_predictions_are_bounded():
    class Blocked(Navigation):
        def plan_remaining(self, *args):
            super().plan_remaining(*args)
            self.prediction['status'] = 'blocked'
            return []

    class Repeating(Client):
        async def predict(self, *args):
            self.calls.append(self.scene.observe())
            return Prediction(self.path, False, 1.)

    async def scenario():
        nav = Blocked(model_replan_limit=2)
        r, scene, client = make_runtime(nav, client_type=Repeating)
        await r.initialize()
        try:
            await r.start('lightnav', 'go')
            await r.task
            assert r.state['stop_reason'] == 'model_replan_limit', r.state
            assert len(client.calls) == 3  # Initial + two permitted replacements.
            assert not any(v or w for v, w, _ in scene.moves)
        finally:
            await r.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('timeout', [.5, 60.])
def test_no_motion_or_expired_prediction_triggers_fallback(timeout):
    class Stalled(KinematicScene):
        def step(self, v, w, dt):
            return super().step(0., 0., dt)

    async def scenario():
        r, _, client = make_runtime(Navigation(path_no_progress_s=1.), Stalled(), path_timeout=timeout)
        await r.initialize()
        try:
            await r.start('lightnav', 'go')
            await r.task
            assert r.state['stop_reason'] == 'model_stop', r.state
            assert len(client.calls) == 2
            assert r.state['tracking']['completed_paths'] == 0
            assert r.state['tracking']['model_replans'] == 1
        finally:
            await r.close()
    asyncio.run(scenario())


def test_stop_waits_for_local_replan_and_never_sends_new_request():
    entered, release = threading.Event(), threading.Event()

    class Slow(Navigation):
        def plan_remaining(self, *args):
            entered.set()
            release.wait(3.)
            return super().plan_remaining(*args)

    async def scenario():
        r, scene, client = make_runtime(Slow())
        await r.initialize()
        try:
            await r.start('lightnav', 'go')
            for _ in range(200):
                if entered.is_set():
                    break
                await asyncio.sleep(.005)
            assert entered.is_set()
            stop = asyncio.create_task(r.stop())
            await asyncio.sleep(.01)
            assert not stop.done()
            release.set()
            await stop
            assert r.state['stop_reason'] == 'user_stop'
            assert len(client.calls) == 1 and not scene.moves
        finally:
            release.set()
            await r.close()
    asyncio.run(scenario())


def test_step_limit_prevents_fallback_request():
    async def scenario():
        r, _, client = make_runtime(Navigation(), max_steps=1)
        await r.initialize()
        try:
            await r.start('lightnav', 'go')
            await r.task
            assert r.state['stop_reason'] == 'step_limit' and len(client.calls) == 1
        finally:
            await r.close()
    asyncio.run(scenario())


def test_successful_recovery_keeps_world_goal_without_model_request():
    class Recoverable(Navigation):
        recovered = False

        def plan_remaining(self, *args):
            path = super().plan_remaining(*args)
            if not self.recovered:
                self.prediction['status'] = 'blocked'
                return []
            return path

    class Recovering(Runtime):
        async def recover_blocked(self):
            self.recoveries += 1
            self.scene.yaw += .3
            self.scene.time += 1.
            self._publish(await self._sim_call(self.scene.observe))
            self.local_navigation.recovered = True
            return True

    async def scenario():
        nav, scene = Recoverable(), KinematicScene()
        client = Client(scene, [(i / 10, 0., 0.) for i in range(1, 11)])
        r = Recovering(lambda: scene, {'lightnav': 'unused'}, local_navigation=nav,
            use_mpc=True, realtime_update=False, client_factory=lambda *args: client)
        await r.initialize()
        try:
            await r.start('lightnav', 'go')
            await asyncio.wait_for(r.task, 10)
            assert r.state['stop_reason'] == 'model_stop', r.state
            assert r.recoveries == 1 and len(client.calls) == 2
            assert r.state['tracking']['model_replans'] == 0
            assert scene.x >= .9 - 1e-6 and abs(scene.y) < .1
            assert all(raw == nav.plans[0][0] for raw, _, _ in nav.plans)
        finally:
            await r.close()
    asyncio.run(scenario())


def test_pose_discontinuity_stops_instead_of_reusing_stale_world_coordinates():
    class Resetting(Navigation):
        def observe(self, frame):
            super().observe(frame)
            if self.revision > 2:
                self.session = 'two'

    async def scenario():
        r, _, client = make_runtime(Resetting())
        await r.initialize()
        try:
            await r.start('lightnav', 'go')
            await r.task
            assert r.state['stop_reason'] == 'navigation_session_changed'
            assert len(client.calls) == 1
        finally:
            await r.close()
    asyncio.run(scenario())


def test_new_obstacle_during_motion_stops_commands_before_reprediction():
    class NewObstacle(Navigation):
        blocked_commands = 0

        def guard(self, command, *args, **kwargs):
            if self.pose[0] >= .2:
                self.blocked_commands += 1
                return 0., 0.
            return super().guard(command, *args, **kwargs)

    async def scenario():
        nav = NewObstacle()
        r, scene, client = make_runtime(nav)
        await r.initialize()
        try:
            await r.start('lightnav', 'go')
            await r.task
            assert r.state['stop_reason'] == 'model_stop', r.state
            assert .2 <= scene.x < .3
            assert nav.blocked_commands >= nav.config.path_failure_confirmations
            assert r.recoveries == 1 and len(client.calls) == 2
            assert r.state['tracking']['model_replans'] == 1
            assert r.state['tracking']['completed_paths'] == 0
            assert all(v == w == 0 for v, w, _ in scene.moves[-2:])
        finally:
            await r.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('options', [dict(path_failure_confirmations=1),
    dict(path_failure_confirmations=2.5), dict(path_replan_interval_s=0),
    dict(path_no_progress_s=.1), dict(model_replan_limit=0)])
def test_incremental_config_rejects_invalid_limits(options):
    with pytest.raises(ValueError):
        NavigationConfig(**options)
