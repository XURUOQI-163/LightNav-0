"""A delayed model must not stall valid control or escape STOP/RESET."""
import asyncio
from copy import deepcopy

import pytest
from habitat_web.clients import Prediction
from isaac_web.runtime import IsaacRuntime
from isaac_web.tracking import PathTracker
from test_tracking_execution import KinematicScene


class DelayedClient:
    def __init__(self, scene, path=None):
        self.scene = scene
        self.path = path or [(i * .5, 0., 0.) for i in range(1, 11)]
        self.started, self.release = asyncio.Event(), asyncio.Event()
        self.calls, self.active, self.peak = [], 0, 0
        self.closed = self.cancelled = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        self.closed = True

    async def predict(self, jpeg, instruction):
        self.calls.append((jpeg, deepcopy(self.scene.observe()['pose'])))
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            if len(self.calls) > 1:
                self.started.set()
                await self.release.wait()
            return Prediction(self.path, len(self.calls) > 1, 1.)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        finally:
            self.active -= 1


async def until(predicate):
    async def wait():
        while not predicate():
            await asyncio.sleep(.001)
    await asyncio.wait_for(wait(), 3)


@pytest.mark.parametrize('mpc', [False, True])
def test_control_continues_while_single_prediction_is_pending(mpc):
    async def scenario():
        scene = KinematicScene()
        client = DelayedClient(scene)
        r = IsaacRuntime(lambda: scene, {'lightnav': 'unused'}, use_mpc=mpc,
                         realtime_update=True, client_factory=lambda *args: client)
        await r.initialize()
        try:
            await r.start('lightnav', 'forward')
            # The first real CasADi/IPOPT load can exceed three seconds on the
            # shared data disk. Test concurrent motion after it has initialized,
            # without imposing a cold-start performance limit on this test.
            await asyncio.wait_for(client.started.wait(), 30)
            await until(lambda: len(scene.moves) >= 5)
            assert len(client.calls) == 2 and client.peak == 1
            assert scene.x > client.calls[1][1]['x']
            assert r.view_jpeg == client.calls[0][0]  # predictions keep their input view
            client.release.set()
            await asyncio.wait_for(r.task, 3)
            assert r.state['stop_reason'] == 'model_stop'
            assert client.active == 0 and client.closed
        finally:
            await r.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('operation', ['stop', 'reset'])
def test_user_stop_or_reset_drains_pending_prediction(operation):
    async def scenario():
        scene = KinematicScene()
        client = DelayedClient(scene)
        r = IsaacRuntime(lambda: scene, {'lightnav': 'unused'}, use_mpc=True,
                         client_factory=lambda *args: client)
        await r.initialize()
        try:
            await r.start('lightnav', 'forward')
            await asyncio.wait_for(client.started.wait(), 3)
            await until(lambda: len(scene.moves) >= 3)
            await getattr(r, operation)()
            count = len(scene.moves)
            client.release.set()
            await asyncio.sleep(.02)
            assert len(scene.moves) == count
            assert client.cancelled and client.closed and client.active == 0
            assert r.state['status'] == ('idle' if operation == 'reset' else 'stopped')
        finally:
            await r.close()
    asyncio.run(scenario())


def test_guard_failure_still_stops_and_cancels_inference():
    class Guarded(IsaacRuntime):
        async def filter_command(self, v, w, dt, **kwargs):
            if self.state['step'] >= 3:
                self.state['navigation_reason'] = 'blocked_command_occupied'
                return 0., 0.
            return v, w

    async def scenario():
        scene = KinematicScene()
        client = DelayedClient(scene)
        r = Guarded(lambda: scene, {'lightnav': 'unused'}, use_mpc=True,
                    client_factory=lambda *args: client)
        await r.initialize()
        try:
            await r.start('lightnav', 'forward')
            await asyncio.wait_for(r.task, 3)
            assert len(scene.moves) == 3
            assert r.state['stop_reason'] == 'blocked_command_occupied'
            assert client.cancelled and client.active == 0
        finally:
            await r.close()
    asyncio.run(scenario())


def test_completed_old_path_waits_without_extrapolating():
    async def scenario():
        scene = KinematicScene()
        client = DelayedClient(scene, [(0.3, 0., 0.)])
        r = IsaacRuntime(lambda: scene, {'lightnav': 'unused'}, use_mpc=True,
                         client_factory=lambda *args: client)
        await r.initialize()
        try:
            await r.start('lightnav', 'forward')
            await until(lambda: r.state.get('tracking', {}).get('completed_paths', 0) == 1)
            count = len(scene.moves)
            await asyncio.sleep(.02)
            assert len(scene.moves) == count and scene.x <= .31
            assert not r.task.done()
            assert r.state['tracking']['command'] == [0., 0.]
            client.release.set()
            await asyncio.wait_for(r.task, 3)
        finally:
            await r.close()
    asyncio.run(scenario())


def test_model_error_cleans_up_pending_request():
    class Failing(DelayedClient):
        async def predict(self, *args):
            result = await super().predict(*args)
            if len(self.calls) > 1:
                raise RuntimeError('model disconnected')
            return result

    async def scenario():
        scene = KinematicScene()
        client = Failing(scene)
        r = IsaacRuntime(lambda: scene, {'lightnav': 'unused'},
                         client_factory=lambda *args: client)
        await r.initialize()
        try:
            await r.start('lightnav', 'forward')
            await until(lambda: len(scene.moves) >= 3)
            client.release.set()
            await asyncio.wait_for(r.task, 3)
            count = len(scene.moves)
            await asyncio.sleep(.02)
            assert len(scene.moves) == count
            assert r.state['status'] == 'error'
            assert 'model disconnected' in r.state['error']
            assert client.closed and client.active == 0
            assert not r.state['tracking']['inference_pending']
        finally:
            await r.close()
    asyncio.run(scenario())


def test_stop_arriving_during_control_solve_prevents_stale_step(monkeypatch):
    async def scenario():
        scene = KinematicScene()
        client = DelayedClient(scene)
        calls = 0

        async def command(self, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                client.release.set()
                await until(lambda: client.active == 0)
            return .15, 0.

        monkeypatch.setattr(PathTracker, 'command', command)
        r = IsaacRuntime(lambda: scene, {'lightnav': 'unused'},
                         client_factory=lambda *args: client)
        await r.initialize()
        try:
            await r.start('lightnav', 'forward')
            await asyncio.wait_for(r.task, 3)
            assert calls == 2 and len(scene.moves) == 1
            assert r.state['stop_reason'] == 'model_stop'
        finally:
            await r.close()
    asyncio.run(scenario())


def test_step_limit_cancels_inflight_prediction():
    async def scenario():
        scene = KinematicScene()
        client = DelayedClient(scene)
        r = IsaacRuntime(lambda: scene, {'lightnav': 'unused'}, max_steps=4,
                         client_factory=lambda *args: client)
        await r.initialize()
        try:
            await r.start('lightnav', 'forward')
            await asyncio.wait_for(r.task, 3)
            assert len(scene.moves) == 4
            assert r.state['stop_reason'] == 'step_limit'
            assert client.cancelled and client.closed and client.active == 0
        finally:
            await r.close()
    asyncio.run(scenario())


def test_mapping_and_guard_continue_during_inference():
    from test_incremental_execution import Navigation

    class MappedNavigation(Navigation):
        def plan(self, points, capture_pose):
            return self.plan_remaining(points, capture_pose, 0)

    async def scenario():
        nav = MappedNavigation()
        scene = KinematicScene()
        client = DelayedClient(scene)
        r = IsaacRuntime(lambda: scene, {'lightnav': 'unused'}, local_navigation=nav,
                         use_mpc=True, client_factory=lambda *args: client)
        await r.initialize()
        try:
            await r.start('lightnav', 'forward')
            await until(lambda: len(scene.moves) >= 5)
            assert len(client.calls) == 2 and client.active == 1
            assert nav.revision >= 5 and len(nav.commands) >= len(scene.moves)
            assert len(nav.plans) == 1
            client.release.set()
            await asyncio.wait_for(r.task, 3)
            assert r.state['stop_reason'] == 'model_stop'
        finally:
            await r.close()
    asyncio.run(scenario())


def test_delayed_path_uses_capture_pose_after_robot_moved(monkeypatch):
    commands = []

    async def command(self, points, pose, **kwargs):
        commands.append(deepcopy(points))
        return .15, 0.

    monkeypatch.setattr(PathTracker, 'command', command)

    class Replacement(DelayedClient):
        async def predict(self, jpeg, instruction):
            result = await super().predict(jpeg, instruction)
            if len(self.calls) == 2:
                return Prediction([(1., 0., 0.)], False, 1.)
            return result

    async def scenario():
        scene = KinematicScene()
        client = Replacement(scene)
        r = IsaacRuntime(lambda: scene, {'lightnav': 'unused'}, use_mpc=True,
                         client_factory=lambda *args: client)
        await r.initialize()
        try:
            await r.start('lightnav', 'forward')
            await until(lambda: len(scene.moves) >= 5)
            captured_x = client.calls[1][1]['x']
            assert scene.x > captured_x
            client.release.set()
            await asyncio.wait_for(r.task, 3)
            short_paths = [p for p in commands if len(p) == 1]
            assert short_paths and short_paths[0][-1][0] == pytest.approx(captured_x + 1.)
            assert len(client.calls) == 3 and client.peak == 1
        finally:
            await r.close()
    asyncio.run(scenario())
