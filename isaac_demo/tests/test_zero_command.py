"""A zero crossing is an executed control tick, not a navigation STOP."""
import asyncio

import pytest
from isaac_web.execution import execute_path
from isaac_web.runtime import IsaacRuntime
from isaac_web.tracking import PathTracker
from test_tracking_execution import Client, KinematicScene


@pytest.mark.parametrize('realtime', [False, True])
@pytest.mark.parametrize('with_map', [False, True])
def test_actual_mpc_zero_crossing_executes_then_turns(realtime, with_map):
    async def scenario():
        from test_incremental_execution import Navigation
        class Map(Navigation):
            def plan(self, points, capture_pose):
                return self.plan_remaining(points, capture_pose, 0)
        scene = KinematicScene()
        nav = Map() if with_map else None
        r = IsaacRuntime(lambda: scene, {}, local_navigation=nav, use_mpc=True, realtime_update=realtime,
                         max_angular=1., max_steps=2)
        await r.initialize()
        tracker = PathTracker(True, .3, 1., .1)
        tracker.previous = (0., .5)
        r.state['tracking'] = dict(completed_paths=0)
        class Pending:
            def ready(self): return False
            def request(self): pass
        try:
            await execute_path(r, tracker, [(-.04, .002, -.3*i) for i in range(1, 11)],
                               r.state['pose'], refresh=Pending() if realtime else None)
            assert len(scene.moves) == 2
            assert abs(scene.moves[0][1]) < 1e-6
            assert scene.moves[1][1] < -.49
            assert r.state['stop_reason'] == 'step_limit'
        finally:
            tracker.close()
            await r.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('realtime', [False, True])
def test_persistent_zero_requests_new_prediction_without_turning(monkeypatch, realtime):
    async def zero(*args, **kwargs): return (0., 0.)
    monkeypatch.setattr(PathTracker, 'command', zero)
    async def scenario():
        scene = KinematicScene()
        client = Client(scene, [(1., 0., 0.)])
        r = IsaacRuntime(lambda: scene, {'lightnav': 'unused'}, use_mpc=True,
                         realtime_update=realtime, client_factory=lambda *args: client)
        await r.initialize()
        try:
            await r.start('lightnav', 'go')
            await asyncio.wait_for(r.task, 5)
            assert r.state['stop_reason'] == 'model_stop'
            assert len(client.calls) == 2
            assert scene.moves and all(v == w == 0 for v,w,_ in scene.moves)
            if not realtime:
                assert len(scene.moves) == 3
                assert r.state['tracking']['last_reprediction_reason'] == 'zero_command_persistent'
        finally:
            await r.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('realtime', [False, True])
def test_endless_zero_predictions_have_bounded_retries(monkeypatch, realtime):
    async def zero(*args, **kwargs): return (0., 0.)
    monkeypatch.setattr(PathTracker, 'command', zero)
    class Endless(Client):
        async def predict(self, *args):
            result = await super().predict(*args)
            result.stop = False
            return result
    async def scenario():
        scene = KinematicScene()
        client = Endless(scene, [(1., 0., 0.)])
        r = IsaacRuntime(lambda: scene, {'lightnav': 'unused'}, use_mpc=True,
                         realtime_update=realtime, client_factory=lambda *args: client)
        await r.initialize()
        try:
            await r.start('lightnav', 'go')
            await asyncio.wait_for(r.task, 5)
            assert r.state['stop_reason'] == 'model_replan_limit', r.state
            assert len(scene.moves) == 12
            assert r.state['tracking']['zero_command_replans'] == 3
            assert all(v == w == 0 for v,w,_ in scene.moves)
        finally:
            await r.close()
    asyncio.run(scenario())
