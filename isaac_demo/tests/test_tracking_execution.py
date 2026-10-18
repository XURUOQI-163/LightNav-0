"""CPU integration of request scheduling with actual official CasADi solves."""

import asyncio
import math
import threading

import pytest
from habitat_web.clients import Prediction
from isaac_web.runtime import IsaacRuntime
from isaac_web.tracking import PathTracker, WorldPath


class KinematicScene:
    def reset(self):
        self.x, self.y, self.yaw, self.time = 0., 0., 0., 0.
        self.moves = []
        return self.observe()

    def observe(self):
        return dict(jpeg=str(self.time).encode(), sim_time=self.time, collided=False,
                    pose=dict(x=self.x, y=self.y, yaw=self.yaw, z=0.))

    def step(self, v, w, dt):
        self.moves.append((v, w, dt))
        self.x += v * math.cos(self.yaw) * dt
        self.y += v * math.sin(self.yaw) * dt
        self.yaw += w * dt
        self.time += dt
        return self.observe()

    def close(self):
        pass


class Client:
    def __init__(self, scene, path):
        self.scene, self.path = scene, path
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def predict(self, *args):
        self.calls.append(self.scene.observe())
        return Prediction(self.path, len(self.calls) == 2, 1.)


@pytest.mark.parametrize('mpc', [False, True])
@pytest.mark.parametrize('realtime', [False, True])
def test_four_modes_gate_requests_on_motion(mpc, realtime):
    async def scenario():
        scene = KinematicScene()
        client = Client(scene, [(i / 10, 0., 0.) for i in range(1, 11)])
        r = IsaacRuntime(lambda: scene, {'lightnav': 'unused'}, use_mpc=mpc,
            realtime_update=realtime, max_linear=.3, client_factory=lambda *args: client)
        await r.initialize()
        try:
            await r.start('lightnav', 'forward')
            await asyncio.wait_for(r.task, 30)
            assert r.state['stop_reason'] == 'model_stop', r.state
            assert len(client.calls) == 2
            if realtime:
                assert len(scene.moves) == 1
                assert client.calls[-1]['pose']['x'] < .2
            else:
                assert len(scene.moves) > 1
                assert client.calls[-1]['pose']['x'] >= .9 - 1e-6
                assert r.state['tracking']['completed_paths'] == 1
            assert all(0 <= v <= .3 and abs(w) <= .5 for v, w, _ in scene.moves)
            if mpc:
                assert all(dt == .1 for _, _, dt in scene.moves)
        finally:
            await r.close()
    asyncio.run(scenario())


def test_safe_prefix_extends_without_requerying_all_ten_waypoints():
    from local_navigation.models import NavigationConfig, transform_path

    class Navigation:
        config = NavigationConfig()
        def reset(self):
            self.prediction = None

        def observe(self, frame):
            p = frame['pose']
            self.pose = (p['x'], p['y'], p['yaw'])

        def snapshot(self):
            return dict(mode='guarded', status='ready', reason='', prediction=self.prediction)

        def plan_remaining(self, points, capture_pose, start_index):
            assert len(points) == 10
            world = transform_path(points, capture_pose)[start_index:start_index + 3]
            self.prediction = dict(status='unchanged' if start_index + 3 >= len(points) else 'safe_prefix')
            return transform_path(world, self.pose, inverse=True)

        def guard(self, command, *args, **kwargs):
            return command

        def clear_prediction(self):
            self.prediction = None

    async def scenario():
        scene, nav = KinematicScene(), Navigation()
        nav.reset()
        client = Client(scene, [(i / 10, 0., 0.) for i in range(1, 11)])
        r = IsaacRuntime(lambda: scene, {'lightnav': 'unused'}, local_navigation=nav,
            realtime_update=False, client_factory=lambda *args: client)
        await r.initialize()
        try:
            await r.start('lightnav', 'forward')
            await r.task
            assert r.state['stop_reason'] == 'model_stop', r.state
            assert len(client.calls) == 2
            assert r.state['tracking']['completed_paths'] == 1
            assert scene.x >= .9 - 1e-6
            assert r.state['tracking']['local_replans'] > 1
        finally:
            await r.close()
    asyncio.run(scenario())


def test_recovery_does_not_request_new_model_path_in_batch_mode():
    class RecoveringRuntime(IsaacRuntime):
        recovered = False

        async def filter_command(self, v, w, dt, **kwargs):
            return (0., 0.) if not self.recovered else (v, w)

        async def recover_blocked(self):
            # Rotate through a full turn; a body-frame path must not drift.
            self.scene.yaw += 2 * math.pi
            self.scene.time += 1.
            self._publish(self.scene.observe())
            self.recovered = True
            return True

    async def scenario():
        scene = KinematicScene()
        client = Client(scene, [(i / 10, 0., 0.) for i in range(1, 11)])
        r = RecoveringRuntime(lambda: scene, {'lightnav': 'unused'}, realtime_update=False,
                              client_factory=lambda *args: client)
        await r.initialize()
        try:
            await r.start('lightnav', 'forward')
            await r.task
            assert r.recovered
            assert r.state['stop_reason'] == 'model_stop', r.state
            assert len(client.calls) == 2
            assert client.calls[-1]['pose']['x'] >= .9 - 1e-6
            assert abs(client.calls[-1]['pose']['y']) < .01
        finally:
            await r.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('mpc', [False, True])
def test_rotation_path_requires_final_heading_before_next_request(mpc):
    async def scenario():
        scene = KinematicScene()
        client = Client(scene, [(0., 0., i * .1) for i in range(1, 11)])
        r = IsaacRuntime(lambda: scene, {'lightnav': 'unused'}, use_mpc=mpc,
                         realtime_update=False, client_factory=lambda *args: client)
        await r.initialize()
        try:
            await r.start('lightnav', 'turn left')
            await asyncio.wait_for(r.task, 30)
            assert r.state['stop_reason'] == 'model_stop', r.state
            assert abs(scene.yaw - 1.) <= math.radians(10)
            assert math.hypot(scene.x, scene.y) < .01
            assert len(client.calls) == 2
        finally:
            await r.close()
    asyncio.run(scenario())


def test_world_path_keeps_capture_origin_and_wraps_final_heading():
    route = WorldPath([(1., 0., .1), (2., 0., .2)], (10., 20., math.pi / 2))
    assert route.points[-1][:2] == pytest.approx([10., 22.])
    assert not route.complete((10., 21., math.pi / 2))
    assert route.complete((10., 22., math.pi / 2 + .2 - 2 * math.pi))


def test_mpc_tracks_curved_path_with_pose_feedback():
    async def scenario():
        scene = KinematicScene()
        # A radius-two left turn, with mutually consistent positions/headings.
        path = [(2 * math.sin(i * .05), 2 * (1 - math.cos(i * .05)), i * .05)
                for i in range(1, 11)]
        client = Client(scene, path)
        r = IsaacRuntime(lambda: scene, {'lightnav': 'unused'}, use_mpc=True,
            realtime_update=False, max_linear=.3, client_factory=lambda *args: client)
        await r.initialize()
        try:
            await r.start('lightnav', 'curve')
            await asyncio.wait_for(r.task, 30)
            assert r.state['stop_reason'] == 'model_stop', r.state
            assert math.dist((scene.x, scene.y), path[-1][:2]) <= .10
            assert abs(scene.yaw - path[-1][2]) <= math.radians(10)
        finally:
            await r.close()
    asyncio.run(scenario())


def test_solver_failure_stops_without_moving_or_requesting(monkeypatch):
    def fail(*args):
        raise RuntimeError('solver unavailable')
    monkeypatch.setattr(PathTracker, '_solve', fail)

    async def scenario():
        scene = KinematicScene()
        client = Client(scene, [(1., 0., 0.)])
        r = IsaacRuntime(lambda: scene, {'lightnav': 'unused'}, use_mpc=True,
            realtime_update=False, client_factory=lambda *args: client)
        await r.initialize()
        try:
            await r.start('lightnav', 'forward')
            await r.task
            assert r.state['status'] == 'error'
            assert 'solver unavailable' in r.state['error']
            assert not scene.moves and len(client.calls) == 1
        finally:
            await r.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('limit,reason', [(0, 'path_execution_timeout'), (2, 'step_limit')])
def test_failed_motion_never_generates_second_request(limit, reason):
    class Stalled(KinematicScene):
        def step(self, v, w, dt):
            return super().step(0., 0., dt)

    async def scenario():
        scene = Stalled()
        client = Client(scene, [(1., 0., 0.)])
        r = IsaacRuntime(lambda: scene, {'lightnav': 'unused'}, realtime_update=False,
            path_timeout=.8, max_steps=limit, client_factory=lambda *args: client)
        await r.initialize()
        try:
            await r.start('lightnav', 'forward')
            await r.task
            assert len(client.calls) == 1
            assert r.state['stop_reason'] == reason
        finally:
            await r.close()
    asyncio.run(scenario())


def test_model_stop_does_not_solve_or_move():
    async def scenario():
        scene = KinematicScene()
        client = Client(scene, [(1., 0., 0.)])
        client.calls.append(None)  # First reply is STOP.
        r = IsaacRuntime(lambda: scene, {'lightnav': 'unused'}, use_mpc=True,
            realtime_update=False, client_factory=lambda *args: client)
        await r.initialize()
        try:
            await r.start('lightnav', 'stop')
            await r.task
            assert scene.moves == []
            assert r.state['stop_reason'] == 'model_stop'
        finally:
            await r.close()
    asyncio.run(scenario())


def test_stop_during_solver_waits_and_never_executes_stale_solution(monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def solve(*args):
        entered.set()
        release.wait(3)
        return .1, 0.

    monkeypatch.setattr(PathTracker, '_solve', solve)

    async def scenario():
        scene = KinematicScene()
        client = Client(scene, [(1., 0., 0.)])
        r = IsaacRuntime(lambda: scene, {'lightnav': 'unused'}, use_mpc=True,
            realtime_update=False, client_factory=lambda *args: client)
        await r.initialize()
        try:
            await r.start('lightnav', 'forward')
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
            assert scene.moves == [] and len(client.calls) == 1
            assert r.state['stop_reason'] == 'user_stop'
        finally:
            release.set()
            await r.close()
    asyncio.run(scenario())


def test_omnivla_preserves_original_loop_with_new_flags():
    async def scenario():
        scene = KinematicScene()
        client = Client(scene, [(1., 0., 0.)])
        r = IsaacRuntime(lambda: scene, {'omnivla': 'unused'}, use_mpc=True,
            realtime_update=False, client_factory=lambda *args: client)
        await r.initialize()
        try:
            await r.start('omnivla', 'forward')
            await r.task
            assert len(scene.moves) == 1
            assert scene.moves[0][2] == pytest.approx(1 / 3)
            assert 'tracking' not in r.state
        finally:
            await r.close()
    asyncio.run(scenario())
