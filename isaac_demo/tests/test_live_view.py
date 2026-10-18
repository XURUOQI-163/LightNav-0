"""Display updates must not change policy observations or physical step counts."""
import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
from aiohttp.test_utils import TestClient, TestServer
from habitat_web.server import create_app
from isaac_web.runtime import IsaacRuntime
from test_http import SensorScene


def test_preview_never_replaces_model_or_memory_observation():
    async def scenario():
        runtime = IsaacRuntime(SensorScene, {"lightnav": "ws://test"})
        async with TestClient(TestServer(create_app(runtime))) as client:
            original = runtime.observation
            frame = runtime.observation_frame_id
            runtime.view_jpeg = b"model-input"
            runtime.pointing = {"apos": [2, 3]}
            runtime._publish_preview({"jpeg": b"live-first", "third_jpeg": b"live-third",
                                      "sim_time": 3.1})
            for view, expected in (("first", b"live-first"), ("third", b"live-third")):
                response = await client.get('/api/display.jpg?view=' + view)
                assert response.status == 200
                assert await response.read() == expected
                assert response.headers['X-Sim-Time'] == '3.1'
                assert response.headers.get('X-Pointing') is None
                unchanged = await client.get('/api/display.jpg?view=' + view +
                                             '&after=' + response.headers['X-Frame-Id'])
                assert unchanged.status == 204
            assert (await client.get('/api/display.jpg?view=invalid')).status == 400
            assert runtime.observation is original
            assert runtime.jpeg == original['jpeg']
            assert runtime.observation_frame_id == frame
            assert runtime.view_jpeg == b"model-input"
            assert runtime.pointing == {"apos": [2, 3]}
            assert runtime.scene.moves == []
    asyncio.run(scenario())


def test_chase_camera_looks_at_robot_for_multiple_headings():
    from isaac_web.geometry import chase_camera_pose
    for yaw in (0., np.pi / 2, -np.pi / 2, np.pi):
        robot = np.array([3., 4., .24])
        position, q = chase_camera_pose(robot, yaw, 1.8, 1.3, .3)
        w, x, y, z = q
        forward = np.array([1 - 2*(y*y+z*z), 2*(x*y+w*z), 2*(x*z-w*y)])
        target = robot + [0., 0., .3]
        expected = target - position
        np.testing.assert_allclose(forward, expected / np.linalg.norm(expected), atol=1e-10)
        assert np.linalg.norm(q) == pytest.approx(1.)


def test_intermediate_frames_preserve_step_count_and_zero_wheels_on_failure():
    from isaac_web.scene import CarterScene
    scene = CarterScene.__new__(CarterScene)
    scene.cfg = {"physics_dt": 1/60, "display_fps": 15}
    scene.sim_time = 0.
    scene.cached = None
    count = []
    wheels = []
    frames = []
    scene.world = SimpleNamespace(step=lambda render: count.append(render))
    scene.robot = SimpleNamespace(get_world_pose=lambda: (np.array([len(count)*.01, 0., .24]), None))
    scene._wheels = lambda v, w: wheels.append((v, w))
    scene._capture_preview = lambda: {"jpeg": b"first", "third_jpeg": b"third",
                                     "sim_time": scene.sim_time}
    scene._capture = lambda: {"sim_time": scene.sim_time}
    scene.preview_callback = lambda frame: frames.append((len(count), frame['sim_time']))
    result = scene.step(.15, .2, 1/3)
    assert len(count) == 20
    assert all(render is False for render in count)
    assert [n for n, _ in frames] == [4, 8, 12, 16]
    assert [t for _, t in frames] == pytest.approx([4/60, 8/60, 12/60, 16/60])
    assert result['sim_time'] == pytest.approx(1/3)
    assert wheels == [(.15, .2), (0., 0.)]
    assert result['timing']['step_wall_ms'] >= result['timing']['render_ms'] >= 0
    scene.preview_callback = lambda frame: (_ for _ in ()).throw(RuntimeError('closed pipe'))
    with pytest.raises(RuntimeError, match='closed pipe'):
        scene.step(.15, 0., 1/3)
    assert wheels[-1] == (0., 0.)


def test_http_can_read_preview_while_physics_step_is_pending():
    import threading

    entered, release = threading.Event(), threading.Event()

    class MovingScene(SensorScene):
        def step(self, linear, angular, duration):
            self.preview_callback({'jpeg': b'intermediate', 'third_jpeg': b'chase',
                                   'sim_time': self.sim_time + duration / 2})
            entered.set()
            if not release.wait(3):
                raise RuntimeError('test release timed out')
            return super().step(linear, angular, duration)

    async def scenario():
        runtime = IsaacRuntime(MovingScene, {'lightnav': 'ws://test'})
        async with TestClient(TestServer(create_app(runtime))) as client:
            task = asyncio.create_task(runtime.manual(.1, 0., .33))
            try:
                for _ in range(100):
                    if entered.is_set() and runtime.display_frame['jpeg'] == b'intermediate':
                        break
                    await asyncio.sleep(.01)
                assert not task.done()
                response = await client.get('/api/display.jpg?view=third')
                assert await response.read() == b'chase'
                assert runtime.observation['sim_time'] == 3.0
                assert runtime.state['motion_phase'] == 'moving'
            finally:
                release.set()
                await task
            assert runtime.observation['sim_time'] == pytest.approx(3.33)
            assert runtime.display_frame['jpeg'] == runtime.observation['jpeg']
    asyncio.run(scenario())
