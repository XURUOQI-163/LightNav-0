"""Prediction pixels must use capture-time calibration and never alter control."""

import asyncio
import json

import pytest
from aiohttp.test_utils import TestClient, TestServer
from habitat_web.clients import Prediction
from habitat_web.runtime import Runtime
from habitat_web.server import create_app
from habitat_web.waypoint_overlay import project_waypoints


def calibration(height=1.0):
    return dict(width=640, height=480, fx=320, fy=320, cx=320, cy=240,
                camera_to_robot=[[0, 0, 1, .2], [-1, 0, 0, 0],
                                 [0, -1, 0, height], [0, 0, 0, 1]])


def test_projection_uses_ground_plane_camera_offset_and_left_sign():
    result = project_waypoints([[2.2, .5, 0]] * 10, calibration())
    assert result['frame_size'] == [640, 480]
    assert result['count'] == result['visible_count'] == 10
    assert [p['index'] for p in result['points']] == list(range(1, 11))
    assert result['points'][0]['pixel'] == pytest.approx([240, 400])
    assert project_waypoints([[2.2, .5, 0]], calibration(.5))['points'][0]['pixel'] == [240, 320]


def test_offscreen_and_behind_points_keep_indices_without_clamping():
    result = project_waypoints([[0, 0, 0], [.3, 0, 0], [2.2, 0, 0]], calibration())
    assert result['visible_count'] == 1
    assert [p['pixel'] for p in result['points']] == [None, None, [320, 400]]
    assert result['points'][2]['index'] == 3
    assert project_waypoints([[1, 0, 0]], None) is None
    bad = calibration()
    bad['camera_to_robot'] = [[0] * 4] * 4
    assert project_waypoints([[1, 0, 0]], bad) is None


def test_http_prediction_image_and_calibration_remain_paired_without_pointing():
    class Scene:
        def reset(self):
            self.time = 0
            return self.observe()

        def observe(self):
            return dict(jpeg=f'frame-{self.time}'.encode(), pose={}, sim_time=self.time,
                        collided=False, depth_camera=calibration(1 + self.time))

        def step(self, *args):
            self.time += 1
            return self.observe()

        def close(self):
            pass

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def predict(self, jpeg, instruction):
            assert jpeg == b'frame-0'
            return Prediction([[2.2, .5, 0]] * 10, False, 1)

    async def scenario():
        runtime = Runtime(Scene, {'lightnav': 'unused'}, max_steps=1,
                          client_factory=lambda *args: Client())
        async with TestClient(TestServer(create_app(runtime))) as client:
            await runtime.start('lightnav', 'forward')
            await runtime.task
            assert runtime.state['status'] == 'stopped', runtime.state
            assert runtime.jpeg == b'frame-1'
            response = await client.get('/api/camera.jpg')
            assert await response.read() == b'frame-0'
            assert json.loads(response.headers['X-Pointing']) is None
            overlay = json.loads(response.headers['X-Waypoints'])
            assert overlay['points'][0]['pixel'] == [240, 400]
            assert runtime.state['waypoints'] == [[2.2, .5, 0]] * 10
            assert response.headers['X-Frame-Id'] == str(runtime.state['frame_id'])
            await runtime.reset()
            response = await client.get('/api/camera.jpg')
            assert json.loads(response.headers['X-Waypoints']) is None

    asyncio.run(scenario())
