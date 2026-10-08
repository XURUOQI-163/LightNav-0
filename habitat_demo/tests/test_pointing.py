import asyncio

import pytest
from aiohttp.test_utils import TestClient, TestServer
from habitat_web.clients import Prediction, parse_pointing
from habitat_web.runtime import Runtime
from habitat_web.server import create_app
from test_runtime import Pending, Scene

POINTS = {
    "frame_size": [640, 480],
    "apos_px": [120, 80],
    "opos_px": [400, 300],
    "apos_state": "point",
    "opos_state": "point",
    "apos_clamped": False,
    "opos_clamped": True,
}


def test_invalid_and_missing_coordinates_do_not_invent_points():
    assert parse_pointing(None) is None
    assert parse_pointing({"frame_size": [0, 480]}) is None
    points = parse_pointing({**POINTS, "apos_px": [True, 2], "opos_px": [640, 10]})
    assert points["apos_px"] is None and points["opos_px"] is None
    points = parse_pointing(
        {**POINTS, "apos_px": [float("nan"), 2], "opos_px": None, "opos_state": "not_visible"}
    )
    assert points["apos_px"] is None and points["opos_state"] == "not_visible"


@pytest.mark.asyncio
async def test_camera_binds_pointing_to_prediction_input_then_reset_clears():
    class Frames(Scene):
        def observe(self):
            return {**super().observe(), "jpeg": f"frame-{self.moves}".encode()}

    class PointClient(Pending):
        async def predict(self, jpeg, *args):
            assert jpeg == b"frame-0"
            return Prediction(((1.0, 0.0, 0.0),), False, 1.0, pointing=POINTS)

    scene = Frames()
    runtime = Runtime(
        lambda: scene,
        {"lightnav": "ws://test"},
        max_steps=1,
        client_factory=lambda *a: PointClient(),
    )
    async with TestClient(TestServer(create_app(runtime))) as client:
        await runtime.start("lightnav", "go")
        await asyncio.wait_for(runtime.task, 1)
        assert runtime.jpeg == b"frame-1"  # Model's next input remains live and unannotated.
        response = await client.get("/api/camera.jpg")
        assert await response.read() == b"frame-0"
        assert "120" in response.headers["X-Pointing"]
        await runtime.stop()
        response = await client.get("/api/camera.jpg")
        assert await response.read() == b"frame-1"
        assert response.headers["X-Pointing"] == "null"
        await runtime.reset()
        assert runtime.pointing is None
