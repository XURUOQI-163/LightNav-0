import pytest
from aiohttp.test_utils import TestClient, TestServer
from habitat_web.runtime import Runtime
from habitat_web.server import create_app
from test_runtime import Pending, Scene


@pytest.mark.asyncio
async def test_http_controls_latest_camera_and_model_selection():
    scene, model = Scene(), Pending()
    runtime = Runtime(
        lambda: scene,
        {"lightnav": "ws://test", "omnivla": "http://test"},
        client_factory=lambda *a: model,
    )
    async with TestClient(TestServer(create_app(runtime))) as client:
        assert (await client.get("/")).status == 200
        assert await (await client.get("/api/camera.jpg")).read() == b"jpeg"
        for _ in range(3):
            await client.get("/api/state")
            await client.get("/api/camera.jpg")
        assert scene.moves == 0
        assert (
            await client.post("/api/start", json={"backend": "other", "instruction": "go"})
        ).status == 400
        assert (
            await client.post(
                "/api/start",
                json={"backend": "omnivla", "instruction": "go"},
                headers={"Origin": "http://example.com"},
            )
        ).status == 403
        assert (
            await client.post("/api/start", json={"backend": "omnivla", "instruction": "go"})
        ).status == 200
        await model.began.wait()
        assert (await client.get("/api/state")).status == 200
        assert (await client.post("/api/stop", json={})).status == 200
        assert scene.moves == 0
        assert (await client.post("/api/reset", json={})).status == 200
        state = await (await client.get("/api/state")).json()
        assert state["step"] == 0 and state["status"] == "idle"


@pytest.mark.asyncio
async def test_detection_image_is_completed_source_not_live_camera():
    from types import SimpleNamespace

    runtime = Runtime(lambda: Scene(), {"lightnav": "ws://test"}, client_factory=lambda *a: Pending())
    async with TestClient(TestServer(create_app(runtime))) as client:
        assert (await client.get('/api/room-memory/detection.jpg')).status == 503
        runtime.room_memory = SimpleNamespace(semantic_backend='groundingdino',
            projection=SimpleNamespace(detection_frame=dict(jpeg=b'captured-and-annotated',
                frame_id='session-1.0', timestamp=1.0)))
        try:
            state = await (await client.get('/api/state')).json()
            assert state['room_detection_enabled'] is True
            response = await client.get('/api/room-memory/detection.jpg')
            assert await response.read() == b'captured-and-annotated'
            assert response.headers['X-Sim-Time'] == '1.0'
            assert response.headers['X-Frame-Id'] == 'session-1.0'
            assert (await client.get('/api/room-memory/detection.jpg?after=session-1.0')).status == 204
            runtime.room_memory.projection.detection_frame = None
            assert (await client.get('/api/room-memory/detection.jpg')).status == 503
        finally:
            runtime.room_memory = None

