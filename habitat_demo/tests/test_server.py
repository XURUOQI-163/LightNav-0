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
