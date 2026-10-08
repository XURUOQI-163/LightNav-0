"""CPU contract checks for the full-model adapter, without loading weights."""

import asyncio
import base64
import io
import threading

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from habitat_web.omnivla_service import create_app
from PIL import Image


def payload(session="episode-a"):
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), "red").save(buffer, format="JPEG")
    return {
        "image_jpeg_base64": base64.b64encode(buffer.getvalue()).decode(),
        "prompt": "go to the door",
        "session_id": session,
    }


def test_http_contract_and_validation():
    class Policy:
        calls = []

        def predict(self, image, prompt, session_id):
            self.calls.append((image.mode, image.size, prompt, session_id))
            return [[10.0, 2.0, 1.0, 0.0]]

    async def check():
        policy = Policy()
        async with TestClient(TestServer(create_app(policy))) as client:
            assert await (await client.get("/health")).json() == {"ok": True, "model": "omnivla"}
            for session in ("episode-a", "episode-b"):
                response = await client.post("/predict", json=payload(session))
                assert response.status == 200
                result = await response.json()
                assert result["waypoints"] == [[10.0, 2.0, 1.0, 0.0]]
                assert result["stop"] is False
                assert result["latency_ms"] >= 0
                reset = await client.post("/reset", json={"session_id": session})
                assert await reset.json() == {"ok": True, "session_id": session}
            assert policy.calls == [
                ("RGB", (16, 16), "go to the door", "episode-a"),
                ("RGB", (16, 16), "go to the door", "episode-b"),
            ]
            for invalid in (
                {},
                [],
                {**payload(), "image_jpeg_base64": "garbage"},
                {**payload(), "prompt": "x" * 1025},
            ):
                assert (await client.post("/predict", json=invalid)).status == 400
            assert (await client.post("/reset", json={})).status == 400

    asyncio.run(check())


def test_cancelled_request_keeps_worker_reserved():
    entered, release = threading.Event(), threading.Event()

    class Policy:
        calls = 0

        def predict(self, image, prompt, session_id):
            self.calls += 1
            entered.set()
            assert release.wait(5)
            return [[1.0, 0.0, 1.0, 0.0]]

    class Request:
        async def json(self):
            return payload()

    async def check():
        policy = Policy()
        app = create_app(policy)
        handler = next(
            route.handler for route in app.router.routes() if route.resource.canonical == "/predict"
        )
        task = asyncio.create_task(handler(Request()))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            try:
                await handler(Request())
            except web.HTTPServiceUnavailable:
                pass
            else:
                raise AssertionError("Cancelled request released the model too early")
            assert policy.calls == 1
        finally:
            release.set()
            app.freeze()
            await app.cleanup()

    asyncio.run(check())
