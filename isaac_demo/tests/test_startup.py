"""Regression cases for early HTTP readiness; no Isaac/GPU imports required."""

import asyncio
import threading

from aiohttp.test_utils import TestClient, TestServer
from habitat_web.server import create_app
from isaac_web.runtime import IsaacRuntime


class Scene:
    def reset(self):
        return dict(jpeg=b"jpeg", sim_time=0., collided=False,
                    pose=dict(x=0., y=0., yaw=0.))

    def close(self):
        pass


def test_http_available_during_loading_and_controls_unlock_only_after_initial_observation():
    async def scenario():
        release = threading.Event()

        def factory():
            runtime.report_startup_progress(dict(phase="first_frame", message="waiting for RGB-D"))
            if not release.wait(3):
                raise TimeoutError("test did not release loading")
            return Scene()

        runtime = IsaacRuntime(factory, {"lightnav": "ws://test"}, background_startup=True)
        try:
            async with TestClient(TestServer(create_app(runtime))) as client:
                assert (await client.get("/")).status == 200
                state = await (await client.get("/api/state")).json()
                assert state["status"] == "loading" and state["scene_ready"] is False
                assert "elapsed_s" in state["startup"]
                assert (await client.get("/api/camera.jpg")).status == 503
                for method in ("start", "stop", "reset", "manual"):
                    assert (await client.post(f"/api/{method}", json={})).status == 400
                release.set()
                await asyncio.wait_for(runtime.initialization_task, 2)
                state = await (await client.get("/api/state")).json()
                assert state["status"] == "idle" and state["scene_ready"] is True
                assert state["startup"]["phase"] == "ready"
                assert (await client.get("/api/camera.jpg")).status == 200
                runtime._apply_startup_progress(dict(phase="engine"))
                assert runtime.state["startup"]["phase"] == "ready"
        finally:
            release.set()
    asyncio.run(scenario())


def test_startup_error_keeps_http_available_with_controls_blocked():
    async def scenario():
        def factory():
            raise RuntimeError("missing texture fixture")

        runtime = IsaacRuntime(factory, {"lightnav": "ws://test"}, background_startup=True)
        async with TestClient(TestServer(create_app(runtime))) as client:
            await runtime.initialization_task
            state = await (await client.get("/api/state")).json()
            assert state["status"] == "error"
            assert state["scene_ready"] is False
            assert "missing texture fixture" in state["error"]
            assert (await client.post("/api/reset", json={})).status == 400
    asyncio.run(scenario())


def test_closing_loading_runtime_signals_constructor_cancellation():
    async def scenario():
        entered = threading.Event()

        def factory():
            entered.set()
            if not runtime.startup_cancel.wait(3):
                raise AssertionError("startup cancellation was not signalled")
            raise RuntimeError("cancelled constructor")

        runtime = IsaacRuntime(factory, {}, background_startup=True)
        await runtime.begin_initialize()
        while not entered.is_set():
            await asyncio.sleep(.001)
        await asyncio.wait_for(runtime.close(), 1)
        assert runtime.initialization_task.done()
        assert runtime._initialization_cleaned
        await runtime.close()
    asyncio.run(scenario())
