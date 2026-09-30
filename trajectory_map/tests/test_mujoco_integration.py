"""Host integration without an OpenGL renderer or inference server."""

import asyncio

import pytest

pytest.importorskip("vln_mujoco")
from aiohttp.test_utils import TestClient, TestServer
from vln_mujoco.server import VlnMujocoServer


class SimulationInput:
    robot_name = "test"

    def __init__(self):
        self.t, self.x, self.y, self.yaw = 0.0, 6.0, 8.0, 0.0

    def start(self):
        pass

    def stop(self):
        pass

    def set_velocity(self, *_):
        pass

    def frame(self):
        return None

    def third_person_frame(self):
        return None

    def reset(self):
        self.__init__()

    def snapshot(self):
        return {
            "sim_time": self.t,
            "pose": {"x": self.x, "y": self.y, "z": 0, "yaw": self.yaw},
            "velocity": {"linear": 0, "angular": 0},
            "command": {"linear": 0, "angular": 0},
            "camera": {"ready": False, "stamp_ns": None, "fps": 0},
        }


@pytest.mark.asyncio
async def test_route_records_without_camera_or_model_and_reset_is_atomic():
    simulation = SimulationInput()
    server = VlnMujocoServer(simulation=simulation, enable_trajectory_map=True)
    async with TestClient(TestServer(server.app())) as client:
        response = await client.get("/api/trajectory")
        assert response.status == 200
        assert response.headers["Cache-Control"] == "no-store"
        first = await response.json()
        assert first["start"] == [0, 0]
        simulation.t, simulation.x, simulation.yaw = 1.0, 6.5, 1.57
        await asyncio.sleep(0.15)
        moved = await (await client.get("/api/trajectory")).json()
        assert moved["current"]["x"] == 0.5
        assert moved["current"]["yaw"] == pytest.approx(1.57)
        assert len(moved["segments"][0]) >= 2
        server.stop_vln()
        assert (await (await client.get("/api/trajectory")).json())["session_id"] == first[
            "session_id"
        ]
        async with client.ws_connect("/ws") as ws:
            await ws.send_json({"type": "reset"})
            while (await ws.receive_json())["type"] != "command_result":
                pass
            reset = await (await client.get("/api/trajectory")).json()
            assert reset["session_id"] != first["session_id"]
            assert reset["segments"] == [[[0, 0]]]
        await asyncio.sleep(1.1)
        assert (await (await client.get("/api/trajectory")).json())["status"] == "stale"


@pytest.mark.asyncio
async def test_cached_page_restoration_keeps_live_map_updates():
    import os

    browser_api = pytest.importorskip("playwright.async_api")
    simulation = SimulationInput()
    server = VlnMujocoServer(simulation=simulation, enable_trajectory_map=True)
    async with TestClient(TestServer(server.app())) as client:
        async with browser_api.async_playwright() as p:
            browser = await p.chromium.launch(
                executable_path=os.environ.get("TRAJECTORY_BROWSER"),
                headless=True,
                args=["--no-sandbox"],
            )
            try:
                page = await browser.new_page()
                await page.goto(str(client.make_url("/")), wait_until="domcontentloaded")
                await page.wait_for_function(
                    "document.querySelector('#trajectory-metrics').textContent.includes('X 0.00')"
                )
                await page.evaluate(
                    "dispatchEvent(new PageTransitionEvent('pagehide', {persisted:true}))"
                )
                simulation.t, simulation.x = 1.0, 6.5
                await page.evaluate(
                    "dispatchEvent(new PageTransitionEvent('pageshow', {persisted:true}))"
                )
                await page.wait_for_function(
                    "document.querySelector('#trajectory-metrics').textContent.includes('X 0.50')",
                    timeout=2500,
                )
            finally:
                await browser.close()


@pytest.mark.asyncio
async def test_disabled_map_has_no_routes_markup_or_sampling():
    server = VlnMujocoServer(simulation=SimulationInput())
    async with TestClient(TestServer(server.app())) as client:
        assert server.trajectory is None
        def unexpected_sample():
            raise AssertionError("disabled mode must not sample trajectories")
        server._sample_trajectory = unexpected_sample
        body = await (await client.get("/")).text()
        assert "trajectory-map" not in body
        assert "trajectory-static" not in body
        assert (await client.get("/api/trajectory")).status == 404
        assert (await client.get("/trajectory-static/trajectory_map.js")).status == 404
        async with client.ws_connect("/ws") as ws:
            assert (await ws.receive_json())["type"] == "snapshot"
            await ws.send_json({"type": "reset"})
            while True:
                message = await ws.receive_json()
                if message["type"] == "command_result":
                    assert message["ok"]
                    break
        assert server.trajectory is None


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_browser_toggle_controls_resources_and_keeps_websocket(enabled):
    import os

    browser_api = pytest.importorskip("playwright.async_api")
    server = VlnMujocoServer(simulation=SimulationInput(), enable_trajectory_map=enabled)
    async with TestClient(TestServer(server.app())) as client:
        async with browser_api.async_playwright() as p:
            browser = await p.chromium.launch(
                executable_path=os.environ.get("TRAJECTORY_BROWSER"),
                headless=True, args=["--no-sandbox"],
            )
            try:
                page = await browser.new_page()
                requests, errors = [], []
                page.on("request", lambda r: requests.append(r.url))
                page.on("pageerror", lambda e: errors.append(str(e)))
                await page.goto(str(client.make_url("/")))
                await page.wait_for_function(
                    "!document.querySelector('#control-toggle').disabled"
                )
                if enabled:
                    await page.wait_for_function(
                        "document.querySelector('#trajectory-metrics').textContent.includes('X 0.00')"
                    )
                await page.wait_for_timeout(300)
                assert await page.locator("#trajectory-map").count() == int(enabled)
                assert any("/api/trajectory" in u for u in requests) == enabled
                assert any("/trajectory-static/" in u for u in requests) == enabled
                assert not errors
            finally:
                await browser.close()
