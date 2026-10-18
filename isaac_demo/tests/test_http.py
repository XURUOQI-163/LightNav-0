"""HTTP boundaries for opt-in Isaac controls and calibrated RGB-D reads."""

import asyncio
import io
import os

import numpy as np
import pytest
from aiohttp.test_utils import TestClient, TestServer
from habitat_web.runtime import Runtime
from habitat_web.server import create_app
from isaac_web.runtime import IsaacRuntime
from PIL import Image


class SensorScene:
    def reset(self):
        self.moves = []
        self.sim_time = 3.0
        return self.observe()

    def step(self, linear, angular, duration):
        self.moves.append((linear, angular, duration))
        self.sim_time += duration
        return self.observe()

    def observe(self):
        image = io.BytesIO()
        Image.new("RGB", (2, 2), (30, 40, 50)).save(image, "JPEG")
        return {
            "jpeg": image.getvalue(),
            "pose": {"x": 1.0, "y": 2.0, "z": 0.4, "yaw": 0.5},
            "sim_time": self.sim_time,
            "collided": False,
            "depth": np.array([[0.0, 1.25], [2.5, 5.0]], dtype=np.float32),
            "depth_camera": {
                "width": 2, "height": 2, "fx": 1.0, "fy": 1.0,
                "cx": 0.5, "cy": 0.5,
                "camera_to_robot": [[0, 0, 1, 0], [-1, 0, 0, 0],
                                    [0, -1, 0, 0.45], [0, 0, 0, 1]],
            },
            "depth_definition": "camera_forward_z_meters",
        }

    def close(self):
        pass


def test_manual_rejects_cross_origin_and_forwards_finite_step():
    async def scenario():
        runtime = IsaacRuntime(SensorScene, {"lightnav": "ws://test"})
        async with TestClient(TestServer(create_app(runtime))) as client:
            data = {"linear": 0.1, "angular": -0.2, "duration": 0.33}
            response = await client.post(
                "/api/manual", json=data, headers={"Origin": "https://other.example"}
            )
            assert response.status == 403
            assert runtime.scene.moves == []
            response = await client.post("/api/manual", json=data)
            assert response.status == 200
            assert await response.json() == {"ok": True}
            assert runtime.scene.moves == [(0.1, -0.2, 0.33)]
            assert (await client.post("/api/manual", json=[])).status == 400
            for bad in ({}, {**data, "duration": 2}, {**data, "linear": float("nan")}):
                assert (await client.post("/api/manual", json=bad)).status == 400
            assert runtime.scene.moves == [(0.1, -0.2, 0.33)]
    asyncio.run(scenario())


def test_sensor_metadata_and_depth_are_read_only_latest_observation():
    async def scenario():
        runtime = IsaacRuntime(SensorScene, {"lightnav": "ws://test"})
        async with TestClient(TestServer(create_app(runtime))) as client:
            frame_id = runtime.state["frame_id"]
            response = await client.get("/api/sensors")
            assert response.status == 200
            payload = await response.json()
            assert payload["pose"] == {"x": 1.0, "y": 2.0, "z": 0.4, "yaw": 0.5}
            assert payload["depth_camera"]["fx"] == 1.0
            assert payload["depth_camera"]["camera_to_robot"][2] == [0, -1, 0, 0.45]
            assert payload["depth_definition"] == "camera_forward_z_meters"
            assert payload["sim_time"] == 3.0
            response = await client.get("/api/depth.npy")
            assert response.status == 200
            assert response.headers["Cache-Control"] == "no-store"
            depth = np.load(io.BytesIO(await response.read()), allow_pickle=False)
            assert depth.dtype == np.float32
            np.testing.assert_array_equal(depth, [[0.0, 1.25], [2.5, 5.0]])
            assert runtime.state["frame_id"] == frame_id
    asyncio.run(scenario())


def test_browser_manual_buttons_and_simulator_title():
    playwright = pytest.importorskip("playwright.async_api")
    executable = os.environ.get("TEST_CHROMIUM")
    if not executable:
        pytest.skip("Set TEST_CHROMIUM to an installed Chromium executable")

    async def scenario():
        runtime = IsaacRuntime(SensorScene, {"lightnav": "ws://test"})
        async with TestServer(create_app(runtime)) as server, playwright.async_playwright() as pw:
            browser = await pw.chromium.launch(
                executable_path=executable, headless=True, args=["--no-sandbox"]
            )
            try:
                page = await browser.new_page(viewport={"width": 1280, "height": 800})
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                await page.goto(str(server.make_url("/")))
                await page.wait_for_function(
                    "document.querySelector('h1').textContent.includes('Isaac Sim')"
                )
                assert "Carter" in await page.title()
                await page.locator("#manual-forward").click()
                await page.wait_for_function(
                    "document.querySelector('#run-detail').textContent.includes('步进完成')"
                )
                assert runtime.scene.moves == [(0.15, 0.0, 0.33)]
                await page.locator("#manual-left").click()
                await page.wait_for_function("document.querySelector('#step-count').textContent === '2'")
                assert runtime.scene.moves[-1] == (0.0, 0.5, 0.33)
                runtime.state["status"] = "running"
                await page.wait_for_function("document.querySelector('#manual-forward').disabled")
                runtime.state["status"] = "stopped"
                runtime.state["manual_control"] = False
                await page.wait_for_function("document.querySelector('#manual-controls').hidden")
                assert not errors
            finally:
                await browser.close()
    asyncio.run(scenario())


def test_habitat_does_not_gain_isaac_only_routes():
    async def scenario():
        runtime = Runtime(SensorScene, {"lightnav": "ws://test"})
        async with TestClient(TestServer(create_app(runtime))) as client:
            for route in ("/api/sensors", "/api/depth.npy", "/api/rgb.jpg", "/api/observation.npz"):
                assert (await client.get(route)).status == 404
            assert (await client.post("/api/manual", json={})).status == 404
            assert (await client.get("/api/camera.jpg")).status == 200
    asyncio.run(scenario())


def test_latest_rgb_depth_and_atomic_bundle_do_not_use_pinned_pointing_frame():
    async def scenario():
        runtime = IsaacRuntime(SensorScene, {"lightnav": "ws://test"})
        async with TestClient(TestServer(create_app(runtime))) as client:
            runtime.view_jpeg = b"previous-prediction-view"
            runtime.state["frame_id"] += 4
            response = await client.get("/api/rgb.jpg")
            assert response.status == 200
            assert await response.read() == runtime.observation["jpeg"]
            assert response.headers["X-Sim-Time"] == "3.0"
            assert response.headers["X-Frame-Id"] == str(runtime.observation_frame_id)
            depth = await client.get("/api/depth.npy")
            assert depth.headers["X-Frame-Id"] == response.headers["X-Frame-Id"]
            metadata = await (await client.get("/api/sensors")).json()
            assert metadata["frame_id"] == runtime.observation_frame_id
            bundle_response = await client.get("/api/observation.npz")
            assert bundle_response.status == 200
            with np.load(io.BytesIO(await bundle_response.read()), allow_pickle=False) as bundle:
                assert bundle["rgb"].dtype == np.uint8
                assert bundle["rgb"].shape == (2, 2, 3)
                np.testing.assert_allclose(bundle["rgb"][0, 0], [30, 40, 50], atol=2)
                np.testing.assert_array_equal(bundle["depth"], [[0, 1.25], [2.5, 5]])
                np.testing.assert_array_equal(bundle["K"], [[1, 0, .5], [0, 1, .5], [0, 0, 1]])
                np.testing.assert_array_equal(bundle["pose"], [1, 2, .4, .5])
                np.testing.assert_array_equal(bundle["camera_to_robot"][2], [0, -1, 0, .45])
                assert float(bundle["sim_time"]) == 3.0
                assert int(bundle["frame_id"]) == runtime.observation_frame_id
            assert await (await client.get("/api/camera.jpg")).read() == b"previous-prediction-view"
    asyncio.run(scenario())


def test_browser_three_views_update_independently_without_switching():
    playwright = pytest.importorskip("playwright.async_api")
    executable = os.environ.get("TEST_CHROMIUM")
    if not executable:
        pytest.skip("Set TEST_CHROMIUM")

    async def scenario():
        runtime = IsaacRuntime(SensorScene, {"lightnav": "ws://test"})
        async with TestServer(create_app(runtime)) as server, playwright.async_playwright() as pw:
            browser = await pw.chromium.launch(executable_path=executable, headless=True,
                                               args=["--no-sandbox"])
            try:
                runtime._publish_preview({**runtime.observation, "third_jpeg": runtime.jpeg})
                page = await browser.new_page()
                errors = []
                page.on('pageerror', lambda error: errors.append(str(error)))
                await page.goto(str(server.make_url('/')))
                for selector in ('#first-camera-image', '#third-camera-image', '#camera-image'):
                    await page.locator(selector).wait_for(state='visible')
                assert await page.locator('[data-camera-view]').count() == 0
                assert await page.locator('#first-view svg, #third-view svg').count() == 0
                assert await page.locator('#pointing-overlay').is_hidden()
                boxes = [await page.locator(selector).bounding_box()
                         for selector in ('#first-view', '#third-view', '#prediction-view')]
                assert boxes[0]['x'] < boxes[1]['x'] < boxes[2]['x']
                assert max(box['y'] for box in boxes) - min(box['y'] for box in boxes) < 2
                frame_id = runtime.state['frame_id']
                old_url = await page.locator('#third-camera-image').get_attribute('src')
                prediction_url = await page.locator('#camera-image').get_attribute('src')
                runtime._publish_preview({**runtime.observation, "third_jpeg": runtime.jpeg})
                await page.wait_for_function("old => document.querySelector('#third-camera-image').src !== old", arg=old_url)
                assert runtime.state['frame_id'] == frame_id
                assert await page.locator('#camera-image').get_attribute('src') == prediction_url
                await page.wait_for_timeout(2300)
                assert await page.locator('#display-fps').inner_text() == '0.0 FPS'
                assert runtime.scene.moves == []
                assert not errors
            finally:
                await browser.close()
    asyncio.run(scenario())
