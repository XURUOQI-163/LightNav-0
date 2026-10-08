"""Optional Chromium check: pixel alignment, missing points, reset and model switch."""

import io
import os

import pytest
from aiohttp.test_utils import TestServer
from habitat_web.clients import Prediction
from habitat_web.runtime import Runtime
from habitat_web.server import create_app
from PIL import Image
from test_runtime import Pending, Scene


@pytest.mark.asyncio
async def test_pointing_browser():
    playwright = pytest.importorskip("playwright.async_api")
    executable = os.environ.get("TEST_CHROMIUM")
    if not executable:
        pytest.skip("Set TEST_CHROMIUM to an installed Chromium executable")
    image = io.BytesIO()
    Image.new("RGB", (640, 360), (30, 40, 50)).save(image, "JPEG")

    class WideScene(Scene):
        def observe(self):
            return {**super().observe(), "jpeg": image.getvalue()}

    class PointClient(Pending):
        visible = True

        async def predict(self, *args):
            return Prediction(
                (),
                True,
                1.0,
                pointing={
                    "frame_size": [640, 360],
                    "apos_px": [120, 80],
                    "opos_px": [400, 300] if self.visible else None,
                    "apos_state": "point",
                    "opos_state": "point" if self.visible else "not_visible",
                },
            )

    client = PointClient()
    runtime = Runtime(
        WideScene,
        {"lightnav": "ws://test", "omnivla": "http://test"},
        client_factory=lambda *args: client,
    )
    async with TestServer(create_app(runtime)) as server, playwright.async_playwright() as pw:
        browser = await pw.chromium.launch(
            executable_path=executable, headless=True, args=["--no-sandbox"]
        )
        try:
            page = await browser.new_page(viewport={"width": 1280, "height": 800})
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            await page.goto(str(server.make_url("/")))
            await page.locator("#instruction").fill("go forward")
            await page.locator("#start-button").click()
            await page.wait_for_function(
                "document.querySelectorAll('#pointing-overlay circle').length === 2"
            )
            assert "APOS (120, 80)" in await page.locator("#pointing-info").inner_text()
            assert "OPOS (400, 300)" in await page.locator("#pointing-info").inner_text()
            await page.wait_for_function(
                "document.querySelector('#run-detail').textContent.includes('返回停止信号')"
            )

            # A 16:9 image letterboxed inside the 4:3 viewer: compare SVG pixel
            # transform to the image's actual object-fit contain transform.
            for width in (1280, 390):
                await page.set_viewport_size({"width": width, "height": 850})
                delta = await page.evaluate("""() => {
                    const img = document.querySelector('#camera-image');
                    const rect = img.getBoundingClientRect();
                    const scale = Math.min(rect.width/640, rect.height/360);
                    const point = new DOMPoint(120,80).matrixTransform(
                        document.querySelector('#pointing-overlay').getScreenCTM());
                    return Math.max(Math.abs(point.x-(rect.x+(rect.width-640*scale)/2+120*scale)),
                        Math.abs(point.y-(rect.y+(rect.height-360*scale)/2+80*scale)));
                }""")
                assert delta < 1
                assert await page.locator("#pointing-overlay").is_visible()
                assert await page.evaluate(
                    "document.documentElement.scrollWidth <= window.innerWidth"
                )

            client.visible = False
            await page.locator("#start-button").click()
            await page.wait_for_function(
                "document.querySelectorAll('#pointing-overlay circle').length === 1"
            )
            assert "OPOS: 不可见" in await page.locator("#pointing-info").inner_text()
            await page.locator("#reset-button").click()
            await page.wait_for_function(
                "document.querySelector('#pointing-overlay').hasAttribute('hidden')"
            )
            await page.locator("#backend").select_option("omnivla")
            await page.locator("#start-button").click()
            await page.wait_for_function("""async () => {
                const state = await (await fetch('/api/state')).json();
                return state.backend === 'omnivla' && state.status === 'stopped';
            }""")
            assert await page.locator("#pointing-info").is_hidden()
            assert await page.locator("#pointing-overlay circle").count() == 0
            assert not errors
        finally:
            await browser.close()
