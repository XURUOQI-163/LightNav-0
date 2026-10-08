"""Actual SVG contract: partial regions, unknown holes, selection and reset."""

import os
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from test_room_map_browser import prospective_snapshot


@pytest.mark.asyncio
async def test_room_regions_visible_selectable_and_cleared_on_reset():
    playwright = pytest.importorskip("playwright.async_api")
    executable = os.environ.get("TEST_CHROMIUM")
    if not executable:
        pytest.skip("TEST_CHROMIUM required")
    web_dir = Path(__file__).parents[1] / "habitat_web/web"
    state = prospective_snapshot()
    outer = [{"x": -3, "y": -2}, {"x": 3, "y": -2}, {"x": 3, "y": 1}, {"x": -3, "y": 1}]
    hole = [{"x": 1, "y": 0}, {"x": 2, "y": 0}, {"x": 2, "y": 0.5}, {"x": 1, "y": 0.5}]
    state["rooms"][0].update(
        region_polygons=[dict(outer=outer, holes=[hole])],
        boundary_segments=[dict(a=outer[0], b=outer[1], kind="wall")],
        boundary_state="partial",
        region_area_m2=17.5,
        geometry_confidence=0.3,
    )
    state["structure"] = dict(enabled=True, status="ready", error=None)
    app = web.Application()

    async def index(_):
        return web.FileResponse(web_dir / "index.html")

    app.router.add_get("/", index)
    app.router.add_static("/static", web_dir)

    async def api(route):
        if route.request.url.endswith("/room-memory"):
            await route.fulfill(json=state)
        elif "/state" in route.request.url:
            await route.fulfill(json=dict(status="idle", backend="lightnav", frame_id=0, step=0))
        else:
            await route.fulfill(status=204)

    async with TestServer(app) as server, playwright.async_playwright() as pw:
        browser = await pw.chromium.launch(
            executable_path=executable, headless=True, args=["--no-sandbox"]
        )
        try:
            page = await browser.new_page(viewport={"width": 1280, "height": 900})
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            await page.route("**/api/**", api)
            await page.goto(str(server.make_url("/")))
            region = page.locator(".room-region[data-room-id='R0']")
            await region.wait_for(state="visible", timeout=5000)
            assert await region.get_attribute("fill-rule") == "evenodd"
            assert (await region.get_attribute("d")).count("M") == 2
            assert await region.evaluate("n => getComputedStyle(n).strokeDasharray") != "none"
            assert await page.locator(".room-structural-wall").count() == 1
            # Click known interior away from the node, portal and unknown hole.
            await region.click(position={"x": 12, "y": 12})
            assert "R0" in await page.locator("#room-detail-title").inner_text()
            assert "17.50" in await page.locator("#room-detail-area").inner_text()
            assert "部分" in await page.locator("#room-detail-boundary").inner_text()
            assert await region.evaluate(
                "n => { const b=n.getBBox(), s=n.ownerSVGElement.viewBox.baseVal; return b.x>=0 && b.y>=0 && b.x+b.width<=s.width && b.y+b.height<=s.height; }"
            )
            await page.locator("#room-regions-toggle").uncheck()
            assert await page.locator(".room-region").count() == 0
            assert await page.locator(".room-portal").count() == 1
            await page.locator("#room-regions-toggle").check()
            state["structure"]["status"] = "floor_mismatch"
            await page.wait_for_function(
                "document.querySelector('#room-memory-error').textContent.includes('高度')"
            )
            state.update(
                session_id="new",
                rooms=[],
                prospective_rooms=[],
                portals=[],
                trajectory=[],
                start_pose=None,
                current_pose=None,
                current_room_id=None,
            )
            await page.wait_for_function("document.querySelectorAll('.room-region').length === 0")
            assert not errors
        finally:
            await browser.close()
