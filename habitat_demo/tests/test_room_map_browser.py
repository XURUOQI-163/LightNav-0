"""Browser contracts for the optional room-memory view; no simulator required."""

import asyncio
import copy
import io
import math
import os
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from PIL import Image


def snapshot():
    """Two visited rooms joined by a traversed portal, using local XY coordinates."""
    rooms = []
    for number, y in ((1, 0), (2, 4)):
        rooms.append(
            {
                "room_id": f"room-{number}",
                "room_type": "unknown" if number == 1 else "<img src=x onerror=alert(1)>",
                "semantic_confidence": None if number == 1 else 0.8,
                "observation_state": "partial",
                "visited": True,
                "search_count": number,
                "first_seen_time": number,
                "last_seen_time": number + 3,
                "entry_pose": {"x": 0, "y": y, "yaw": 0},
                "exit_pose": None,
                "trajectory_indices": [number - 1],
                "entry_portal_ids": [] if number == 1 else ["portal-1"],
                "connected_room_ids": [f"room-{3 - number}"],
                "representative_frame_ids": [f"frame-{number}"],
                "representative_frame_id": f"frame-{number}",
                "semantic_status": "unconfigured" if number == 1 else "ready",
            }
        )
    return {
        "enabled": True,
        "status": "recording",
        "error": None,
        "semantic_status": "unconfigured",
        "session_id": "session-1",
        "current_room_id": "room-2",
        "current_pose": {"x": 1, "y": 5, "yaw": math.pi / 2},
        "start_pose": {"x": -1, "y": -1, "yaw": 0},
        "trajectory": [
            {"index": 0, "timestamp": 1, "x": 0, "y": 0, "yaw": 0, "room_id": "room-1"},
            {"index": 1, "timestamp": 2, "x": 0, "y": 4, "yaw": 0, "room_id": "room-2"},
            {
                "index": 2,
                "timestamp": 3,
                "x": 1,
                "y": 5,
                "yaw": math.pi / 2,
                "room_id": "room-2",
            },
        ],
        "rooms": rooms,
        "portals": [
            {
                "portal_id": "portal-1",
                "x": 0,
                "y": 2,
                "normal_yaw": math.pi / 2,
                "width": 0.9,
                "room_a": "room-1",
                "room_b": "room-2",
                "traversal_count": 1,
                "confidence": 0.7,
            }
        ],
        "candidate": None,
        "dropped_frames": 0,
        "live_semantic": {
            "enabled": True,
            "status": "ready",
            "room_type": "candy_store",
            "confidence": 0.162055,
            "source_timestamp": 3.0,
            "age_seconds": 0.2,
            "latency_ms": 82.1,
            "completed_count": 1,
            "error": None,
        },
    }


@pytest.mark.asyncio
async def test_room_map_browser():
    """Catch unsafe labels, reversed XY/yaw, broken selection and poll recovery."""
    playwright = pytest.importorskip("playwright.async_api")
    executable = os.environ.get("TEST_CHROMIUM")
    if not executable:
        pytest.skip("Set TEST_CHROMIUM to an installed Chromium executable")
    web_dir = Path(__file__).parents[1] / "habitat_web" / "web"
    app = web.Application()

    async def index(_request):
        return web.FileResponse(web_dir / "index.html")

    app.router.add_get("/", index)
    app.router.add_static("/static", web_dir)
    image = io.BytesIO()
    Image.new("RGB", (64, 48), (30, 40, 50)).save(image, "JPEG")
    next_snapshot = snapshot()
    failing = False
    release_memory = None
    memory_started = asyncio.Event()
    memory_requests = 0
    frame_requests = []
    state = {
        "status": "idle",
        "backend": "lightnav",
        "instruction": "go forward",
        "sim_time": 3,
        "step": 2,
        "model_latency_ms": None,
        "frame_id": 1,
        "error": None,
    }

    async def api(route):
        nonlocal memory_requests
        path = route.request.url.split("/api/", 1)[1].split("?", 1)[0]
        if path == "room-memory":
            memory_requests += 1
            memory_started.set()
            if release_memory is not None:
                await release_memory.wait()
            if failing:
                await route.fulfill(status=503, body="temporary room-memory failure")
            else:
                await route.fulfill(json=next_snapshot)
        elif path.startswith("room-memory/frames/"):
            frame_requests.append(path)
            await route.fulfill(content_type="image/jpeg", body=image.getvalue())
        elif path == "state":
            await route.fulfill(json=state)
        elif path == "camera.jpg":
            await route.fulfill(content_type="image/jpeg", body=image.getvalue())
        else:
            await route.fulfill(json={"ok": True})

    async with TestServer(app) as server, playwright.async_playwright() as pw:
        browser = await pw.chromium.launch(
            executable_path=executable, headless=True, args=["--no-sandbox"]
        )
        try:
            page = await browser.new_page(viewport={"width": 1280, "height": 900})
            errors = []
            dialogs = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.on("dialog", lambda dialog: dialogs.append(dialog.message))
            await page.route("**/api/**", api)
            await page.goto(str(server.make_url("/")))
            await page.locator("#room-memory-panel").wait_for(state="visible", timeout=5000)
            assert "candy_store" in await page.locator("#room-live-result").inner_text()
            assert "16.2%" in await page.locator("#room-live-result").inner_text()
            assert "82 ms" in await page.locator("#room-live-meta").inner_text()
            assert "3.0 s" in await page.locator("#room-live-meta").inner_text()
            assert await page.locator("#room-map .room-node").count() == 2
            assert await page.locator("#room-map .room-portal-edge").count() == 1
            assert await page.locator("#room-map .room-portal.is-explored").count() == 1
            assert await page.locator("#room-map .room-trajectory").count() == 1
            assert await page.locator("#room-map .room-start").count() == 1
            assert await page.locator("#room-map .room-current-pose").count() == 1
            assert "不完整" in await page.locator("#room-map-note").inner_text()
            assert "<img src=x onerror=alert(1)>" in await page.locator("#room-map").text_content()
            assert (
                await page.locator(
                    "#room-map img, #room-detail img:not(#room-representative)"
                ).count()
                == 0
            )
            assert not dialogs
            coordinates = await page.locator("#room-map .room-node circle").evaluate_all(
                "nodes => nodes.map(node => Number(node.getAttribute('cy')))"
            )
            assert coordinates[1] < coordinates[0], "Positive Y must point up"
            assert await page.evaluate("""() => {
                const start = document.querySelector('#room-map .room-start').getBBox();
                const first = document.querySelector('#room-map .room-node circle');
                return start.x + start.width / 2 < Number(first.getAttribute('cx'))
                    && start.y + start.height / 2 > Number(first.getAttribute('cy'));
            }"""), "Start pose must remain distinct from a pruned trajectory's first sample"
            assert "rotate(-90)" in await page.locator(
                "#room-map .room-current-pose"
            ).get_attribute("transform")
            await page.locator('#room-map .room-node[data-room-id="room-1"]').click()
            assert "room-1" in await page.locator("#room-detail").inner_text()
            assert "未配置" in await page.locator("#room-detail").inner_text()
            assert "未知房间" not in await page.locator("#room-memory-panel").inner_text()
            assert await page.locator("#room-detail-confidence").inner_text() == "—"
            await page.wait_for_function(
                "document.querySelector('#room-representative').naturalWidth === 64"
            )
            assert (await page.locator("#room-representative").get_attribute("src")).endswith(
                "/frame-1.jpg"
            )
            assert await page.locator("#room-memory-export").get_attribute("href") == (
                "/api/room-memory/export"
            )

            next_snapshot["rooms"][0].update(
                room_type="palace",
                semantic_status="ready",
                semantic_confidence=0.528,
                topological_status="provisional",
            )
            await page.wait_for_function(
                "document.querySelector('#room-detail-type').textContent === 'palace'"
            )
            assert "52.8%" in await page.locator("#room-detail-confidence").inner_text()
            assert "尚未确认门户穿越" in await page.locator("#room-detail-topology").inner_text()
            assert "空间节点编号" in await page.locator("#room-detail").inner_text()

            # The optional Qwen provider may legitimately abstain; it is not a
            # stale Places365 process and restarting it is not a remedy.
            next_snapshot["semantic_backend"] = "qwen"
            next_snapshot["rooms"][0]["room_type"] = "unknown"
            await page.wait_for_function(
                "document.querySelector('#room-detail-semantic').textContent.includes('暂无明确类别')"
            )
            assert "重启" not in await page.locator("#room-detail-semantic").inner_text()
            next_snapshot["semantic_backend"] = "clip"
            next_snapshot["rooms"][0]["room_type"] = "palace"
            await page.wait_for_function(
                "document.querySelector('#room-live-title').textContent.includes('CLIP')"
            )
            assert "CLIP" in await page.locator("#room-memory-status").inner_text()

            next_snapshot["live_semantic"].update(status="processing", processing_seconds=2.1)
            await page.wait_for_function(
                "document.querySelector('#room-live-status').textContent.includes('识别中')"
            )
            assert "上次结果" in await page.locator("#room-live-result").inner_text()
            next_snapshot["live_semantic"].update(status="error", error="worker unavailable")
            await page.locator("#room-live-error").wait_for(state="visible", timeout=5000)
            assert "worker unavailable" in await page.locator("#room-live-error").inner_text()
            next_snapshot["live_semantic"].update(status="ready", error=None)
            await page.locator("#room-live-error").wait_for(state="hidden", timeout=5000)

            state.update(status="stopped", stop_reason="step_limit", max_steps=200)
            await page.wait_for_function(
                "document.querySelector('#run-detail').textContent.includes('200 步上限')"
            )
            assert "未判定找到目标" in await page.locator("#run-detail").inner_text()

            failing = True
            await page.locator("#room-memory-error").wait_for(state="visible", timeout=5000)
            assert await page.locator("#connection").inner_text() == "已连接"
            assert await page.locator("#start-button").is_enabled()
            assert await page.locator("#error-message").is_hidden()
            failing = False
            await page.locator("#room-memory-error").wait_for(state="hidden", timeout=5000)
            assert frame_requests.count("room-memory/frames/frame-1.jpg") == 1

            # A stalled module request must time out while other controls stay live;
            # no second memory request may start while the first is still pending.
            release_memory = asyncio.Event()
            memory_started.clear()
            await asyncio.wait_for(memory_started.wait(), timeout=2)
            request_count = memory_requests
            await page.locator("#room-memory-error").wait_for(state="visible", timeout=6000)
            assert "超时" in await page.locator("#room-memory-error").inner_text()
            assert memory_requests == request_count
            assert await page.locator("#start-button").is_enabled()
            release_memory.set()
            release_memory = None
            await page.locator("#room-memory-error").wait_for(state="hidden", timeout=5000)

            await page.set_viewport_size({"width": 390, "height": 844})
            assert await page.locator("#room-map").is_visible()
            assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            next_snapshot = {"enabled": False}
            await page.locator("#room-memory-panel").wait_for(state="hidden", timeout=5000)
            next_snapshot = snapshot()
            next_snapshot.update(
                session_id="session-2",
                rooms=[],
                portals=[],
                trajectory=[],
                current_pose=None,
                current_room_id=None,
                start_pose=None,
            )
            await page.locator("#room-memory-panel").wait_for(state="visible", timeout=5000)
            assert await page.locator("#room-map .room-node").count() == 0
            assert await page.locator("#room-representative").is_hidden()
            assert await page.locator("#room-map-empty").is_visible()
            assert not errors
        finally:
            await browser.close()


def prospective_snapshot():
    """A visible doorway creates a candidate while every real pose stays in R0."""
    result = snapshot()
    origin = result["rooms"][0]
    origin.update(
        room_id="R0",
        connected_room_ids=["R1"],
        visit_count=1,
        representative_frame_ids=[],
        representative_frame_id=None,
    )
    result.update(
        rooms=[origin],
        current_room_id="R0",
        current_pose={"x": 0, "y": 0, "yaw": 0},
        trajectory=[{"index": 0, "timestamp": 1, "x": 0, "y": 0, "yaw": 0, "room_id": "R0"}],
        prospective_rooms=[
            {
                "room_id": "R1",
                "room_type": "unknown",
                "semantic_status": "pending",
                "semantic_confidence": None,
                "visited": False,
                "visit_count": 0,
                "observation_state": "unseen",
                "topological_status": "prospective",
                "entry_pose": None,
                "candidate_pose": {"x": 0, "y": 3, "yaw": math.pi / 2},
                "first_seen_time": 1,
                "last_seen_time": 2,
                "connected_room_ids": ["R0"],
                "entry_portal_ids": ["P1"],
                "representative_frame_ids": ["F1"],
            }
        ],
        portals=[
            {
                "portal_id": "P1",
                "x": 0,
                "y": 2,
                "normal_yaw": math.pi / 2,
                "width": 0.9,
                "room_a": "R0",
                "room_b": "R1",
                "confidence": 0.7,
                "traversal_count": 0,
                "status": "unexplored",
                "first_seen_time": 1,
                "last_seen_time": 2,
                "observation_count": 2,
                "source": "yoloe",
                "geometry_supported": True,
            }
        ],
    )
    return result


@pytest.mark.asyncio
async def test_prospective_room_is_visible_before_entry_and_promotes_in_place():
    """Catch hidden unentered doors, semantic-as-visit mistakes and selection lost on entry."""
    playwright = pytest.importorskip("playwright.async_api")
    executable = os.environ.get("TEST_CHROMIUM")
    if not executable:
        pytest.skip("Set TEST_CHROMIUM to an installed Chromium executable")
    web_dir = Path(__file__).parents[1] / "habitat_web" / "web"
    app = web.Application()

    async def index(_request):
        return web.FileResponse(web_dir / "index.html")

    app.router.add_get("/", index)
    app.router.add_static("/static", web_dir)
    image = io.BytesIO()
    Image.new("RGB", (64, 48), (30, 40, 50)).save(image, "JPEG")
    next_snapshot = prospective_snapshot()

    async def api(route):
        path = route.request.url.split("/api/", 1)[1].split("?", 1)[0]
        if path == "room-memory":
            await route.fulfill(json=next_snapshot)
        elif path.endswith(".jpg"):
            await route.fulfill(content_type="image/jpeg", body=image.getvalue())
        elif path == "state":
            await route.fulfill(
                json={
                    "status": "idle",
                    "backend": "lightnav",
                    "instruction": "go forward",
                    "sim_time": 3,
                    "step": 2,
                    "model_latency_ms": None,
                    "frame_id": 1,
                    "error": None,
                }
            )
        else:
            await route.fulfill(json={"ok": True})

    async with TestServer(app) as server, playwright.async_playwright() as pw:
        browser = await pw.chromium.launch(
            executable_path=executable, headless=True, args=["--no-sandbox"]
        )
        try:
            page = await browser.new_page(viewport={"width": 1280, "height": 900})
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            await page.route("**/api/**", api)
            await page.goto(str(server.make_url("/")))
            await page.locator("#room-memory-panel").wait_for(state="visible", timeout=5000)
            assert await page.locator("#room-map .room-node").count() == 2
            assert await page.locator('#room-map .is-current[data-room-id="R0"]').count() == 1
            assert await page.locator('#room-map .is-prospective[data-room-id="R1"]').count() == 1
            assert await page.locator("#room-map .room-trajectory").count() == 1
            assert "已进入空间 1" in await page.locator("#room-memory-status").inner_text()
            assert "候选空间 1" in await page.locator("#room-memory-status").inner_text()
            assert "门口 1" in await page.locator("#room-memory-status").inner_text()
            portal = page.locator('#room-map .room-portal[data-portal-id="P1"]')
            assert (
                await portal.locator(".room-portal-marker").evaluate(
                    "node => getComputedStyle(node).stroke"
                )
                == "rgb(96, 165, 250)"
            )
            assert (
                await page.locator("#room-map .room-portal-edge.is-unexplored").evaluate(
                    "node => getComputedStyle(node).strokeDasharray"
                )
                != "none"
            )
            await portal.click()
            assert "R1" in await page.locator("#room-detail-title").inner_text()
            assert "未进入" in await page.locator("#room-detail-visits").inner_text()
            assert "进入 0 次" in await page.locator("#room-detail-visits").inner_text()
            assert await page.locator("#room-detail-entry").inner_text() == "—"
            assert "(0.00, 3.00)" in await page.locator("#room-detail-anchor").inner_text()
            assert "70%" in await page.locator("#room-detail-portal-confidence").inner_text()
            assert "0.90 m" in await page.locator("#room-detail-portal-width").inner_text()
            assert "未进入" in await page.locator("#room-detail-portal-status").inner_text()
            await page.wait_for_function(
                "document.querySelector('#room-representative').naturalWidth === 64"
            )

            next_snapshot["prospective_rooms"][0].update(
                room_type="office", semantic_status="tentative", semantic_confidence=0.8
            )
            await page.wait_for_function(
                "document.querySelector('#room-detail-type').textContent.includes('office?')"
            )
            assert "暂定" in await page.locator("#room-detail-semantic").inner_text()
            assert "进入 0 次" in await page.locator("#room-detail-visits").inner_text()
            assert await page.locator('#room-map .is-current[data-room-id="R0"]').count() == 1
            assert (
                await portal.locator(".room-portal-marker").evaluate(
                    "node => getComputedStyle(node).stroke"
                )
                == "rgb(96, 165, 250)"
            )

            promoted = copy.deepcopy(next_snapshot["prospective_rooms"][0])
            promoted.update(
                visited=True,
                visit_count=1,
                observation_state="glanced",
                topological_status="crossing_confirmed",
                semantic_status="pending",
                room_type="unknown",
                entry_pose={"x": 0, "y": 2.5, "yaw": math.pi / 2},
            )
            next_snapshot["rooms"].append(promoted)
            next_snapshot["prospective_rooms"] = []
            next_snapshot["current_room_id"] = "R1"
            next_snapshot["portals"][0].update(status="explored", traversal_count=1)
            await page.wait_for_function(
                "document.querySelector('#room-detail-visits').textContent.includes('进入 1 次')"
            )
            assert await page.locator('#room-map [data-room-id="R1"]').count() == 1
            assert await page.locator('#room-map [data-portal-id="P1"]').count() == 1
            assert "R1" in await page.locator("#room-detail-title").inner_text()
            assert await portal.get_attribute("aria-pressed") == "true"
            assert "已穿越" in await page.locator("#room-detail-portal-status").inner_text()
            assert (
                await portal.locator(".room-portal-marker").evaluate(
                    "node => getComputedStyle(node).stroke"
                )
                == "rgb(248, 113, 113)"
            )
            assert "候选空间 0" in await page.locator("#room-memory-status").inner_text()

            next_snapshot.update(error="storage unavailable", doorway_error="detector unavailable")
            await page.locator("#room-memory-error").wait_for(state="visible", timeout=5000)
            module_error = await page.locator("#room-memory-error").inner_text()
            assert "storage unavailable" in module_error
            assert "detector unavailable" in module_error
            assert await page.locator("#start-button").is_enabled()

            next_snapshot = prospective_snapshot()
            next_snapshot.update(session_id="session-reset", prospective_rooms=[], portals=[])
            next_snapshot["rooms"][0]["connected_room_ids"] = []
            await page.wait_for_function(
                "document.querySelectorAll('#room-map .room-node').length === 1"
            )
            assert await page.locator("#room-map .room-portal").count() == 0
            assert "R0" in await page.locator("#room-detail-title").inner_text()
            assert await page.locator("#room-detail-portal").is_hidden()
            assert await page.locator("#room-representative").is_hidden()
            assert await page.locator("#room-memory-error").is_hidden()
            assert not errors
        finally:
            await browser.close()
