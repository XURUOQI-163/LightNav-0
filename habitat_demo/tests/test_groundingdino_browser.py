"""The first viewer uses source-aligned detections, alongside live/model views."""
import io
import os
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from PIL import Image
from room_memory.groundingdino_recognizer import annotate


@pytest.mark.asyncio
async def test_detection_source_frame_display_and_reset(tmp_path):
    playwright = pytest.importorskip('playwright.async_api')
    executable = os.environ.get('TEST_CHROMIUM')
    if not executable:
        pytest.skip('TEST_CHROMIUM required')
    image = io.BytesIO()
    Image.new('RGB', (640, 480), (55, 65, 75)).save(image, 'JPEG')
    jpeg = annotate(image.getvalue(), [dict(bbox_xyxy=[30, 20, 400, 420],
        room_type='kitchen', score=.75)])
    state = dict(status='idle', backend='lightnav', simulator='Isaac Sim', frame_id=0,
        step=0, scene_ready=True, live_view=True, room_detection_enabled=True,
        room_detection_session_id='first')
    requests = []
    ready = True
    web_dir = Path(__file__).parents[1] / 'habitat_web/web'
    app = web.Application()
    async def index(_):
        return web.FileResponse(web_dir / 'index.html')
    app.router.add_get('/', index)
    app.router.add_static('/static', web_dir)

    async def api(route):
        url = route.request.url
        requests.append(url)
        if '/api/state' in url:
            await route.fulfill(json=state)
        elif '/detection.jpg' in url:
            if ready:
                await route.fulfill(body=jpeg, content_type='image/jpeg', headers={
                    'X-Frame-Id': 'first-1.25', 'X-Sim-Time': '1.25'})
            else:
                await route.fulfill(status=503)
        elif '.jpg' in url:
            await route.fulfill(body=image.getvalue(), content_type='image/jpeg', headers={'X-Frame-Id': '0'})
        else:
            await route.fulfill(json={'enabled': False})

    async with TestServer(app) as server, playwright.async_playwright() as pw:
        browser = await pw.chromium.launch(executable_path=executable, headless=True, args=['--no-sandbox'])
        try:
            page = await browser.new_page(viewport={'width': 1500, 'height': 950})
            errors = []
            page.on('pageerror', lambda e: errors.append(str(e)))
            await page.route('**/api/**', api)
            await page.goto(str(server.make_url('/')))
            await page.wait_for_function("document.querySelector('#room-detection-frame-info').textContent.includes('1.25 s')")
            for selector in ('#first-camera-image', '#third-camera-image', '#camera-image'):
                assert await page.locator(selector).is_visible()
            assert 'Grounding DINO-T' in await page.locator('#first-view-title').text_content()
            assert not any('/api/display.jpg?view=first' in url for url in requests)
            await page.locator('#viewer-panel').screenshot(path=str(tmp_path / 'detection-viewer.png'))
            ready = False
            state['room_detection_session_id'] = 'second'
            await page.wait_for_function("document.querySelector('#first-camera-image').hidden")
            assert '等待本会话检测帧' in await page.locator('#room-detection-frame-info').text_content()
            # CLIP sessions restore the raw live image, not the last DINO source frame.
            state['room_detection_enabled'] = False
            await page.wait_for_function(
                "!document.querySelector('#first-camera-image').hidden && "
                "!document.querySelector('#first-view-title').textContent.includes('Grounding DINO')"
            )
            assert any('/api/display.jpg?view=first' in url for url in requests)

            assert not errors
        finally:
            await browser.close()
