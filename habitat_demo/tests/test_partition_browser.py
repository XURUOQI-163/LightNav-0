"""The page uses independent geometric rooms/centroids, not legacy R0 anchors."""
import base64
import io
import os
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from PIL import Image
from room_memory.partition import GeometricRoomPartition
from test_room_map_browser import snapshot
from test_room_partition import structure


@pytest.mark.asyncio
async def test_partition_regions_centroid_labels_and_dynamic_semantics(tmp_path):
    playwright = pytest.importorskip('playwright.async_api')
    executable = os.environ.get('TEST_CHROMIUM')
    if not executable:
        pytest.skip('TEST_CHROMIUM required')
    state = snapshot()
    state['semantic_backend'] = 'clip'
    state['room_partition'] = GeometricRoomPartition().update(structure())
    for r in state['room_partition']['rooms']:
        r.update(room_type='bedroom', semantic_status='ready', semantic_cells=10)
    directory = Path(__file__).parents[1]/'habitat_web/web'
    app = web.Application()
    async def index(_):
        return web.FileResponse(directory/'index.html')
    app.router.add_get('/', index)
    app.router.add_static('/static', directory)
    image = io.BytesIO()
    Image.new('RGB', (64, 48)).save(image, 'JPEG')
    png = io.BytesIO()
    Image.new('RGB', (16, 16), (120, 80, 200)).save(png, 'PNG')
    state['semantic_map'] = {'enabled': True, 'image': 'data:image/png;base64,' + base64.b64encode(png.getvalue()).decode(), 'bounds': [0, 0, 16, 16]}
    async def api(route):
        url = route.request.url
        if '/api/room-memory' in url and not url.endswith('.jpg'):
            await route.fulfill(json=state)
        elif '.jpg' in url:
            await route.fulfill(body=image.getvalue(), content_type='image/jpeg')
        else:
            await route.fulfill(json={'status': 'idle', 'backend': 'lightnav', 'frame_id': 1})
    async with TestServer(app) as server, playwright.async_playwright() as pw:
        browser = await pw.chromium.launch(executable_path=executable, headless=True, args=['--no-sandbox'])
        try:
            page = await browser.new_page(viewport={'width': 1500, 'height': 1000})
            errors = []
            page.on('pageerror', lambda e: errors.append(str(e)))
            await page.route('**/api/**', api)
            await page.goto(str(server.make_url('/')))
            await page.wait_for_function("document.querySelectorAll('#room-map [data-partition-boundary]').length === 2")
            nodes = page.locator('#room-map .room-node')
            assert await nodes.count() == 2
            assert await page.locator('#room-map .room-semantic-raster').get_attribute('clip-path') == 'url(#room-partition-clip)'
            assert 'bedroom' in await nodes.first.text_content()
            assert await nodes.first.evaluate("n => n.querySelector('circle').getAttribute('cx') === n.querySelector('text').getAttribute('x')")
            assert await page.locator('#room-map .room-portal').count() == 1
            assert await page.locator('#room-map .room-portal-label').text_content() == 'P0'
            await nodes.first.click()
            assert '几何分割候选' in await page.locator('#room-detail-topology').inner_text()
            state['room_partition']['rooms'][0]['room_type'] = 'kitchen'
            await page.wait_for_function("document.querySelector('#room-map .room-node').textContent.includes('kitchen')")
            assert await page.locator('#room-map .room-node').count() == 2
            await page.locator('#room-memory-panel').screenshot(path=str(tmp_path/'partition.png'))
            assert not errors
        finally:
            await browser.close()
