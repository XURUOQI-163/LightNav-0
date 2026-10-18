"""Real browser renders a world-aligned raster without room/portal dependencies."""
import os
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from test_room_map_browser import snapshot
from test_room_regions import rgbd
from test_semantic_projection import merge, setup_map


@pytest.mark.asyncio
async def test_semantic_layer_colors_alignment_toggle_and_session_reset(tmp_path):
    playwright = pytest.importorskip('playwright.async_api')
    executable = os.environ.get('TEST_CHROMIUM')
    if not executable:
        pytest.skip('TEST_CHROMIUM required')
    _, evidence = setup_map()
    merge(evidence, rgbd(), {'kitchen': .9, 'bedroom': .1})
    state = snapshot()
    state.update(rooms=[], portals=[], trajectory=[], current_pose={'x': 0, 'y': 0, 'yaw': 0},
                 start_pose={'x': 0, 'y': 0}, semantic_map=evidence.snapshot())
    web_dir = Path(__file__).parents[1] / 'habitat_web/web'
    app = web.Application()

    async def index(_):
        return web.FileResponse(web_dir / 'index.html')

    async def api(route):
        if route.request.url.endswith('/room-memory'):
            await route.fulfill(json=state)
        elif '/state' in route.request.url:
            await route.fulfill(json=dict(status='idle', backend='lightnav', frame_id=0, step=0))
        else:
            await route.fulfill(status=204)

    app.router.add_get('/', index)
    app.router.add_static('/static', web_dir)
    async with TestServer(app) as server, playwright.async_playwright() as pw:
        browser = await pw.chromium.launch(executable_path=executable, headless=True, args=['--no-sandbox'])
        try:
            page = await browser.new_page(viewport={'width': 1280, 'height': 900})
            errors = []
            page.on('pageerror', lambda e: errors.append(str(e)))
            await page.route('**/api/**', api)
            await page.goto(str(server.make_url('/')))
            layer = page.locator('.room-semantic-raster')
            await layer.wait_for(state='visible', timeout=5000)
            assert '整图' in await page.locator('#room-semantic-status').text_content()
            assert await page.locator('#room-semantic-legend span').count() == 2
            assert await page.locator('#room-semantic-legend i').first.evaluate('n=>getComputedStyle(n).backgroundColor') == 'rgb(239, 68, 68)'
            bounds = state['semantic_map']['bounds']
            assert float(await layer.get_attribute('width')) / float(await layer.get_attribute('height')) == pytest.approx((bounds[2]-bounds[0])/(bounds[3]-bounds[1]))
            await page.locator('#room-memory-panel').screenshot(path=str(tmp_path / 'semantic-map.png'))
            await page.locator('#room-semantic-toggle').uncheck()
            assert await layer.count() == 0
            await page.locator('#room-semantic-toggle').check()
            await layer.wait_for(state='visible')
            state['semantic_map'].update(status='filtered', accepted_views=0, rejected_views=1,
                                         filter_timestamp=1)
            await page.wait_for_function("document.querySelector('#room-semantic-status').textContent.includes('采纳 0 / 过滤 1')")
            assert await layer.count() == 1  # rejected new evidence does not erase history
            state.update(session_id='reset-session', semantic_map={'enabled': True, 'status': 'waiting'})
            await page.wait_for_function("document.querySelectorAll('.room-semantic-raster').length===0")
            assert not errors
        finally:
            await browser.close()
