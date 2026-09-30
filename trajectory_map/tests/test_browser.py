"""Execute the real reusable canvas component in Chromium, without a server."""

import os
from pathlib import Path

import pytest

playwright = pytest.importorskip("playwright.sync_api")
WEB = Path(__file__).parents[1] / "src/lightnav_trajectory/web"


@pytest.fixture
def page():
    with playwright.sync_playwright() as p:
        executable = os.environ.get("TRAJECTORY_BROWSER")
        browser = p.chromium.launch(
            executable_path=executable, headless=True, args=["--no-sandbox"]
        )
        page = browser.new_page(viewport={"width": 900, "height": 600}, device_scale_factor=2)
        page.set_content(
            '<canvas id="map" style="width:600px;height:320px"></canvas>'
            '<p id="status"></p><p id="metrics"></p><p id="source"></p>'
        )
        yield page
        browser.close()


def install(page):
    assert (WEB / "trajectory_map.js").is_file(), "trajectory component not implemented"
    page.add_script_tag(path=str(WEB / "trajectory_map.js"))
    page.evaluate("""() => { window.view = new LightNavTrajectory.TrajectoryView(
        document.querySelector('#map'), {status: document.querySelector('#status'),
        metrics: document.querySelector('#metrics'), source: document.querySelector('#source')}); }""")


def test_projection_equal_scale_and_positive_y_up(page):
    install(page)
    result = page.evaluate("""() => {
      const p = LightNavTrajectory.createProjection([[0,0],[2,1]], 600, 320);
      return [p.point(0,0),p.point(1,0),p.point(0,1)];
    }""")
    origin, right, up = result
    assert right[0] - origin[0] == pytest.approx(origin[1] - up[1])
    assert up[1] < origin[1]


def test_heading_reset_invalid_input_and_stale_display(page):
    install(page)
    page.evaluate("""() => {
      view.setSnapshot({schema_version:1,session_id:'a',revision:1,source:'mujoco_ground_truth',
        status:'live',age_s:0,stale_after_s:1, start:[0,0],segments:[[[0,0],[1,0]]],
        current:{x:1,y:0,yaw:Math.PI/2}});
    }""")
    assert "90.0" in page.locator("#metrics").inner_text()
    assert "MuJoCo" in page.locator("#source").inner_text()
    assert page.locator("#map").get_attribute("width") == "1200"
    assert page.evaluate("view.data.current.x") == 1
    page.evaluate("view.setSnapshot({...view.data,revision:0,current:{x:99,y:0,yaw:0}})")
    assert page.evaluate("view.data.current.x") == 1
    page.evaluate("view.setSnapshot({...view.data,revision:2,current:{x:NaN,y:0,yaw:0}})")
    assert page.evaluate("view.data.current.x") == 1
    page.wait_for_timeout(1200)
    assert "未更新" in page.locator("#status").inner_text()
    page.evaluate("view.invalidate()")
    assert page.evaluate("view.data") is None
    assert "等待" in page.locator("#status").inner_text()
    page.evaluate("view.destroy()")


def test_canvas_arrow_points_up_for_positive_quarter_turn(page):
    install(page)
    pixel = page.evaluate("""() => {
      view.setSnapshot({schema_version:1,session_id:'a',revision:1,source:'test',
        status:'live',age_s:0,stale_after_s:1,start:[0,0],segments:[[[0,0],[1,0]]],
        current:{x:1,y:0,yaw:Math.PI/2}});
      const p = LightNavTrajectory.createProjection([[0,0],[1,0]],600,320).point(1,0);
      return [...view.canvas.getContext('2d').getImageData(Math.round(p[0]*2),Math.round((p[1]-9)*2),1,1).data];
    }""")
    assert pixel == [255, 203, 114, 255]
    page.evaluate("view.destroy()")


def test_late_poll_cannot_repopulate_reset_history(page):
    install(page)
    page.evaluate("""() => {
      window.fetch = () => new Promise(resolve => { window.finishOldRead = resolve; });
      window.pendingRead = view.poll();
      view.invalidate();
      finishOldRead({ok:true,json:async () => ({schema_version:1,session_id:'old',revision:1,
        source:'test', status:'live',age_s:0,stale_after_s:1,start:[0,0],segments:[[[0,0]]],
        current:{x:0,y:0,yaw:0}})});
    }""")
    page.evaluate("window.pendingRead")
    assert page.evaluate("view.data") is None
    page.evaluate("view.destroy()")
