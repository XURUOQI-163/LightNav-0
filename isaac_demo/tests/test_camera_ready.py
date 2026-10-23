"""Sensor buffers, not a fixed render count, determine readiness (CPU fakes)."""
from types import SimpleNamespace

import numpy as np
import pytest
from isaac_web.scene import CarterScene


def make_scene():
    scene = CarterScene.__new__(CarterScene)
    scene.cfg = dict(width=2, height=2, camera_ready_max_renders=3, camera_ready_timeout_s=10.)
    from isaac_web.stability import DriveLimiter
    scene.drive_limiter = DriveLimiter({})
    scene.sim_time = 0.
    scene.progress_callback = None
    scene.camera = SimpleNamespace(get_rgb=lambda: np.ones((2, 2, 3), dtype=np.uint8),
                                   get_depth=lambda: np.ones((2, 2), dtype=np.float32))
    scene.third_camera = SimpleNamespace(get_rgb=lambda: np.ones((2, 2, 3), dtype=np.uint8))
    return scene


def test_pending_front_rgb_gets_explicit_capture_then_reads_all_buffers_again():
    scene = make_scene()
    captures = []
    scene.camera.get_rgb = lambda: None if not captures else np.ones((2, 2, 3), dtype=np.uint8)
    scene._render_sensor_frame = lambda: captures.append(True)
    buffers = scene._ready_camera_buffers(require_depth=True)
    assert len(captures) == 1
    assert buffers['front_rgb'].shape == (2, 2, 3)
    assert buffers['depth'].shape == (2, 2)
    assert scene.sim_time == 0.


def test_missing_depth_or_third_view_cannot_publish_partial_rgbd():
    for name in ('depth', 'third_rgb'):
        scene = make_scene()
        captures = []
        if name == 'depth':
            scene.camera.get_depth = lambda: None
        else:
            scene.third_camera.get_rgb = lambda: np.empty((0,))
        scene._render_sensor_frame = lambda: captures.append(True)
        with pytest.raises(RuntimeError, match=name):
            scene._ready_camera_buffers(require_depth=True)
        assert len(captures) == 3


def test_preview_does_not_require_depth_and_ready_buffers_do_not_trigger_extra_capture():
    scene = make_scene()
    scene.camera.get_depth = lambda: (_ for _ in ()).throw(AssertionError('preview must not read depth'))
    scene._render_sensor_frame = lambda: (_ for _ in ()).throw(AssertionError('already ready'))
    assert set(scene._ready_camera_buffers(require_depth=False)) == {'front_rgb', 'third_rgb'}


def test_nonempty_wrong_resolution_is_a_clear_error_not_a_silent_resize():
    scene = make_scene()
    scene.camera.get_rgb = lambda: np.ones((1, 2, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match='front_rgb.*shape'):
        scene._ready_camera_buffers(require_depth=False)


def test_explicit_capture_uses_zero_delta_and_restores_physics_guard_on_error(monkeypatch):
    import sys

    calls = []
    values = {'/app/player/playSimulations': True}
    settings = SimpleNamespace(get=lambda k: values[k], set=lambda k, v: values.__setitem__(k, v))

    def step(**kwargs):
        assert values['/app/player/playSimulations'] is False
        calls.append(kwargs)
        raise RuntimeError('fixture annotator failure')

    core = SimpleNamespace(orchestrator=SimpleNamespace(step=step))
    replicator = SimpleNamespace(core=core)
    monkeypatch.setitem(sys.modules, 'carb', SimpleNamespace(settings=SimpleNamespace(get_settings=lambda: settings)))
    monkeypatch.setitem(sys.modules, 'omni', SimpleNamespace(replicator=replicator))
    monkeypatch.setitem(sys.modules, 'omni.replicator', replicator)
    monkeypatch.setitem(sys.modules, 'omni.replicator.core', core)
    with pytest.raises(RuntimeError, match='fixture annotator failure'):
        make_scene()._render_sensor_frame()
    assert calls == [dict(delta_time=0.0, pause_timeline=False, wait_for_render=True)]
    assert values['/app/player/playSimulations'] is True



def test_after_fallback_every_observation_schedules_fresh_data_even_when_old_buffer_exists():
    scene = make_scene()
    version = [0]
    scene.camera.get_rgb = lambda: (None if version[0] == 0 else
                                   np.full((2, 2, 3), version[0], dtype=np.uint8))

    def capture():
        version[0] += 1

    scene._render_sensor_frame = capture
    first = scene._ready_camera_buffers(require_depth=True)
    assert first['front_rgb'][0, 0, 0] == 1
    second = scene._ready_camera_buffers(require_depth=False)
    assert second['front_rgb'][0, 0, 0] == 2
    assert first['front_rgb'][0, 0, 0] == 1
    assert scene.sim_time == 0.


def test_soft_time_budget_stops_further_capture_attempts(monkeypatch):
    scene = make_scene()
    now = [0.]
    monkeypatch.setattr('isaac_web.scene.time.monotonic', lambda: now[0])
    scene.camera.get_depth = lambda: None
    calls = []

    def capture():
        calls.append(True)
        now[0] += 11.

    scene._render_sensor_frame = capture
    with pytest.raises(RuntimeError, match='after 1 explicit captures'):
        scene._ready_camera_buffers(require_depth=True)
    assert len(calls) == 1


def test_warmup_renders_four_times_but_normal_capture_only_one_fresh_sensor_step():
    scene = make_scene()
    scene.cfg.update(render_frames=4, camera_offset=[0., 0., .6], floor_z=0.)
    scene.collided = False
    renders, captures = [], []
    scene.world = SimpleNamespace(render=lambda: renders.append(True))
    scene._position_cameras = lambda: (np.array([0., 0., .24]), np.array([1., 0., 0., 0.]), np.eye(3))
    scene.camera.get_intrinsics_matrix = lambda: np.eye(3)
    scene.camera.get_rgb = lambda: np.full((2, 2, 3), len(captures), dtype=np.uint8)
    scene.camera.get_depth = lambda: np.full((2, 2), 1 + len(captures), dtype=np.float32)
    scene.third_camera.get_rgb = scene.camera.get_rgb
    scene._render_sensor_frame = lambda: captures.append(True)
    scene._jpeg = lambda data: bytes([int(data[0, 0, 0])])
    scene._capture(report_progress=True)
    assert len(renders) == 4
    renders.clear()
    first = scene._capture()
    second = scene._capture()
    preview = scene._capture_preview()
    assert not renders  # No redundant world.render before an explicit sensor step.
    assert len(captures) == 3
    assert (first['jpeg'], second['jpeg'], preview['jpeg']) == (b'\x01', b'\x02', b'\x03')
    assert np.all(first['depth'] == 2) and np.all(second['depth'] == 3)
    assert scene.sim_time == 0.
