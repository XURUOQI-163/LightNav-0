import time

import pytest
from isaac_web.bridge import IsaacScene


def echo_worker(pipe, config):
    pipe.send({"ok": True, "result": None})
    while True:
        method, args = pipe.recv()
        if method == "close":
            break
        if method == "step":
            time.sleep(config.get("delay", 0))
        pipe.send({"ok": True, "result": (method, args)})
    pipe.close()


def test_private_worker_roundtrip_and_close():
    scene = IsaacScene({}, worker_target=echo_worker, startup_timeout=10)
    pid = scene.process.pid
    assert pid is not None
    assert scene.reset() == ("reset", ())
    assert scene.step(.1, .2, .3) == ("step", (.1, .2, .3))
    scene.close()
    assert not scene.process.is_alive()
    scene.close()


def test_unresponsive_step_terminates_own_worker():
    scene = IsaacScene({"delay": 10}, worker_target=echo_worker,
                       startup_timeout=10, command_timeout=.05)
    with pytest.raises(TimeoutError):
        scene.step(.1, 0, .3)
    assert not scene.process.is_alive()


def preview_worker(pipe, config):
    pipe.send({"ok": True})
    while True:
        method, args = pipe.recv()
        if method == "close":
            break
        pipe.send({"event": "preview", "result": {"jpeg": b"middle"}})
        pipe.send({"ok": True, "result": {"jpeg": b"final"}})
    pipe.close()


def test_preview_messages_are_delivered_before_final_step_response():
    scene = IsaacScene({}, worker_target=preview_worker, startup_timeout=10)
    frames = []
    scene.preview_callback = frames.append
    try:
        assert scene.step(.1, 0, .3) == {"jpeg": b"final"}
        assert frames == [{"jpeg": b"middle"}]
    finally:
        scene.close()


def endless_preview_worker(pipe, config):
    pipe.send({'ok': True})
    pipe.recv()
    while True:
        pipe.send({'event': 'preview', 'result': {'jpeg': b'preview'}})
        time.sleep(.005)


def test_previews_cannot_extend_command_deadline():
    scene = IsaacScene({}, worker_target=endless_preview_worker, startup_timeout=10,
                       command_timeout=.1)
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        scene.step(.1, 0., .3)
    assert time.monotonic() - started < 3
    assert not scene.process.is_alive()


def progress_worker(pipe, config):
    pipe.send({"event": "progress", "result": {"phase": "engine"}})
    echo_worker(pipe, config)


def endless_progress_worker(pipe, config):
    while True:
        pipe.send({"event": "progress", "result": {"phase": "engine"}})
        time.sleep(.01)


def test_startup_progress_is_separate_from_ready_response():
    progress = []
    scene = IsaacScene({}, worker_target=progress_worker, startup_timeout=10,
                       progress_callback=progress.append)
    try:
        assert progress == [{"phase": "engine"}]
        assert scene.reset() == ("reset", ())
    finally:
        scene.close()


def test_startup_progress_cannot_extend_timeout():
    with pytest.raises(TimeoutError):
        IsaacScene({}, worker_target=endless_progress_worker, startup_timeout=.2)


def test_startup_cancel_event_interrupts_wait_and_terminates_child(monkeypatch):
    import threading

    event = threading.Event()
    children = []
    original_close = IsaacScene.close

    def capture_close(self, force=False):
        children.append(self.process)
        original_close(self, force)

    monkeypatch.setattr(IsaacScene, "close", capture_close)
    event.set()
    with pytest.raises(RuntimeError, match="cancelled"):
        IsaacScene({}, worker_target=endless_progress_worker, cancel_event=event)
    assert len(children) == 1
    assert not children[0].is_alive()
