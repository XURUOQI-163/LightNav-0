"""A constructor error must be reported before Kit's process-level shutdown."""

import sys
import types

import pytest
from isaac_web.scene import CarterScene


def test_constructor_reports_original_error_before_shutdown_can_exit(monkeypatch):
    events = []

    class App:
        def __init__(self, config):
            pass

        def close(self):
            events.append("close")
            raise SystemExit(0)  # Kit fast shutdown may never return to Python.

    def broken_build(self):
        raise ValueError("fixture first frame failed")

    monkeypatch.setitem(sys.modules, "isaacsim", types.SimpleNamespace(SimulationApp=App))
    monkeypatch.setattr(CarterScene, "_build", broken_build)
    with pytest.raises(SystemExit):
        CarterScene(dict(width=640, height=480, gpu=0),
                    error_callback=lambda message: events.append(message))
    assert "ValueError: fixture first frame failed" in events[0]
    assert events[1] == "close"
