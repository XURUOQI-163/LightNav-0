import copy
import json
from pathlib import Path

import numpy as np
import pytest

from isaac_web.robot_profiles import apply_robot_profile, room_geometry_for_robot


CONFIG = Path(__file__).parents[1] / "config/grscenes_carter.json"


def test_half_xy_keeps_camera_height_and_round_wheel_kinematics():
    original = apply_robot_profile(json.loads(CONFIG.read_text()), "original")
    small = apply_robot_profile(json.loads(CONFIG.read_text()), "half_xy")
    np.testing.assert_allclose(small["footprint"], np.array(original["footprint"]) * .5)
    assert small["wheel_radius"] == pytest.approx(original["wheel_radius"] * .5)
    assert small["wheel_base"] == pytest.approx(original["wheel_base"] * .5)
    assert small["camera_offset"] == [.2, 0., .6]
    assert original["camera_offset"] == [.4, 0., .6]
    assert small["spawn"] == original["spawn"]
    assert small["robot_height"] == original["robot_height"]
    assert small["robot_width"] == pytest.approx(.315)


def test_profile_application_is_idempotent_and_does_not_mutate_input():
    cfg = json.loads(CONFIG.read_text())
    before = copy.deepcopy(cfg)
    small = apply_robot_profile(cfg, "half_xy")
    assert cfg == before
    assert apply_robot_profile(small, "half_xy") == small
    assert apply_robot_profile(small, "original")["wheel_radius"] == .24
    with pytest.raises(ValueError):
        apply_robot_profile(cfg, "typo")


def test_both_door_detectors_use_selected_width_but_keep_safety_margin():
    cfg = apply_robot_profile(json.loads(CONFIG.read_text()), "half_xy")
    geometry = {"robot_width": .6, "portal_min_width": .75, "safety_margin": .1}
    doorway = {"depth": dict(geometry), "enabled": True}
    g, d = room_geometry_for_robot(geometry, doorway, cfg)
    for result in (g, d["depth"]):
        assert result["robot_width"] == pytest.approx(.315)
        assert result["portal_min_width"] == pytest.approx(.415)
        assert result["safety_margin"] == .1
    assert geometry["robot_width"] == .6
    assert doorway["depth"]["portal_min_width"] == .75


def test_cli_accepts_profile_override(tmp_path):
    from isaac_web.__main__ import parse_args

    cfg = json.loads(CONFIG.read_text())
    asset = tmp_path / "asset.usd"
    asset.touch()
    cfg.update(scene=str(asset), robot=str(asset), materials=str(tmp_path))
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    _, small = parse_args(["--config", str(path), "--robot-profile", "half_xy"])
    assert small["wheel_base"] == .27
    assert small["robot_profile"] == "half_xy"
