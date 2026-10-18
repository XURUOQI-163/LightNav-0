import json

import pytest
from isaac_web.__main__ import parse_args


def fixture_config(tmp_path):
    scene = tmp_path / "scene.usd"
    scene.touch()
    robot = tmp_path / "carter.usd"
    robot.touch()
    config = json.loads((__import__('pathlib').Path(__file__).parents[1] /
                         'config/grscenes_carter.json').read_text())
    config.update(scene=str(scene), robot=str(robot), materials=str(tmp_path))
    path = tmp_path / "scene.json"
    path.write_text(json.dumps(config))
    return path


def test_config_override_and_distinct_model_urls(tmp_path):
    config = fixture_config(tmp_path)
    a, cfg = parse_args(["--config", str(config), "--gpu", "1", "--port", "8091",
                         "--spawn", "1", "2", ".27", "90"])
    assert cfg["gpu"] == 1 and a.port == 8091
    assert cfg["spawn"][:3] == [1, 2, .27]
    assert cfg["spawn"][3] == pytest.approx(1.57079632679)
    assert a.lightnav_url == "ws://127.0.0.1:8050"


def test_independent_tracking_switches_and_completion_limits(tmp_path):
    config = str(fixture_config(tmp_path))
    defaults, _ = parse_args(['--config', config])
    assert not defaults.mpc and defaults.realtime_update
    for mpc in (False, True):
        for realtime in (False, True):
            args, _ = parse_args(['--config', config, '--mpc' if mpc else '--no-mpc',
                                 '--realtime-update' if realtime else '--no-realtime-update'])
            assert (args.mpc, args.realtime_update) == (mpc, realtime)
    for flag, value in [('--path-timeout', '0'), ('--path-timeout', 'nan'),
                        ('--path-position-tolerance', '1'), ('--path-yaw-tolerance-deg', '90')]:
        with pytest.raises(SystemExit):
            parse_args(['--config', config, flag, value])


@pytest.mark.parametrize("extra", [["--dt", "nan"], ["--gpu", "-1"],
                                  ["--hfov", "180"], ["--max-linear", "-1"],
                                  ["--lightnav-url", "http://localhost:8050"]])
def test_invalid_config_rejected_before_loading_gpu(tmp_path, extra):
    with pytest.raises(SystemExit):
        parse_args(["--config", str(fixture_config(tmp_path)), *extra])


@pytest.mark.parametrize("name,value", [("width", 640.5), ("height", True),
                                      ("gpu", .5), ("gpu", True)])
def test_fractional_or_boolean_pixel_dimensions_and_gpu_rejected(tmp_path, name, value):
    config = fixture_config(tmp_path)
    data = json.loads(config.read_text())
    data[name] = value
    config.write_text(json.dumps(data))
    with pytest.raises(SystemExit):
        parse_args(["--config", str(config)])


def test_display_rate_override_and_disable(tmp_path):
    config = fixture_config(tmp_path)
    for fps in (0, 10, 15):
        _, cfg = parse_args(['--config', str(config), '--display-fps', str(fps)])
        assert cfg['display_fps'] == fps
    for fps in ('-1', 'nan', '61'):
        with pytest.raises(SystemExit):
            parse_args(['--config', str(config), '--display-fps', fps])


def test_local_navigation_explicit_enable_and_radius(tmp_path):
    config = fixture_config(tmp_path)
    # User-edited launch defaults must not change this CLI override test.
    local_config = tmp_path / 'local.json'
    local_config.write_text(json.dumps(dict(window_radius_m=5., recovery_max_turn_deg=0.)))
    base = ['--config', str(config), '--local-navigation-config', str(local_config)]
    args, cfg = parse_args([*base, '--local-navigation'])
    assert args.local_navigation
    assert cfg['local_navigation']['window_radius_m'] == 5.0
    assert cfg['local_navigation']['recovery_max_turn_deg'] == 0.0
    args, cfg = parse_args([*base, '--local-navigation', '--local-map-radius', '3'])
    assert cfg['local_navigation']['window_radius_m'] == 3.0
    assert not parse_args(['--config', str(config)])[0].local_navigation
    for radius in ('0', 'nan', '20'):
        with pytest.raises(SystemExit):
            parse_args(['--config', str(config), '--local-navigation', '--local-map-radius', radius])


def test_mdl_normalization_defaults_on_and_has_comparison_override(tmp_path):
    config = fixture_config(tmp_path)
    assert parse_args(["--config", str(config)])[1]["normalize_mdl_paths"] is True
    assert parse_args(["--config", str(config), "--no-normalize-mdl-paths"])[1]["normalize_mdl_paths"] is False
    data = json.loads(config.read_text())
    data["normalize_mdl_paths"] = "false"
    config.write_text(json.dumps(data))
    with pytest.raises(SystemExit):
        parse_args(["--config", str(config)])


@pytest.mark.parametrize('name,value', [('camera_ready_max_renders', 0),
                                      ('camera_ready_max_renders', True),
                                      ('camera_ready_max_renders', 1.5),
                                      ('camera_ready_timeout_s', 0),
                                      ('camera_ready_timeout_s', True),
                                      ('camera_ready_timeout_s', float('nan'))])
def test_camera_wait_limits_are_validated(tmp_path, name, value):
    config = fixture_config(tmp_path)
    data = json.loads(config.read_text())
    data[name] = value
    config.write_text(json.dumps(data))
    with pytest.raises(SystemExit):
        parse_args(['--config', str(config)])
