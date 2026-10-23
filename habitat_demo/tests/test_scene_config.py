import pytest
from habitat_web.__main__ import parse_args


def test_semantic_backend_cli_selection(tmp_path):
    scene = tmp_path / "test.glb"
    scene.touch()
    assert parse_args(["--scene", str(scene)]).room_semantic_backend is None
    for backend in ("clip", "groundingdino", "places365"):
        args = parse_args(["--scene", str(scene), "--room-semantic-backend", backend])
        assert args.room_semantic_backend == backend


def test_optional_scene_dataset_config(tmp_path):
    scene = tmp_path / "scene.glb"
    scene.touch()
    config = tmp_path / "dataset.scene_dataset_config.json"
    config.write_text('{"stages": {}}')
    assert parse_args(["--scene", str(scene)]).scene_dataset_config is None
    args = parse_args(["--scene", str(scene), "--scene-dataset-config", str(config)])
    assert args.scene_dataset_config == config.resolve()


@pytest.mark.parametrize("contents", [None, "not json", "[]"])
def test_bad_dataset_config_fails_before_loading_habitat(tmp_path, capsys, contents):
    scene = tmp_path / "scene.glb"
    scene.touch()
    config = tmp_path / "bad.json"
    if contents is not None:
        config.write_text(contents)
    with pytest.raises(SystemExit) as error:
        parse_args(["--scene", str(scene), "--scene-dataset-config", str(config)])
    assert error.value.code == 2
    assert "invalid scene dataset config" in capsys.readouterr().err


def test_unlimited_steps_by_default_or_explicit_zero(tmp_path):
    scene = tmp_path / "scene.glb"
    scene.touch()
    assert parse_args(["--scene", str(scene)]).max_steps == 0
    assert parse_args(["--scene", str(scene), "--max-steps", "0"]).max_steps == 0
    assert parse_args(["--scene", str(scene), "--max-steps", "25"]).max_steps == 25
    with pytest.raises(SystemExit):
        parse_args(["--scene", str(scene), "--max-steps", "-1"])
