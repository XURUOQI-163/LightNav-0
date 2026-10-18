"""Check fusion, real checkpoint execution, and worker lifecycle without cloud calls."""

import asyncio
import importlib
import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from room_memory.config import load_config
from room_memory.semantics import SemanticFrame

PLACES = Path.home() / "projects/places365"
PYTHON = Path.home() / ".conda/envs/places365/bin/python"


def module():
    assert importlib.util.find_spec("room_memory.places365_recognizer"), "Places365 adapter missing"
    return importlib.import_module("room_memory.places365_recognizer")


def frames():
    output = io.BytesIO()
    Image.new("RGB", (32, 32), "grey").save(output, "JPEG")
    return [SemanticFrame("F0", output.getvalue()), SemanticFrame("F1", output.getvalue())]


def test_default_provider_is_local_places_without_api_key(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    _, _, config = load_config()
    assert config["backend"] == "places365"
    assert config["device"] == "cpu"
    assert config["mode"] == "subprocess"
    _, _, shipped = load_config(Path(__file__).parents[1] / "room_memory/config/default.json")
    assert config == shipped


def test_fusion_averages_raw_classes_without_mapping_or_thresholds():
    # Keep bedroom and bedchamber distinct; average views rather than taking max.
    result = module().aggregate_predictions(
        frames(),
        ["bedroom", "bedchamber", "office", "forest_path"],
        [[0.4, 0.2, 0.1, 0.3], [0.7, 0.1, 0.1, 0.1]],
    )
    assert result.room_type == "bedroom"
    assert result.confidence == pytest.approx(0.55)
    assert result.probabilities["forest_path"] == pytest.approx(0.2)
    assert result.probabilities["bedchamber"] == pytest.approx(0.15)
    assert result.probabilities["office"] == pytest.approx(0.1)
    assert sum(result.probabilities.values()) == pytest.approx(1)
    assert result.representative_frame_id == "F1"


def test_unmapped_scene_keeps_its_original_top1_label_and_score():
    result = module().aggregate_predictions(
        frames(), ["forest_path", "bedroom"], [[0.9, 0.1], [0.8, 0.2]]
    )
    assert result.room_type == "forest_path"
    assert result.confidence == pytest.approx(0.85)
    assert "unknown" not in result.probabilities
    assert result.probabilities["bedroom"] == pytest.approx(0.15)


@pytest.mark.parametrize("scores", [[[np.nan, 0]], [[-1, 2]], [[0, 0]], [[1, 0], [1, 0]]])
def test_invalid_inference_scores_rejected(scores):
    with pytest.raises(ValueError):
        module().aggregate_predictions(frames()[:1], ["office", "bedroom"], scores)


@pytest.mark.asyncio
async def test_real_checkpoint_worker_reused_and_reaped():
    if not PYTHON.is_file() or not (PLACES / "resnet18_places365.pth.tar").is_file():
        pytest.skip("local Places365 environment and checkpoint not available")
    recognizer = module().Places365Recognizer(python_executable=str(PYTHON), repo_path=str(PLACES))
    image = (PLACES / "cam_example.jpg").read_bytes()
    process = None
    try:
        result = await recognizer.classify([SemanticFrame("F10", image)])
        process = recognizer.process
        assert process is not None and process.returncode is None
        assert result.representative_frame_id == "F10"
        assert sum(result.probabilities.values()) == pytest.approx(1)
        assert result.room_type == "candy_store"
        assert result.confidence == pytest.approx(0.162055, abs=1e-4)
        assert len(result.probabilities) == 365
        second = await recognizer.classify([SemanticFrame("F11", image)])
        assert recognizer.process is process, "Weights should remain loaded between requests"
        assert second.probabilities == pytest.approx(result.probabilities)
        assert second.representative_frame_id == "F11"
    finally:
        await recognizer.aclose()
    assert process.returncode is not None


@pytest.mark.asyncio
async def test_cancelled_request_reaps_worker_and_does_not_block_loop(tmp_path):
    # Substitute only the slow external process; parent protocol/lifecycle remain real.
    worker = tmp_path / "slow_worker.py"
    worker.write_text("import time\nprint('{\"ready\":true}', flush=True)\ntime.sleep(60)\n")
    recognizer = module().Places365Recognizer(
        repo_path=str(PLACES), python_executable=sys.executable
    )
    from unittest.mock import patch

    original = asyncio.create_subprocess_exec

    async def spawn(*args, **kwargs):
        return await original(sys.executable, str(worker), **kwargs)

    with patch("asyncio.create_subprocess_exec", spawn):
        task = asyncio.create_task(recognizer.classify(frames()))
        try:
            for _ in range(100):
                if recognizer.process is not None:
                    break
                await asyncio.sleep(0.01)
            process = recognizer.process
            assert process is not None
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 2)
            assert process.returncode is not None
        finally:
            await recognizer.aclose()


@pytest.mark.asyncio
async def test_missing_weights_report_error_without_disabling_geometry(tmp_path):
    from room_memory.config import MemoryConfig
    from room_memory.service import RoomMemoryService
    from test_room_service import observation

    service = RoomMemoryService(
        MemoryConfig(),
        {},
        {
            "backend": "places365",
            "repo_path": str(tmp_path),
            "python_executable": str(PYTHON),
        },
        tmp_path / "output",
    )
    await service.start()
    try:
        service.offer(observation(0))
        await service.flush()
        await asyncio.wait_for(service.semantic_queue.join(), 15)
        room = service.cache["rooms"][0]
        assert room["room_type"] == "unknown"
        assert "Places365" in room["semantic_status"]
        assert service.cache["trajectory_total"] == 1
        assert service.cache["semantic_backend"] == "places365"
    finally:
        await service.close()


def test_cli_exposes_raw_places_labels_and_unmapped_mass():
    if not PYTHON.is_file() or not (PLACES / "resnet18_places365.pth.tar").is_file():
        pytest.skip("local Places365 environment and checkpoint not available")
    result = subprocess.run(
        [str(PYTHON), "-m", "room_memory.places365_recognizer", str(PLACES / "cam_example.jpg")],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    answer = json.loads(result.stdout)
    assert len(answer["places365_top5"]) == 5
    assert answer["places365_top5"][0]["label"] == "candy_store"
    assert answer["places365_top5"][0]["probability"] == pytest.approx(0.162055, abs=1e-4)
    assert answer["unmapped_probability"] == pytest.approx(0.965029, abs=1e-4)
    assert answer["room_type"] == "candy_store"
    assert answer["confidence"] == pytest.approx(0.162055, abs=1e-4)
