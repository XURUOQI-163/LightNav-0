import asyncio
import importlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from room_memory.config import load_config
from room_memory.semantics import SemanticFrame


def module():
    return importlib.import_module("room_memory.clip_recognizer")


def jpeg():
    out = io.BytesIO()
    Image.new("RGB", (64, 48), "grey").save(out, "JPEG")
    return out.getvalue()


def test_clip_selection_preserves_places_defaults_and_uses_own_environment():
    path = Path(__file__).parents[1] / "room_memory/config/default.json"
    _, _, places = load_config(path)
    _, _, clip = load_config(path, semantic_backend="clip")
    assert places["backend"] == "places365"
    assert clip["backend"] == "clip"
    assert clip["python_executable"].endswith("/envs/omnivla/bin/python")
    assert "repo_path" not in clip and "mapping_path" not in clip
    assert clip["labels"] == [
        "bedroom",
        "living room",
        "bathroom",
        "kitchen",
        "dining room",
        "office room",
        "gym",
        "lounge",
        "laundry room",
        "corridor",
    ]


def test_clip_multiview_averages_probabilities_and_keeps_top1():
    frames = [SemanticFrame("a", b""), SemanticFrame("b", b"")]
    result = module().fuse_predictions(
        frames, ["lounge", "living room", "office room"], [[0.4, 0.5, 0.1], [0.5, 0.1, 0.4]]
    )
    assert result.room_type == "lounge"
    assert result.confidence == pytest.approx(0.45)
    assert result.representative_frame_id == "b"
    assert sum(result.probabilities.values()) == pytest.approx(1)


@pytest.mark.parametrize("scores", [[[np.nan, 0]], [[0, 0]], [[-1, 2]]])
def test_clip_invalid_predictions_are_rejected(scores):
    with pytest.raises(ValueError):
        module().fuse_predictions([SemanticFrame("a", b"")], ["gym", "corridor"], scores)


@pytest.mark.asyncio
async def test_clip_real_worker_reuses_local_model_and_reaps_process():
    m = module()
    if not Path(m.DEFAULT_MODEL).is_file() or not Path(m.DEFAULT_PYTHON).is_file():
        pytest.skip("existing CLIP checkpoint/environment unavailable")
    recognizer = m.CLIPRecognizer()
    frame = SemanticFrame("F0", jpeg())
    try:
        result = await recognizer.classify([frame])
        worker = recognizer.process
        again = await recognizer.classify([frame])
        assert recognizer.process is worker
        assert result.room_type in m.DEFAULT_LABELS
        assert set(result.probabilities) == set(m.DEFAULT_LABELS)
        assert result.confidence == pytest.approx(max(result.probabilities.values()))
        assert again.probabilities == pytest.approx(result.probabilities)
    finally:
        await recognizer.aclose()
    assert worker.returncode is not None


@pytest.mark.asyncio
async def test_missing_clip_model_is_visible_without_stopping_geometry(tmp_path):
    from room_memory.config import MemoryConfig
    from room_memory.service import RoomMemoryService
    from test_room_service import observation

    service = RoomMemoryService(
        MemoryConfig(),
        {},
        {"backend": "clip", "model_path": str(tmp_path / "missing.pt")},
        tmp_path / "run",
    )
    await service.start()
    try:
        service.offer(observation(0))
        await service.flush()
        await asyncio.wait_for(service.semantic_queue.join(), 15)
        await asyncio.wait_for(service.live.queue.join(), 15)
        assert service.cache["trajectory_total"] == 1
        assert service.cache["live_semantic"]["status"] == "error"
        assert "CLIP" in service.cache["live_semantic"]["error"]
        assert service.cache["semantic_backend"] == "clip"
        saved = json.loads((service.manager.output_dir / "graph.json").read_text())
        assert saved["semantic_metadata"]["backend"] == "clip"
        assert saved["semantic_metadata"]["labels"] == list(module().DEFAULT_LABELS)
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_clip_cancel_reaps_worker_and_timeout_reports_provider(tmp_path, monkeypatch):
    import sys

    m = module()
    worker = tmp_path / "slow.py"
    worker.write_text("import time\nprint('{\"ready\":true}',flush=True)\ntime.sleep(60)\n")
    original = asyncio.create_subprocess_exec

    async def spawn(*_args, **kwargs):
        return await original(sys.executable, str(worker), **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    frames = [SemanticFrame("F0", jpeg())]
    recognizer = m.CLIPRecognizer()
    try:
        task = asyncio.create_task(recognizer.classify(frames))
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
        recognizer.timeout = 0.1
        with pytest.raises(m.SemanticClassificationError, match="CLIP inference timed out"):
            await recognizer.classify(frames)
        assert recognizer.process is None
    finally:
        await recognizer.aclose()


@pytest.mark.parametrize(
    "options",
    [
        {"labels": ["bedroom", "bedroom"]},
        {"labels": ["unknown", "bedroom"]},
        {"templates": ["no placeholder"]},
        {"num_threads": True},
        {"timeout": float("nan")},
    ],
)
def test_clip_rejects_invalid_configuration(options):
    with pytest.raises(ValueError):
        module().CLIPRecognizer(**options)


@pytest.mark.asyncio
async def test_compare_runs_both_models_on_identical_images(tmp_path):
    from room_memory.places365_recognizer import DEFAULT_PYTHON, DEFAULT_REPO

    m = module()
    needed = [
        Path(m.DEFAULT_MODEL),
        Path(m.DEFAULT_PYTHON),
        Path(DEFAULT_PYTHON),
        Path(DEFAULT_REPO) / "resnet18_places365.pth.tar",
        Path(DEFAULT_REPO) / "categories_places365.txt",
    ]
    if not all(path.is_file() for path in needed):
        pytest.skip("local CLIP and Places365 checkpoints/environments unavailable")
    compare = importlib.import_module("room_memory.compare_semantics")
    path = tmp_path / "frame.jpg"
    path.write_bytes(jpeg())
    report = await compare.compare_images([path])
    assert set(report["results"]) == {"clip", "places365"}
    assert len(report["images"][0]["sha256"]) == 64
    for backend, result in report["results"].items():
        assert result["representative_frame_id"] == "F0"
        assert result["warm_request_ms"] > 0
        assert result["metadata"]["backend"] == backend
    assert len(report["results"]["clip"]["probabilities"]) == 10
    assert len(report["results"]["places365"]["probabilities"]) == 365


@pytest.mark.asyncio
async def test_close_during_worker_startup_does_not_leak_process(tmp_path, monkeypatch):
    import sys

    worker = tmp_path / "worker.py"
    worker.write_text(
        'import json,sys\nprint(json.dumps({"ready":True}),flush=True)\n'
        'for line in sys.stdin:\n print(json.dumps({"room_type":"gym",'
        '"probabilities":{"gym":1},"confidence":1,"representative_frame_id":"F0"}),flush=True)\n'
    )
    entered, release = asyncio.Event(), asyncio.Event()
    original = asyncio.create_subprocess_exec
    children = []

    async def spawn(*_args, **kwargs):
        entered.set()
        await release.wait()
        child = await original(sys.executable, str(worker), **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    recognizer = module().CLIPRecognizer()
    task = asyncio.create_task(recognizer.classify([SemanticFrame("F0", jpeg())]))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        await asyncio.wait_for(recognizer.aclose(), 1)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
        assert recognizer.process is None
        assert all(child.returncode is not None for child in children)
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await recognizer.aclose()
