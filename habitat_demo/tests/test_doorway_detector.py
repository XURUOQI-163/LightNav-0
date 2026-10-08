"""Exercise the async transport with a real child process, without model weights."""

import asyncio
import sys
from io import BytesIO

import pytest
from PIL import Image
from room_memory.doorway_detector import (
    DoorwayDetectionError,
    YOLOEDoorwayDetector,
)


def _jpeg():
    stream = BytesIO()
    Image.new("RGB", (80, 60)).save(stream, format="JPEG")
    return stream.getvalue()


def _worker(tmp_path, behavior="success"):
    script = tmp_path / "worker.py"
    script.write_text(
        "import json,sys,time\n"
        f"behavior={behavior!r}\n"
        "if behavior == 'startup_hang': time.sleep(30)\n"
        "print(json.dumps({'ready': True}), flush=True)\n"
        "for line in sys.stdin:\n"
        " request=json.loads(line)\n"
        " if behavior == 'hang': time.sleep(30)\n"
        " if behavior == 'exit': sys.exit(7)\n"
        " if behavior == 'error':\n"
        "  print(json.dumps({'error':'model failed'}),flush=True); continue\n"
        " response={'id':request['id'],'image_size':[80,60], 'detections':[\n"
        "  {'bbox_xyxy':[10,5,50,55],'confidence':0.85,'label':'doorway',\n"
        "   'polygon':[[10,5],[50,5],[50,55],[10,55]]}]}\n"
        " if behavior == 'wrong_size': response['image_size']=[640,480]\n"
        " if behavior == 'wrong_id': response['id']+=1\n"
        " if behavior == 'nan': response['detections'][0]['confidence']=float('nan')\n"
        " if behavior == 'outside': response['detections'][0]['bbox_xyxy'][2]=81\n"
        " if behavior == 'unknown': response['detections'][0]['label']='window'\n"
        " print(json.dumps(response),flush=True)\n"
    )

    class Detector(YOLOEDoorwayDetector):
        def _worker_command(self):
            return [sys.executable, "-u", str(script)]

    return Detector({"timeout_seconds": 0.25, "startup_timeout_seconds": 1})


@pytest.mark.asyncio
async def test_original_pixel_coordinates_and_persistent_process(tmp_path):
    detector = _worker(tmp_path)
    try:
        detections = await detector.detect(_jpeg())
        process = detector.process
        assert detections[0].bbox_xyxy == (10.0, 5.0, 50.0, 55.0)
        assert detections[0].polygon == ((10.0, 5.0), (50.0, 5.0), (50.0, 55.0), (10.0, 55.0))
        assert detections[0].label == "doorway"
        assert detections[0].confidence == 0.85
        assert await detector.detect(_jpeg()) == detections
        assert detector.process is process
    finally:
        await detector.aclose()
    assert process.returncode is not None
    with pytest.raises(DoorwayDetectionError, match="closed"):
        await detector.detect(_jpeg())


@pytest.mark.parametrize(
    "behavior", ["wrong_size", "wrong_id", "nan", "outside", "unknown", "error", "exit"]
)
@pytest.mark.asyncio
async def test_bad_worker_results_fail_explicitly_and_stop_worker(tmp_path, behavior):
    detector = _worker(tmp_path, behavior)
    try:
        with pytest.raises(DoorwayDetectionError):
            await detector.detect(_jpeg())
        assert detector.process is None
    finally:
        await detector.aclose()


@pytest.mark.parametrize("behavior", ["hang", "startup_hang"])
@pytest.mark.asyncio
async def test_timeout_kills_child_and_reports_error(tmp_path, behavior):
    detector = _worker(tmp_path, behavior)
    try:
        with pytest.raises(DoorwayDetectionError, match="timed out"):
            await detector.detect(_jpeg())
        assert detector.process is None
    finally:
        await detector.aclose()


@pytest.mark.asyncio
async def test_cancellation_reaps_worker(tmp_path):
    detector = _worker(tmp_path, "hang")
    task = asyncio.create_task(detector.detect(_jpeg()))
    for _ in range(100):
        if detector.process is not None:
            break
        await asyncio.sleep(0.005)
    process = detector.process
    assert process is not None
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert process.returncode is not None
    assert detector.process is None
    await detector.aclose()


@pytest.mark.asyncio
async def test_invalid_image_rejected_before_worker_start(tmp_path):
    detector = _worker(tmp_path)
    with pytest.raises(DoorwayDetectionError, match="image"):
        await detector.detect(b"not a jpeg")
    assert detector.process is None
    await detector.aclose()


@pytest.mark.asyncio
async def test_cancelled_queue_wait_does_not_kill_active_detection(tmp_path):
    detector = _worker(tmp_path, "hang")
    active = asyncio.create_task(detector.detect(_jpeg()))
    for _ in range(100):
        if detector.process is not None:
            break
        await asyncio.sleep(0.005)
    process = detector.process
    assert process is not None
    queued = asyncio.create_task(detector.detect(_jpeg()))
    await asyncio.sleep(0.02)
    queued.cancel()
    with pytest.raises(asyncio.CancelledError):
        await queued
    assert detector.process is process
    assert process.returncode is None
    active.cancel()
    with pytest.raises(asyncio.CancelledError):
        await active
    await detector.aclose()


@pytest.mark.asyncio
async def test_failed_worker_can_restart_on_next_detection(tmp_path):
    detector = _worker(tmp_path, "error")
    with pytest.raises(DoorwayDetectionError):
        await detector.detect(_jpeg())
    script = tmp_path / "worker.py"
    script.write_text(script.read_text().replace("behavior='error'", "behavior='success'"))
    try:
        assert (await detector.detect(_jpeg()))[0].label == "doorway"
    finally:
        await detector.aclose()


@pytest.mark.parametrize(
    "options",
    [
        {"device": "auto"},
        {"cpu_threads": 0},
        {"confidence": float("nan")},
        {"timeout_seconds": 0},
        {"labels": []},
        {"max_detections": -1},
        {"model_path": "yolo11s.pt"},
        {"imgsz": True},
    ],
)
def test_invalid_configuration_is_rejected(options):
    with pytest.raises(ValueError):
        YOLOEDoorwayDetector(options)
