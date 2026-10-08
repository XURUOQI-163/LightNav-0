"""The visual observer must not turn detector latency into navigation latency."""

import asyncio

import pytest
from room_memory.config import MemoryConfig
from room_memory.semantics import RoomSemanticResult
from room_memory.service import RoomMemoryService
from test_room_service import observation
from test_visible_portals import visual


class Classifier:
    async def classify(self, frames):
        return RoomSemanticResult("office", {"office": 1}, 1, frames[0].frame_id)


class SlowDetector:
    def __init__(self):
        self.began = asyncio.Event()
        self.release = asyncio.Event()
        self.closed = False

    async def detect(self, jpeg):
        self.began.set()
        await self.release.wait()
        return [object()]

    async def aclose(self):
        self.closed = True


class Verifier:
    def __init__(self):
        self.frames = []

    def verify(self, frame, detection):
        self.frames.append(frame)
        return visual(), {"reason": "accepted_visual_opening"}


def make_service(tmp_path, detector):
    return RoomMemoryService(
        MemoryConfig(),
        {},
        {},
        tmp_path,
        classifier=Classifier(),
        doorway_config={"enabled": True, "queue_size": 2},
        doorway_detector=detector,
    )


@pytest.mark.asyncio
async def test_slow_detector_keeps_source_pose_and_bounded_queue_without_blocking_observer(
    tmp_path,
):
    detector = SlowDetector()
    service = make_service(tmp_path, detector)
    service.depth_verifier = verifier = Verifier()
    await service.start()
    try:
        service.offer(observation(0))
        await asyncio.wait_for(detector.began.wait(), 1)
        for t in (0.3, 0.6, 0.9, 1.2):
            service.offer(observation(t))
            await asyncio.wait_for(service.queue.join(), 1)
        assert service.cache["current_pose"]["x"] == pytest.approx(0.12)
        assert service.doorway_queue.qsize() == 2
        assert service.doorway_dropped_frames == 2
        detector.release.set()
        await asyncio.wait_for(service.doorway_queue.join(), 2)
        await service.semantic_queue.join()
        assert verifier.frames[0].timestamp == 0
        assert verifier.frames[0].pose.x == 0
        assert service.cache["portals"][0]["first_seen_time"] == 0
        assert len(service.cache["prospective_rooms"]) == 1
        assert service.cache["current_room_id"] == "R0"
    finally:
        await service.close()
    assert detector.closed


@pytest.mark.asyncio
async def test_reset_discards_inflight_visual_result(tmp_path):
    detector = SlowDetector()
    service = make_service(tmp_path, detector)
    service.depth_verifier = verifier = Verifier()
    await service.start()
    try:
        service.offer(observation(0))
        await asyncio.wait_for(detector.began.wait(), 1)
        old = service.session_id
        await service.reset()
        detector.release.set()
        await asyncio.wait_for(service.doorway_queue.join(), 1)
        assert service.session_id != old
        assert not verifier.frames
        assert not service.cache["portals"]
        assert not service.cache["rooms"]
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_detector_failure_is_visible_and_shutdown_cancels_wait(tmp_path):
    class Broken(SlowDetector):
        async def detect(self, jpeg):
            raise RuntimeError("missing model asset")

    detector = Broken()
    service = make_service(tmp_path, detector)
    await service.start()
    try:
        service.offer(observation(0))
        await service.queue.join()
        await service.doorway_queue.join()
        assert service.cache["doorway_status"] == "error"
        assert "missing model asset" in service.cache["doorway_error"]
        assert not service.cache["portals"]
        assert service.cache["current_room_id"] == "R0"
    finally:
        await service.close()
    detector = SlowDetector()
    service = make_service(tmp_path / "slow", detector)
    await service.start()
    service.offer(observation(0))
    await asyncio.wait_for(detector.began.wait(), 1)
    await asyncio.wait_for(service.close(), 1)
    assert detector.closed
