"""A slow recognizer must not backlog camera frames or change room identity."""

import asyncio

import pytest
from room_memory.config import MemoryConfig
from room_memory.semantics import RoomSemanticResult
from room_memory.service import RoomMemoryService
from test_room_service import observation


class ControlledClassifier:
    def __init__(self):
        self.calls = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.error = False

    async def classify(self, frames):
        self.calls.append(frames[0].frame_id)
        self.started.set()
        await self.release.wait()
        if self.error:
            raise RuntimeError("classification unavailable")
        return RoomSemanticResult(
            "palace", {"palace": 0.4, "office": 0.3, "bedroom": 0.3}, 0.4, frames[0].frame_id
        )


def service_for(tmp_path, classifier):
    return RoomMemoryService(
        MemoryConfig(),
        {},
        {"live_enabled": True, "live_interval": 0.01},
        tmp_path,
        classifier=classifier,
    )


@pytest.mark.asyncio
async def test_live_uses_latest_frame_and_geometry_keeps_running(tmp_path):
    classifier = ControlledClassifier()
    service = service_for(tmp_path, classifier)
    await service.start()
    try:
        service.offer(observation(0))
        await asyncio.wait_for(classifier.started.wait(), 1)
        assert service.cache["live_semantic"]["status"] == "processing"
        for i in range(1, 6):
            service.offer(observation(i / 3))
            await service.queue.join()
        assert service.cache["trajectory_total"] >= 3
        classifier.release.set()
        await asyncio.wait_for(service.live.queue.join(), 2)
        live = service.cache["live_semantic"]
        assert live["status"] == "ready" and live["room_type"] == "palace"
        assert live["confidence"] == 0.4 and live["source_timestamp"] == 5 / 3
        assert live["completed_count"] == 2 and live["dropped_frames"] == 4
        assert live["latency_ms"] >= 0 and live["age_seconds"] >= 0
        assert service.cache["rooms"][0]["room_type"] == "unknown"
        assert len(service.cache["rooms"]) == 1
    finally:
        classifier.release.set()
        await service.close()


@pytest.mark.asyncio
async def test_reset_rejects_inflight_live_result_and_error_recovers(tmp_path):
    classifier = ControlledClassifier()
    service = service_for(tmp_path, classifier)
    await service.start()
    try:
        service.offer(observation(0))
        await asyncio.wait_for(classifier.started.wait(), 1)
        await service.queue.join()
        await service.reset()
        classifier.release.set()
        await asyncio.wait_for(service.live.queue.join(), 2)
        assert service.cache["live_semantic"]["status"] == "waiting"
        assert service.cache["live_semantic"]["room_type"] is None
        classifier.error = True
        service.offer(observation(1))
        await asyncio.wait_for(service.live.queue.join(), 2)
        assert service.cache["live_semantic"]["status"] == "error"
        assert "classification unavailable" in service.cache["live_semantic"]["error"]
        assert service.cache["live_semantic"]["confidence"] is None
        classifier.error = False
        service.offer(observation(2))
        await asyncio.wait_for(service.live.queue.join(), 2)
        assert service.cache["live_semantic"]["status"] == "ready"
        assert service.cache["live_semantic"]["error"] is None
    finally:
        classifier.release.set()
        await service.close()


@pytest.mark.asyncio
async def test_pointing_keeps_live_classification_on_the_visible_camera_frame(tmp_path):
    from habitat_web.clients import Prediction
    from habitat_web.runtime import Runtime
    from test_runtime import Pending, Scene

    class MovingScene(Scene):
        def observe(self):
            return {**observation(self.moves * 0.3), "collided": False}

    class PointingClient(Pending):
        async def predict(self, *_args):
            return Prediction(((1.0, 0.0, 0.0),), False, 1.0, pointing={"apos": [2, 3]})

    classifier = ControlledClassifier()
    classifier.release.set()
    memory = service_for(tmp_path, classifier)
    runtime = Runtime(
        MovingScene,
        {"lightnav": "ws://fake"},
        room_memory=memory,
        max_steps=1,
        client_factory=lambda *_args: PointingClient(),
    )
    await runtime.initialize()
    try:
        await memory.live.queue.join()
        await runtime.start("lightnav", "go")
        await asyncio.wait_for(runtime.task, 2)
        await memory.live.queue.join()
        assert runtime.state["sim_time"] == 0.3
        assert memory.cache["live_semantic"]["source_timestamp"] == 0
        assert memory.cache["live_semantic"]["completed_count"] == 1
        await runtime.stop()  # Explicit stop now publishes the actual latest view.
        await memory.live.queue.join()
        assert memory.cache["live_semantic"]["source_timestamp"] == 0.3
        assert memory.cache["live_semantic"]["completed_count"] == 2
    finally:
        await runtime.close()
