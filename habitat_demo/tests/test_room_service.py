import asyncio
import io

import numpy as np
import pytest
from PIL import Image
from room_memory.config import MemoryConfig
from room_memory.semantics import RoomSemanticResult
from room_memory.service import RoomMemoryService


class SlowClassifier:
    def __init__(self):
        self.began = asyncio.Event()
        self.release = asyncio.Event()

    async def classify(self, frames):
        self.began.set()
        await self.release.wait()
        return RoomSemanticResult("office", {"office": 1.0}, 1.0, frames[0].frame_id)


def observation(t):
    output = io.BytesIO()
    Image.new("RGB", (64, 48), "grey").save(output, "JPEG")
    return dict(
        sim_time=t,
        jpeg=output.getvalue(),
        pose=dict(x=0, z=-t * 0.1, y=0, yaw=0),
        depth=np.full((48, 64), 2.0, dtype=np.float32),
        depth_camera=dict(
            fx=32,
            fy=32,
            cx=31.5,
            cy=23.5,
            camera_to_robot=[[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0.45], [0, 0, 0, 1]],
        ),
    )


@pytest.mark.asyncio
async def test_slow_semantics_does_not_block_geometry_and_reset_discards_result(tmp_path):
    classifier = SlowClassifier()
    service = RoomMemoryService(MemoryConfig(), {}, {}, tmp_path, classifier=classifier)
    await service.start()
    try:
        service.offer(observation(0))
        await service.flush()
        await asyncio.wait_for(classifier.began.wait(), 1)
        for i in range(1, 6):
            service.offer(observation(i / 3))
            await service.queue.join()
        assert service.cache["trajectory_total"] >= 3
        old = service.session_id
        await service.reset()
        service.offer(observation(0))
        await service.queue.join()
        classifier.release.set()
        await service.semantic_queue.join()
        assert service.session_id != old
        assert service.cache["rooms"][0]["room_type"] == "unknown"
    finally:
        classifier.release.set()
        await service.close()


@pytest.mark.asyncio
async def test_geometry_failure_is_visible_and_observer_queue_is_bounded(tmp_path, monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    service = RoomMemoryService(MemoryConfig(queue_size=2), {}, {"backend": "qwen"}, tmp_path)
    await service.start()
    try:
        for i in range(10):
            service.offer(observation(i / 3))
        assert service.queue.qsize() == 2 and service.dropped_frames == 8
        await service.queue.join()
        assert service.cache["semantic_status"] == "unconfigured"
        bad = observation(4)
        bad["depth_camera"]["camera_to_robot"] = [[1]]
        service.offer(bad)
        await service.queue.join()
        assert "4x4" in service.cache["error"]
    finally:
        await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("vision_enabled", [False, True])
async def test_memory_is_passive_and_shutdown_does_not_wait_for_cloud(tmp_path, vision_enabled):
    from habitat_web.clients import Prediction
    from habitat_web.runtime import Runtime
    from test_runtime import Pending, Scene

    class MovingScene(Scene):
        def __init__(self):
            super().__init__()
            self.commands = []

        def observe(self):
            return {**observation(self.moves * 0.3), "collided": False}

        def step(self, *args):
            self.commands.append(args)
            return super().step(*args)

    class FixedClient(Pending):
        async def predict(self, *args):
            return Prediction(((1.0, 0.2, 0.0),), False, 1.0)

    class SlowDoorway:
        async def detect(self, _jpeg):
            await asyncio.Event().wait()

        async def aclose(self):
            pass

    classifier = SlowClassifier()
    memory = RoomMemoryService(
        MemoryConfig(),
        {},
        {"live_enabled": True, "live_interval": 0.01},
        tmp_path,
        classifier=classifier,
        doorway_config={"enabled": vision_enabled},
        doorway_detector=SlowDoorway() if vision_enabled else None,
        region_config={"enabled": True, "grid_size": 256},
    )
    commands = []
    for enabled in (False, True):
        scene = MovingScene()
        runtime = Runtime(
            lambda: scene,
            {"lightnav": "ws://fake"},
            max_steps=3,
            client_factory=lambda *a: FixedClient(),
            room_memory=memory if enabled else None,
        )
        await runtime.initialize()
        await runtime.start("lightnav", "go")
        await asyncio.wait_for(runtime.task, 2)
        assert runtime.state["step"] == 3 and runtime.state["status"] == "stopped"
        commands.append(scene.commands)
        if enabled:
            await asyncio.wait_for(classifier.began.wait(), 1)
            assert not classifier.release.is_set()
            assert memory.cache["rooms"][0]["room_type"] == "unknown"
        await asyncio.wait_for(runtime.close(), 1)
    assert commands[0] == commands[1], "Memory must not change base policy commands"
