import asyncio
import threading

import pytest
from isaac_web.runtime import IsaacRuntime


class Scene:
    def __init__(self):
        self.time = 0.

    def reset(self):
        self.time = 0.
        return self.observe()

    def observe(self):
        return dict(jpeg=b"jpeg", sim_time=self.time, collided=False,
                    pose=dict(x=self.time, y=0., z=.25, yaw=0.))

    def step(self, linear, angular, dt):
        self.time += dt
        return self.observe()

    def close(self):
        pass


def test_manual_validation_and_automatic_exclusion():
    async def scenario():
        runtime = IsaacRuntime(Scene, {"lightnav": "ws://test"})
        await runtime.initialize()
        try:
            for args in [(2., 0., .3), (.1, 0., 100.), (float('nan'), 0., .3), (True, 0., .3)]:
                with pytest.raises(ValueError):
                    await runtime.manual(*args)
            assert runtime.state["sim_time"] == 0.
            await runtime.manual(.1, 0., .3)
            assert runtime.state["sim_time"] == .3
            assert runtime.observation["pose"]["x"] == .3
            runtime.state["status"] = "running"
            with pytest.raises(ValueError):
                await runtime.manual(.1, 0., .3)
            assert runtime.state["sim_time"] == .3
            runtime.state["status"] = "stopped"
            await runtime.reset()
            assert runtime.state["sim_time"] == 0.
        finally:
            await runtime.close()
    asyncio.run(scenario())


def test_start_rejects_busy_controls_without_queuing():
    async def scenario():
        runtime = IsaacRuntime(Scene, {"lightnav": "ws://test"})
        await runtime.initialize()
        try:
            async with runtime.commands:
                with pytest.raises(ValueError, match="操作"):
                    await asyncio.wait_for(runtime.start("lightnav", "go forward"), .2)
            assert runtime.task is None
        finally:
            await runtime.close()
    asyncio.run(scenario())


def test_cancelled_startup_closes_scene_returned_by_delayed_factory():
    async def scenario():
        entered, release, closed = threading.Event(), threading.Event(), threading.Event()

        class DelayedScene(Scene):
            def __init__(self):
                entered.set()
                release.wait(2)
                super().__init__()

            def close(self):
                closed.set()

        runtime = IsaacRuntime(DelayedScene, {"lightnav": "ws://test"})
        task = asyncio.create_task(runtime.initialize())
        while not entered.is_set():
            await asyncio.sleep(.001)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert closed.is_set(), "worker constructed after cancellation must still be closed"
    asyncio.run(scenario())
