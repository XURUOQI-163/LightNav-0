import asyncio
import threading

import pytest
from habitat_web.clients import Prediction
from habitat_web.runtime import Runtime


class Scene:
    def __init__(self):
        self.threads = []
        self.moves = 0
        self.closed = False

    def reset(self):
        self.threads.append(threading.get_ident())
        self.moves = 0
        return self.observe()

    def step(self, *args):
        self.threads.append(threading.get_ident())
        self.moves += 1
        return self.observe()

    def observe(self):
        return {
            "jpeg": b"jpeg",
            "pose": {"x": 0, "y": 0, "z": 0, "yaw": 0},
            "sim_time": self.moves * 0.3,
            "collided": False,
        }

    def close(self):
        self.threads.append(threading.get_ident())
        self.closed = True


class Pending:
    def __init__(self):
        self.began = asyncio.Event()
        self.release = asyncio.Event()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def predict(self, *args):
        self.began.set()
        await self.release.wait()
        return Prediction(((1.0, 0.0, 0.0),), False, 1.0)


@pytest.mark.asyncio
async def test_stop_and_reset_discard_inflight_prediction():
    scene, client = Scene(), Pending()
    runtime = Runtime(lambda: scene, {"lightnav": "ws://test"}, client_factory=lambda *a: client)
    await runtime.initialize()
    await runtime.start("lightnav", "go")
    await client.began.wait()
    await asyncio.wait_for(runtime.stop(), 0.5)
    client.release.set()
    await asyncio.sleep(0.01)
    assert scene.moves == 0
    assert runtime.state["stop_reason"] == "user_stop"
    await runtime.reset()
    assert runtime.state["status"] == "idle"
    await runtime.close()
    assert scene.closed
    assert len(set(scene.threads)) == 1


@pytest.mark.asyncio
async def test_step_budget_and_duplicate_start():
    scene, client = Scene(), Pending()
    runtime = Runtime(
        lambda: scene, {"lightnav": "ws://test"}, max_steps=2, client_factory=lambda *a: client
    )
    await runtime.initialize()
    await runtime.start("lightnav", "go")
    with pytest.raises(ValueError):
        await runtime.start("lightnav", "go again")
    client.release.set()
    await asyncio.wait_for(runtime.task, 2)
    assert scene.moves == 2
    assert runtime.state["stop_reason"] == "step_limit"
    assert runtime.state["status"] == "stopped"
    await runtime.close()


@pytest.mark.asyncio
async def test_stop_publishes_observation_after_inflight_scene_step():
    class BlockingScene(Scene):
        def __init__(self):
            super().__init__()
            self.step_entered = threading.Event()
            self.finish_step = threading.Event()

        def observe(self):
            observation = super().observe()
            observation["jpeg"] = f"frame-{self.moves}".encode()
            observation["pose"]["x"] = float(self.moves)
            return observation

        def step(self, *args):
            self.step_entered.set()
            if not self.finish_step.wait(2):
                raise TimeoutError("test did not release the simulation step")
            return super().step(*args)

    scene, client = BlockingScene(), Pending()
    client.release.set()
    runtime = Runtime(lambda: scene, {"lightnav": "ws://test"}, client_factory=lambda *a: client)
    await runtime.initialize()
    initial_frame = runtime.state["frame_id"]
    try:
        await runtime.start("lightnav", "go")
        assert await asyncio.to_thread(scene.step_entered.wait, 1)
        stop = asyncio.create_task(runtime.stop())
        await asyncio.sleep(0.01)
        assert not stop.done(), "STOP must await the executing simulation step"
        scene.finish_step.set()
        await asyncio.wait_for(stop, 1)
        assert scene.moves == 1
        assert runtime.state["status"] == "stopped"
        assert runtime.state["step"] == 1
        assert runtime.jpeg == b"frame-1"
        assert runtime.state["pose"]["x"] == 1.0
        assert runtime.state["sim_time"] == pytest.approx(0.3)
        assert runtime.state["frame_id"] > initial_frame
    finally:
        scene.finish_step.set()
        await runtime.close()


@pytest.mark.asyncio
async def test_default_run_continues_past_200_until_model_stop():
    scene = Scene()

    class LongClient(Pending):
        async def predict(self, *args):
            return Prediction(((1.0, 0.0, 0.0),), scene.moves >= 205, 1.0)

    runtime = Runtime(
        lambda: scene, {"lightnav": "ws://test"}, client_factory=lambda *a: LongClient()
    )
    await runtime.initialize()
    try:
        await runtime.start("lightnav", "find a toilet")
        await asyncio.wait_for(runtime.task, 4)
        assert scene.moves == 205
        assert runtime.state["max_steps"] == 0
        assert runtime.state["stop_reason"] == "model_stop"
        await runtime.stop()
        assert runtime.state["stop_reason"] == "model_stop", (
            "Do not overwrite completed stop reason"
        )
        await runtime.reset()
        assert runtime.state["stop_reason"] is None
    finally:
        await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prediction,reason",
    [
        (Prediction((), False, 1.0), "empty_path"),
        (Prediction(((0.0, 0.0, 0.0),), False, 1.0), "zero_command"),
    ],
)
async def test_stop_reason_distinguishes_empty_path_and_zero_command(prediction, reason):
    class Client(Pending):
        async def predict(self, *args):
            return prediction

    runtime = Runtime(Scene, {"lightnav": "ws://test"}, client_factory=lambda *a: Client())
    await runtime.initialize()
    try:
        await runtime.start("lightnav", "go")
        await runtime.task
        assert runtime.state["stop_reason"] == reason
        assert runtime.state["step"] == 0
    finally:
        await runtime.close()
