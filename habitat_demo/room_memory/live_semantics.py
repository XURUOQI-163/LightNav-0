"""Latest-only scene recognition for the camera view, independent of Room Graph IDs."""

import asyncio
import math
import time

from .semantics import SemanticFrame


class LiveSceneRecognition:
    def __init__(self, classifier, enabled=True, interval=1.0):
        if not isinstance(enabled, bool):
            raise ValueError("semantic.live_enabled must be boolean")
        if (
            isinstance(interval, bool)
            or not isinstance(interval, (int, float))
            or not math.isfinite(interval)
            or interval <= 0
        ):
            raise ValueError("semantic.live_interval must be finite and positive")
        self.classifier = classifier
        self.enabled = enabled
        self.interval = interval
        self.queue = None  # Python 3.9: create on the aiohttp event loop.
        self.generation = 0
        self.sequence = 0
        self.reset()

    def reset(self):
        self.generation += 1
        if self.queue is not None:
            while not self.queue.empty():
                self.queue.get_nowait()
                self.queue.task_done()
        self.last_started = 0.0
        self.result_received = None
        self.processing_started = None
        self.last_offered = None
        self.state = dict(
            enabled=self.enabled,
            status="waiting" if self.enabled else "disabled",
            room_type=None,
            confidence=None,
            top5=[],
            source_timestamp=None,
            frame_id=None,
            completed_at=None,
            latency_ms=None,
            error=None,
            completed_count=0,
            dropped_frames=0,
        )

    def start(self):
        self.queue = asyncio.Queue(maxsize=1)
        return asyncio.create_task(self.run())

    def offer(self, observation):
        if not self.enabled or self.queue is None or not observation.get("jpeg"):
            return
        key = (observation["sim_time"], observation["jpeg"])
        if key == self.last_offered and self.state["status"] != "error":
            return  # Re-publishing the same displayed image is not a new view.
        self.last_offered = key
        self.sequence += 1
        if self.queue.full():
            self.queue.get_nowait()
            self.queue.task_done()
            self.state["dropped_frames"] += 1
        self.queue.put_nowait(
            (
                self.generation,
                SemanticFrame(f"live-{self.sequence}", observation["jpeg"]),
                observation["sim_time"],
                time.monotonic(),
            )
        )

    def snapshot(self):
        now = time.monotonic()
        return dict(
            self.state,
            age_seconds=None if self.result_received is None else now - self.result_received,
            processing_seconds=None
            if self.processing_started is None
            else now - self.processing_started,
        )

    async def run(self):
        while True:
            # Sleep before taking a frame so the one pending slot stays replaceable.
            await asyncio.sleep(max(0.0, self.interval - (time.monotonic() - self.last_started)))
            generation, frame, timestamp, received = await self.queue.get()
            try:
                if generation != self.generation:
                    continue
                started = time.monotonic()
                self.last_started = self.processing_started = started
                self.state.update(status="processing", error=None)
                try:
                    # Places365's lock serializes this with multiview room jobs.
                    # Waiting for that shared worker is included in latency_ms.
                    result = await self.classifier.classify([frame])
                except Exception as exc:
                    if generation == self.generation:
                        self.state.update(
                            status="error", error=f"{type(exc).__name__}: {str(exc)[:160]}"
                        )
                    continue
                if generation != self.generation:
                    continue
                self.result_received = received
                self.state.update(
                    status="ready",
                    room_type=result.room_type,
                    confidence=result.confidence,
                    top5=[
                        dict(label=label, probability=score)
                        for label, score in sorted(
                            result.probabilities.items(), key=lambda pair: pair[1], reverse=True
                        )[:5]
                    ],
                    source_timestamp=timestamp,
                    frame_id=frame.frame_id,
                    completed_at=time.time(),
                    latency_ms=(time.monotonic() - started) * 1000,
                    completed_count=self.state["completed_count"] + 1,
                    error=None,
                )
            finally:
                if generation == self.generation:
                    self.processing_started = None
                self.queue.task_done()
