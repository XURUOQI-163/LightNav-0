"""Bounded observer and semantic workers; neither sends navigation commands."""

import asyncio
import contextlib
import copy
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .geometry import CameraModel, DepthPortalDetector
from .graph import RoomGraphManager
from .live_semantics import LiveSceneRecognition
from .models import HabitatGTPoseSource, RGBDFrame
from .places365_recognizer import Places365Recognizer
from .semantics import QwenRoomSemanticClassifier


class RoomMemoryService:
    def __init__(
        self,
        memory_config,
        geometry_config,
        semantic_config,
        output_root,
        pose_source=None,
        classifier=None,
        doorway_config=None,
        doorway_detector=None,
        region_config=None,
        projection_config=None,
        partition_config=None,
    ):
        from .partition import PartitionConfig

        self.partition_config = partition_config if isinstance(partition_config, PartitionConfig) else PartitionConfig(**(partition_config or {}))
        self.config = memory_config
        DepthPortalDetector(geometry_config)  # Validate before starting workers/EGL.
        self.geometry_config = geometry_config
        from .structure import RegionConfig

        if region_config is None or isinstance(region_config, RegionConfig):
            self.region_config = region_config
        else:
            try:
                self.region_config = RegionConfig(**region_config)
            except TypeError as exc:
                raise ValueError("unknown regions configuration key") from exc
        self.output_root = Path(output_root)
        self.pose_source = pose_source or HabitatGTPoseSource()
        options = dict(semantic_config)
        self.semantic_backend = options.pop("backend", "places365")
        live_enabled = options.pop(
            "live_enabled", self.semantic_backend in ("places365", "clip", "groundingdino") and classifier is None
        )
        live_interval = options.pop("live_interval", 1.0)
        if classifier is not None:
            self.classifier = classifier
            self.semantic_enabled = True
            self.semantic_backend = "custom"
        elif self.semantic_backend == "places365":
            self.classifier = Places365Recognizer(**options)
            self.semantic_enabled = True
        elif self.semantic_backend == "clip":
            from .clip_recognizer import CLIPRecognizer

            self.classifier = CLIPRecognizer(**options)
            self.semantic_enabled = True
        elif self.semantic_backend == "groundingdino":
            from .groundingdino_recognizer import GroundingDINORecognizer

            self.classifier = GroundingDINORecognizer(**options)
            self.semantic_enabled = True
        elif self.semantic_backend == "qwen":
            self.classifier = QwenRoomSemanticClassifier(**options)
            self.semantic_enabled = bool(os.environ.get("DASHSCOPE_API_KEY"))
        else:
            raise ValueError("unknown room semantic backend")
        self.live = LiveSceneRecognition(self.classifier, live_enabled and self.semantic_backend != "groundingdino", live_interval)
        describe = getattr(self.classifier, "metadata", None)
        self.semantic_metadata = describe() if describe else dict(backend=self.semantic_backend)
        from .projection_worker import SemanticProjectionWorker

        self.projection = SemanticProjectionWorker(self, projection_config)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="room-memory")
        self.doorway_config = doorway_config or {"enabled": False}
        self.doorway_detector = doorway_detector
        self.doorway_error = None
        self.doorway_status = "disabled"
        self.doorway_dropped_frames = 0
        self.doorway_diagnostics = {}
        self.doorway_queue = None
        self.depth_verifier = None
        if self.doorway_config.get("enabled"):
            from .doorway_depth import DoorwayDepthVerifier
            from .doorway_detector import YOLOEDoorwayDetector

            self.depth_verifier = DoorwayDepthVerifier(self.doorway_config.get("depth", {}))
            if self.doorway_detector is None:
                self.doorway_detector = YOLOEDoorwayDetector(
                    self.doorway_config.get("detector", {})
                )
            self.doorway_status = "loading"
        # Python 3.9 queues bind their loop at construction. aiohttp.run_app
        # creates its own loop, so construct queues inside the startup callback.
        self.queue = None
        self.semantic_queue = None
        self.manager = None
        self.generation = 0
        self.dropped_frames = 0
        self._cache = {
            "enabled": True,
            "status": "loading",
            "rooms": [],
            "trajectory": [],
            "portals": [],
        }
        self.tasks = []
        self.error = None

    async def _call(self, fn, *args):
        return await asyncio.get_running_loop().run_in_executor(self.executor, fn, *args)

    @property
    def cache(self):
        # Recompute age even when navigation is paused and geometry is unchanged.
        return dict(self._cache, live_semantic=(self.projection.live_snapshot()
            if self.semantic_backend == "groundingdino" else self.live.snapshot()),
            semantic_map=dict(self.projection.state))

    async def start(self):
        self.queue = asyncio.Queue(maxsize=self.config.queue_size)
        self.semantic_queue = asyncio.Queue(maxsize=self.config.semantic_queue_size)
        self.doorway_queue = asyncio.Queue(maxsize=self.doorway_config.get("queue_size", 2))
        await self.reset()
        self.tasks = [
            asyncio.create_task(self._worker()),
            asyncio.create_task(self._semantic_worker()),
        ]
        if self.live.enabled:
            self.tasks.append(self.live.start())
        if self.projection.enabled:
            self.tasks.append(self.projection.start())
        if self.depth_verifier is not None:
            self.tasks.append(asyncio.create_task(self._doorway_worker()))

    @staticmethod
    def _drain(queue):
        while not queue.empty():
            queue.get_nowait()
            queue.task_done()

    async def reset(self):
        self.generation += 1
        self.live.reset()
        self.projection.reset()
        self._drain(self.queue)
        self._drain(self.semantic_queue)
        self._drain(self.doorway_queue)
        if self.manager is not None:
            await self._call(self.manager.persist)
        self.session_id = uuid.uuid4().hex
        self.manager = await self._call(
            lambda: RoomGraphManager(
                self.config,
                DepthPortalDetector(self.geometry_config),
                self.output_root / self.session_id,
                self.semantic_enabled,
                vision_enabled=self.depth_verifier is not None,
                region_config=self.region_config,
                semantic_metadata=self.semantic_metadata,
            )
        )
        self.error = None
        self.dropped_frames = 0
        self.doorway_dropped_frames = 0
        self.doorway_error = None
        self.doorway_diagnostics = {}
        await self._refresh()

    def observe_camera(self, jpeg, timestamp):
        """Observe the displayed RGB independently of the latest geometric pose."""
        self.live.offer({"jpeg": jpeg, "sim_time": timestamp})

    def offer(self, observation, *, live=True):
        if live:
            self.live.offer(observation)
        if observation.get("depth") is None:
            return
        if self.queue.full():
            self.queue.get_nowait()
            self.queue.task_done()
            self.dropped_frames += 1
        if self.projection.enabled:
            observation = copy.deepcopy(observation)
        self.queue.put_nowait((self.generation, self.manager, observation))

    def _frame(self, observation):
        depth = observation["depth"]
        h, w = depth.shape
        intrinsics = observation["depth_camera"]
        extrinsic = np.asarray(intrinsics["camera_to_robot"], dtype=float)
        camera = CameraModel(
            width=w,
            height=h,
            fx=intrinsics["fx"],
            fy=intrinsics["fy"],
            cx=intrinsics["cx"],
            cy=intrinsics["cy"],
            camera_to_robot=extrinsic,
        )
        timestamp = observation["sim_time"]
        return RGBDFrame(
            timestamp,
            observation["jpeg"],
            depth,
            self.pose_source.pose_at(timestamp, observation),
            camera,
            elevation=observation.get("pose", {}).get("y")
            if isinstance(self.pose_source, HabitatGTPoseSource)
            else observation.get("floor_elevation"),
        )

    def _partition_snapshot(self, manager, evidence_map):
        from .partition import GeometricRoomPartition

        if not hasattr(manager, 'room_partition'):
            manager.room_partition = GeometricRoomPartition(self.partition_config)
        partition = manager.room_partition
        try:
            return partition.update(manager.structure, evidence_map, manager.current_pose,
                                    manager.trajectory, manager.output_dir)
        except Exception as exc:
            # A passive observer failure never stops navigation or rewrites room IDs.
            return dict(partition.state, status='error', error=f'{type(exc).__name__}: {str(exc)[:180]}')

    async def _refresh(self):
        manager, generation = self.manager, self.generation
        evidence_map = self.projection.map
        partition = await self._call(self._partition_snapshot, manager, evidence_map)
        snapshot = await self._call(manager.snapshot)
        snapshot["room_partition"] = partition
        if evidence_map is not None:
            raster = await self._call(evidence_map.snapshot)
            if generation == self.generation:
                # Newly confirmed structural walls also mask previously fused color.
                for key in ("image", "bounds", "evidence_cells"):
                    self.projection.state[key] = raster[key]
        if generation == self.generation:
            snapshot.update(
                session_id=self.session_id,
                dropped_frames=self.dropped_frames,
                semantic_status="configured" if self.semantic_enabled else "unconfigured",
                semantic_backend=self.semantic_backend,
                error=self.error,
                doorway_status=self.doorway_status,
                doorway_error=self.doorway_error,
                doorway_dropped_frames=self.doorway_dropped_frames,
                doorway_diagnostics=self.doorway_diagnostics,
            )
            self._cache = snapshot

    async def _enqueue_semantics(self, generation, manager, requests):
        for rid, frames, epoch in requests:
            if generation != self.generation:
                return
            if self.semantic_queue.full():
                await self._call(manager.apply_semantics, rid, None, "semantic queue full", epoch)
            else:
                self.semantic_queue.put_nowait((generation, manager, rid, frames, epoch))

    async def _worker(self):
        while True:
            generation, manager, observation = await self.queue.get()
            try:
                if generation != self.generation:
                    continue
                frame = self._frame(observation)
                requests = await self._call(manager.observe, frame)
                if generation == self.generation:
                    self.projection.offer(frame, manager)
                if generation == self.generation:
                    self.error = None
                await self._enqueue_semantics(generation, manager, requests)
                if self.depth_verifier is not None and generation == self.generation:
                    if self.doorway_queue.full():
                        self.doorway_queue.get_nowait()
                        self.doorway_queue.task_done()
                        self.doorway_dropped_frames += 1
                    self.doorway_queue.put_nowait((generation, manager, frame))
                await self._refresh()
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {str(exc)[:160]}"
                await self._refresh()
            finally:
                self.queue.task_done()

    async def _semantic_worker(self):
        while True:
            generation, manager, rid, frames, epoch = await self.semantic_queue.get()
            try:
                if generation != self.generation:
                    continue
                if manager.rooms[rid].semantic_epoch != epoch:
                    continue
                try:
                    result = await self.classifier.classify(frames)
                    if generation == self.generation:
                        await self._call(manager.apply_semantics, rid, result, None, epoch)
                except Exception as exc:
                    if generation == self.generation:
                        await self._call(
                            manager.apply_semantics,
                            rid,
                            None,
                            f"{type(exc).__name__}: {str(exc)[:120]}",
                            epoch,
                        )
                await self._refresh()
            except Exception as exc:
                self.error = f"Semantic worker: {type(exc).__name__}"
                await self._refresh()
            finally:
                self.semantic_queue.task_done()

    def _verify_doorways(self, manager, frame, detections):
        from dataclasses import replace

        from .models import distance

        measurements, diagnostics = [], []
        geometry = manager.detector.detect(frame.depth, frame.camera, frame.pose).candidate
        for detection in detections:
            measurement, diagnostic = self.depth_verifier.verify(frame, detection)
            diagnostics.append(diagnostic)
            if measurement is not None:
                supported = (
                    geometry is not None
                    and distance(measurement.candidate, geometry)
                    < self.config.portal_match_distance
                )
                measurements.append(replace(measurement, geometry_supported=supported))
        requests = manager.observe_portals(frame, measurements)
        return requests, dict(
            timestamp=frame.timestamp,
            rgb_detections=len(detections),
            accepted=len(measurements),
            checks=diagnostics,
        )

    async def _doorway_worker(self):
        while True:
            generation, manager, frame = await self.doorway_queue.get()
            try:
                if generation != self.generation:
                    continue
                self.doorway_status = "processing"
                detections = await self.doorway_detector.detect(frame.rgb)
                if generation != self.generation:
                    continue
                requests, diagnostics = await self._call(
                    self._verify_doorways, manager, frame, detections
                )
                if generation == self.generation:
                    self.doorway_status = "ready"
                    self.doorway_error = None
                    self.doorway_diagnostics = diagnostics
                await self._enqueue_semantics(generation, manager, requests)
                await self._refresh()
            except Exception as exc:
                if generation == self.generation:
                    self.doorway_status = "error"
                    self.doorway_error = f"{type(exc).__name__}: {str(exc)[:180]}"
                    await self._refresh()
            finally:
                self.doorway_queue.task_done()

    async def flush(self):
        try:
            await self.queue.join()
            requests = await self._call(self.manager.flush)
            await self._enqueue_semantics(self.generation, self.manager, requests)
        except Exception as exc:
            self.error = f"Memory persistence: {type(exc).__name__}"
        await self._refresh()

    async def image(self, fid):
        return await self._call(self.manager.image, fid)

    async def close(self):
        # Persist observations; shutdown must not wait for a remote model timeout.
        if self.manager is None:
            await self._close_classifier()
            self.executor.shutdown(wait=True)
            return
        await self.flush()
        for task in self.tasks:
            task.cancel()
        for task in self.tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        try:
            if self.manager is not None:
                await self._call(self.manager.persist)
        finally:
            try:
                await self._close_classifier()
            finally:
                self.executor.shutdown(wait=True)

    async def _close_classifier(self):
        if self.doorway_detector is not None:
            await self.doorway_detector.aclose()
        close = getattr(self.classifier, "aclose", None)
        if close is not None:
            await close()
