"""Latest-only RGB-D classification; frozen patches survive motion, not session reset."""
import asyncio
import copy
import json
import time

from .semantic_projection import ROOM_COLORS, ProjectionConfig, SemanticEvidenceMap


class CLIPProjectionWorker:
    def __init__(self, service, config=None):
        self.service = service
        self.config = config if isinstance(config, ProjectionConfig) else ProjectionConfig(**(config or {}))
        if self.config.view_mode == 'detections':
            from dataclasses import replace
            self.config = replace(self.config, view_mode='full', filter_low_confidence=True)
        # Places365/room multiview/live RGB remain unchanged. This layer is CLIP only.
        self.enabled = self.config.enabled and service.semantic_backend == 'clip'
        self.queue = None
        self.generation = 0
        self.map = None
        self.reset()

    def reset(self):
        self.generation += 1
        if self.queue is not None:
            while not self.queue.empty():
                self.queue.get_nowait()
                self.queue.task_done()
        self.map = None
        self.detection_frame = None
        self.last_offered = -float('inf')
        self.last_started = 0.
        self.state = dict(enabled=self.enabled, status='waiting' if self.enabled else 'disabled',
                          error=None, completed_count=0, dropped_frames=0, view_mode=self.config.view_mode,
                          source_timestamp=None, processing_timestamp=None,
                          labels=list(self.service.semantic_metadata.get('labels', ())),
                          colors=[ROOM_COLORS.get(label, '#94a3b8')
                                  for label in self.service.semantic_metadata.get('labels', ())])

    def start(self):
        self.queue = asyncio.Queue(maxsize=1)
        return asyncio.create_task(self.run())

    def offer(self, frame, manager):
        if not self.enabled or self.queue is None or frame.timestamp <= self.last_offered:
            return
        if manager.structure is None or not manager.structure.config.enabled:
            self.state.update(status='geometry_disabled')
            return
        self.last_offered = frame.timestamp
        if self.queue.full():
            self.queue.get_nowait()
            self.queue.task_done()
            self.state['dropped_frames'] += 1
        # No shared mutable depth array, camera transform or pose can be relabelled
        # while the subprocess is busy. Keep at most in-flight + one pending frame.
        self.queue.put_nowait((self.generation, manager, copy.deepcopy(frame)))

    def _prepare(self, generation, manager, frame):
        evidence_map = self.map
        if evidence_map is None or evidence_map.structure is not manager.structure:
            evidence_map = SemanticEvidenceMap(manager.structure, self.service.semantic_metadata['labels'], self.config)
        patches = evidence_map.project(frame)
        if generation == self.generation:
            self.map = evidence_map
        return evidence_map, patches

    def _commit(self, generation, manager, evidence_map, frame, predictions):
        if generation != self.generation:
            return None
        from dataclasses import asdict

        changed = evidence_map.fuse(frame.timestamp, predictions, asdict(frame.pose))
        snapshot = evidence_map.snapshot()
        if changed:
            path = manager.output_dir / 'semantic_map.npz'
            evidence_map.save(path.with_suffix('.tmp'))
            path.with_suffix('.tmp').replace(path)
        if predictions:
            path = manager.output_dir / 'semantic_map.json'
            path.with_suffix('.tmp').write_text(json.dumps(snapshot, ensure_ascii=False, allow_nan=False))
            path.with_suffix('.tmp').replace(path)
        return snapshot

    async def run(self):
        s = self.service
        while True:
            await asyncio.sleep(max(0., self.config.interval - (time.monotonic() - self.last_started)))
            generation, manager, frame = await self.queue.get()
            try:
                if generation != self.generation:
                    continue
                self.last_started = time.monotonic()
                self.state.update(status='processing', processing_timestamp=frame.timestamp, error=None)
                evidence_map, patches = await s._call(self._prepare, generation, manager, frame)
                predictions = []
                for patch in patches:
                    if generation != self.generation:
                        break
                    result = await s.classifier.classify([patch.frame])
                    predictions.append((patch, result.probabilities))
                if generation != self.generation:
                    continue
                snapshot = await s._call(self._commit, generation, manager, evidence_map, frame, predictions)
                if generation == self.generation and snapshot is not None:
                    self.state.update(snapshot, status=snapshot['status'] if predictions else 'no_visible_depth',
                                      processing_timestamp=None, error=None,
                                      completed_count=self.state['completed_count'] + 1,
                                      latency_ms=(time.monotonic() - self.last_started) * 1000)
                    await s._refresh()
            except Exception as exc:
                if generation == self.generation:
                    self.state.update(status='error', error=f'{type(exc).__name__}: {str(exc)[:180]}',
                                      processing_timestamp=None)
                    await s._refresh()
            finally:
                self.queue.task_done()
