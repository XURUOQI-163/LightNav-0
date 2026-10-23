"""Latest-only room detection with capture-bound RGB-D and frozen geometry."""
import asyncio
import copy
import json
import time
from dataclasses import asdict, replace

from .groundingdino_recognizer import annotate, summarize_frames
from .semantic_projection import ROOM_COLORS, ProjectionConfig, SemanticEvidenceMap
from .semantics import SemanticFrame


class GroundingDINOProjectionWorker:
    def __init__(self, service, config=None):
        self.service = service
        self.config = config if isinstance(config, ProjectionConfig) else ProjectionConfig(**(config or {}))
        self.config = replace(self.config, view_mode='detections', filter_low_confidence=False)
        self.enabled = service.semantic_backend == 'groundingdino'
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
        self.completed_at = None
        self.live_result = None
        self.state = dict(enabled=self.enabled and self.config.enabled,
            status='waiting' if self.enabled else 'disabled', error=None,
            completed_count=0, dropped_frames=0, view_mode='detections', detections=[],
            source_timestamp=None, processing_timestamp=None,
            labels=list(self.service.semantic_metadata.get('labels', ())),
            colors=[ROOM_COLORS.get(label, '#94a3b8') for label in self.service.semantic_metadata.get('labels', ())])

    def live_snapshot(self):
        result = self.live_result
        return dict(enabled=self.enabled, status=self.state['status'],
            room_type=result.room_type if result else None, confidence=result.confidence if result else None,
            top5=[] if result is None else [dict(label=k, probability=v) for k, v in
                sorted(result.probabilities.items(), key=lambda p: p[1], reverse=True)[:5]],
            detections=self.state['detections'], source_timestamp=self.state.get('detection_timestamp'),
            frame_id=None if self.detection_frame is None else self.detection_frame['frame_id'],
            age_seconds=None if self.completed_at is None else time.monotonic() - self.completed_at,
            latency_ms=self.state.get('latency_ms'), completed_count=self.state['completed_count'],
            error=self.state['error'], score_kind='mean_sigmoid_phrase_tokens')

    def start(self):
        self.queue = asyncio.Queue(maxsize=1)
        return asyncio.create_task(self.run())

    def offer(self, frame, manager):
        if not self.enabled or self.queue is None or frame.timestamp <= self.last_offered:
            return
        self.last_offered = frame.timestamp
        if self.queue.full():
            self.queue.get_nowait()
            self.queue.task_done()
            self.state['dropped_frames'] += 1
        self.queue.put_nowait((self.generation, manager, copy.deepcopy(frame)))

    def _prepare(self, generation, manager):
        if (not self.config.enabled or manager.structure is None
                or not manager.structure.config.enabled):
            return None, None
        evidence_map = self.map
        if evidence_map is None or evidence_map.structure is not manager.structure:
            evidence_map = SemanticEvidenceMap(manager.structure, self.service.semantic_metadata['labels'], self.config)
        frozen = copy.deepcopy(manager.structure)
        if generation == self.generation:
            self.map = evidence_map
        return evidence_map, frozen

    def _commit(self, generation, manager, evidence_map, frozen, frame, result):
        if generation != self.generation:
            return None
        predictions = []
        if evidence_map is not None:
            for detection in result['detections']:
                patches = evidence_map.project(frame, crops=[detection['bbox_xyxy']], structure=frozen)
                for patch in patches:
                    # One class vote per candidate; absolute detector score weights
                    # its evidence. Overlapping boxes share at most one frame vote.
                    patch = replace(patch, weights=patch.weights * detection['score'])
                    predictions.append((patch, {detection['room_type']: 1.}))
            changed = evidence_map.fuse(frame.timestamp, predictions, asdict(frame.pose))
            snapshot = evidence_map.snapshot()
            if changed:
                path = manager.output_dir / 'semantic_map.npz'
                evidence_map.save(path.with_suffix('.tmp'))
                path.with_suffix('.tmp').replace(path)
            path = manager.output_dir / 'semantic_map.json'
            path.with_suffix('.tmp').write_text(json.dumps(snapshot, ensure_ascii=False, allow_nan=False))
            path.with_suffix('.tmp').replace(path)
        else:
            snapshot = dict(status='geometry_disabled', enabled=False)
        annotated = annotate(frame.rgb, result['detections'])
        record = dict(result, timestamp=frame.timestamp, pose=asdict(frame.pose),
                      session_id=manager.output_dir.name)
        path = manager.output_dir / 'room_detections.json'
        path.with_suffix('.tmp').write_text(json.dumps(record, ensure_ascii=False, allow_nan=False))
        path.with_suffix('.tmp').replace(path)
        return snapshot, annotated, record

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
                evidence_map, frozen = await s._call(self._prepare, generation, manager)
                source = SemanticFrame(f'{manager.output_dir.name}-{frame.timestamp}', frame.rgb)
                result = await s.classifier.detect(source)
                if generation != self.generation:
                    continue
                committed = await s._call(self._commit, generation, manager, evidence_map, frozen, frame, result)
                if generation == self.generation and committed is not None:
                    snapshot, jpeg, record = committed
                    self.detection_frame = dict(jpeg=jpeg, frame_id=source.frame_id,
                                                timestamp=frame.timestamp, record=record)
                    self.live_result = summarize_frames([result], s.semantic_metadata['labels'])
                    self.completed_at = time.monotonic()
                    self.state.update(snapshot, status=('ready' if result['detections'] else 'no_detections'),
                        detections=result['detections'], detection_timestamp=frame.timestamp,
                        processing_timestamp=None, error=None,
                        completed_count=self.state['completed_count'] + 1,
                        latency_ms=(self.completed_at - self.last_started) * 1000)
                    await s._refresh()
            except Exception as exc:
                if generation == self.generation:
                    self.state.update(status='error', error=f'{type(exc).__name__}: {str(exc)[:180]}',
                                      processing_timestamp=None)
                    await s._refresh()
            finally:
                self.queue.task_done()
