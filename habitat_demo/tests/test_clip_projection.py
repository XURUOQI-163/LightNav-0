import math

import numpy as np
import pytest
from room_memory.models import Pose2D
from test_semantic_projection import evidence, rgbd, setup_map


def test_view_sectors_can_have_different_labels_and_persist(tmp_path):
    _, m = setup_map(view_mode='left_right')
    patches = m.project(rgbd())
    assert len(patches) == 2
    assert [p.crop for p in patches] == [(0, 0, 40, 60), (40, 0, 80, 60)]
    m.fuse(0, [(p, {'kitchen': float(i != 1), 'bedroom': float(i == 1)})
               for i, p in enumerate(patches)])
    assert evidence(m, 1.5, .9)[0][0] > .9  # optical left = robot left
    assert evidence(m, 1.5, -.9)[0][1] > .9
    m.save(tmp_path / 'semantics.npz')
    with np.load(tmp_path / 'semantics.npz', allow_pickle=False) as saved:
        assert saved['probabilities'].shape[1] == 2
        np.testing.assert_allclose(saved['probabilities'].sum(axis=1), 1, atol=1e-6)
    snapshot = m.snapshot()
    assert snapshot['image'].startswith('data:image/png;base64,')
    assert snapshot['source_timestamp'] == 0


@pytest.mark.asyncio
async def test_service_binds_delayed_result_to_capture_and_reset_discards_it(tmp_path, monkeypatch):
    import asyncio

    from room_memory import clip_recognizer
    from room_memory.config import MemoryConfig
    from room_memory.semantics import RoomSemanticResult
    from room_memory.service import RoomMemoryService
    from test_room_service import observation

    class Classifier:
        def __init__(self, **_):
            self.started, self.release = asyncio.Event(), asyncio.Event()

        def metadata(self):
            return {'labels': ['kitchen', 'bedroom'], 'backend': 'clip'}

        async def classify(self, frames):
            self.started.set()
            await self.release.wait()
            return RoomSemanticResult('kitchen', {'kitchen': 1., 'bedroom': 0.}, 1., frames[0].frame_id)

    monkeypatch.setattr(clip_recognizer, 'CLIPRecognizer', Classifier)
    service = RoomMemoryService(MemoryConfig(), {}, {'backend': 'clip', 'live_enabled': False},
                                tmp_path, region_config={'grid_size': 256},
                                projection_config={'interval': .1})
    await service.start()
    try:
        original = observation(0)
        service.offer(original)
        original['depth'][:] = 0  # sensor buffer reused before the observer runs
        original['pose']['x'] = 40
        await service.queue.join()
        await asyncio.wait_for(service.classifier.started.wait(), 2)
        evidence_map = service.projection.map
        original['depth'][:] = 0  # caller reuses sensor buffer after capture
        service.manager.current_pose = Pose2D(4, 4, math.pi)
        service.classifier.release.set()
        await asyncio.wait_for(service.projection.queue.join(), 3)
        assert evidence(evidence_map, 1., 0)[1] > 0
        assert evidence(evidence_map, 4., 3.)[1] == 0
        assert service.cache['semantic_map']['source_timestamp'] == 0
        assert service.cache['semantic_map']['view_mode'] == 'full'
        assert len(service.cache['semantic_map']['views']) == 1
        assert service.cache['semantic_map']['source_pose'] == {'x': 0., 'y': 0., 'yaw': 0.}
        assert service.cache['portals'] == []
        assert (service.manager.output_dir / 'semantic_map.npz').is_file()
        # RGB-only display updates must never use the last available depth/pose.
        count = service.cache['semantic_map']['completed_count']
        service.observe_camera(original['jpeg'], 100)
        await asyncio.sleep(.05)
        assert service.cache['semantic_map']['completed_count'] == count

        service.classifier.started.clear()
        service.classifier.release.clear()
        service.offer(observation(1))
        await service.queue.join()
        await asyncio.wait_for(service.classifier.started.wait(), 2)
        await service.reset()
        service.classifier.release.set()
        await asyncio.wait_for(service.projection.queue.join(), 3)
        assert service.cache['semantic_map']['status'] == 'waiting'
        assert service.cache['semantic_map']['source_timestamp'] is None
        assert service.projection.map is None
    finally:
        service.classifier.release.set()
        await service.close()


def test_default_full_image_preserves_both_sides_and_uses_one_distribution():
    import io

    from PIL import Image

    _, m = setup_map()
    frame = rgbd()
    pixels = np.zeros((60, 80, 3), np.uint8)
    pixels[:, :40, 0] = 255
    pixels[:, 40:, 2] = 255
    out = io.BytesIO()
    Image.fromarray(pixels).save(out, 'JPEG')
    frame.rgb = out.getvalue()
    patches = m.project(frame)
    assert len(patches) == 1
    assert patches[0].crop == (0, 0, 80, 60)
    sent = np.array(Image.open(io.BytesIO(patches[0].frame.jpeg)))
    assert sent[40, 5, 0] > 220 and sent[40, 75, 2] > 220
    assert m.fuse(0, [(patches[0], {'kitchen': .9, 'bedroom': .1})])
    for y in (-.9, 0, .9):
        assert evidence(m, 1.5, y)[1] > 0
        np.testing.assert_allclose(evidence(m, 1.5, y)[0], [.9, .1], atol=1e-6)
    assert evidence(m, 1.5, 0)[1] > evidence(m, 1.5, .9)[1]
    assert m.snapshot()['view_mode'] == 'full'
