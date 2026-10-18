"""Projection evidence must stay at capture coordinates, independent of Room IDs."""
import math

import numpy as np
import pytest
from room_memory.models import Pose2D
from room_memory.semantic_projection import SemanticEvidenceMap
from test_room_regions import builder, rgbd


def setup_map(**options):
    structure = builder()
    structure.observe(rgbd())
    structure.coverage[:] = 1
    structure.wall_hits[:] = 0
    return structure, SemanticEvidenceMap(structure, ('kitchen', 'bedroom'), options)


def evidence(m, x, y):
    ix, iy = m.structure.cell(x, y)
    return m.values[iy, ix], m.weights[iy, ix]


def merge(m, frame, probabilities):
    patches = m.project(frame)
    m.fuse(frame.timestamp, [(p, probabilities) for p in patches])
    return patches


def test_depth_and_known_walls_limit_projection_without_any_portals():
    s, m = setup_map()
    frame = rgbd(np.full((60, 80), 1.5, np.float32))
    merge(m, frame, {'kitchen': .9, 'bedroom': .1})
    assert evidence(m, 1., 0)[1] > 0
    assert evidence(m, 2., 0)[1] == 0
    assert evidence(m, -1., 0)[1] == 0
    s, m = setup_map()
    ix, _ = s.cell(1, 0)
    s.wall_hits[:, ix] = 3
    merge(m, rgbd(), {'kitchen': 1., 'bedroom': 0.})
    assert evidence(m, .7, 0)[1] > 0
    assert evidence(m, 1.5, 0)[1] == 0


def test_turn_does_not_recolor_previous_room_and_movement_keeps_world_cells():
    _, m = setup_map()
    old = rgbd(t=1)
    patches = m.project(old)
    # Result arrives after a different view has been projected: apply OLD patch.
    m.project(rgbd(t=2, pose=Pose2D(3, 3, math.pi / 2)))
    m.fuse(1, [(p, {'kitchen': 1., 'bedroom': 0.}) for p in patches])
    before = evidence(m, 1., 0)[0].copy()
    merge(m, rgbd(t=3, pose=Pose2D(0, 0, math.pi)), {'kitchen': 0., 'bedroom': 1.})
    np.testing.assert_array_equal(evidence(m, 1., 0)[0], before)
    assert evidence(m, -1., 0)[0][1] == pytest.approx(1)
    assert evidence(m, 3., 4.)[1] == 0


def test_multiview_fuses_whole_distribution_instead_of_last_top1():
    _, m = setup_map()
    merge(m, rgbd(t=0), {'kitchen': .8, 'bedroom': .2})
    merge(m, rgbd(t=1), {'kitchen': .2, 'bedroom': .8})
    np.testing.assert_allclose(evidence(m, 1., 0)[0], [.5, .5], atol=1e-6)
    old = m.weights.copy()
    merge(m, rgbd(t=1), {'kitchen': 1., 'bedroom': 0.})
    np.testing.assert_array_equal(m.weights, old)  # duplicate frame does not vote twice


@pytest.mark.parametrize('depth', [0., np.nan, np.inf, 100.])
def test_invalid_depth_never_paints_unknown_area(depth):
    _, m = setup_map()
    merge(m, rgbd(np.full((60, 80), depth)), {'kitchen': 1., 'bedroom': 0.})
    assert not m.weights.any()


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


def test_projection_has_no_three_metre_cap_and_uses_long_term_coverage():
    s, m = setup_map()
    frame = rgbd(np.full((60, 80), 5.8, np.float32))
    merge(m, frame, {'kitchen': .9, 'bedroom': .1})
    assert evidence(m, 4.5, 1.)[1] > 0
    assert evidence(m, 6.1, 1.)[1] == 0
    s, m = setup_map()
    ix, iy = s.cell(4.5, 1.)
    s.coverage[iy, ix] = 0
    merge(m, frame, {'kitchen': .9, 'bedroom': .1})
    assert evidence(m, 4.5, 1.)[1] == 0


def test_low_close_top3_is_rejected_before_fusion_without_erasing_old_evidence():
    from room_memory.clip_recognizer import DEFAULT_LABELS

    s, _ = setup_map()
    m = SemanticEvidenceMap(s, DEFAULT_LABELS)
    patches = m.project(rgbd())
    low = dict(zip(DEFAULT_LABELS, [.19, .18, .17, .12, .11, .09, .08, .06]))
    assert not m.fuse(0, [(p, low) for p in patches])
    assert not m.weights.any()
    assert m.snapshot()['rejected_views'] == 1
    assert all(v['reason'] == 'low_or_ambiguous_top3' for v in m.snapshot()['views'])
    assert m.snapshot()['image'] is None
    high = {label: float(label == 'kitchen') for label in DEFAULT_LABELS}
    assert m.fuse(1, [(p, high) for p in patches])
    weights, values = m.weights.copy(), m.values.copy()
    assert not m.fuse(2, [(p, low) for p in patches])
    np.testing.assert_array_equal(m.weights, weights)
    np.testing.assert_array_equal(m.values, values)
    assert m.snapshot()['source_timestamp'] == 1  # last accepted evidence
    assert m.snapshot()['filter_timestamp'] == 2


@pytest.mark.parametrize('scores', [
    [.32, .20, .16, .1, .08, .06, .05, .03],  # low top1, but clearly separated
    [.36, .35, .29, 0, 0, 0, 0, 0],  # close top3, but not all low
])
def test_filter_rejects_either_low_or_close_without_erasing_history(scores):
    from room_memory.clip_recognizer import DEFAULT_LABELS

    s, _ = setup_map()
    m = SemanticEvidenceMap(s, DEFAULT_LABELS)
    patches = m.project(rgbd())
    predictions = [(p, dict(zip(DEFAULT_LABELS, scores))) for p in patches]
    assert not m.fuse(0, predictions)
    assert not m.weights.any()
    assert m.snapshot()['rejected_views'] == 1
    high = {label: float(label == 'kitchen') for label in DEFAULT_LABELS}
    assert m.fuse(1, [(p, high) for p in patches])
    weights, values = m.weights.copy(), m.values.copy()
    assert not m.fuse(2, predictions)
    np.testing.assert_array_equal(m.weights, weights)
    np.testing.assert_array_equal(m.values, values)
    assert m.snapshot()['source_timestamp'] == 1


def test_filter_is_per_view_not_whole_frame():
    from room_memory.clip_recognizer import DEFAULT_LABELS

    s, _ = setup_map()
    m = SemanticEvidenceMap(s, DEFAULT_LABELS, {'view_mode': 'left_right'})
    left, right = m.project(rgbd())
    strong = {label: float(label == 'kitchen') for label in DEFAULT_LABELS}
    weak = dict(zip(DEFAULT_LABELS, [.19, .18, .17, .12, .11, .09, .08, .06]))
    assert m.fuse(0, [(left, strong), (right, weak)])
    assert evidence(m, 1.5, .9)[1] > 0
    assert evidence(m, 1.5, -.9)[1] == 0
    assert m.snapshot()['accepted_views'] == m.snapshot()['rejected_views'] == 1


def test_low_confidence_filter_can_be_disabled_and_legacy_radius_does_not_limit():
    from room_memory.clip_recognizer import DEFAULT_LABELS

    s, _ = setup_map()
    m = SemanticEvidenceMap(s, DEFAULT_LABELS, {'filter_low_confidence': False, 'max_range': 3.})
    patches = m.project(rgbd(np.full((60, 80), 5.8, np.float32)))
    assert m.fuse(0, [(p, dict.fromkeys(DEFAULT_LABELS, 1 / 8)) for p in patches])
    assert evidence(m, 4.5, 1.)[1] > 0
    assert m.snapshot()['rejected_views'] == 0


@pytest.mark.parametrize('options', [
    {'filter_low_confidence': 'true'}, {'low_top1_threshold': -1},
    {'top3_spread_threshold': 1.1}, {'low_top1_threshold': float('nan')},
])
def test_reject_invalid_filter_configuration(options):
    from room_memory.semantic_projection import ProjectionConfig

    with pytest.raises(ValueError):
        ProjectionConfig(**options)


def test_semantics_reaches_far_structure_above_near_furniture():
    depth = np.full((60, 80), np.nan, np.float32)
    depth[20:26] = 4.0  # far wall seen above a nearer low obstacle
    depth[27:33] = 1.0
    frame = rgbd(depth)
    s = builder()
    s.observe(frame)
    m = SemanticEvidenceMap(s, ('kitchen', 'bedroom'))
    merge(m, frame, {'kitchen': 1., 'bedroom': 0.})
    for x in (2., 3.5, 3.95):
        ix, iy = s.cell(x, .5)
        assert s.coverage[iy, ix]
        assert evidence(m, x, .5)[1] > 0
    assert evidence(m, 4.2, .2)[1] == 0


def test_semantics_uses_structural_height_depth_and_sampling_options():
    # Only a high wall sample, missed by the old semantic height limit/row grid.
    depth = np.full((60, 80), np.nan, np.float32)
    depth[9] = 4.0
    frame = rgbd(depth)
    s = builder(height_max=2.8, sample_rows=60, depth_max=8.)
    s.observe(frame)
    # Old projection-specific settings must not override the structural domain.
    m = SemanticEvidenceMap(s, ('kitchen', 'bedroom'),
                            {'height_max': 1.6, 'depth_max': 2., 'sample_rows': 8})
    merge(m, frame, {'kitchen': 1., 'bedroom': 0.})
    ix, iy = s.cell(3.5, .5)
    assert s.coverage[iy, ix]
    assert evidence(m, 3.5, .5)[1] > 0


def test_farthest_projection_still_respects_captured_depth_not_old_coverage():
    s = builder()
    s.observe(rgbd(np.full((60, 80), 5., np.float32)))
    m = SemanticEvidenceMap(s, ('kitchen', 'bedroom'))
    captured = rgbd(np.full((60, 80), 2., np.float32), t=1)
    patches = m.project(captured)
    captured.pose = Pose2D(3, 3, math.pi)
    assert m.fuse(1, [(p, {'kitchen': 1., 'bedroom': 0.}) for p in patches])
    assert evidence(m, 1., .2)[1] > 0
    assert evidence(m, 3.5, .2)[1] == 0


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


@pytest.mark.parametrize('mode', ['left', '', None, 1])
def test_view_mode_rejects_invalid_values(mode):
    from room_memory.semantic_projection import ProjectionConfig

    with pytest.raises(ValueError):
        ProjectionConfig(view_mode=mode)
