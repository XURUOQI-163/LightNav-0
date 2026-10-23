import numpy as np
import pytest
from room_memory.groundingdino_recognizer import decode_regions, summarize_frames


def test_normalized_boxes_are_converted_to_original_pixels_and_centres():
    boxes = np.array([[.5, .5, .4, .6], [.9, .5, .4, 1.], [.2, .2, .1, .1]])
    scores = np.array([[.8, .1], [.1, .7], [.12, .13]])
    regions = decode_regions(boxes, scores, ['kitchen', 'bedroom'], 1000, 500, .25)
    assert len(regions) == 2
    assert regions[0]['bbox_xyxy'] == pytest.approx([300, 100, 700, 400])
    assert regions[0]['center_pixel'] == pytest.approx([500, 250])
    assert regions[0]['room_type'] == 'kitchen'
    assert regions[0]['score'] == pytest.approx(.8)
    assert regions[1]['bbox_xyxy'] == pytest.approx([700, 0, 1000, 500])
    assert regions[1]['center_pixel'] == pytest.approx([850, 250])


def test_no_detection_is_not_a_fabricated_room():
    assert decode_regions([[.5, .5, 1, 1]], [[.1, .12]], ['kitchen', 'bedroom'], 80, 60, .25) == []
    result = summarize_frames([{'frame_id': 'F0', 'detections': []}], ['kitchen', 'bedroom'])
    assert result.room_type == 'unknown' and result.confidence == 0


def test_invalid_boxes_and_scores_are_rejected():
    with pytest.raises(ValueError):
        decode_regions([[float('nan'), 0, 1, 1]], [[.5, .2]], ['kitchen', 'bedroom'], 80, 60, .25)


def test_backend_override_does_not_reuse_another_environment(tmp_path):
    import json

    from room_memory.config import load_config
    path = tmp_path / 'config.json'
    path.write_text(json.dumps({'semantic': {'backend': 'places365', 'repo_path': '/old',
        'python_executable': '/wrong/python', 'live_interval': 3.}}))
    _, _, options = load_config(path, semantic_backend='groundingdino')
    assert options['python_executable'].endswith('/groundingdino/bin/python')
    assert 'repo_path' not in options and options['live_interval'] == 3.
    assert options['device'] == 'cuda:0'
    with pytest.raises(ValueError, match='semantic.backend'):
        load_config(path, semantic_backend='not_a_backend')


@pytest.mark.asyncio
async def test_timeout_stops_worker_and_is_explicit(monkeypatch):
    import asyncio
    import io

    from PIL import Image
    from room_memory.groundingdino_recognizer import GroundingDINORecognizer
    from room_memory.semantics import SemanticClassificationError, SemanticFrame
    r = GroundingDINORecognizer(timeout=.02)
    stopped = []
    async def remote(_):
        await asyncio.Event().wait()
    async def stop():
        stopped.append(True)
    monkeypatch.setattr(r, '_remote', remote)
    monkeypatch.setattr(r, '_stop_process', stop)
    output = io.BytesIO()
    Image.new('RGB', (32, 32)).save(output, 'JPEG')
    with pytest.raises(SemanticClassificationError, match='timed out'):
        await r.detect(SemanticFrame('test', output.getvalue()))
    assert stopped
    await r.aclose()
