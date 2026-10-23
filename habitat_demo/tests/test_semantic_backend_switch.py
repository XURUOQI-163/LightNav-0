import json

import pytest
from room_memory.config import load_config, load_projection_config


def test_named_backend_configs_survive_cli_switch_without_leaking_options(tmp_path):
    path = tmp_path / 'memory.json'
    path.write_text(json.dumps({
        'semantic': {'backend': 'clip'},
        'semantic_backends': {
            'clip': {'labels': ['bedroom', 'corridor'], 'templates': ['a photo of a {}.'], 'device': 'cpu'},
            'groundingdino': {'labels': ['kitchen', 'bathroom'], 'box_threshold': .3, 'device': 'cuda:0'},
        },
        'semantic_projection': {'interval': 2., 'backends': {
            'clip': {'view_mode': 'left_right', 'filter_low_confidence': True},
            'groundingdino': {'view_mode': 'detections', 'filter_low_confidence': False}}},
    }))
    _, _, clip = load_config(path, semantic_backend='clip')
    _, _, dino = load_config(path, semantic_backend='groundingdino')
    assert clip['labels'] == ['bedroom', 'corridor'] and 'box_threshold' not in clip
    assert dino['labels'] == ['kitchen', 'bathroom'] and 'templates' not in dino
    assert clip['python_executable'].endswith('/omnivla/bin/python')
    assert dino['python_executable'].endswith('/groundingdino/bin/python')
    c = load_projection_config(path, semantic_backend='clip')
    d = load_projection_config(path, semantic_backend='groundingdino')
    assert c.view_mode == 'left_right' and c.filter_low_confidence
    assert d.view_mode == 'detections' and not d.filter_low_confidence
    assert c.interval == d.interval == 2.


def test_legacy_dino_projection_switch_restores_clip_defaults(tmp_path):
    path = tmp_path / 'old.json'
    path.write_text(json.dumps({'semantic': {'backend': 'groundingdino'},
        'semantic_projection': {'view_mode': 'detections', 'filter_low_confidence': False}}))
    c = load_projection_config(path, semantic_backend='clip')
    assert c.view_mode == 'full' and c.filter_low_confidence


@pytest.mark.parametrize('backend', ['clip', 'groundingdino'])
def test_backend_specific_worker_is_selected(backend, tmp_path):
    from room_memory.config import MemoryConfig
    from room_memory.service import RoomMemoryService
    s = RoomMemoryService(MemoryConfig(), {}, {'backend': backend}, tmp_path)
    try:
        assert backend in type(s.projection).__module__
        assert s.projection.enabled
    finally:
        s.executor.shutdown(wait=True)
