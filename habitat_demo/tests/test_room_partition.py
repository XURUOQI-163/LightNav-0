"""Geometry-only rooms, thin unseen candidates, stable IDs and cell semantics."""
from types import SimpleNamespace

import numpy as np
import pytest
from room_memory.partition import GeometricRoomPartition
from room_memory.structure import StructuralMapBuilder


def structure():
    s = StructuralMapBuilder({'grid_size': 160, 'cell_size': .1})
    s.origin = (0., 0.)
    s.coverage[20:60, 10:50] = 1
    s.coverage[20:60, 60:100] = 1
    s.coverage[36:44, 50:60] = 1  # .8 m opening
    s.revision = 1
    return s


def room_at(p, x, y):
    return int(p.owners[y, x])


def test_opening_splits_without_portal_or_semantics_and_conserves_cells():
    s = structure()
    p = GeometricRoomPartition({'cleanup_enabled': False, 'suppress_thin_regions': False})
    out = p.update(s)
    assert len(out['rooms']) == 2
    assert room_at(p, 30, 40) != room_at(p, 80, 40)
    assert np.array_equal(p.owners >= 0, s.coverage.astype(bool))
    assert len(out['portals']) == 1
    assert out['portals'][0]['source'] == 'geometry_partition'
    assert all(r['room_type'] == 'unknown' for r in out['rooms'])
    assert all(r['centroid'] and r['region_polygons'] for r in out['rooms'])


def test_thin_branches_and_tiny_disconnected_candidates_are_not_removed():
    s = structure()
    s.coverage[:] = 0
    s.coverage[60:110, 50:100] = 1
    for y in (65, 80, 95):
        s.coverage[y:y+2, 10:50] = 1
    s.coverage[130, 130] = 1
    p = GeometricRoomPartition({'cleanup_enabled': False, 'suppress_thin_regions': False})
    out = p.update(s)
    ids = {room_at(p, 15, y) for y in (65, 80, 95)}
    assert len(ids) == 3
    assert room_at(p, 75, 85) not in ids
    assert room_at(p, 130, 130) >= 0
    assert np.array_equal(p.owners >= 0, s.coverage.astype(bool))
    assert len(out['rooms']) >= 5


def test_semantics_never_affect_geometry_and_aggregate_all_class_evidence():
    s = structure()
    p = GeometricRoomPartition({'cleanup_enabled': False, 'suppress_thin_regions': False})
    p.update(s)
    before = p.owners.copy()
    evidence = SimpleNamespace(structure=s, revision=1, labels=['bedroom', 'kitchen'],
                              values=np.zeros((160, 160, 2)), weights=np.ones((160, 160)))
    evidence.values[..., 0] = .8
    evidence.values[..., 1] = .2
    out = p.update(s, evidence)
    assert all(r['room_type'] == 'bedroom' for r in out['rooms'])
    evidence.values[:] = [.1, .9]
    evidence.revision += 1
    out = p.update(s, evidence)
    assert all(r['room_type'] == 'kitchen' for r in out['rooms'])
    np.testing.assert_array_equal(p.owners, before)


def test_growth_split_merge_keep_ids_and_record_lineage():
    s = structure()
    p = GeometricRoomPartition({'cleanup_enabled': False, 'suppress_thin_regions': False})
    p.update(s)
    left, right = room_at(p, 30, 40), room_at(p, 80, 40)
    s.coverage[15:20, 10:50] = 1
    s.revision += 1
    p.update(s)
    assert room_at(p, 30, 40) == left and room_at(p, 80, 40) == right
    s.coverage[20:60, 50:60] = 1  # remove bottleneck: one open space
    s.revision += 1
    out = p.update(s)
    assert len(out['rooms']) == 1
    assert room_at(p, 30, 40) in (left, right)
    assert out['retired_room_ids']


def test_wall_and_hole_are_not_filled():
    s = structure()
    s.coverage[30:33, 30:33] = 0
    s.wall_hits[20:60, 70] = 3
    p = GeometricRoomPartition({'cleanup_enabled': False, 'suppress_thin_regions': False})
    p.update(s)
    assert (p.owners[30:33, 30:33] == -1).all()
    assert (p.owners[20:60, 70] == -1).all()


def test_diagonal_one_cell_ray_stays_one_candidate():
    s = structure()
    s.coverage[:] = 0
    s.coverage[np.arange(15, 40), np.arange(15, 40)] = 1
    p = GeometricRoomPartition({'cleanup_enabled': False, 'suppress_thin_regions': False})
    out = p.update(s)
    assert len(out['rooms']) == 1
    assert (p.owners >= 0).sum() == 25


def test_split_inherits_one_id_and_semantic_weighted_mean(tmp_path):
    s = structure()
    s.coverage[20:60, 50:60] = 1
    p = GeometricRoomPartition({'cleanup_enabled': False, 'suppress_thin_regions': False})
    p.update(s)
    old = room_at(p, 30, 40)
    s.coverage[20:36, 50:60] = 0
    s.coverage[44:60, 50:60] = 0
    s.revision += 1
    out = p.update(s, output_dir=tmp_path)
    assert len(out['rooms']) == 2
    assert old in (room_at(p, 30, 40), room_at(p, 80, 40))
    assert all(f'R{old}' in r['parent_room_ids'] for r in out['rooms'])
    assert (tmp_path/'room_partition.json').exists()
    assert (tmp_path/'room_partition.npz').exists()


@pytest.mark.asyncio
async def test_service_exposes_partition_without_changing_legacy_graph(tmp_path):
    from room_memory.config import MemoryConfig
    from room_memory.service import RoomMemoryService
    from test_room_service import observation

    service = RoomMemoryService(MemoryConfig(), {}, {'backend': 'qwen'}, tmp_path,
                                region_config={'grid_size': 256})
    await service.start()
    try:
        service.offer(observation(0))
        await service.queue.join()
        result = service.cache
        assert result['room_partition']['enabled']
        assert result['room_partition']['rooms']
        assert len(result['rooms']) == 1  # old online Portal graph remains independent
        assert (service.manager.output_dir/'room_partition.json').exists()
        await service.reset()
        assert service.cache['room_partition']['rooms'] == []
    finally:
        await service.close()


def test_two_separate_openings_between_same_rooms_get_two_portals():
    s = structure()
    s.coverage[36:44, 50:60] = 0
    s.coverage[25:31, 50:60] = 1
    s.coverage[49:55, 50:60] = 1
    p = GeometricRoomPartition({'cleanup_enabled': False, 'suppress_thin_regions': False})
    out = p.update(s)
    assert len(out['rooms']) == 2
    assert len(out['portals']) == 2
    assert len({v['portal_id'] for v in out['portals']}) == 2
