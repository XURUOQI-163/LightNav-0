import numpy as np
from room_memory.partition import GeometricRoomPartition
from room_memory.structure import StructuralMapBuilder


def test_cleanup_fills_small_holes_removes_specks_preserves_walls_and_long_arms():
    s = StructuralMapBuilder({'grid_size': 160, 'cell_size': .05})
    s.origin = (0., 0.)
    s.coverage[30:100, 40:110] = 1
    s.coverage[60:62, 10:40] = 1  # thin long branch
    s.coverage[50:52, 70:72] = 0  # small enclosed unknown hole
    s.coverage[80:82, 80:82] = 0
    s.wall_hits[80:82, 80:82] = 3  # must never fill wall hole
    s.coverage[120, 120] = 1
    s.coverage[70:72, 110:112] = 1  # tiny protrusion
    before = s.coverage.copy()
    p = GeometricRoomPartition({'fill_internal_wall_islands': False})
    p.update(s)
    assert (p.owners[50:52, 70:72] >= 0).all()
    assert (p.owners[80:82, 80:82] < 0).all()
    assert p.owners[120, 120] == -1
    assert (p.owners[60:62, 15:35] >= 0).all()
    assert p.owners[70, 111] == -1
    np.testing.assert_array_equal(s.coverage, before)


def test_cleanup_does_not_sever_narrow_connection_or_fill_large_unknown():
    s = StructuralMapBuilder({'grid_size': 160, 'cell_size': .05})
    s.origin = (0., 0.)
    s.coverage[20:60, 20:60] = 1
    s.coverage[20:60, 64:104] = 1
    s.coverage[40:42, 60:64] = 1
    s.coverage[25:55, 25:55] = 0
    p = GeometricRoomPartition()
    p.update(s)
    assert (p.owners[40:42, 60:64] >= 0).all()
    assert (p.owners[25:55, 25:55] < 0).all()


def test_internal_large_hole_filled_but_exterior_gap_and_walls_stay():
    s = StructuralMapBuilder({'grid_size':160, 'cell_size':.05})
    s.origin=(0.,0.)
    s.coverage[10:130,10:130]=1
    s.coverage[50:60,30:90]=0  # 1.5 m2 enclosed
    s.coverage[80:90,10:70]=0  # same size, open to exterior
    s.coverage[100:110,90:110]=0
    s.wall_hits[100:110,90:110]=3
    p=GeometricRoomPartition()
    p.update(s)
    assert (p.owners[50:60,30:90]>=0).all()
    assert (p.owners[82:88,20:65]<0).all()
    assert (p.owners[100:110,90:110]<0).all()


def test_long_thin_arm_is_covered_but_no_extra_portal():
    s=StructuralMapBuilder({'grid_size':160,'cell_size':.05})
    s.origin=(0.,0.)
    s.coverage[40:110,70:140]=1
    s.coverage[65:68,10:70]=1
    p=GeometricRoomPartition()
    r=p.update(s)
    assert p.owners[66,20]>=0
    assert p.owners[66,20] == p.owners[75,100]
    assert not r['portals']


def test_small_enclosed_wall_islands_are_filled_only_in_room_layer():
    s = StructuralMapBuilder({'grid_size': 160, 'cell_size': .05})
    s.origin = (0., 0.)
    s.coverage[20:120, 20:120] = 1
    s.wall_hits[60:63, 60:63] = 3
    s.coverage[60:63, 60:63] = 0
    s.wall_hits[20:70, 90] = 3  # boundary-connected partition remains
    s.wall_hits[80:100, 70] = 3  # long internal wall remains
    original = s.wall_hits.copy()
    p = GeometricRoomPartition()
    p.update(s)
    assert (p.owners[60:63, 60:63] >= 0).all()
    assert (p.owners[20:70, 90] < 0).all()
    assert (p.owners[80:100, 70] < 0).all()
    np.testing.assert_array_equal(s.wall_hits, original)
    assert not s.coverage[60:63, 60:63].any()


def test_narrow_cracks_do_not_generate_portals_but_real_door_survives():
    s = StructuralMapBuilder({'grid_size': 160, 'cell_size': .05})
    s.origin = (0., 0.)
    s.coverage[20:100, 20:60] = 1
    s.coverage[20:100, 65:105] = 1
    s.coverage[35:37, 60:65] = 1  # 10 cm crack
    s.coverage[70:82, 60:65] = 1  # 60 cm doorway
    p = GeometricRoomPartition({'cleanup_enabled': False})
    result = p.update(s)
    assert len(result['rooms']) == 2
    assert len(result['portals']) == 1
    assert result['portals'][0]['width'] >= .4


def test_new_long_wall_evidence_restores_previously_filled_island():
    s = StructuralMapBuilder({'grid_size': 160, 'cell_size': .05})
    s.origin = (0., 0.)
    s.coverage[20:120, 20:120] = 1
    s.wall_hits[60:63, 60:63] = 3
    p = GeometricRoomPartition()
    p.update(s)
    assert p.owners[61, 61] >= 0
    s.wall_hits[20:63, 60:63] = 3
    s.revision += 1
    p.update(s)
    assert p.owners[61, 61] == -1
