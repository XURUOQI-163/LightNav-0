"""Furniture slits must not create room area; tall openings must survive."""

import numpy as np
import pytest
from room_memory.semantic_projection import SemanticEvidenceMap
from test_room_regions import builder, cell_value, rgbd


def recess(rows):
    depth = np.full((60, 80), 2., np.float32)
    depth[rows, 30:50] = 4.
    return rgbd(depth)


def test_shallow_height_recess_stops_at_cabinet_front():
    m = builder()
    m.observe(recess(slice(23, 26)))  # Only 0.2 m tall at the back of a shelf.
    assert cell_value(m, m.coverage, 1.5, 0)
    assert not cell_value(m, m.coverage, 3., 0)


def test_tall_opening_still_extends_room_range():
    m = builder()
    m.observe(recess(slice(15, 30)))
    assert cell_value(m, m.coverage, 3., 0)


def test_separate_shelves_do_not_add_up_to_one_tall_opening():
    m = builder()
    m.observe(recess([15, 16, 22, 23, 29, 30]))
    assert not cell_value(m, m.coverage, 3., 0)


def test_invalid_rows_break_support_and_no_supported_return_means_unknown():
    depth = np.full((60, 80), np.nan, np.float32)
    depth[[15, 16, 22, 23, 29, 30]] = 4.
    m = builder()
    m.observe(rgbd(depth))
    assert not m.coverage.any()


def test_semantics_cannot_repaint_rejected_recess_even_on_old_coverage():
    m = builder()
    m.observe(rgbd())
    m.coverage[:] = 1
    semantics = SemanticEvidenceMap(m, ('bedroom', 'office room'))
    frame = recess(slice(23, 26))
    patches = semantics.project(frame)
    semantics.fuse(0, [(p, {'bedroom': 1.}) for p in patches])
    assert cell_value(m, semantics.weights, 1.5, 0) > 0
    assert cell_value(m, semantics.weights, 3., 0) == 0


def test_filter_can_be_disabled_for_comparison():
    m = builder(far_support_enabled=False)
    m.observe(recess(slice(23, 26)))
    assert cell_value(m, m.coverage, 3., 0)


@pytest.mark.parametrize('options', [
    {'far_support_enabled': 'yes'}, {'far_support_min_height': 0},
    {'far_support_min_samples': 1}, {'far_support_min_samples': 2.5},
    {'far_support_depth_tolerance': -1},
])
def test_invalid_vertical_support_config(options):
    with pytest.raises(ValueError):
        builder(**options)
