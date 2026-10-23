import copy

import numpy as np
from room_memory.models import Pose2D
from room_memory.semantic_projection import SemanticEvidenceMap
from test_room_regions import builder, rgbd


def test_detection_bbox_only_projects_corresponding_capture_sector():
    s = builder()
    frame = rgbd(t=1)
    s.observe(frame)
    s.coverage[:] = 1
    frozen = copy.deepcopy(s)
    m = SemanticEvidenceMap(s, ('kitchen', 'bedroom'))
    # An inference result arrives after the live geometry and pose have changed.
    s.wall_hits[:] = 3
    patches = m.project(frame, crops=[(0, 0, 40, 60)], structure=frozen)
    assert len(patches) == 1
    indices = patches[0].indices
    ix, iy = frozen.cell(1.5, .9)
    assert iy * s.config.grid_size + ix in indices
    ix, iy = frozen.cell(1.5, -.9)
    assert iy * s.config.grid_size + ix not in indices
    frame.pose = Pose2D(4, 4, 3.)
    m.fuse(1, [(patches[0], {'kitchen': 1.})])
    assert m.weights.any()


def test_detection_with_no_valid_depth_does_not_paint():
    s = builder()
    s.observe(rgbd())
    m = SemanticEvidenceMap(s, ('kitchen', 'bedroom'))
    assert m.project(rgbd(np.zeros((60, 80))), crops=[(0, 0, 40, 60)]) == []
    assert m.project(rgbd(), crops=[]) == []
