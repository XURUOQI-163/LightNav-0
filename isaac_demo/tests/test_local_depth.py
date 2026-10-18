import numpy as np
from local_navigation.depth_mapping import DepthMapper
from local_navigation.models import NavigationConfig
from local_navigation.rolling_grid import RollingGrid


def observation(depth):
    h, w = depth.shape
    return dict(depth=depth, sim_time=0., pose=dict(x=0., y=0., z=0., yaw=0.),
                depth_camera=dict(width=w, height=h, fx=w/2, fy=w/2,
                                  cx=(w-1)/2, cy=(h-1)/2,
                                  camera_to_robot=[[0,0,1,0],[-1,0,0,0],
                                                   [0,-1,0,.5],[0,0,0,1]]))


def test_depth_obstacles_do_not_expire_and_invalid_depth_does_not_clear():
    cfg = NavigationConfig()
    grid = RollingGrid(cfg)
    mapper = DepthMapper(cfg, .663)
    frame = observation(np.ones((32, 32), dtype=np.float32))
    mapper.update(grid, frame)
    assert grid.hits.any()
    before = grid.hits.copy()
    frame['depth'][:] = 0
    frame['sim_time'] = 100000
    mapper.update(grid, frame)
    np.testing.assert_array_equal(before, grid.hits)


def test_matching_height_clearance_needs_new_frames_not_duplicate_reads():
    cfg = NavigationConfig()
    grid = RollingGrid(cfg)
    mapper = DepthMapper(cfg, .663)
    frame = observation(np.ones((32, 32), dtype=np.float32))
    mapper.update(grid, frame)
    ix, iy = grid.cell(1., 0.)
    old = int(grid.hits[iy, ix])
    assert old
    frame['depth'][:] = 2.
    frame['sim_time'] = 1.
    mapper.update(grid, frame)
    first = grid.hits.copy()
    mapper.update(grid, frame)
    np.testing.assert_array_equal(first, grid.hits)
    frame['sim_time'] = 2.
    mapper.update(grid, frame)
    assert int(grid.hits[iy, ix]) != old


def test_ground_is_not_obstacle_and_produces_supported_free_cells():
    depth = np.zeros((64, 64), dtype=np.float32)
    rows = np.arange(64)
    slope = (rows - 31.5) / 32
    depth[slope > 0] = (.5 / slope[slope > 0])[:, None]
    cfg = NavigationConfig()
    grid = RollingGrid(cfg)
    DepthMapper(cfg, .663).update(grid, observation(depth))
    assert not grid.hits.any()
    assert grid.free.any()
