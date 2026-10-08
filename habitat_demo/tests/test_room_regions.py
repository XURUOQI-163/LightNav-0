"""Metric fixtures exercise observed space, boundaries and portal ownership."""

import io
import math

import numpy as np
import pytest
from PIL import Image
from room_memory.geometry import CameraModel
from room_memory.models import PortalEdge, Pose2D, ProspectiveRoom, RGBDFrame, RoomNode


def rgbd(depth=None, t=0, pose=None):
    out = io.BytesIO()
    Image.new("RGB", (80, 60), "grey").save(out, "JPEG")
    camera = CameraModel(
        80,
        60,
        40,
        40,
        39.5,
        29.5,
        np.array(
            [
                [0, 0, 1, 0],
                [-1, 0, 0, 0],
                [0, -1, 0, 0.45],
                [0, 0, 0, 1],
            ],
            dtype=float,
        ),
    )
    return RGBDFrame(
        t,
        out.getvalue(),
        np.full((60, 80), 3.0, np.float32) if depth is None else depth,
        pose or Pose2D(0, 0, 0),
        camera,
    )


def builder(**changes):
    from room_memory.structure import StructuralMapBuilder

    return StructuralMapBuilder(dict(grid_size=256, cell_size=0.05, **changes))


def cell_value(m, array, x, y):
    ix, iy = m.cell(x, y)
    return array[iy, ix]


def test_observed_fan_ends_at_surface_and_never_clears_invalid_depth():
    m = builder()
    m.observe(rgbd())
    assert cell_value(m, m.coverage, 2, 0)
    assert not cell_value(m, m.coverage, 3.5, 0)
    assert not cell_value(m, m.coverage, -1, 0)
    before = m.coverage.copy()
    for t, value in enumerate((0, np.nan, np.inf, 30), 1):
        m.observe(rgbd(np.full((60, 80), value), t))
    np.testing.assert_array_equal(m.coverage, before)


def test_wall_requires_vertical_extent_and_repeated_frames():
    m = builder()
    m.observe(rgbd())
    assert not m.walls.any()
    m.observe(rgbd(t=1))
    assert cell_value(m, m.walls, 3, 0)
    # A low cabinet visible only below 0.7 m supplies coverage, not a room wall.
    low = np.full((60, 80), np.nan)
    low[27:35] = 1.0
    n = builder()
    n.observe(rgbd(low))
    n.observe(rgbd(low, 1))
    assert n.coverage.any() and not n.walls.any()


def test_pose_rotation_calibration_and_map_bounds():
    m = builder()
    m.observe(rgbd(pose=Pose2D(2, 3, math.pi / 2)))
    assert cell_value(m, m.coverage, 2, 5)
    assert not cell_value(m, m.coverage, 4, 3)
    m.observe(rgbd(t=1, pose=Pose2D(200, 200, 0)))
    assert m.diagnostics()["status"] == "capacity_reached"
    assert m.coverage.shape == (256, 256)


def test_floor_change_and_disabled_regions_do_not_mix_geometry():
    m = builder()
    f = rgbd()
    f.elevation = 1.0
    m.observe(f)
    before = m.coverage.copy()
    f = rgbd(t=1, pose=Pose2D(1, 1, 0))
    f.elevation = 3.0
    m.observe(f)
    assert m.diagnostics()["status"] == "floor_mismatch"
    np.testing.assert_array_equal(before, m.coverage)
    off = builder(enabled=False)
    off.observe(rgbd())
    assert not off.coverage.any()


@pytest.mark.parametrize(
    "options",
    [
        {"cell_size": 0},
        {"grid_size": 100000},
        {"wall_confirm_frames": 0},
        {"enabled": "yes"},
        {"depth_max": float("nan")},
        {"unknown": 1},
        {"wall_bin_size": 1e-9},
        {"wall_padding": 1000000},
        {"seed_radius": 1000000},
    ],
)
def test_invalid_config_rejected(options):
    from room_memory.structure import StructuralMapBuilder

    with pytest.raises(ValueError):
        StructuralMapBuilder(options)


def two_rooms():
    m = builder()
    m.observe(rgbd())  # Sets origin only; the fixture below is two measured boxes.
    m.coverage[:] = 0
    m.wall_hits[:] = 0
    for x in np.arange(-2, 4, 0.05):
        for y in np.arange(-1.5, 1.5, 0.05):
            ix, iy = m.cell(x + 0.025, y + 0.025)
            m.coverage[iy, ix] = 1
    # Dividing wall at x=1, opening spans y=-.5..+.5.
    for y in np.arange(-1.5, 1.55, 0.05):
        if abs(y) > 0.5:
            ix, iy = m.cell(1, y)
            m.wall_hits[iy, ix] = 3
    rooms = {
        "R0": RoomNode("R0", entry_pose=Pose2D(0, 0, 0)),
        "R1": ProspectiveRoom("R1", candidate_pose=Pose2D(1.4, 0, 0)),
    }
    portal = PortalEdge("P0", 1, 0, 0, 1.0, "R0", "R1", 0.9, 0, "unexplored")
    return m, rooms, {"P0": portal}


def test_portal_divides_observed_space_before_entry_and_preserves_room_id():
    from room_memory.regions import RoomRegionBuilder

    m, rooms, portals = two_rooms()
    b = RoomRegionBuilder(m.config)
    regions = b.build(m, rooms, portals, [])
    assert regions["R0"]["region_area_m2"] > 6
    assert regions["R1"]["region_area_m2"] > 6
    assert max(p["x"] for r in regions["R0"]["region_polygons"] for p in r["outer"]) < 1.1
    assert min(p["x"] for r in regions["R1"]["region_polygons"] for p in r["outer"]) > 0.9
    assert regions["R1"]["boundary_state"] == "partial"
    rooms["R1"] = RoomNode("R1", entry_pose=Pose2D(1.4, 0, 0))
    promoted = b.build(m, rooms, portals, [])
    assert promoted["R1"]["region_polygons"] == regions["R1"]["region_polygons"]


def test_late_portal_repartitions_without_extending_into_unknown():
    from room_memory.regions import RoomRegionBuilder

    m, rooms, portals = two_rooms()
    b = RoomRegionBuilder(m.config)
    early = b.build(m, {"R0": rooms["R0"]}, {}, [])
    late = b.build(m, rooms, portals, [])
    assert early["R0"]["region_area_m2"] > late["R0"]["region_area_m2"]
    for region in late.values():
        assert region["boundary_state"] == "partial"
        for polygon in region["region_polygons"]:
            assert all(-2.1 <= p["x"] <= 4.1 and -1.6 <= p["y"] <= 1.6 for p in polygon["outer"])


def test_crossing_hysteresis_samples_do_not_seed_old_room_behind_portal():
    from room_memory.models import TrajectoryPoint
    from room_memory.regions import RoomRegionBuilder

    m, rooms, portals = two_rooms()
    # Graph deliberately waits until >0.20m beyond the door before switching;
    # the 0.15m point is therefore still R0, but must not seed the whole far side.
    points = [TrajectoryPoint(0, 0, 1.15, 0, 0, "R0")]
    regions = RoomRegionBuilder(m.config).build(m, rooms, portals, points)
    assert max(p["x"] for r in regions["R0"]["region_polygons"] for p in r["outer"]) < 1.1


def test_unknown_interior_remains_a_hole_not_invented_observation():
    from room_memory.regions import RoomRegionBuilder

    m, rooms, _ = two_rooms()
    a, b = m.cell(-1.5, -0.5), m.cell(-0.5, 0.5)
    m.coverage[a[1] : b[1], a[0] : b[0]] = 0
    region = RoomRegionBuilder(m.config).build(m, {"R0": rooms["R0"]}, {}, [])["R0"]
    assert any(p["holes"] for p in region["region_polygons"])
    assert region["boundary_state"] == "partial"


def test_manager_exports_regions_and_only_structural_evidence(tmp_path):
    import json

    from room_memory.config import MemoryConfig
    from room_memory.geometry import DepthPortalDetector
    from room_memory.graph import RoomGraphManager

    m = RoomGraphManager(
        MemoryConfig(),
        DepthPortalDetector({}),
        tmp_path,
        region_config={"enabled": True, "grid_size": 256},
    )
    m.observe(rgbd())
    m.observe(rgbd(t=1))
    m.flush()
    graph = json.loads((tmp_path / "graph.json").read_text())
    assert graph["rooms"][0]["region_area_m2"] > 1
    assert graph["rooms"][0]["boundary_state"] == "partial"
    assert graph["structure"]["wall_cells"] > 0
    assert graph["schema_version"] == 3
    evidence = np.load(tmp_path / "structure.npz")
    assert set(evidence.files) == {"coverage", "wall_hits", "origin", "cell_size", "revision"}
    assert "depth" not in graph and "objects" not in graph


def test_portal_callback_cannot_clear_a_structural_failure_without_valid_depth(tmp_path):
    from dataclasses import replace

    from room_memory.config import MemoryConfig
    from room_memory.geometry import DepthPortalDetector
    from room_memory.graph import RoomGraphManager

    m = RoomGraphManager(
        MemoryConfig(),
        DepthPortalDetector({}),
        tmp_path,
        region_config={"enabled": True, "grid_size": 256},
    )
    m.observe(rgbd())
    bad = rgbd(t=1)
    bad.camera = replace(bad.camera, fx=0)
    m._update_regions(bad)
    assert "calibration" in m.snapshot()["structure"]["error"]
    m.observe_portals(bad, [])
    assert "calibration" in m.snapshot()["structure"]["error"]
    m.observe(rgbd(t=2))
    assert m.snapshot()["structure"]["error"] is None


@pytest.mark.asyncio
async def test_service_region_reset_floor_guard_and_toggle(tmp_path, monkeypatch):
    from room_memory.config import MemoryConfig
    from room_memory.service import RoomMemoryService
    from test_room_service import observation

    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    service = RoomMemoryService(
        MemoryConfig(),
        {},
        {"backend": "qwen"},
        tmp_path,
        region_config={"enabled": True, "grid_size": 256},
    )
    await service.start()
    try:
        service.offer(observation(0))
        await service.queue.join()
        assert service.cache["rooms"][0]["region_polygons"]
        revision = service.cache["structure"]["revision"]
        upstairs = observation(1)
        upstairs["pose"]["y"] = 3
        service.offer(upstairs)
        await service.queue.join()
        assert service.cache["structure"]["status"] == "floor_mismatch"
        assert service.cache["structure"]["revision"] == revision
        await service.reset()
        assert not service.cache["rooms"]
        assert service.cache["structure"]["observed_area_m2"] == 0
    finally:
        await service.close()


def test_regions_config_is_independent_and_legacy_config_loads(tmp_path):
    import json

    from room_memory.config import load_config, load_region_config

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"regions": {"enabled": False}}))
    assert len(load_config(path)) == 3
    assert load_region_config(path).enabled is False
    assert load_region_config().enabled is True
