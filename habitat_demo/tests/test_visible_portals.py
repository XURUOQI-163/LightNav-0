import io

from PIL import Image
from room_memory.config import MemoryConfig
from room_memory.doorway_depth import VisualPortalObservation
from room_memory.geometry import PortalCandidate
from room_memory.graph import RoomGraphManager
from room_memory.models import Pose2D, RGBDFrame
from room_memory.semantics import RoomSemanticResult
from test_room_graph import Door


def frame(x, t, y=0, yaw=0):
    out = io.BytesIO()
    Image.new("RGB", (160, 120), "grey").save(out, "JPEG")
    return RGBDFrame(t, out.getvalue(), None, Pose2D(x, y, yaw), None)


def visual(x=1.0, y=0.0, yaw=0.0):
    return VisualPortalObservation(
        PortalCandidate(x, y, yaw, 1.0, 0.8, "visual_opening"), (30, 10, 90, 100), 0.8, 0.9
    )


def manager(path):
    return RoomGraphManager(
        MemoryConfig(), Door(), path, semantic_enabled=True, vision_enabled=True
    )


def test_visible_opening_creates_unvisited_room_before_crossing_and_deduplicates(tmp_path):
    m = manager(tmp_path)
    f = frame(0, 0)
    m.observe(f)
    requests = m.observe_portals(f, [visual()])
    s = m.snapshot()
    assert len(s["rooms"]) == 1 and len(s["prospective_rooms"]) == 1
    prospective = s["prospective_rooms"][0]
    p = s["portals"][0]
    assert prospective["room_id"] == "R1" and not prospective["visited"]
    assert prospective["entry_pose"] is None and prospective["visit_count"] == 0
    assert p["status"] == "unexplored" and p["traversal_count"] == 0
    assert m.current_room_id == "R0" and len(requests) == 1
    for i in range(1, 5):
        f = frame(0.05 * i, i / 3)
        m.observe(f)
        m.observe_portals(f, [visual(1.02)])
    assert len(m.portals) == 1 and len(m.rooms) == 2
    assert m.portals["P0"].observation_count == 5
    assert 1 < m.portals["P0"].x < 1.03


def test_crossing_promotes_same_room_return_reuses_it_and_stale_roi_cannot_overwrite(tmp_path):
    m = manager(tmp_path)
    f = frame(0, 0)
    m.observe(f)
    req = m.observe_portals(f, [visual()])[0]
    rid, frames, epoch = req
    for i, x in enumerate([0.3, 0.6, 0.9, 1.1, 1.3], 1):
        m.observe(frame(x, i / 3))
    assert m.current_room_id == rid == "R1"
    assert m.portals["P0"].status == "explored" and m.portals["P0"].traversal_count == 1
    assert not m.snapshot()["prospective_rooms"]
    assert m.rooms[rid].visited and m.rooms[rid].visit_count == 1
    m.apply_semantics(
        rid,
        RoomSemanticResult("office", {"office": 1}, 1, frames[0].frame_id),
        semantic_epoch=epoch,
    )
    assert m.rooms[rid].room_type == "unknown"
    for i, x in enumerate([1.0, 0.7], 6):
        m.observe(frame(x, i / 3, yaw=3.14159))
    assert m.current_room_id == "R0" and len(m.rooms) == 2
    assert m.portals["P0"].traversal_count == 2


def test_delayed_visual_result_uses_source_pose_and_recognizes_already_completed_crossing(tmp_path):
    m = manager(tmp_path)
    source = frame(0, 0)
    m.observe(source)
    for i, x in enumerate([0.3, 0.6, 0.9, 1.1, 1.3], 1):
        m.observe(frame(x, i / 3))
    m.observe_portals(source, [visual()])
    assert m.current_room_id == "R1"
    assert m.portals["P0"].x == 1.0
    assert m.rooms["R1"].entry_pose.x == 1.3
    assert m.trajectory[-1].room_id == "R1"


def test_rotation_or_teleport_is_not_a_crossing(tmp_path):
    m = manager(tmp_path)
    f = frame(0, 0)
    m.observe(f)
    m.observe_portals(f, [visual()])
    m.observe(frame(0, 0.3, yaw=3.14))
    m.observe(frame(1.3, 20))
    assert m.current_room_id == "R0"
    assert m.portals["P0"].status == "unexplored"


def test_two_visible_neighbor_doors_keep_distinct_candidates_and_tentative_labels(tmp_path):
    m = manager(tmp_path)
    f = frame(0, 0)
    m.observe(f)
    requests = m.observe_portals(f, [visual(2, 0), visual(2, 1.5)])
    assert len(m.portals) == 2 and len(m.snapshot()["prospective_rooms"]) == 2
    rid, frames, epoch = requests[0]
    m.apply_semantics(
        rid,
        RoomSemanticResult("office", {"office": 1}, 1, frames[0].frame_id),
        semantic_epoch=epoch,
    )
    assert m.rooms[rid].room_type == "office" and not m.rooms[rid].visited
    assert m.rooms[rid].semantic_status == "tentative"
    assert m.current_room_id == "R0"
