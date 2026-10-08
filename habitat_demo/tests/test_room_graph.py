import io
from types import SimpleNamespace

from PIL import Image
from room_memory.config import MemoryConfig
from room_memory.graph import RoomGraphManager
from room_memory.models import Pose2D, RGBDFrame
from room_memory.semantics import RoomSemanticResult


def jpeg():
    out = io.BytesIO()
    Image.new("RGB", (48, 32), "grey").save(out, "JPEG")
    return out.getvalue()


class Door:
    def detect(self, depth, camera, pose):
        candidate = (
            SimpleNamespace(
                x=1.0, y=0.0, normal_yaw=0.0, width=1.0, confidence=0.8, kind="doorway_like"
            )
            if pose.x < 0.8
            else None
        )
        return SimpleNamespace(candidate=candidate, diagnostics={})


def frame(x, t, y=0, yaw=0):
    return RGBDFrame(t, jpeg(), None, Pose2D(x, y, yaw), None)


def test_crossing_and_revisit_do_not_duplicate_rooms(tmp_path):
    m = RoomGraphManager(MemoryConfig(), Door(), tmp_path)
    for i, x in enumerate([0, 0.1, 0.2, 0.5, 0.8, 1.05, 1.25]):
        m.observe(frame(x, i / 3))
    assert m.current_room_id == "R1" and len(m.portals) == 1
    for i, x in enumerate([1.0, 0.7], 7):
        m.observe(frame(x, i / 3, yaw=3.14))
    assert m.current_room_id == "R0" and len(m.rooms) == 2
    assert m.portals["P0"].traversal_count == 2
    assert m.rooms["R0"].visit_count == 2
    assert m.rooms["R1"].visited and m.rooms["R1"].observation_state in (
        "entered",
        "glanced",
        "partial",
    )


def test_looking_at_door_and_semantic_change_never_switch_room(tmp_path):
    m = RoomGraphManager(MemoryConfig(), Door(), tmp_path)
    for i in range(5):
        m.observe(frame(0.1, i / 3))
    assert len(m.rooms) == 1 and not m.portals
    room = m.rooms["R0"]
    result = RoomSemanticResult(
        "bedroom", {"bedroom": 0.9, "unknown": 0.1}, 0.9, room.representative_frame_id
    )
    m.apply_semantics("R0", result)
    assert m.rooms["R0"].room_type == "bedroom" and len(m.rooms) == 1
    m.observe(frame(1.5, 50))  # Missing observations / teleport cannot prove a crossing.
    assert len(m.rooms) == 1 and not m.portals


def test_low_score_top1_is_kept_even_with_legacy_threshold_config(tmp_path):
    m = RoomGraphManager(MemoryConfig(semantic_min_confidence=0.99), Door(), tmp_path)
    m.observe(frame(0, 0))
    room = m.rooms["R0"]
    result = RoomSemanticResult(
        "palace",
        {"palace": 0.4, "museum/indoor": 0.35, "ballroom": 0.25},
        0.4,
        room.representative_frame_ids[0],
    )
    m.apply_semantics("R0", result)
    assert room.room_type == "palace"
    assert room.semantic_confidence == 0.4
    assert m.current_room_id == "R0" and len(m.rooms) == 1


def test_bounded_trajectory_keeps_stable_indices_and_disk_history(tmp_path):
    m = RoomGraphManager(MemoryConfig(max_trajectory_points=5), Door(), tmp_path)
    for i in range(20):
        m.observe(frame(-i * 0.1, i / 3))
    assert len(m.trajectory) <= 5
    assert m.trajectory[-1].index == 19
    assert all(i >= 15 for i in m.rooms["R0"].trajectory_indices)
    assert len((tmp_path / "trajectory.jsonl").read_text().splitlines()) == 20
    assert len(m.frames["R0"]) <= 5


def test_confirmed_portal_survives_a_different_opening_before_crossing(tmp_path):
    class ChangingDoor(Door):
        def detect(self, depth, camera, pose):
            result = super().detect(depth, camera, pose)
            if 0.4 <= pose.x < 1.0:
                result.candidate = SimpleNamespace(
                    x=2.0, y=0.0, normal_yaw=0.0, width=1.0, confidence=0.8, kind="doorway_like"
                )
            return result

    m = RoomGraphManager(MemoryConfig(portal_confirm_frames=3), ChangingDoor(), tmp_path)
    for i, x in enumerate([0, 0.1, 0.2, 0.5, 0.8, 1.05, 1.25]):
        m.observe(frame(x, i / 3))
    assert len(m.portals) == 1
    assert m.portals["P0"].x == 1.0
