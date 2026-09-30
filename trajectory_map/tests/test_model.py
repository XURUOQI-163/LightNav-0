import math

import pytest
from lightnav_trajectory import Pose2D, TrajectoryMap


def pose(t, x=10.0, y=20.0, yaw=0.0, frame="world", source="test"):
    return Pose2D(t, x, y, yaw, frame, source)


def test_origin_translation_current_endpoint_and_rotation():
    trail = TrajectoryMap()
    assert trail.snapshot()["current"] is None
    assert trail.update(pose(0), received_s=0)
    trail.update(pose(1, 10.1, 20.2), received_s=0.1)
    trail.update(pose(2, 10.11, 20.2, math.pi / 2), received_s=0.2)
    snap = trail.snapshot(now_s=0.3)
    assert snap["origin"] == [10.0, 20.0]
    assert snap["start"] == [0.0, 0.0]
    assert snap["current"]["x"] == pytest.approx(0.11)
    assert snap["current"]["yaw"] == pytest.approx(math.pi / 2)
    assert snap["segments"][0][-1] == pytest.approx([0.11, 0.2])
    assert snap["status"] == "live"


def test_stationary_samples_do_not_grow_path_but_refresh_freshness():
    trail = TrajectoryMap()
    for t in range(100):
        trail.update(pose(t, yaw=t / 10), received_s=t)
    snap = trail.snapshot(now_s=99.5)
    assert snap["point_count"] == 1
    assert snap["status"] == "live"
    assert trail.snapshot(now_s=102)["status"] == "stale"


@pytest.mark.parametrize(
    "bad",
    [
        pose(1, x=float("nan")),
        pose(1, yaw=float("inf")),
        pose(-1),
        pose(True),
        pose(1, frame=""),
        pose(1, source=""),
    ],
)
def test_invalid_observations_do_not_mutate_state(bad):
    trail = TrajectoryMap()
    trail.update(pose(0), received_s=0)
    with pytest.raises(ValueError):
        trail.update(bad, received_s=0.1)
    assert trail.snapshot(now_s=0.1)["revision"] == 1


def test_out_of_order_and_frame_switch_require_reset():
    trail = TrajectoryMap()
    trail.update(pose(10), received_s=0)
    assert not trail.update(pose(10, x=11), received_s=1)
    assert not trail.update(pose(9, x=11), received_s=1)
    with pytest.raises(ValueError):
        trail.update(pose(11, frame="odom"), received_s=1)
    assert trail.snapshot(now_s=2)["status"] == "stale"
    old_session = trail.snapshot()["session_id"]
    trail.reset()
    trail.update(pose(0, x=100, frame="odom"), received_s=3)
    snap = trail.snapshot(now_s=3)
    assert snap["session_id"] != old_session
    assert snap["segments"] == [[[0, 0]]]


def test_jump_breaks_path_without_losing_start():
    trail = TrajectoryMap()
    trail.update(pose(0), received_s=0)
    trail.update(pose(1, 10.1), received_s=1)
    trail.update(pose(2, 30), received_s=2)
    snap = trail.snapshot(now_s=2)
    assert len(snap["segments"]) == 2
    assert snap["segments"][0][-1] == pytest.approx([0.1, 0])
    assert snap["segments"][1] == [[20, 0]]
    assert snap["start"] == [0, 0]


def test_capacity_decimates_old_points_preserves_full_extent():
    trail = TrajectoryMap(max_points=16)
    for t in range(300):
        trail.update(pose(t, x=10 + t * 0.1), received_s=t)
    snap = trail.snapshot(now_s=299)
    assert snap["point_count"] <= 16
    assert snap["segments"][0][0] == [0, 0]
    assert snap["segments"][0][-1] == pytest.approx([29.9, 0])
    assert snap["decimated"]


def test_excessive_discontinuities_are_bounded_not_joined():
    trail = TrajectoryMap(max_points=8)
    for t in range(30):
        trail.update(pose(t, x=t * 10), received_s=t)
    snap = trail.snapshot(now_s=29)
    assert snap["point_count"] <= 8
    assert snap["history_full"]
    assert snap["current"]["x"] == 290
    assert all(len(segment) == 1 for segment in snap["segments"])


def test_snapshots_are_detached_and_source_change_rejected():
    trail = TrajectoryMap()
    trail.update(pose(0), received_s=0)
    snap = trail.snapshot(now_s=0)
    snap["segments"][0][0][0] = 900
    assert trail.snapshot(now_s=0)["start"] == [0, 0]
    with pytest.raises(ValueError):
        trail.update(pose(1, source="other"), received_s=1)
