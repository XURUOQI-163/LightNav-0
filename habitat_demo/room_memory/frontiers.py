"""Visible-portal lifecycle layered onto the existing RoomGraphManager.

Only measured RGB-D observations create prospective nodes. Semantics never
creates topology. Recent synchronized poses handle asynchronous detector delay.
"""

import io
import math

from PIL import Image

from .models import PortalEdge, Pose2D, ProspectiveRoom, RoomNode, distance
from .semantics import SemanticFrame


class PortalMemory:
    def __init__(self, manager):
        self.manager = manager
        self.anchors = {}

    def remember(self, frame, measurements):
        m = self.manager
        cfg = m.config
        source = next(
            (p for p in m.pose_history if abs(p["timestamp"] - frame.timestamp) < 1e-6), None
        )
        if source is None:
            return  # No matching observation/expired history is not latest pose.
        for measurement in measurements:
            c = measurement.candidate
            matches = [
                e
                for e in m.portals.values()
                if distance(c, e) < cfg.portal_match_distance
                and abs(math.cos(c.normal_yaw - e.normal_yaw))
                > math.cos(math.radians(cfg.portal_match_angle_deg))
                and abs(c.width - e.width) < max(0.35, e.width * 0.5)
                and source["room_id"] in (e.room_a, e.room_b)
            ]
            if matches:
                portal = min(matches, key=lambda e: distance(c, e))
                if frame.timestamp <= portal.last_seen_time:
                    continue
                # Freeze the crossing geometry near the threshold; otherwise updates
                # can move a plane through a stationary robot and invent a crossing.
                if (
                    abs(m._coordinates(portal, m.current_pose)[0])
                    > cfg.portal_update_freeze_distance
                ):
                    alpha = cfg.portal_update_alpha
                    portal.x += (c.x - portal.x) * alpha
                    portal.y += (c.y - portal.y) * alpha
                    yaw = c.normal_yaw
                    if math.cos(yaw - portal.normal_yaw) < 0:
                        yaw += math.pi
                    delta = math.atan2(
                        math.sin(yaw - portal.normal_yaw), math.cos(yaw - portal.normal_yaw)
                    )
                    portal.normal_yaw += alpha * delta
                    portal.width += (c.width - portal.width) * alpha
                portal.confidence += (c.confidence - portal.confidence) * cfg.portal_update_alpha
                portal.last_seen_time = frame.timestamp
                portal.observation_count += 1
                portal.geometry_supported |= measurement.geometry_supported
                room = m.rooms[portal.room_b]
                if isinstance(room, ProspectiveRoom):
                    room.last_seen_time = frame.timestamp
                    room.candidate_pose = self._candidate_pose(portal)
                continue
            if len(m.rooms) >= cfg.max_rooms or len(m.portals) >= cfg.max_portals:
                m.status = "capacity_reached"
                continue
            signed, _ = m._coordinates(c, frame.pose)
            if signed >= -cfg.crossing_confirm_distance:
                continue
            rid = f"R{m.next_room}"
            m.next_room += 1
            pid = f"P{len(m.portals)}"
            portal = PortalEdge(
                pid,
                c.x,
                c.y,
                c.normal_yaw,
                c.width,
                source["room_id"],
                rid,
                c.confidence,
                traversal_count=0,
                status="unexplored",
                first_seen_time=frame.timestamp,
                last_seen_time=frame.timestamp,
                source="yoloe",
                geometry_supported=measurement.geometry_supported,
            )
            room = ProspectiveRoom(
                rid,
                first_seen_time=frame.timestamp,
                last_seen_time=frame.timestamp,
                candidate_pose=self._candidate_pose(portal),
                entry_portal_ids=[pid],
            )
            m.rooms[rid] = room
            m.frames[rid] = []
            m.frame_poses[rid] = []
            m.portals[pid] = portal
            self._roi(rid, frame, measurement.bbox_xyxy)
            # Use a real observation as the initial side witness.
            self.anchors[pid] = (source["timestamp"], source["pose"], source["room_id"])
        # A result may arrive after crossing. Walk only continuous recorded poses,
        # beginning at its original source time. Never stamp it with a newer pose.
        for record in list(m.pose_history):
            if record["timestamp"] < frame.timestamp:
                continue
            f = type(frame)(record["timestamp"], b"", None, record["pose"], None)
            if self.advance(f, record["continuous"]):
                self._repair_recent_labels(record["timestamp"])
        m.persist()

    @staticmethod
    def _candidate_pose(portal):
        return Pose2D(
            portal.x + 0.35 * math.cos(portal.normal_yaw),
            portal.y + 0.35 * math.sin(portal.normal_yaw),
            portal.normal_yaw,
        )

    def _roi(self, rid, frame, box):
        m = self.manager
        with Image.open(io.BytesIO(frame.rgb)) as image:
            x1, y1, x2, y2 = box
            pad = 0.10
            crop = image.convert("RGB").crop(
                (
                    x1 + (x2 - x1) * pad,
                    y1 + (y2 - y1) * pad,
                    x2 - (x2 - x1) * pad,
                    y2 - (y2 - y1) * pad,
                )
            )
            crop.thumbnail((448, 448))
            out = io.BytesIO()
            crop.save(out, "JPEG", quality=85)
        key = SemanticFrame(f"F{m.next_frame}", out.getvalue())
        m.next_frame += 1
        m.frames[rid] = [key]
        m.rooms[rid].representative_frame_ids = [key.frame_id]
        m.rooms[rid].representative_frame_id = key.frame_id
        m._save_representative(rid, key.frame_id)
        m._request_semantics(rid, force=True)

    def _promote(self, rid, frame):
        m = self.manager
        old = m.rooms[rid]
        if not isinstance(old, ProspectiveRoom):
            return
        m.rooms[rid] = RoomNode(
            rid,
            first_seen_time=old.first_seen_time,
            last_seen_time=frame.timestamp,
            entry_pose=frame.pose,
            entry_portal_ids=list(old.entry_portal_ids),
            semantic_epoch=old.semantic_epoch + 1,
            tentative_room_type=old.room_type,
            tentative_confidence=old.semantic_confidence,
        )
        # ROI is evidence from outside. Formal semantics starts from entered views.
        m.frames[rid] = []
        m.frame_poses[rid] = []
        m.classified_visits.pop(rid, None)

    def advance(self, frame, continuous):
        m = self.manager
        cfg = m.config
        if not continuous:
            self.anchors.clear()
            return False
        switched = False
        for portal in m.portals.values():
            if m.current_room_id not in (portal.room_a, portal.room_b):
                continue
            side = -1 if m.current_room_id == portal.room_a else 1
            anchor = self.anchors.get(portal.portal_id)
            signed, lateral = m._coordinates(portal, frame.pose)
            if signed * side > cfg.crossing_confirm_distance:
                self.anchors[portal.portal_id] = (frame.timestamp, frame.pose, m.current_room_id)
                continue
            if anchor is None or frame.timestamp <= anchor[0] or anchor[2] != m.current_room_id:
                continue
            a, al = m._coordinates(portal, anchor[1])
            if a * side <= cfg.crossing_confirm_distance:
                continue
            # Check intersection of the actual traversed segment at the doorway,
            # not merely the robot's endpoint after a large lateral movement.
            if signed * side >= -cfg.crossing_confirm_distance:
                continue
            fraction = a / (a - signed)
            intersection = al + (lateral - al) * fraction
            if abs(intersection) >= portal.width / 2 - cfg.crossing_lateral_margin:
                continue
            if abs(lateral) >= portal.width / 2:
                continue
            target = portal.room_b if side == -1 else portal.room_a
            self._promote(target, frame)
            portal.status = "explored"
            portal.traversal_count += 1
            m._switch(target, frame, portal)
            self.anchors[portal.portal_id] = (frame.timestamp, frame.pose, target)
            switched = True
        return switched

    def _repair_recent_labels(self, timestamp):
        m = self.manager
        for record in m.pose_history:
            if record["timestamp"] >= timestamp:
                record["room_id"] = m.current_room_id
        for point in m.trajectory:
            if point.timestamp >= timestamp:
                point.room_id = m.current_room_id
        for room in m.rooms.values():
            room.trajectory_indices = []
        for point in m.trajectory:
            m.rooms[point.room_id].trajectory_indices.append(point.index)
        # Only asynchronous crossings need historical correction; rewrite streamed
        # JSONL atomically so disk and browser agree without retaining all history.
        path = m.output_dir / "trajectory.jsonl"
        if path.exists():
            import json

            temp = path.with_suffix(".tmp")
            with path.open() as source, temp.open("w") as target:
                for line in source:
                    point = json.loads(line)
                    if point["timestamp"] >= timestamp:
                        point["room_id"] = m.current_room_id
                    target.write(json.dumps(point) + "\n")
            temp.replace(path)
