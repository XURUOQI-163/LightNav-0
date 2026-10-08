"""Passive room graph: geometric crossings own identity; semantics only labels nodes."""

import io
import json
import math
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path

from PIL import Image

from .models import PortalEdge, ProspectiveRoom, RoomNode, TrajectoryPoint, angle_distance, distance
from .semantics import SemanticFrame


class RoomGraphManager:
    def __init__(
        self,
        config,
        detector,
        output_dir,
        semantic_enabled=False,
        vision_enabled=False,
        region_config=None,
        semantic_metadata=None,
    ):
        self.config, self.detector = config, detector
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "frames").mkdir(exist_ok=True)
        self.semantic_enabled = semantic_enabled
        self.semantic_metadata = dict(semantic_metadata or {})
        self.trajectory = []
        self.rooms = {}
        self.portals = {}
        self.frames = {}
        self.frame_poses = {}
        self.classified_visits = {}
        self.current_room_id = None
        self.current_pose = None
        self.last_timestamp = None
        self.last_persist = -float("inf")
        self.next_index = 0
        self.next_frame = 0
        self.pending = None
        self.armed = set()
        self.status = "ready"
        self.diagnostics = {}
        self.requests = []
        self.next_room = 0
        self.pose_history = deque(maxlen=config.max_trajectory_points)
        self.frontiers = None
        self.structure = None
        self.region_builder = None
        self.region_error = None
        self.region_build_ms = 0.0
        self.region_updates = 0
        self._region_signature = None
        self._saved_structure_revision = -1
        if region_config is not None:
            from .regions import RoomRegionBuilder
            from .structure import StructuralMapBuilder

            self.structure = StructuralMapBuilder(region_config)
            self.region_builder = RoomRegionBuilder(self.structure.config)
        if vision_enabled:
            from .frontiers import PortalMemory

            self.frontiers = PortalMemory(self)

    def _new_room(self, frame):
        rid = f"R{self.next_room}"
        self.next_room += 1
        self.rooms[rid] = RoomNode(
            rid,
            first_seen_time=frame.timestamp,
            last_seen_time=frame.timestamp,
            entry_pose=frame.pose,
        )
        self.frames[rid] = []
        self.frame_poses[rid] = []
        self.current_room_id = rid
        return rid

    @staticmethod
    def _coordinates(portal, pose):
        dx, dy = pose.x - portal.x, pose.y - portal.y
        c, s = math.cos(portal.normal_yaw), math.sin(portal.normal_yaw)
        return c * dx + s * dy, -s * dx + c * dy

    def _crossed(self, portal, pose, side):
        signed, lateral = self._coordinates(portal, pose)
        return (
            signed * side < -self.config.crossing_confirm_distance
            and abs(lateral) < portal.width / 2 - self.config.crossing_lateral_margin
        )

    def _reassociate(self, frame, candidate):
        # Known portal endpoints are handled first. For a new portal, require
        # several previously traversed nearby samples and consistent approach.
        # This is conservative, not a replacement for loop closure / room polygons.
        candidates = []
        for rid, room in self.rooms.items():
            if rid == self.current_room_id or isinstance(room, ProspectiveRoom):
                continue
            points = [
                p
                for p in self.trajectory
                if p.room_id == rid
                and distance(p, frame.pose) < self.config.reassociation_distance
                and angle_distance(p.yaw, frame.pose.yaw)
                < math.radians(self.config.reassociation_heading_deg)
            ]
            if len(points) < 2:
                continue
            # Do not jump across a known boundary just because poses are near.
            inconsistent = False
            for edge in self.portals.values():
                if rid in (edge.room_a, edge.room_b):
                    expected = -1 if rid == edge.room_a else 1
                    d, _ = self._coordinates(edge, frame.pose)
                    anchor, _ = self._coordinates(edge, room.entry_pose)
                    if (
                        abs(anchor) < 2
                        and abs(d) < 2
                        and d * expected < -self.config.crossing_confirm_distance
                    ):
                        inconsistent = True
            if not inconsistent:
                candidates.append((min(distance(p, frame.pose) for p in points), rid))
        if len(candidates) == 1:
            return candidates[0][1]
        return None  # Ambiguous evidence must not merge two nearby offices.

    def _switch(self, target, frame, portal):
        old = self.rooms[self.current_room_id]
        old.exit_pose = frame.pose
        old.last_seen_time = frame.timestamp
        if self.frames[old.room_id]:
            old.observation_state = "partial"
        self._request_semantics(old.room_id, force=True)
        new = self.rooms[target]
        if self.frames[target]:
            new.visit_count += 1
        new.visited = True
        new.last_seen_time = frame.timestamp
        new.topological_status = "crossing_confirmed"
        old.topological_status = "crossing_confirmed"
        for a, b in ((old, new), (new, old)):
            if b.room_id not in a.connected_room_ids:
                a.connected_room_ids.append(b.room_id)
            if portal.portal_id not in a.entry_portal_ids:
                a.entry_portal_ids.append(portal.portal_id)
        self.current_room_id = target
        self.pending = None
        self.armed = {portal.portal_id}

    def _topology(self, frame, candidate, continuous):
        cfg = self.config
        if not continuous:
            self.pending = None
            self.armed.clear()
            self.status = "pose_discontinuity"
            return False
        for edge in self.portals.values():
            if self.current_room_id not in (edge.room_a, edge.room_b):
                continue
            side = -1 if self.current_room_id == edge.room_a else 1
            signed, lateral = self._coordinates(edge, frame.pose)
            if (
                cfg.crossing_confirm_distance < signed * side < cfg.max_pose_jump * 4
                and abs(lateral) < edge.width
            ):
                self.armed.add(edge.portal_id)
            if edge.portal_id in self.armed and self._crossed(edge, frame.pose, side):
                target = edge.room_b if side == -1 else edge.room_a
                edge.traversal_count += 1
                self._switch(target, frame, edge)
                return True
        if candidate is not None:
            # Re-observing an existing portal must never create a duplicate edge.
            known = any(
                math.hypot(candidate.x - e.x, candidate.y - e.y) < cfg.portal_match_distance
                and abs(math.cos(candidate.normal_yaw - e.normal_yaw))
                > math.cos(math.radians(cfg.portal_match_angle_deg))
                for e in self.portals.values()
            )
            if not known:
                signed, _ = self._coordinates(candidate, frame.pose)
                same = (
                    self.pending is not None
                    and distance(candidate, self.pending["candidate"]) < cfg.portal_match_distance
                    and abs(math.cos(candidate.normal_yaw - self.pending["candidate"].normal_yaw))
                    > math.cos(math.radians(cfg.portal_match_angle_deg))
                )
                if same:
                    self.pending["count"] += 1
                    self.pending["last"] = frame.timestamp
                    # Keep the first measured plane: fitting it to the robot's
                    # viewpoint while crossing can drag the boundary backwards.
                elif signed < -cfg.crossing_confirm_distance and (
                    self.pending is None or self.pending["count"] < cfg.portal_confirm_frames
                ):
                    self.pending = {"candidate": candidate, "count": 1, "last": frame.timestamp}
        if self.pending is not None:
            pending = self.pending
            if frame.timestamp - pending["last"] > cfg.portal_candidate_ttl:
                self.pending = None
            elif pending["count"] >= cfg.portal_confirm_frames and self._crossed(
                pending["candidate"], frame.pose, -1
            ):
                if len(self.rooms) >= cfg.max_rooms or len(self.portals) >= cfg.max_portals:
                    self.status = "capacity_reached"
                    self.pending = None
                    return False
                source = self.current_room_id
                target = self._reassociate(frame, pending["candidate"])
                if target is None:
                    target = self._new_room(frame)
                    self.current_room_id = source
                c = pending["candidate"]
                edge = PortalEdge(
                    f"P{len(self.portals)}",
                    c.x,
                    c.y,
                    c.normal_yaw,
                    c.width,
                    source,
                    target,
                    c.confidence,
                )
                self.portals[edge.portal_id] = edge
                self._switch(target, frame, edge)
                return True
        return False

    def _trajectory(self, frame, force=False):
        cfg = self.config
        last = self.trajectory[-1] if self.trajectory else None
        keep = (
            force
            or last is None
            or distance(last, frame.pose) >= cfg.trajectory_min_translation
            or angle_distance(last.yaw, frame.pose.yaw)
            >= math.radians(cfg.trajectory_min_rotation_deg)
            or frame.timestamp - last.timestamp >= cfg.trajectory_max_interval
        )
        if not keep:
            return
        point = TrajectoryPoint(
            self.next_index,
            frame.timestamp,
            frame.pose.x,
            frame.pose.y,
            frame.pose.yaw,
            self.current_room_id,
        )
        self.next_index += 1
        self.trajectory.append(point)
        self.rooms[self.current_room_id].trajectory_indices.append(point.index)
        with (self.output_dir / "trajectory.jsonl").open("a") as f:
            f.write(json.dumps(asdict(point)) + "\n")
        if len(self.trajectory) > cfg.max_trajectory_points:
            self.trajectory = self.trajectory[-cfg.max_trajectory_points :]
            first = self.trajectory[0].index
            for room in self.rooms.values():
                room.trajectory_indices = [i for i in room.trajectory_indices if i >= first]

    def _keyframe(self, frame):
        cfg = self.config
        rid = self.current_room_id
        frames, poses = self.frames[rid], self.frame_poses[rid]
        if len(frames) >= cfg.max_representative_frames:
            return
        if poses:
            if frame.timestamp - poses[-1][0] < cfg.representative_min_interval:
                return
            # Reject duplicate views even after a revisit.
            if any(
                distance(frame.pose, p) < cfg.representative_min_translation
                and angle_distance(frame.pose.yaw, p.yaw)
                < math.radians(cfg.representative_min_rotation_deg)
                for _, p in poses
            ):
                return
        image = Image.open(io.BytesIO(frame.rgb)).convert("RGB")
        image.thumbnail((448, 448))
        output = io.BytesIO()
        image.save(output, "JPEG", quality=85)
        key = SemanticFrame(f"F{self.next_frame}", output.getvalue())
        self.next_frame += 1
        frames.append(key)
        poses.append((frame.timestamp, frame.pose))
        room = self.rooms[rid]
        room.representative_frame_ids = [f.frame_id for f in frames]
        if room.representative_frame_id is None:
            room.representative_frame_id = key.frame_id
            self._save_representative(rid, key.frame_id)
        # These are evidence states, not estimates of room coverage.
        room.observation_state = (
            "entered" if len(frames) == 1 else "glanced" if len(frames) == 2 else "partial"
        )
        self._request_semantics(rid)

    def _request_semantics(self, rid, force=False):
        room = self.rooms[rid]
        if self.classified_visits.get(rid) == room.visit_count or not self.frames[rid]:
            return
        if not force and len(self.frames[rid]) < self.config.semantic_min_frames:
            return
        if not self.semantic_enabled:
            room.semantic_status = "unconfigured"
            return
        self.classified_visits[rid] = room.visit_count
        room.semantic_status = "pending"
        self.requests.append((rid, list(self.frames[rid]), room.semantic_epoch))

    def observe_portals(self, frame, measurements):
        if self.frontiers is not None:
            self.frontiers.remember(frame, measurements)
        self._update_regions()
        if measurements:
            self.persist()
        requests, self.requests = self.requests, []
        return requests

    def _update_regions(self, frame=None):
        if self.structure is None or not self.structure.config.enabled:
            return
        start = time.perf_counter()
        try:
            updated = self.structure.observe(frame) if frame is not None else False
            signature = (
                self.structure.revision,
                tuple(
                    (p.portal_id, p.x, p.y, p.width, p.normal_yaw, p.room_a, p.room_b)
                    for p in self.portals.values()
                ),
                tuple((r.room_id, r.visited) for r in self.rooms.values()),
            )
            if signature != self._region_signature:
                regions = self.region_builder.build(
                    self.structure, self.rooms, self.portals, self.trajectory
                )
                for rid, fields in regions.items():
                    for key, value in fields.items():
                        setattr(self.rooms[rid], key, value)
                self._region_signature = signature
                self.region_updates += 1
                self.region_build_ms = (time.perf_counter() - start) * 1000
            if updated:
                self.region_error = None
        except Exception as exc:
            # Geometry is a passive optional observer; preserve the live graph
            # and report an error rather than interrupting semantic/topology work.
            self.region_error = f"{type(exc).__name__}: {str(exc)[:160]}"

    def flush(self):
        if self.current_room_id is not None:
            self._request_semantics(self.current_room_id, force=True)
        self.persist()
        requests, self.requests = self.requests, []
        return requests

    def observe(self, frame):
        if not all(
            math.isfinite(v) for v in (frame.timestamp, frame.pose.x, frame.pose.y, frame.pose.yaw)
        ):
            raise ValueError("room observation needs finite synchronized timestamp and pose")
        if self.last_timestamp is not None and frame.timestamp <= self.last_timestamp:
            return []
        if self.current_room_id is None:
            self._new_room(frame)
        continuous = self.current_pose is None or (
            distance(frame.pose, self.current_pose) <= self.config.max_pose_jump
            and frame.timestamp - self.last_timestamp <= self.config.max_frame_gap
        )
        result = self.detector.detect(frame.depth, frame.camera, frame.pose)
        self.diagnostics = result.diagnostics
        switched = (
            self.frontiers.advance(frame, continuous)
            if self.frontiers is not None
            else self._topology(frame, result.candidate, continuous)
        )
        if continuous and self.status == "pose_discontinuity":
            self.status = "ready"
        self.current_pose = frame.pose
        self.last_timestamp = frame.timestamp
        self.pose_history.append(
            dict(
                timestamp=frame.timestamp,
                pose=frame.pose,
                room_id=self.current_room_id,
                continuous=continuous,
            )
        )
        self.rooms[self.current_room_id].last_seen_time = frame.timestamp
        self._trajectory(frame, switched)
        self._update_regions(frame)
        crossing_zone = (
            self.pending is not None
            and self.pending["count"] >= self.config.portal_confirm_frames
            and abs(self._coordinates(self.pending["candidate"], frame.pose)[0])
            < self.config.crossing_confirm_distance
        )
        if not crossing_zone:
            # Do not mix doorway views from the other side into the old room's
            # semantic batch while waiting for crossing-distance hysteresis.
            self._keyframe(frame)
        if frame.timestamp - self.last_persist >= self.config.persist_interval or switched:
            self.persist()
            self.last_persist = frame.timestamp
        requests, self.requests = self.requests, []
        return requests

    def _save_representative(self, rid, fid):
        image = next(f.jpeg for f in self.frames[rid] if f.frame_id == fid)
        target = self.output_dir / "frames" / f"{rid}.jpg"
        target.write_bytes(image)

    def apply_semantics(self, rid, result=None, error=None, semantic_epoch=None):
        room = self.rooms[rid]
        if semantic_epoch is not None and room.semantic_epoch != semantic_epoch:
            return
        if error:
            room.semantic_status = "error: " + str(error)[:160]
        else:
            room.room_type_probabilities = dict(result.probabilities)
            room.semantic_confidence = result.confidence
            room.room_type = result.room_type
            room.semantic_status = "tentative" if isinstance(room, ProspectiveRoom) else "ready"
            if result.representative_frame_id in room.representative_frame_ids:
                room.representative_frame_id = result.representative_frame_id
                self._save_representative(rid, result.representative_frame_id)
        self.persist()

    def image(self, fid):
        return next(
            (f.jpeg for frames in self.frames.values() for f in frames if f.frame_id == fid), None
        )

    def snapshot(self):
        candidate = self.pending["candidate"] if self.pending else None
        return dict(
            start_pose=asdict(self.rooms["R0"].entry_pose) if "R0" in self.rooms else None,
            enabled=True,
            status=self.status,
            error=None,
            current_room_id=self.current_room_id,
            current_pose=asdict(self.current_pose) if self.current_pose else None,
            trajectory=[asdict(p) for p in self.trajectory],
            rooms=[asdict(r) for r in self.rooms.values() if not isinstance(r, ProspectiveRoom)],
            prospective_rooms=[
                asdict(r) for r in self.rooms.values() if isinstance(r, ProspectiveRoom)
            ],
            schema_version=3,
            semantic_metadata=self.semantic_metadata,
            structure=dict(
                self.structure.diagnostics(),
                error=self.region_error,
                build_ms=round(self.region_build_ms, 2),
                update_count=self.region_updates,
            )
            if self.structure is not None
            else dict(enabled=False, status="disabled"),
            portals=[asdict(p) for p in self.portals.values()],
            candidate={k: getattr(candidate, k) for k in ("x", "y", "width")}
            if candidate
            else None,
            diagnostics=self.diagnostics,
            trajectory_total=self.next_index,
            trajectory_archived=max(0, self.next_index - len(self.trajectory)),
        )

    def persist(self):
        if (
            self.structure is not None
            and self.structure.config.enabled
            and self.structure.origin is not None
            and self.structure.revision != self._saved_structure_revision
        ):
            structural = self.output_dir / "structure.npz"
            self.structure.save(structural.with_suffix(".tmp"))
            structural.with_suffix(".tmp").replace(structural)
            self._saved_structure_revision = self.structure.revision
        target = self.output_dir / "graph.json"
        temp = target.with_suffix(".tmp")
        temp.write_text(json.dumps(self.snapshot(), ensure_ascii=False, allow_nan=False))
        temp.replace(target)
