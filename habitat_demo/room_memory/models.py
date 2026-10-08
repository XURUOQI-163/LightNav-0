"""Transport-independent metric and topological records (metres, radians, seconds)."""

import math
from dataclasses import dataclass, field
from typing import Optional, Protocol


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float


class PoseSource(Protocol):
    def pose_at(self, timestamp: float, observation: dict) -> Pose2D:
        """Return the pose synchronized to this observation, or raise if unavailable."""
        ...


class HabitatGTPoseSource:
    def pose_at(self, timestamp, observation):
        if abs(timestamp - observation["sim_time"]) > 1e-6:
            raise ValueError("unsynchronized Habitat pose")
        p = observation["pose"]
        # map +X = Habitat -Z (initial forward), map +Y = Habitat -X (left).
        pose = Pose2D(-p["z"], -p["x"], p["yaw"])
        if not all(math.isfinite(v) for v in (pose.x, pose.y, pose.yaw)):
            raise ValueError("nonfinite pose")
        return pose


@dataclass
class RGBDFrame:
    timestamp: float
    rgb: bytes
    depth: object
    pose: Pose2D
    camera: object
    elevation: Optional[float] = None


@dataclass
class TrajectoryPoint:
    index: int
    timestamp: float
    x: float
    y: float
    yaw: float
    room_id: str


@dataclass
class RoomNode:
    room_id: str
    room_type: str = "unknown"
    room_type_probabilities: dict = field(default_factory=lambda: {"unknown": 1.0})
    semantic_confidence: float = 0.0
    observation_state: str = "entered"
    first_seen_time: float = 0.0
    last_seen_time: float = 0.0
    entry_pose: Optional[Pose2D] = None
    exit_pose: Optional[Pose2D] = None
    trajectory_indices: list = field(default_factory=list)
    entry_portal_ids: list = field(default_factory=list)
    connected_room_ids: list = field(default_factory=list)
    representative_frame_ids: list = field(default_factory=list)
    representative_frame_id: Optional[str] = None
    visited: bool = True
    search_count: int = 0
    visit_count: int = 1
    semantic_status: str = "collecting"
    topological_status: str = "provisional"
    semantic_epoch: int = 0
    tentative_room_type: str = "unknown"
    tentative_confidence: float = 0.0
    region_polygons: list = field(default_factory=list)
    boundary_segments: list = field(default_factory=list)
    region_area_m2: float = 0.0
    boundary_state: str = "unobserved"
    geometry_confidence: float = 0.0


@dataclass
class ProspectiveRoom(RoomNode):
    visited: bool = False
    visit_count: int = 0
    observation_state: str = "unseen"
    topological_status: str = "prospective"
    candidate_pose: Optional[Pose2D] = None


@dataclass
class PortalEdge:
    portal_id: str
    x: float
    y: float
    normal_yaw: float
    width: float
    room_a: str
    room_b: str
    confidence: float
    traversal_count: int = 1
    status: str = "explored"
    first_seen_time: float = 0.0
    last_seen_time: float = 0.0
    observation_count: int = 1
    source: str = "depth"
    geometry_supported: bool = False


def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def angle_distance(a, b):
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))
