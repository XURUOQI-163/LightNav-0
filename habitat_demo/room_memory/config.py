"""All heuristic thresholds live here or in geometry defaults; JSON can override them."""

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class MemoryConfig:
    trajectory_min_translation: float = 0.05
    trajectory_min_rotation_deg: float = 5.0
    trajectory_max_interval: float = 2.0
    max_trajectory_points: int = 5000
    max_rooms: int = 200
    max_portals: int = 400
    portal_confirm_frames: int = 2
    portal_match_distance: float = 0.55
    portal_match_angle_deg: float = 30.0
    portal_candidate_ttl: float = 15.0
    crossing_confirm_distance: float = 0.20
    crossing_lateral_margin: float = 0.05
    max_pose_jump: float = 0.75
    max_frame_gap: float = 2.0
    reassociation_distance: float = 0.35
    reassociation_heading_deg: float = 70.0
    representative_min_translation: float = 0.6
    representative_min_rotation_deg: float = 25.0
    representative_min_interval: float = 0.5
    max_representative_frames: int = 5
    semantic_min_frames: int = 3
    semantic_min_confidence: float = 0.0  # Legacy config accepted, no longer filters labels.
    persist_interval: float = 1.0
    queue_size: int = 4
    semantic_queue_size: int = 8
    portal_update_alpha: float = 0.20
    portal_update_freeze_distance: float = 0.60

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name == "semantic_min_confidence" and value == 0 and not isinstance(value, bool):
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"invalid memory config: {name}")
        for name in (
            "max_trajectory_points",
            "max_rooms",
            "max_portals",
            "portal_confirm_frames",
            "max_representative_frames",
            "semantic_min_frames",
            "queue_size",
            "semantic_queue_size",
        ):
            if not isinstance(getattr(self, name), int):
                raise ValueError(f"{name} must be integer")
        if not 1 <= self.semantic_min_frames <= self.max_representative_frames <= 5:
            raise ValueError("semantic frames must be between 1 and 5")
        if not 0 <= self.semantic_min_confidence <= 1:
            raise ValueError("legacy semantic_min_confidence must be in [0,1]")
        if self.portal_confirm_frames < 2:
            raise ValueError("portal_confirm_frames must be at least 2")
        if self.portal_update_alpha > 1:
            raise ValueError("portal_update_alpha must be in (0,1]")


def load_config(path=None, semantic_backend=None):
    value = json.loads(Path(path).read_text()) if path else {}
    if not isinstance(value, dict) or set(value) - {
        "memory",
        "geometry",
        "semantic",
        "doorway",
        "regions",
        "semantic_projection",
    }:
        raise ValueError(
            "room memory config requires memory/geometry/semantic/doorway/regions objects"
        )
    for section in value.values():
        if not isinstance(section, dict):
            raise ValueError("room memory config sections must be objects")
    memory = MemoryConfig(**value.get("memory", {}))
    from .places365_recognizer import DEFAULT_PYTHON, DEFAULT_REPO

    provided = value.get("semantic", {})
    if semantic_backend is not None and semantic_backend != provided.get("backend", "places365"):
        # Backend-specific paths/environments must never leak into another provider.
        provided = {
            "backend": semantic_backend,
            **{k: provided[k] for k in ("live_enabled", "live_interval") if k in provided},
        }
    backend = provided.get("backend", "places365")
    if backend == "places365":
        semantic = {
            "backend": backend,
            "repo_path": DEFAULT_REPO,
            "python_executable": DEFAULT_PYTHON,
            "mode": "subprocess",
            "device": "cpu",
            "num_threads": 2,
            "timeout": 60.0,
            "mapping_path": None,
            "live_enabled": True,
            "live_interval": 1.0,
        }
    elif backend == "clip":
        from .clip_recognizer import DEFAULT_LABELS, DEFAULT_MODEL, DEFAULT_TEMPLATES
        from .clip_recognizer import DEFAULT_PYTHON as CLIP_PYTHON

        semantic = {
            "backend": backend,
            "model_path": DEFAULT_MODEL,
            "python_executable": CLIP_PYTHON,
            "labels": list(DEFAULT_LABELS),
            "templates": list(DEFAULT_TEMPLATES),
            "device": "cpu",
            "num_threads": 2,
            "timeout": 60.0,
            "live_enabled": True,
            "live_interval": 1.0,
        }
    elif backend == "qwen":
        semantic = {
            "backend": backend,
            "model": "qwen-vl-plus",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "timeout": 20.0,
        }
    else:
        raise ValueError("semantic.backend must be places365, clip or qwen")
    if set(provided) - set(semantic):
        raise ValueError(
            f"unknown {backend} semantic config key; API keys belong in environment variables"
        )
    semantic.update(provided)
    return memory, value.get("geometry", {}), semantic


def load_region_config(path=None):
    """Keep the existing three-return-value config interface compatible."""
    from .structure import RegionConfig

    value = json.loads(Path(path).read_text()) if path else {}
    options = value.get("regions", {})
    if not isinstance(options, dict):
        raise ValueError("regions must be a configuration object")
    try:
        return RegionConfig(**options)
    except TypeError as exc:
        raise ValueError("unknown regions configuration key") from exc


def load_doorway_config(path=None):
    """Separate loader preserves the existing three-value memory config interface."""
    value = json.loads(Path(path).read_text()) if path else {}
    default = json.loads((Path(__file__).parent / "config/default.json").read_text())["doorway"]
    provided = value.get("doorway", {})
    if not isinstance(provided, dict) or set(provided) - {
        "enabled",
        "queue_size",
        "detector",
        "depth",
    }:
        raise ValueError("unknown doorway configuration key")
    result = {**default, **provided}
    for section in ("detector", "depth"):
        if not isinstance(result[section], dict):
            raise ValueError("doorway detector/depth must be objects")
        result[section] = {**default[section], **provided.get(section, {})}
    if not isinstance(result["enabled"], bool):
        raise ValueError("doorway.enabled must be boolean")
    if type(result["queue_size"]) is not int or not 1 <= result["queue_size"] <= 8:
        raise ValueError("doorway.queue_size must be 1..8")
    return result


def load_projection_config(path=None):
    from .semantic_projection import ProjectionConfig

    value = json.loads(Path(path).read_text()) if path else {}
    try:
        return ProjectionConfig(**value.get("semantic_projection", {}))
    except TypeError as exc:
        raise ValueError("unknown semantic_projection configuration key") from exc
