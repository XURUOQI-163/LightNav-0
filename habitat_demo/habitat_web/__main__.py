import argparse
import json
import math
from pathlib import Path
from urllib.parse import urlsplit

from aiohttp import web

from .runtime import Runtime
from .server import create_app
from .simulator import HabitatScene


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Headless Habitat RGB and language navigation viewer")
    p.add_argument(
        "--scene", type=Path, required=True, help="Existing Habitat scene .glb with navmesh"
    )
    p.add_argument(
        "--scene-dataset-config",
        type=Path,
        help="Optional .scene_dataset_config.json (use the matching config for HM3D)",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--lightnav-url", default="ws://127.0.0.1:8050")
    p.add_argument("--omnivla-url", default="http://127.0.0.1:8891")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--room-memory", action="store_true", help="Observe RGB-D and build a passive room graph"
    )
    p.add_argument(
        "--room-memory-config", type=Path, help="Room geometry/memory/semantic JSON config"
    )
    p.add_argument("--room-memory-output", type=Path, default=Path("runs/room-memory"))
    p.add_argument(
        "--room-semantic-backend",
        choices=("clip", "groundingdino", "places365"),
        help="Override room classifier; other memory/geometry settings stay unchanged",
    )
    p.add_argument(
        "--dt", type=float, default=1 / 3, help="Simulated action duration per prediction (s)"
    )
    p.add_argument("--max-linear", type=float, default=0.15, help="m/s")
    p.add_argument("--max-angular", type=float, default=0.5, help="rad/s")
    p.add_argument(
        "--max-steps", type=int, default=0, help="Action limit; 0 means unlimited (default)"
    )
    p.add_argument(
        "--timeout", type=float, default=30.0, help="Model request timeout (wall seconds)"
    )
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--camera-height", type=float, default=0.45)
    p.add_argument("--hfov", type=float, default=90.0)
    a = p.parse_args(argv)
    if not a.scene.is_file():
        p.error("scene file not found")
    if a.scene_dataset_config is not None:
        try:
            a.scene_dataset_config = a.scene_dataset_config.resolve(strict=True)
            dataset = json.loads(a.scene_dataset_config.read_text(encoding="utf-8"))
            if not isinstance(dataset, dict):
                raise ValueError("expected a JSON object")
        except (OSError, ValueError) as exc:
            p.error(f"invalid scene dataset config: {exc}")
    for key in ("dt", "max_linear", "max_angular", "timeout", "camera_height", "hfov"):
        if not math.isfinite(getattr(a, key)) or getattr(a, key) <= 0:
            p.error(f"{key} must be positive and finite")
    if not (
        0 < a.dt <= 1
        and 1 < a.hfov < 179
        and 1 <= a.port <= 65535
        and a.max_steps >= 0
        and 16 <= a.width <= 1920
        and 16 <= a.height <= 1080
    ):
        p.error("invalid control, port or camera bounds")
    for url, schemes in [(a.lightnav_url, ("ws", "wss")), (a.omnivla_url, ("http", "https"))]:
        if urlsplit(url).scheme not in schemes or not urlsplit(url).hostname:
            p.error("invalid model URL scheme or hostname")
    return a


def main():
    a = parse_args()
    memory = None
    if a.room_memory:
        from room_memory.config import (
            load_config,
            load_doorway_config,
            load_partition_config,
            load_projection_config,
            load_region_config,
        )
        from room_memory.service import RoomMemoryService

        try:
            memory_config, geometry_config, semantic_config = load_config(
                a.room_memory_config, semantic_backend=a.room_semantic_backend
            )
            memory = RoomMemoryService(
                memory_config,
                geometry_config,
                semantic_config,
                a.room_memory_output,
                doorway_config=load_doorway_config(a.room_memory_config),
                region_config=load_region_config(a.room_memory_config),
                projection_config=load_projection_config(a.room_memory_config, semantic_backend=a.room_semantic_backend),
                partition_config=load_partition_config(a.room_memory_config),
            )
        except (OSError, ValueError, TypeError) as exc:
            raise SystemExit(f"Invalid room memory configuration: {exc}") from exc
    runtime = Runtime(
        lambda: HabitatScene(
            a.scene,
            a.seed,
            a.width,
            a.height,
            a.camera_height,
            a.hfov,
            scene_dataset_config=a.scene_dataset_config,
            enable_depth=a.room_memory,
        ),
        {"lightnav": a.lightnav_url, "omnivla": a.omnivla_url},
        scene_name=a.scene.name,
        dt=a.dt,
        max_linear=a.max_linear,
        max_angular=a.max_angular,
        max_steps=a.max_steps,
        timeout=a.timeout,
        room_memory=memory,
    )
    web.run_app(create_app(runtime), host=a.host, port=a.port)


if __name__ == "__main__":
    main()
