"""Start the existing navigation web UI against a headless Carter process."""

import argparse
import json
import math
from pathlib import Path
from urllib.parse import urlsplit

from aiohttp import web
from habitat_web.server import create_app

from .bridge import IsaacScene
from .geometry import IsaacGTPoseSource
from .robot_profiles import PROFILES, apply_robot_profile, room_geometry_for_robot
from .runtime import IsaacRuntime
from .stability import StabilityConfig

ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="GRScenes / Carter RGB-D language navigation")
    p.add_argument("--config", type=Path, default=ROOT / "config/grscenes_carter.json")
    p.add_argument("--scene", type=Path, help="Override USD; supply --spawn for the new scene")
    p.add_argument("--robot", type=Path)
    p.add_argument("--robot-profile", choices=PROFILES,
                   help="Carter original or half length/width with unchanged overall height")
    p.add_argument("--spawn", type=float, nargs=4, metavar=("X", "Y", "Z", "YAW_DEG"))
    p.add_argument("--gpu", type=int)
    p.add_argument("--width", type=int)
    p.add_argument("--height", type=int)
    p.add_argument("--hfov", type=float)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8091)
    p.add_argument("--lightnav-url", default="ws://127.0.0.1:8050")
    p.add_argument("--omnivla-url", default="http://127.0.0.1:8891")
    p.add_argument("--dt", type=float, default=1 / 3)
    p.add_argument('--mpc', action=argparse.BooleanOptionalAction, default=False,
                   help='LightNav: use the official MPC (fixed 0.1 s control step)')
    p.add_argument('--realtime-update', action=argparse.BooleanOptionalAction, default=True,
                   help='LightNav: predict asynchronously while tracking; disable to finish each path first')
    p.add_argument('--path-position-tolerance', type=float, default=.10, help='Path completion distance in metres')
    p.add_argument('--path-yaw-tolerance-deg', type=float, default=10., help='Path completion heading tolerance')
    p.add_argument('--path-timeout', type=float, default=60.,
                   help='Per-path simulated seconds; local batch mode requests replacement on expiry, otherwise stops')
    p.add_argument("--display-fps", type=float, help="intermediate display frames per simulated second (0 disables)")
    p.add_argument("--no-normalize-mdl-paths", action="store_true",
                   help="Disable runtime MDL path normalization for comparison")
    p.add_argument("--max-linear", type=float, default=.15)
    p.add_argument("--max-angular", type=float, default=.5)
    p.add_argument("--max-steps", type=int, default=0, help="0 = unlimited")
    p.add_argument("--timeout", type=float, default=60., help="model request wall time")
    p.add_argument("--startup-timeout", type=float, default=1800.)
    p.add_argument("--command-timeout", type=float, default=120.)
    p.add_argument("--room-memory", action="store_true")
    p.add_argument('--local-navigation', action='store_true', help='Enable spatial costmap, path correction and command guard')
    p.add_argument('--local-navigation-config', type=Path, default=ROOT / 'config/local_navigation.json')
    p.add_argument('--local-map-radius', type=float, help='Override circular window radius in metres (default 5)')
    p.add_argument("--room-memory-config", type=Path, default=ROOT / "config/room_memory.json")
    p.add_argument("--room-memory-output", type=Path, default=ROOT / "runs/room-memory")
    p.add_argument("--room-semantic-backend", choices=("clip", "groundingdino", "places365"))
    a = p.parse_args(argv)
    try:
        cfg = json.loads(a.config.read_text())
        cfg = apply_robot_profile(cfg, a.robot_profile)
        stability = StabilityConfig(**cfg["stability"])
        if a.local_navigation:
            from dataclasses import asdict

            from local_navigation.models import NavigationConfig
            options = json.loads(a.local_navigation_config.read_text())
            if a.local_map_radius is not None:
                options['window_radius_m'] = a.local_map_radius
            cfg['local_navigation'] = asdict(NavigationConfig(**options))
        elif a.local_map_radius is not None:
            raise ValueError('--local-map-radius requires --local-navigation')
        cfg.setdefault("camera_ready_max_renders", 60)
        cfg.setdefault("camera_ready_timeout_s", 30.)
        if (type(cfg["camera_ready_max_renders"]) is not int or
                not 1 <= cfg["camera_ready_max_renders"] <= 600):
            raise ValueError("camera_ready_max_renders must be an integer between 1 and 600")
        if (isinstance(cfg["camera_ready_timeout_s"], bool) or
                not math.isfinite(cfg["camera_ready_timeout_s"]) or
                not 0 < cfg["camera_ready_timeout_s"] <= 300):
            raise ValueError("camera_ready_timeout_s must be between 0 and 300 seconds")
        cfg.setdefault("normalize_mdl_paths", True)
        if type(cfg["normalize_mdl_paths"]) is not bool:
            raise ValueError("normalize_mdl_paths must be a boolean")
        if a.no_normalize_mdl_paths:
            cfg["normalize_mdl_paths"] = False
        cfg.setdefault("display_fps", 15.)
        cfg.setdefault("chase_distance", 1.8)
        cfg.setdefault("chase_height", 1.3)
        cfg.setdefault("chase_target_height", .3)
        if a.display_fps is not None:
            cfg["display_fps"] = a.display_fps
        for name in ("display_fps", "chase_distance", "chase_height", "chase_target_height"):
            if isinstance(cfg[name], bool) or not math.isfinite(cfg[name]):
                raise ValueError(f"invalid {name}")
        if not 0 <= cfg["display_fps"] <= 60:
            raise ValueError("display_fps must be between 0 and 60")
        if not (0 < cfg["chase_distance"] <= 10 and
                0 < cfg["chase_height"] <= 10 and 0 <= cfg["chase_target_height"] <= 3):
            raise ValueError("invalid chase camera distances")
        if a.scene and not a.spawn:
            raise ValueError("--scene requires --spawn X Y Z YAW_DEG; positions differ per scene")
        for name in ("scene", "robot", "gpu", "width", "height", "hfov"):
            if getattr(a, name) is not None:
                cfg[name] = str(getattr(a, name)) if name in ("scene", "robot") else getattr(a, name)
        if a.spawn:
            cfg["spawn"] = [*a.spawn[:3], math.radians(a.spawn[3])]
        for name in ("scene", "robot"):
            cfg[name] = str(Path(cfg[name]).expanduser().resolve(strict=True))
        if cfg.get("materials") and not Path(cfg["materials"]).is_dir():
            raise ValueError("materials directory not found")
        for name in ("dt", "max_linear", "max_angular", "timeout", "startup_timeout", "command_timeout",
                     'path_position_tolerance', 'path_yaw_tolerance_deg', 'path_timeout'):
            if not math.isfinite(getattr(a, name)) or getattr(a, name) <= 0:
                raise ValueError(f"invalid {name}")
        if a.path_position_tolerance > .5 or a.path_yaw_tolerance_deg > 45:
            raise ValueError('path completion tolerances must be <= 0.5 m and <= 45 degrees')
        if a.mpc:
            from .tracking import official_mpc
            try:
                official_mpc()
            except ImportError as exc:
                raise ValueError('MPC requires CasADi in ISAAC_PYTHON; existing simple mode is unchanged') from exc
        if not 0 < a.dt <= 1 or a.max_steps < 0 or not 1 <= a.port <= 65535:
            raise ValueError("invalid dt/step limit/port")
        for name in ("wheel_radius", "wheel_base", "physics_dt", "hfov"):
            if not math.isfinite(cfg[name]) or cfg[name] <= 0:
                raise ValueError(f"invalid scene config {name}")
        for name in ("width", "height", "gpu"):
            if type(cfg[name]) is not int:
                raise ValueError(f"{name} must be an integer")
        if not (1 < cfg["hfov"] < 179 and 16 <= cfg["width"] <= 1920 and
                16 <= cfg["height"] <= 1080 and cfg["gpu"] >= 0 and
                .001 <= cfg["physics_dt"] <= .05):
            raise ValueError("invalid camera/GPU/physics bounds")
        for name, length in (("spawn", 4), ("camera_offset", 3)):
            if len(cfg[name]) != length or not all(math.isfinite(x) for x in cfg[name]):
                raise ValueError(f"invalid {name}")
        for name in ("settle_steps", "render_frames"):
            if type(cfg[name]) is not int or not 1 <= cfg[name] <= 600:
                raise ValueError(f"invalid {name}")
        if not math.isfinite(cfg["floor_z"]) or not math.isfinite(cfg["ambient_light"]):
            raise ValueError("invalid floor or lighting")
        for url, schemes in ((a.lightnav_url, ("ws", "wss")), (a.omnivla_url, ("http", "https"))):
            if urlsplit(url).scheme not in schemes or not urlsplit(url).hostname:
                raise ValueError("invalid model URL scheme or hostname")
    except (OSError, KeyError, TypeError, ValueError) as exc:
        p.error(str(exc))
    requested = (a.max_linear, a.max_angular)
    a.max_linear = min(a.max_linear, stability.max_linear)
    a.max_angular = min(a.max_angular, stability.max_angular)
    if requested != (a.max_linear, a.max_angular):
        print(f"[isaac-web] {cfg['robot_profile']} dynamics limits: requested v/w={requested}, "
              f"effective v/w={(a.max_linear, a.max_angular)}", flush=True)
    return a, cfg


def main():
    a, cfg = parse_args()
    navigation = None
    if a.local_navigation:
        from local_navigation.models import NavigationConfig
        from local_navigation.service import LocalNavigation
        navigation = LocalNavigation(NavigationConfig(**cfg['local_navigation']),
                                     cfg['footprint'], cfg['robot_height'], cfg['robot_profile'])
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

        memory_cfg, geometry_cfg, semantic_cfg = load_config(
            a.room_memory_config, semantic_backend=a.room_semantic_backend
        )
        geometry_cfg, doorway_cfg = room_geometry_for_robot(
            geometry_cfg, load_doorway_config(a.room_memory_config), cfg,
        )
        memory = RoomMemoryService(
            memory_cfg, geometry_cfg, semantic_cfg, a.room_memory_output,
            pose_source=IsaacGTPoseSource(),
            doorway_config=doorway_cfg,
            region_config=load_region_config(a.room_memory_config),
            projection_config=load_projection_config(a.room_memory_config, semantic_backend=a.room_semantic_backend),
            partition_config=load_partition_config(a.room_memory_config),
        )
    runtime = IsaacRuntime(
        lambda: IsaacScene(
            cfg, startup_timeout=a.startup_timeout, command_timeout=a.command_timeout,
            progress_callback=runtime.report_startup_progress, cancel_event=runtime.startup_cancel,
        ),
        {"lightnav": a.lightnav_url, "omnivla": a.omnivla_url},
        scene_name=Path(cfg["scene"]).parent.name, dt=a.dt,
        max_linear=a.max_linear, max_angular=a.max_angular,
        max_steps=a.max_steps, timeout=a.timeout, room_memory=memory, background_startup=True,
        local_navigation=navigation,
        use_mpc=a.mpc, realtime_update=a.realtime_update,
        path_position_tolerance=a.path_position_tolerance,
        path_yaw_tolerance=math.radians(a.path_yaw_tolerance_deg), path_timeout=a.path_timeout,
    )
    runtime.state["stability"] = cfg["stability"]
    runtime.state["robot_geometry"] = {key: cfg[key] for key in (
        "robot_profile", "footprint", "robot_width", "robot_height",
        "wheel_radius", "wheel_base", "camera_offset",
    )}
    print(f"Carter profile={cfg['robot_profile']}, width={cfg['robot_width']:.3f} m, "
          f"wheel radius={cfg['wheel_radius']:.3f} m, track={cfg['wheel_base']:.3f} m, "
          f"camera offset={cfg['camera_offset']}", flush=True)
    print(f"LightNav controller={'official MPC (dt=0.1s)' if a.mpc else 'simple'}, "
          f"prediction={'asynchronous (single in-flight request)' if a.realtime_update else 'after path completion'}", flush=True)
    print(f"Loading Carter / GRScenes on GPU {cfg['gpu']}; web port {a.port} shows loading progress before RGB-D is ready.", flush=True)
    web.run_app(create_app(runtime), host=a.host, port=a.port)


if __name__ == "__main__":
    main()
