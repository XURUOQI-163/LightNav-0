"""Offline Habitat GT-path replay for room-memory diagnostics, NOT a local policy.

Only this validation script queries navmesh paths; RoomGraphManager sees RGB-D+pose.
It writes no raw depth/cloud and never contacts a model or cloud API.
"""

import argparse
import json
import math
import time
from pathlib import Path

import habitat_sim
import numpy as np
from habitat_web.simulator import HabitatScene
from room_memory.config import load_config
from room_memory.geometry import CameraModel, DepthPortalDetector
from room_memory.graph import RoomGraphManager
from room_memory.models import HabitatGTPoseSource, RGBDFrame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--scene-dataset-config", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples", default=250, type=int)
    parser.add_argument("--sample-step", default=0.04, type=float)
    args = parser.parse_args()
    scene = HabitatScene(
        args.scene, scene_dataset_config=args.scene_dataset_config, enable_depth=True
    )
    config, geometry, _ = load_config()
    manager = RoomGraphManager(config, DepthPortalDetector(geometry), args.output)
    source = HabitatGTPoseSource()
    observations = 0
    timings = []
    candidates = 0
    evidence = []
    try:
        initial = scene.observe()
        camera = CameraModel(width=640, height=480, **initial["depth_camera"])
        # Seek several long connected routes, then walk the sampled polyline and
        # reverse it. No semantic labels or true room annotations are inspected.
        paths = []
        for _ in range(15):
            route = habitat_sim.ShortestPath()
            route.requested_start = scene.position
            route.requested_end = scene.sim.pathfinder.get_random_navigable_point()
            if scene.sim.pathfinder.find_path(route) and route.geodesic_distance > 3:
                paths.append(
                    (route.geodesic_distance, [np.asarray(p).copy() for p in route.points])
                )
        if not paths:
            raise RuntimeError("no validation route found")
        points = max(paths, key=lambda p: p[0])[1]
        points = points + points[-2::-1]
        for a, b in zip(points, points[1:]):
            delta = b - a
            horizontal = math.hypot(delta[0], delta[2])
            yaw = math.atan2(-delta[0], -delta[2])
            for alpha in np.linspace(0, 1, max(2, math.ceil(horizontal / args.sample_step))):
                scene.position = a + (b - a) * alpha
                scene.yaw = yaw
                scene.sim_time += 1 / 3
                scene._set()
                obs = scene.observe()
                frame = RGBDFrame(
                    obs["sim_time"],
                    obs["jpeg"],
                    obs["depth"],
                    source.pose_at(obs["sim_time"], obs),
                    camera,
                )
                start = time.perf_counter()
                manager.observe(frame)
                timings.append((time.perf_counter() - start) * 1000)
                candidates += bool(manager.pending)
                if manager.pending:
                    pending = manager.pending
                    c = pending["candidate"]
                    signed, lateral = manager._coordinates(c, frame.pose)
                    evidence.append(
                        dict(
                            t=frame.timestamp,
                            count=pending["count"],
                            signed=signed,
                            lateral=lateral,
                            width=c.width,
                            x=c.x,
                            y=c.y,
                            pose_x=frame.pose.x,
                            pose_y=frame.pose.y,
                        )
                    )
                observations += 1
                if observations >= args.samples:
                    break
            if observations >= args.samples:
                break
        manager.flush()
        report = dict(
            observations=observations,
            rooms=len(manager.rooms),
            portals=len(manager.portals),
            candidate_frames=candidates,
            mean_memory_ms=float(np.mean(timings)),
            p95_memory_ms=float(np.percentile(timings, 95)),
            route_geodesic=max(paths, key=lambda p: p[0])[0],
            semantic_api="not_called",
            evaluation="unlabelled_GT_path_replay",
        )
        (args.output / "portal-evidence.json").write_text(json.dumps(evidence, indent=2))
        (args.output / "report.json").write_text(json.dumps(report, indent=2))
        (args.output / "last-view.jpg").write_bytes(obs["jpeg"])
        print(json.dumps(report, indent=2))
    finally:
        scene.close()


if __name__ == "__main__":
    main()
