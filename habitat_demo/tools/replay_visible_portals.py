"""Replay recorded poses with real Habitat RGB-D and YOLOE, without a VLN policy.

Run from habitat_demo with PYTHONPATH=. in the Habitat environment. Output contains
the graph and scalar detections; raw depth/point clouds are never archived.
"""

import argparse
import asyncio
import json
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
from habitat_web.simulator import HabitatScene
from room_memory.config import load_config, load_doorway_config, load_region_config
from room_memory.doorway_depth import DoorwayDepthVerifier
from room_memory.doorway_detector import YOLOEDoorwayDetector
from room_memory.geometry import CameraModel, DepthPortalDetector
from room_memory.graph import RoomGraphManager
from room_memory.models import HabitatGTPoseSource, RGBDFrame
from room_memory.places365_recognizer import Places365Recognizer


async def replay(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    memory, geometry, semantic = load_config(args.config)
    visual = load_doorway_config(args.config)
    detector = YOLOEDoorwayDetector(visual["detector"])
    verifier = DoorwayDepthVerifier(visual["depth"])
    semantic.pop("backend")
    classifier = Places365Recognizer(**semantic) if args.semantics else None
    manager = RoomGraphManager(
        memory,
        DepthPortalDetector(geometry),
        output,
        semantic_enabled=classifier is not None,
        vision_enabled=True,
        region_config=load_region_config(args.config),
    )
    points = [json.loads(line) for line in Path(args.trajectory).read_text().splitlines()]
    scene = HabitatScene(
        args.scene, scene_dataset_config=args.scene_dataset_config, enable_depth=True
    )
    pose_source = HabitatGTPoseSource()
    rows = []

    async def classify(requests):
        if classifier is not None:
            for rid, frames, epoch in requests:
                manager.apply_semantics(
                    rid, await classifier.classify(frames), semantic_epoch=epoch
                )

    try:
        for i, point in enumerate(points):
            scene.position = np.array(
                [-point["y"], point.get("elevation", scene.start[1]), -point["x"]]
            )
            scene.yaw = point["yaw"]
            scene.sim_time = point["timestamp"]
            scene._set()
            obs = scene.observe()
            camera = CameraModel(
                width=obs["depth"].shape[1], height=obs["depth"].shape[0], **obs["depth_camera"]
            )
            frame = RGBDFrame(
                obs["sim_time"],
                obs["jpeg"],
                obs["depth"],
                pose_source.pose_at(obs["sim_time"], obs),
                camera,
                elevation=float(scene.position[1]),
            )
            await classify(manager.observe(frame))
            start = time.monotonic()
            detections = await detector.detect(frame.rgb)
            elapsed = time.monotonic() - start
            measurements, checks = [], []
            for detection in detections:
                measured, check = verifier.verify(frame, detection)
                checks.append(dict(box=list(detection.bbox_xyxy), label=detection.label, **check))
                if measured:
                    measurements.append(measured)
            await classify(manager.observe_portals(frame, measurements))
            row = dict(
                index=i,
                timestamp=frame.timestamp,
                seconds=elapsed,
                checks=checks,
                current_room=manager.current_room_id,
                portals=[asdict(p) for p in manager.portals.values()],
                structure=manager.snapshot()["structure"],
            )
            rows.append(row)
            if args.save_rgb and detections:
                (output / f"view{i:03d}.jpg").write_bytes(frame.rgb)
            print(
                json.dumps(
                    dict(
                        index=i,
                        room=manager.current_room_id,
                        portals=len(manager.portals),
                        checks=checks,
                    )
                ),
                flush=True,
            )
        await classify(manager.flush())
        summary = dict(
            frames=len(rows),
            rooms=len(manager.snapshot()["rooms"]),
            prospective_rooms=len(manager.snapshot()["prospective_rooms"]),
            portals=len(manager.portals),
            explored_portals=sum(p.status == "explored" for p in manager.portals.values()),
            reasons=dict(Counter(c["reason"] for r in rows for c in r["checks"])),
            median_detection_ms=1000 * float(np.median([r["seconds"] for r in rows[1:]])),
            structure=manager.snapshot()["structure"],
            region_areas={r.room_id: r.region_area_m2 for r in manager.rooms.values()},
        )
        (output / "detections.json").write_text(json.dumps(rows, indent=2))
        (output / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)
    finally:
        scene.close()
        await detector.aclose()
        if classifier is not None:
            await classifier.aclose()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--scene-dataset-config", required=True)
    parser.add_argument("--config", default="room_memory/config/default.json")
    parser.add_argument("--output", required=True, help="New directory, never overwrite a session")
    parser.add_argument("--save-rgb", action="store_true")
    parser.add_argument("--semantics", action="store_true")
    asyncio.run(replay(parser.parse_args()))


if __name__ == "__main__":
    main()
