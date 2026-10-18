"""Opt-in real GPU/PhysX check, not collected by CPU pytest. Saves reviewable evidence."""

import argparse
import json
from pathlib import Path

import numpy as np
from isaac_web.bridge import IsaacScene


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/grscenes_carter.json")
    p.add_argument("--output", default="runs/smoke")
    p.add_argument("--startup-timeout", type=float, default=1800)
    args = p.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    scene = IsaacScene(json.loads(Path(args.config).read_text()), startup_timeout=args.startup_timeout)
    records = []
    previews = []
    def preview(frame):
        index = len(previews)
        (output / f"preview-{index:03d}.jpg").write_bytes(frame["third_jpeg"])
        previews.append(frame["sim_time"])
    scene.preview_callback = preview
    try:
        for name, command in [("start", None), ("forward", (.15, 0., 1.)),
                              ("left", (0., .5, 1.)), ("reset", None)]:
            obs = scene.reset() if command is None else scene.step(*command)
            (output / f"{name}.jpg").write_bytes(obs["jpeg"])
            (output / f"{name}-third.jpg").write_bytes(obs["third_jpeg"])
            np.save(output / f"{name}-depth.npy", obs["depth"], allow_pickle=False)
            valid = obs["depth"][obs["depth"] > 0]
            assert valid.size > obs["depth"].size * .3, "depth mostly invalid"
            records.append(dict(name=name, pose=obs["pose"], sim_time=obs["sim_time"],
                                camera=obs["depth_camera"], valid_fraction=valid.size / obs["depth"].size,
                                depth_median=float(np.median(valid)), timing=obs.get("timing")))
            print(json.dumps(records[-1]), flush=True)
        p0, p1, p2 = [r["pose"] for r in records[:3]]
        forward = ((p1["x"] - p0["x"]) * np.cos(p0["yaw"]) +
                   (p1["y"] - p0["y"]) * np.sin(p0["yaw"]))
        turn = np.arctan2(np.sin(p2["yaw"] - p1["yaw"]), np.cos(p2["yaw"] - p1["yaw"]))
        assert forward > .05, f"Carter did not move forward: {forward}"
        assert turn > .15, f"Carter did not turn left: {turn}"
        assert abs(records[-1]["pose"]["x"] - p0["x"]) < .03, "reset drift"
        (output / "report.json").write_text(json.dumps(records, indent=2))
        assert len(previews) >= 2, "no intermediate motion frames"
        assert all(b > a for a, b in zip(previews, previews[1:])), "preview time out of order"
        (output / "preview-times.json").write_text(json.dumps(previews))
        print("PASS: RGB-D, chase camera, intermediate frames, physical forward/left, reset", flush=True)
    finally:
        scene.close()


if __name__ == "__main__":
    main()
