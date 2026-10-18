"""Opt-in check against an already running Isaac web service; moves/resets Carter."""

import argparse
import io
import json
import math
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8091")
    parser.add_argument("--output", type=Path, default=Path("runs/smoke/http"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    opener = build_opener(ProxyHandler({}))

    def get(path):
        return opener.open(args.url + path, timeout=180).read()

    def post(path, payload):
        request = Request(args.url + path, data=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"})
        with opener.open(request, timeout=180) as response:
            assert json.load(response)["ok"]

    records = []
    post("/api/stop", {})
    for name, action in [("start", None), ("forward", (.15, 0.)),
                         ("left", (0., .5)), ("reset", None)]:
        if action is None:
            post("/api/reset", {})
        else:
            post("/api/manual", dict(linear=action[0], angular=action[1], duration=1.))
        packet = get("/api/observation.npz")
        (args.output / f"{name}.npz").write_bytes(packet)
        (args.output / f"{name}.jpg").write_bytes(get("/api/rgb.jpg"))
        with np.load(io.BytesIO(packet), allow_pickle=False) as obs:
            valid = obs["depth"] > 0
            record = dict(name=name, pose=obs["pose"].tolist(),
                          sim_time=float(obs["sim_time"]), K=obs["K"].tolist(),
                          valid_fraction=float(valid.mean()),
                          depth_median=float(np.median(obs["depth"][valid])))
            assert record["valid_fraction"] > .3, "depth mostly invalid"
            assert np.isfinite(obs["depth"]).all()
            assert obs["rgb"].shape == (*obs["depth"].shape, 3)
        records.append(record)
        print(json.dumps(record), flush=True)
    p0, p1, p2, p3 = [r["pose"] for r in records]
    forward = (p1[0] - p0[0]) * math.cos(p0[3]) + (p1[1] - p0[1]) * math.sin(p0[3])
    turn = math.atan2(math.sin(p2[3] - p1[3]), math.cos(p2[3] - p1[3]))
    (args.output / "report.json").write_text(json.dumps(records, indent=2))
    assert forward > .05, f"forward movement failed: {forward}"
    assert turn > .15, f"left turn failed: {turn}"
    assert np.linalg.norm(np.array(p3[:3]) - p0[:3]) < .03, "reset position drift"
    print(f"PASS: synchronized RGB-D, forward={forward:.3f}m, left={turn:.3f}rad, reset")


if __name__ == "__main__":
    main()
