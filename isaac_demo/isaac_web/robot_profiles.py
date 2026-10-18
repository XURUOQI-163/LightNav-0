"""One physical profile drives Carter geometry, camera and passive door checks.

Dimensions are metres in the unscaled Carter root frame, not a centred box.
The half_xy robot keeps circular wheels and the original ground-to-roof height.
"""

import copy
import json
from pathlib import Path


PROFILE_DIR = Path(__file__).resolve().parents[1] / "config/robots"
PROFILES = ("original", "half_xy")


def apply_robot_profile(config, name=None):
    name = name or config.get("robot_profile", "original")
    if name not in PROFILES:
        raise ValueError(f"unknown Carter robot profile: {name}")
    result = copy.deepcopy(config)
    profile = json.loads((PROFILE_DIR / f"carter_{name}.json").read_text())
    result.update(profile)
    ys = [point[1] for point in result["footprint"]]
    result["robot_width"] = max(ys) - min(ys)
    return result


def room_geometry_for_robot(geometry, doorway, config):
    """Preserve physical margins; scale the old minimum opening, not the margin."""
    geometry, doorway = copy.deepcopy(geometry), copy.deepcopy(doorway)
    for options in (geometry, doorway["depth"]):
        options["robot_width"] = config["robot_width"]
        options["portal_min_width"] = max(
            config["robot_width"] + options.get("safety_margin", .1),
            options.get("portal_min_width", .75) * config["robot_xy_scale"],
        )
    return geometry, doorway
