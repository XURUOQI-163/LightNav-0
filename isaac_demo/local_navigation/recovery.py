"""Choose an in-place recovery turn from image-left/right depth, not semantics."""
import numpy as np


def choose_depth_turn(observation, config):
    result = dict(direction=0, left_m=None, right_m=None,
                  left_valid_fraction=0., right_valid_fraction=0.)
    if observation is None or observation.get('depth') is None:
        return result
    depth = np.asarray(observation['depth'], dtype=float)
    if depth.ndim != 2 or min(depth.shape) < 4:
        return result
    h, w = depth.shape
    top, bottom = int(h * config.recovery_roi_top), int(h * config.recovery_roi_bottom)
    # Ignore image edges, the central seam, and the near-floor bottom band.
    # A low percentile avoids letting a small distant hole dominate a close wall.
    for name, lo, hi in (('left', .05, .45), ('right', .55, .95)):
        roi = depth[top:bottom, int(w * lo):int(w * hi)]
        valid = np.isfinite(roi) & (roi >= config.depth_min_m) & (roi <= config.depth_max_m)
        fraction = float(valid.mean()) if roi.size else 0.
        result[name + '_valid_fraction'] = fraction
        if valid.any() and fraction >= config.recovery_min_valid_fraction:
            result[name + '_m'] = float(np.percentile(roi[valid], config.recovery_depth_percentile))
    left, right = result['left_m'], result['right_m']
    if left is not None or right is not None:
        # Image left is positive yaw for the calibrated forward camera.
        # Exact ties select left deterministically, avoiding frame-to-frame jitter.
        result['direction'] = 1 if right is None or (left is not None and left >= right) else -1
    return result
