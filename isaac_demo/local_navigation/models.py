"""Bounded configuration and coordinate contracts (metres, radians)."""
import math
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class NavigationConfig:
    window_radius_m: float = 5.0
    initial_clear_radius_m: float = 2.0
    near_blind_clear_distance_m: float = 1.5
    resolution_m: float = .05
    safety_margin_m: float = .05
    soft_clearance_m: float = .25
    max_goal_adjustment_m: float = .75
    max_detour_extra_m: float = 2.
    max_search_expansions: int = 20000
    search_timeout_s: float = 1.0
    heuristic_weight: float = 2.0
    heading_bins: int = 32
    primitive_length_m: float = .15
    depth_min_m: float = .15
    depth_max_m: float = 8.
    sample_columns: int = 160
    sample_rows: int = 96
    ground_tolerance_m: float = .04
    height_bin_m: float = .1
    height_margin_m: float = .05
    clear_confirm_frames: int = 2
    max_pose_jump_m: float = 1.0
    floor_tolerance_m: float = .15
    braking_deceleration: float = .5
    braking_angular_deceleration: float = 1.
    reaction_time_s: float = .1
    recovery_max_turn_deg: float = 0.
    recovery_angular_speed: float = .35
    recovery_forward_distance_m: float = .5
    recovery_reset_translation_m: float = .2
    recovery_min_valid_fraction: float = .2
    recovery_depth_percentile: float = 30.
    recovery_roi_top: float = .25
    recovery_roi_bottom: float = .65
    path_replan_interval_s: float = .5
    path_failure_confirmations: int = 3
    path_no_progress_s: float = 8.
    path_progress_translation_m: float = .05
    model_replan_limit: int = 3

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError('invalid local navigation option: ' + name)
            if value <= 0 and name not in ('safety_margin_m', 'height_margin_m', 'initial_clear_radius_m',
                                          'near_blind_clear_distance_m', 'recovery_max_turn_deg', 'recovery_roi_top'):
                raise ValueError('local navigation option must be positive: ' + name)
        for name in ('max_search_expansions', 'heading_bins', 'sample_columns', 'sample_rows', 'clear_confirm_frames',
                     'path_failure_confirmations', 'model_replan_limit'):
            if type(getattr(self, name)) is not int:
                raise ValueError('integer required: ' + name)
        if not (1 <= self.window_radius_m <= 10 and .025 <= self.resolution_m <= .2
                and 0 <= self.initial_clear_radius_m <= 3
                and 0 <= self.near_blind_clear_distance_m <= 3
                and self.window_radius_m / self.resolution_m <= 200
                and 0 <= self.safety_margin_m <= .3 and 0 <= self.height_margin_m <= .3
                and .05 <= self.height_bin_m <= .25
                and 8 <= self.heading_bins <= 64 and 1 <= self.clear_confirm_frames <= 8
                and 8 <= self.sample_columns <= 320 and 8 <= self.sample_rows <= 160
                and 100 <= self.max_search_expansions <= 50000
                and .05 <= self.search_timeout_s <= 5
                and .01 <= self.braking_deceleration <= 10
                and .01 <= self.braking_angular_deceleration <= 20
                and .01 <= self.reaction_time_s <= 2
                and 1 <= self.heuristic_weight <= 5
                and .05 <= self.primitive_length_m <= .3
                and 0 <= self.recovery_max_turn_deg <= 720
                and .05 <= self.recovery_angular_speed <= 2
                and .1 <= self.recovery_forward_distance_m <= min(2., self.window_radius_m)
                and .05 <= self.recovery_reset_translation_m <= 2
                and .05 <= self.recovery_min_valid_fraction <= 1
                and 1 <= self.recovery_depth_percentile <= 50
                and 0 <= self.recovery_roi_top < self.recovery_roi_bottom <= 1
                and .1 <= self.path_replan_interval_s <= 5
                and 2 <= self.path_failure_confirmations <= 20
                and 1 <= self.path_no_progress_s <= 120
                and .01 <= self.path_progress_translation_m <= .5
                and 1 <= self.model_replan_limit <= 10
                and self.depth_min_m < self.depth_max_m <= 20):
            raise ValueError('local navigation options exceed bounded limits')


def pose_tuple(pose):
    result = tuple(float(pose[k]) for k in ('x', 'y', 'yaw'))
    if not all(math.isfinite(x) for x in result):
        raise ValueError('nonfinite navigation pose')
    return result


def transform_path(points, pose, inverse=False):
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    co, si = math.cos(pose[2]), math.sin(pose[2])
    rotation = np.array([[co, -si], [si, co]])
    xy = ((points[:, :2] - pose[:2]) @ rotation if inverse
          else points[:, :2] @ rotation.T + pose[:2])
    yaw = points[:, 2] + (-pose[2] if inverse else pose[2])
    return np.column_stack((xy, np.arctan2(np.sin(yaw), np.cos(yaw)))).tolist()
