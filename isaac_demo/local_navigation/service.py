"""Single-worker navigation state, published as atomic JSON-friendly snapshots."""
import base64
import io
import math
import time
import uuid

import numpy as np
from PIL import Image

from .blindspot import initialize_near_blindspot
from .collision import CollisionChecker
from .command_guard import guard_command
from .depth_mapping import DepthMapper
from .models import pose_tuple, transform_path
from .path_corrector import advisory_path, correct_path
from .rolling_grid import RollingGrid


class LocalNavigation:
    def __init__(self, config, footprint, robot_height, profile):
        self.config, self.footprint = config, footprint
        self.robot_height, self.profile = robot_height, profile
        self.reset()

    def reset(self, *, allow_initial_free=True):
        self.mode = 'planning_only'
        self.initial_free_pending = allow_initial_free and self.config.initial_clear_radius_m > 0
        self.initial_free_center = None
        self.session = uuid.uuid4().hex
        self.mapper = DepthMapper(self.config, self.robot_height)
        self.grid = RollingGrid(self.config, self.mapper.bins)
        self.checker = None
        self.prediction = None
        self.command = None
        self.floor_z = None
        self.status, self.reason = 'waiting_observation', 'waiting_rgbd'
        self.observation_id = 0
        self.prediction_id = 0
        self.mapping_ms = 0.
        self._snapshot = self._build_snapshot()

    def observe(self, observation):
        started = time.monotonic()
        if observation.get('depth') is None or observation.get('depth_camera') is None:
            self.status, self.reason = 'waiting_observation', 'missing_rgbd'
            self.checker = None
            self._snapshot = self._build_snapshot()
            return
        pose = pose_tuple(observation['pose'])
        floor = float(observation['pose']['z'])
        if not math.isfinite(floor):
            raise ValueError('nonfinite floor')
        if self.floor_z is not None and (abs(floor - self.floor_z) > self.config.floor_tolerance_m
                or math.hypot(pose[0] - self.grid.pose[0], pose[1] - self.grid.pose[1]) > self.config.max_pose_jump_m):
            # A tracking discontinuity is not a newly authorized clear birth.
            self.reset(allow_initial_free=False)
        self.floor_z = floor
        if self.mapper.update(self.grid, observation):
            if self.initial_free_pending:
                self.initial_free_pending = False
                self.initial_free_center = list(pose[:2])
                yy, xx = np.indices(self.grid.free.shape)
                wx, wy = self.grid.world(xx, yy)
                self.grid.initial_free = (
                    (np.hypot(wx - pose[0], wy - pose[1]) <= self.config.initial_clear_radius_m)
                    & self.grid.valid & ~self.grid.ground & (self.grid.hits == 0)
                )
            initialize_near_blindspot(self.grid, observation['depth_camera'])
            self.observation_id += 1
            self.checker = CollisionChecker(self.grid, self.footprint)
            self.command = None
            self.mapping_ms = round((time.monotonic() - started) * 1000, 2)
        self.status, self.reason = ('observe_only', 'manual_unprotected') if self.mode == 'observe_only' else ('ready', 'rgbd_updated')
        self._snapshot = self._build_snapshot()

    def set_mode(self, mode):
        if mode not in ('planning_only', 'guarded', 'observe_only'):
            raise ValueError('mode must be planning_only, guarded or observe_only')
        self.mode = mode
        self.prediction = self.command = None
        self.status, self.reason = ('observe_only', 'manual_unprotected') if mode == 'observe_only' else (
            'ready', 'trajectory_correction_only' if mode == 'planning_only' else 'protection_enabled')
        self._snapshot = self._build_snapshot()

    def plan(self, waypoints, capture_pose):
        return self.plan_remaining(waypoints, capture_pose, 0)

    def plan_remaining(self, waypoints, capture_pose, start_index):
        """Replan the original prediction, retaining all blue points on the map."""
        raw = transform_path(waypoints, capture_pose)
        if not 0 <= start_index < len(raw) and raw:
            raise ValueError('invalid remaining prediction index')
        self.prediction_id += 1
        if self.mode == 'observe_only':
            result = dict(raw=raw, corrected=[], status='blocked', reason='manual_collection_only',
                          point_status=[], segment_status=[], planning_ms=0.)
        elif self.checker is None:
            result = dict(raw=raw, corrected=[], status='waiting_observation', reason=self.reason,
                          point_status=[], segment_status=[], planning_ms=0.)
        else:
            result = correct_path(raw[start_index:], self.grid.pose, self.checker)
            if self.mode == 'planning_only':
                result = advisory_path(raw[start_index:], self.grid.pose, self.checker, result)
        if start_index:
            result['raw'] = [tuple(p) for p in raw]
        for key in ('point_status', 'segment_status'):
            result[key] = ['passed'] * start_index + result[key]
        if 'goal_index' in result:
            result['goal_index'] += start_index
        result['remaining_start_index'] = start_index
        result.update(prediction_id=self.prediction_id, map_revision=self.grid.revision,
                      session_id=self.session, capture_pose=list(capture_pose))
        self.prediction = result
        self.status, self.reason = result['status'], result['reason']
        self._snapshot = self._build_snapshot()
        return transform_path(result['corrected'][1:], self.grid.pose, inverse=True)

    def recovery_forward_clear(self):
        """Exit recovery only when a short forward body sweep is known clear.

        Rotation itself is exempt by the user's simulation assumption; this
        method does not relax the existing rules for forward translation.
        """
        if self.mode != 'guarded' or self.checker is None:
            return False
        from .path_corrector import advance

        end = advance(self.grid.pose, self.config.recovery_forward_distance_m, 0.)
        return self.checker.segment(self.grid.pose, end) == 'clear'

    def guard(self, command, duration, *, max_angular=.5, manual=False):
        if self.mode == 'planning_only':
            # Occupancy is used upstream for correction, never to veto execution.
            if len(command) != 2 or not all(math.isfinite(v) for v in command):
                raise ValueError('command must contain finite velocities')
            accepted = (command[0], max(-max_angular, min(max_angular, command[1])))
            detail = dict(reason='trajectory_correction_only', sweep=[])
        elif self.mode == 'observe_only':
            accepted = command if manual else (0., 0.)
            detail = dict(reason='manual_unprotected' if manual else 'manual_collection_only', sweep=[])
        elif self.checker is None:
            accepted, detail = (0., 0.), dict(reason='blocked_command_missing_map', sweep=[])
        else:
            path = [] if manual or not self.prediction else self.prediction['corrected']
            accepted, detail = guard_command(command, duration, self.grid.pose, self.checker,
                                             path, adjust=not manual)
            if abs(accepted[1]) > max_angular:
                accepted, detail = guard_command((accepted[0], math.copysign(max_angular, accepted[1])),
                    duration, self.grid.pose, self.checker, [], adjust=False)
        self.command = dict(detail, velocity=list(accepted), map_revision=self.grid.revision)
        if accepted == (0., 0.) and command != (0., 0.):
            self.status, self.reason = 'blocked', detail['reason']
        self._snapshot = self._build_snapshot()
        return accepted

    def clear_prediction(self):
        self.prediction = self.command = None
        self.status, self.reason = ('observe_only', 'manual_unprotected') if self.mode == 'observe_only' else ('ready' if self.checker else 'waiting_observation', 'stopped')
        self._snapshot = self._build_snapshot()

    def _build_snapshot(self):
        g, c = self.grid, self.config
        image = np.zeros((g.n, g.n, 4), dtype=np.uint8)
        labels = g.labels()
        image[g.valid] = [97, 105, 118, 255]
        image[labels == 1] = [228, 234, 231, 255]
        image[g.initial_free & (labels == 1)] = [184, 220, 229, 255]
        image[g.blind_free & (labels == 1)] = [184, 220, 229, 255]
        if self.checker is not None:
            hard, _ = self.checker.layers(g.pose[2])
            image[g.valid & hard] = [204, 143, 74, 255]
            near = (self.checker.clearance < c.soft_clearance_m) & (labels == 1) & ~hard
            image[near] = [224, 204, 158, 255]
        image[labels == 2] = [33, 38, 46, 255]
        output = io.BytesIO()
        Image.fromarray(np.flipud(image)).save(output, format='PNG')
        return dict(enabled=True, mode=self.mode,
                    execution_collision_check=self.mode == 'guarded',
                    recovery_enabled=self.mode == 'guarded' and c.recovery_max_turn_deg > 0,
                    initial_clearance=dict(center=self.initial_free_center,
                        radius=c.initial_clear_radius_m,
                        remaining_cells=int(np.count_nonzero(g.initial_free))),
                    near_blind_clearance=dict(max_distance=c.near_blind_clear_distance_m,
                        remaining_cells=int(np.count_nonzero(g.blind_free))),
                    start_check=self.checker.reason(g.pose) if self.checker else 'missing_map',
                    radius=c.window_radius_m, resolution=c.resolution_m,
                    origin=[v * c.resolution_m for v in g.origin] if g.origin else None,
                    size=g.n, center=list(g.pose[:2]), pose=list(g.pose), profile=self.profile,
                    footprint=self.footprint, margin=c.safety_margin_m,
                    map_revision=g.revision, session_id=self.session, observation_id=self.observation_id,
                    mapping_ms=self.mapping_ms,
                    status=self.status, reason=self.reason, prediction=self.prediction, command=self.command,
                    occupied_count=int(np.count_nonzero(labels == 2)),
                    free_count=int(np.count_nonzero(labels == 1)),
                    png='data:image/png;base64,' + base64.b64encode(output.getvalue()).decode())

    def snapshot(self):
        return self._snapshot
