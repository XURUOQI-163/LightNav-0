import asyncio
import math
import threading

import numpy as np
import pytest
from habitat_web.clients import Prediction
from isaac_web.runtime import IsaacRuntime
from local_navigation.models import NavigationConfig
from local_navigation.service import LocalNavigation
from test_local_depth import observation
from test_local_navigation import SMALL


def test_side_depth_selects_image_left_as_positive_yaw_and_ignores_invalid():
    from local_navigation.recovery import choose_depth_turn

    depth = np.ones((40, 80), dtype=np.float32)
    depth[:, :40] = 4.
    decision = choose_depth_turn(observation(depth), NavigationConfig())
    assert decision['direction'] == 1 and decision['left_m'] > decision['right_m']
    depth[:, :40] = np.inf
    decision = choose_depth_turn(observation(depth), NavigationConfig())
    assert decision['direction'] == -1 and decision['left_m'] is None
    depth[:] = 0
    assert choose_depth_turn(observation(depth), NavigationConfig())['direction'] == 0


def test_side_score_does_not_choose_single_far_pixel_or_bottom_floor():
    from local_navigation.recovery import choose_depth_turn

    depth = np.ones((40, 80), dtype=np.float32)
    depth[:, 40:] = 3.
    depth[12, 10] = 8.
    depth[30:, 40:] = .2
    assert choose_depth_turn(observation(depth), NavigationConfig())['direction'] == -1


class RecoveryScene:
    def reset(self):
        self.time, self.x, self.yaw = 0., 0., 0.
        self.moves = []
        return self.observe()

    def observe(self):
        depth = np.ones((32, 64), dtype=np.float32)
        depth[:, :32] = 4.
        return dict(observation(depth), jpeg=f'frame-{self.time}'.encode(),
                    sim_time=self.time, collided=False,
                    pose=dict(x=self.x, y=0., z=0., yaw=self.yaw))

    def step(self, linear, angular, duration):
        self.moves.append((linear, angular, duration))
        self.time += duration
        self.x += linear * duration
        self.yaw += angular * duration
        return self.observe()

    def close(self):
        pass


class RecoveryNavigation(LocalNavigation):
    """Controlled map availability isolates runtime flow from scene geometry."""
    clear_after = .6

    def observe(self, frame):
        self.time = frame['sim_time']
        super().observe(frame)

    def plan(self, *args):
        return [(1., 0., 0.)] if self.time >= self.clear_after else []

    def recovery_forward_clear(self):
        return self.time >= self.clear_after

    def guard(self, command, *args, **kwargs):
        # The pure rotation must bypass this guard, even with an unknown body.
        assert command[0] > 0, 'recovery rotation was sent through the collision guard'
        return command


@pytest.mark.parametrize('blocked_by_command', [False, True])
def test_blocked_path_turns_without_collision_guard_and_requests_fresh_waypoints(blocked_by_command):
    class Client:
        def __init__(self):
            self.frames = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def predict(self, jpeg, instruction):
            self.frames.append(jpeg)
            return Prediction([(1., 0., 0.)], len(self.frames) == 3, 1.)

    async def scenario():
        client = Client()

        class Navigation(RecoveryNavigation):
            def plan(self, *args):
                return [(1., 0., 0.)] if blocked_by_command else super().plan(*args)

            def guard(self, command, *args, **kwargs):
                super().guard(command, *args, **kwargs)
                return (0., 0.) if self.time < self.clear_after else command

        nav = Navigation(NavigationConfig(initial_clear_radius_m=0, recovery_max_turn_deg=360.), SMALL, .663, 'half_xy')
        runtime = IsaacRuntime(RecoveryScene, {'lightnav': 'unused'}, local_navigation=nav,
                               client_factory=lambda *args: client)
        await runtime.initialize()
        await runtime.navigation_mode('guarded')  # Explicit legacy recovery tests.
        try:
            await runtime.start('lightnav', 'go')
            await asyncio.wait_for(runtime.task, 4.)
            moves = runtime.scene.moves
            assert len(moves) == 3 and moves[0][0] == moves[1][0] == 0
            assert moves[0][1] > 0 and moves[1][1] > 0 and moves[2][0] > 0
            assert client.frames[0] != client.frames[1]  # no stale prediction replay
            assert runtime.state['stop_reason'] == 'model_stop'
        finally:
            await runtime.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('disabled, invalid, max_steps', [(True, False, 0), (False, True, 0),
                                                        (False, False, 1)])
def test_recovery_respects_disabled_invalid_depth_and_step_limit(disabled, invalid, max_steps):
    class Scene(RecoveryScene):
        def observe(self):
            frame = super().observe()
            if invalid:
                frame['depth'][:] = 0.
            return frame

    async def scenario():
        nav = RecoveryNavigation((NavigationConfig() if disabled else NavigationConfig(recovery_max_turn_deg=360.)),
                                 SMALL, .663, 'half_xy')
        nav.clear_after = math.inf
        runtime = IsaacRuntime(Scene, {'lightnav': 'unused'}, local_navigation=nav, max_steps=max_steps)
        await runtime.initialize()
        await runtime.navigation_mode('guarded')  # Explicit legacy recovery tests.
        try:
            assert not await runtime.recover_blocked()
            assert len(runtime.scene.moves) == (1 if max_steps else 0)
            if max_steps:
                assert runtime.state['navigation_reason'] == 'step_limit'
            if invalid:
                assert runtime.state['navigation_reason'] == 'recovery_insufficient_depth'
        finally:
            await runtime.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('options', [dict(recovery_max_turn_deg=-1), dict(recovery_max_turn_deg=721),
                                    dict(recovery_angular_speed=0), dict(recovery_min_valid_fraction=1.1),
                                    dict(recovery_roi_top=.8, recovery_roi_bottom=.7)])
def test_recovery_configuration_rejects_invalid_limits(options):
    with pytest.raises(ValueError):
        NavigationConfig(**options)


def test_recovery_is_bounded_and_reentry_does_not_reset_turn_budget():
    async def scenario():
        nav = RecoveryNavigation(NavigationConfig(recovery_max_turn_deg=15.), SMALL, .663, 'half_xy')
        nav.clear_after = math.inf
        runtime = IsaacRuntime(RecoveryScene, {'lightnav': 'unused'}, local_navigation=nav)
        await runtime.initialize()
        await runtime.navigation_mode('guarded')  # Explicit legacy recovery tests.
        try:
            assert not await runtime.recover_blocked()
            moves = list(runtime.scene.moves)
            assert moves and all(v == 0 for v, _, _ in moves)
            assert sum(abs(w) * dt for _, w, dt in moves) == pytest.approx(math.radians(15))
            assert not await runtime.recover_blocked()
            assert runtime.scene.moves == moves
            assert runtime.state['navigation_reason'] == 'recovery_turn_limit'
        finally:
            await runtime.close()
    asyncio.run(scenario())


def test_stop_during_recovery_waits_for_one_inflight_step_without_new_motion():
    entered, release = threading.Event(), threading.Event()

    class DelayedScene(RecoveryScene):
        def step(self, *args):
            entered.set()
            release.wait(2.)
            return super().step(*args)

    async def scenario():
        nav = RecoveryNavigation(NavigationConfig(recovery_max_turn_deg=360.), SMALL, .663, 'half_xy')
        runtime = IsaacRuntime(DelayedScene, {'lightnav': 'unused'}, local_navigation=nav)
        await runtime.initialize()
        await runtime.navigation_mode('guarded')  # Explicit legacy recovery tests.
        try:
            runtime.state['status'] = 'running'
            runtime.task = asyncio.create_task(runtime.recover_blocked())
            for _ in range(200):
                if entered.is_set() or runtime.task.done():
                    break
                await asyncio.sleep(.005)
            assert entered.is_set()
            stop = asyncio.create_task(runtime.stop())
            await asyncio.sleep(.01)
            release.set()
            await stop
            assert len(runtime.scene.moves) == 1
            assert runtime.state['status'] == 'stopped'
            assert not runtime.state['recovery']['active']
        finally:
            release.set()
            await runtime.close()
    asyncio.run(scenario())


def test_recovery_forward_test_keeps_unknown_and_occupied_as_blocked():
    from local_navigation.collision import CollisionChecker

    nav = LocalNavigation(NavigationConfig(initial_clear_radius_m=0), SMALL, .663, 'half_xy')
    nav.set_mode('guarded')
    nav.observe(observation(np.zeros((32, 64), dtype=np.float32)))
    assert not nav.recovery_forward_clear()
    nav.grid.free[nav.grid.valid] = True
    nav.checker = CollisionChecker(nav.grid, SMALL)
    assert nav.recovery_forward_clear()
    ix, iy = nav.grid.cell(.4, 0.)
    nav.grid.hits[iy, ix] = 1
    nav.checker = CollisionChecker(nav.grid, SMALL)
    assert not nav.recovery_forward_clear()


def test_physical_translation_stall_also_enters_recovery():
    class StalledScene(RecoveryScene):
        def step(self, linear, angular, duration):
            frame = super().step(linear, angular, duration)
            frame['collided'] = linear > 0
            return frame

    class Client:
        calls = 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def predict(self, *args):
            self.calls += 1
            return Prediction([(1., 0., 0.)], self.calls == 2, 1.)

    async def scenario():
        nav = RecoveryNavigation(NavigationConfig(recovery_max_turn_deg=360.), SMALL, .663, 'half_xy')
        nav.clear_after = 0.
        runtime = IsaacRuntime(StalledScene, {'lightnav': 'unused'}, local_navigation=nav,
                               client_factory=lambda *args: Client())
        await runtime.initialize()
        await runtime.navigation_mode('guarded')  # Explicit legacy recovery tests.
        try:
            await runtime.start('lightnav', 'go')
            await asyncio.wait_for(runtime.task, 4.)
            moves = runtime.scene.moves
            assert len(moves) == 2 and moves[0][0] > 0 and moves[1][0] == 0
            assert moves[1][1] > 0
        finally:
            await runtime.close()
    asyncio.run(scenario())
