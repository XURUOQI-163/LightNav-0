"""Carter navigation-proxy dynamics; all changes stay in anonymous USD layers."""
import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class StabilityConfig:
    max_linear: float = .6
    max_angular: float = 1.
    linear_acceleration: float = .6
    linear_deceleration: float = .8
    angular_acceleration: float = 1.5
    max_lateral_acceleration: float = .5
    wheel_max_torque: float = 12.
    wheel_damping: float = 20.
    com_lowering: float = .04
    max_depenetration_velocity: float = .5
    solver_position_iterations: int = 64
    solver_velocity_iterations: int = 4

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError('invalid stability.' + name)
            if name == 'com_lowering':
                if not 0 <= value <= .12:
                    raise ValueError('stability.com_lowering must be between 0 and .12 m')
            elif value <= 0:
                raise ValueError('stability.' + name + ' must be positive')
        if self.max_linear > 2 or self.max_angular > 3:
            raise ValueError('stability velocity limits exceed supported range')
        for name in ('solver_position_iterations', 'solver_velocity_iterations'):
            if type(getattr(self, name)) is not int or getattr(self, name) > 255:
                raise ValueError('invalid stability.' + name)


class DriveLimiter:
    """Rate-limit physical targets across commands, including zero/reverse requests.

    Joint target zeroing between paused render calls does not reset this state.
    Only a physics reset resets it. No physical pose or velocity is overwritten.
    """
    def __init__(self, config):
        self.config = config if isinstance(config, StabilityConfig) else StabilityConfig(**config)
        self.command = (0., 0.)

    def reset(self):
        self.command = (0., 0.)

    def advance(self, linear, angular, dt):
        c = self.config
        if not all(math.isfinite(x) for x in (linear, angular, dt)) or dt <= 0:
            raise ValueError('invalid drive target or timestep')
        v = max(-c.max_linear, min(c.max_linear, linear))
        w = max(-c.max_angular, min(c.max_angular, angular))
        old_v, old_w = self.command
        # Brake to zero before reversing; never spend the whole tick accelerating
        # in the opposite direction while still translating at the old speed.
        if old_v * v < 0:
            v = 0.
        acceleration = c.linear_deceleration if abs(v) < abs(old_v) else c.linear_acceleration
        v = old_v + max(-acceleration*dt, min(acceleration*dt, v-old_v))
        # Limit turning at the CURRENT ramped speed, not merely at target speed.
        # This also handles a request to stop and spin while still braking.
        lateral_limit = c.max_lateral_acceleration / max(abs(v), abs(old_v), 1e-6)
        w = max(-lateral_limit, min(lateral_limit, w))
        w = old_w + max(-c.angular_acceleration*dt, min(c.angular_acceleration*dt, w-old_w))
        # Do not accelerate translation faster than the current turn allows.
        if abs(w) > 1e-6 and abs(v) > abs(old_v):
            v = math.copysign(min(abs(v), c.max_lateral_acceleration / abs(w)), v)
        self.command = (v, w)
        return self.command


def configure_carter_dynamics(root, options):
    from pxr import Gf, Usd, UsdPhysics

    c = options if isinstance(options, StabilityConfig) else StabilityConfig(**options)
    stage = root.GetStage()
    if not stage.GetRootLayer().anonymous or not stage.GetSessionLayer().anonymous:
        raise ValueError('stability overrides require anonymous runtime layers')
    if root.GetCustomDataByKey('lightnav_stability'):
        raise ValueError('stability overrides already applied')
    bodies = [p for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    joints = [p for p in Usd.PrimRange(root) if p.GetName() in ('left_wheel', 'right_wheel')]
    if not bodies or len(joints) != 2:
        raise ValueError('Carter wheel joints/rigid bodies missing')
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        # The original asset has 64/32 iterations. Keep position convergence,
        # use four velocity iterations for the TGS solver.
        root.GetAttribute('physxArticulation:solverPositionIterationCount').Set(c.solver_position_iterations)
        root.GetAttribute('physxArticulation:solverVelocityIterationCount').Set(c.solver_velocity_iterations)
        for body in bodies:
            body.GetAttribute('physxRigidBody:maxDepenetrationVelocity').Set(c.max_depenetration_velocity)
            if body.GetName() in ('chassis_link', 'com_offset'):
                mass = UsdPhysics.MassAPI(body)
                com = mass.GetCenterOfMassAttr().Get()
                mass.CreateCenterOfMassAttr(Gf.Vec3f(com[0], com[1], com[2]-c.com_lowering))
                mass.CreateDiagonalInertiaAttr(Gf.Vec3f(0))
                mass.CreatePrincipalAxesAttr(Gf.Quatf(1))
        for joint in joints:
            drive = UsdPhysics.DriveAPI(joint, 'angular')
            drive.CreateTypeAttr('force')
            drive.CreateStiffnessAttr(0.)
            drive.CreateDampingAttr(c.wheel_damping)
            drive.CreateMaxForceAttr(c.wheel_max_torque)
        root.SetCustomDataByKey('lightnav_stability', True)
