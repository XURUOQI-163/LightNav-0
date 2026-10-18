"""Resize the referenced Carter in anonymous runtime layers, before PhysX starts.

An articulation-wide [s,s,1] scale would make the rolling wheels elliptical.
Instead reshape body geometry, keep wheels circular, lower their axles by the
lost radius, and re-anchor the joints. No source USD is saved or overwritten.
This is a navigation proxy, not a dynamically identified Limo model.
"""

import math


def resize_carter(root, xy_scale):
    if xy_scale == 1.:
        return
    if xy_scale != .5:
        raise ValueError("Carter supports only original and half_xy profiles")
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = root.GetStage()
    layer = stage.GetSessionLayer()
    dynamic_layer = stage.GetRootLayer()
    if layer is None or not layer.anonymous or not dynamic_layer.anonymous:
        raise ValueError("Carter resize requires anonymous runtime layers")
    names = ("chassis_link", "left_wheel_link", "right_wheel_link",
             "rear_pivot_link", "rear_wheel_link", "com_offset", "imu")
    bodies = {name: root.GetChild(name) for name in names}
    if not all(p and p.HasAPI(UsdPhysics.RigidBodyAPI) for p in bodies.values()):
        raise ValueError("half_xy requires the Carter V1 articulated asset")
    if root.GetCustomDataByKey("lightnav_half_xy"):
        raise ValueError("Carter half_xy already applied")
    s = xy_scale
    # Bodies must be direct children: local coordinates then equal robot coordinates.
    matrices = {n: UsdGeom.Xformable(p).GetLocalTransformation() for n, p in bodies.items()}
    for body in bodies.values():
        if list(body.GetAttribute("xformOpOrder").Get()) != [
                "xformOp:translate", "xformOp:orient", "xformOp:scale"]:
            raise ValueError("Carter dynamic bodies require standard translate/orient/scale ops")
    positions = {n: m.ExtractTranslation() for n, m in matrices.items()}
    shape_scales = {n: Gf.Vec3d(s, s, 1.) for n in names}
    new_positions = {n: Gf.Vec3d(v[0] * s, v[1] * s, v[2]) for n, v in positions.items()}
    for name in ("left_wheel_link", "right_wheel_link", "rear_wheel_link"):
        collider = bodies[name].GetChild("cylinder")
        if not collider or not collider.IsA(UsdGeom.Cylinder):
            raise ValueError(f"Carter wheel collider missing: {name}")
        relative = UsdGeom.Xformable(collider).GetLocalTransformation()
        radius = UsdGeom.Cylinder(collider).GetRadiusAttr().Get() * relative.TransformDir(Gf.Vec3d(1, 0, 0)).GetLength()
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("invalid Carter wheel radius")
        new_positions[name][2] -= radius * (1. - s)
        shape_scales[name] = Gf.Vec3d(s)
    # Keep the caster fork attached to the now-lower, smaller rear wheel.
    old_drop = positions["rear_wheel_link"][2] - positions["rear_pivot_link"][2]
    new_drop = new_positions["rear_wheel_link"][2] - new_positions["rear_pivot_link"][2]
    shape_scales["rear_pivot_link"][2] = new_drop / old_drop
    joints = [p for p in Usd.PrimRange(root) if p.IsA(UsdPhysics.Joint)]
    for joint in joints:
        for side in (0, 1):
            targets = joint.GetRelationship(f"physics:body{side}").GetTargets()
            if len(targets) != 1 or stage.GetPrimAtPath(targets[0]) not in bodies.values():
                raise ValueError("unexpected Carter joint body; refusing partial resize")

    def set_matrix(prim, matrix):
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddTransformOp(opSuffix="robotProfile").Set(matrix)

    with Usd.EditContext(stage, layer):
        for name, body in bodies.items():
            factors = shape_scales[name]
            scale = Gf.Matrix4d().SetScale(factors)
            if not body.IsA(UsdGeom.Gprim):
                for child in body.GetChildren():
                    xform = UsdGeom.Xformable(child)
                    if xform:
                        set_matrix(child, xform.GetLocalTransformation() * scale)
            # PhysX writes poses into the normal runtime edit layer. Never put a
            # stronger, static matrix/op-order on a dynamic rigid body: that masks
            # updates and leaves the rendered car behind while its camera moves.
            # Keep canonical TRS also to avoid scale-orientation/skew on the IMU.
            with Usd.EditContext(stage, dynamic_layer):
                body.GetAttribute("xformOp:translate").Set(new_positions[name])
                if body.IsA(UsdGeom.Gprim):
                    old_scale = body.GetAttribute("xformOp:scale").Get()
                    body.GetAttribute("xformOp:scale").Set(
                        Gf.Vec3d(*[old_scale[i] * factors[i] for i in range(3)]))
                rigid = UsdPhysics.RigidBodyAPI(body)
                rigid.CreateVelocityAttr(Gf.Vec3f(0))
                rigid.CreateAngularVelocityAttr(Gf.Vec3f(0))
            mass = UsdPhysics.MassAPI(body)
            old_mass = mass.GetMassAttr().Get()
            if old_mass and old_mass > 0:
                mass.CreateMassAttr(old_mass * factors[0] * factors[1] * factors[2])
            com = mass.GetCenterOfMassAttr().Get()
            if com is not None and all(math.isfinite(v) for v in com):
                mass.CreateCenterOfMassAttr(Gf.Vec3f(*[com[i] * factors[i] for i in range(3)]))
            # PhysX recomputes inertia from the resized collision geometry and mass.
            mass.CreateDiagonalInertiaAttr(Gf.Vec3f(0))
            mass.CreatePrincipalAxesAttr(Gf.Quatf(1))
        # Choose each child-side anchor, then put both sides at exactly that point.
        # LocalRot values and the wheel spin axes remain unchanged.
        for joint in joints:
            child = stage.GetPrimAtPath(joint.GetRelationship("physics:body1").GetTargets()[0])
            parent = stage.GetPrimAtPath(joint.GetRelationship("physics:body0").GetTargets()[0])
            factors = shape_scales[child.GetName()]
            old_anchor = joint.GetAttribute("physics:localPos1").Get()
            anchor = Gf.Vec3d(*[old_anchor[i] * factors[i] for i in range(3)])
            child_matrix = UsdGeom.Xformable(child).GetLocalTransformation()
            parent_matrix = UsdGeom.Xformable(parent).GetLocalTransformation()
            parent_anchor = parent_matrix.GetInverse().Transform(child_matrix.Transform(anchor))
            joint.GetAttribute("physics:localPos0").Set(Gf.Vec3f(parent_anchor))
            joint.GetAttribute("physics:localPos1").Set(Gf.Vec3f(anchor))
        root.SetCustomDataByKey("lightnav_half_xy", True)
