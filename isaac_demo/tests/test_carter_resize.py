"""Read-only USD asset checks; these do not start Kit, PhysX, or a GPU renderer."""

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pxr.Usd")
from isaac_web.carter_resize import resize_carter
from pxr import Gf, Usd, UsdGeom, UsdPhysics

ASSET = Path(os.environ.get("CARTER_USD", "/data/disk1/xrq/datasets/omninavbench/Assets/robots/Carter/carter_v1.usd"))


@pytest.fixture
def carter():
    if not ASSET.is_file():
        pytest.skip("Carter asset not installed; set CARTER_USD")
    stage = Usd.Stage.CreateInMemory()
    root = stage.DefinePrim("/Robot", "Xform")
    root.GetReferences().AddReference(str(ASSET))
    return stage, root


def visual_size(root):
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"])
    return np.array(cache.ComputeWorldBound(root).ComputeAlignedRange().GetSize())


def test_original_profile_leaves_asset_untouched(carter):
    stage, root = carter
    before = stage.GetRootLayer().ExportToString(), stage.GetSessionLayer().ExportToString()
    resize_carter(root, 1.)
    assert before == (stage.GetRootLayer().ExportToString(), stage.GetSessionLayer().ExportToString())


def test_half_xy_preserves_height_round_wheels_and_joint_alignment(carter):
    stage, root = carter
    before = visual_size(root)
    disk_hash = hashlib.sha256(ASSET.read_bytes()).hexdigest()
    resize_carter(root, .5)
    after = visual_size(root)
    np.testing.assert_allclose(after[:2], before[:2] * .5, atol=.003)
    assert after[2] == pytest.approx(before[2], abs=.003)
    assert hashlib.sha256(ASSET.read_bytes()).hexdigest() == disk_hash
    # Cylinder's local X/Y are the rolling circle; they must remain equal.
    for name in ("left_wheel_link", "right_wheel_link", "rear_wheel_link"):
        prim = stage.GetPrimAtPath(f"/Robot/{name}/cylinder")
        matrix = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        scales = np.linalg.norm(matrix[:3, :3], axis=1)
        assert scales[0] == pytest.approx(scales[1], abs=1e-6)
        radius = UsdGeom.Cylinder(prim).GetRadiusAttr().Get() * scales[0]
        assert radius == pytest.approx(.038 if name == "rear_wheel_link" else .12, abs=1e-5)
    for p in Usd.PrimRange(root):
        if not p.IsA(UsdPhysics.Joint):
            continue
        anchors = []
        for side in (0, 1):
            body = stage.GetPrimAtPath(p.GetRelationship(f"physics:body{side}").GetTargets()[0])
            matrix = UsdGeom.Xformable(body).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            anchors.append(matrix.Transform(Gf.Vec3d(p.GetAttribute(f"physics:localPos{side}").Get())))
        assert np.linalg.norm(np.array(anchors[0]) - anchors[1]) < 1e-6


def test_physics_pose_updates_reach_render_geometry_after_resize(carter):
    stage, root = carter
    resize_carter(root, .5)
    body = stage.GetPrimAtPath("/Robot/chassis_link")
    visual = stage.GetPrimAtPath("/Robot/chassis_link/chassis_obj")
    before = UsdGeom.Xformable(visual).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    # PhysX writes dynamic transforms into the runtime stage's normal edit layer.
    # A stronger session-layer transform must not mask these updates.
    pos = body.GetAttribute("xformOp:translate").Get()
    with Usd.EditContext(stage, stage.GetRootLayer()):
        body.GetAttribute("xformOp:translate").Set(Gf.Vec3d(pos) + Gf.Vec3d(1, 2, 0))
    after = UsdGeom.Xformable(visual).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    np.testing.assert_allclose(after.ExtractTranslation() - before.ExtractTranslation(), [1, 2, 0], atol=1e-6)
    for name in ("chassis_link", "left_wheel_link", "right_wheel_link", "imu"):
        prim = stage.GetPrimAtPath(f"/Robot/{name}")
        assert list(prim.GetAttribute("xformOpOrder").Get()) == [
            "xformOp:translate", "xformOp:orient", "xformOp:scale"]
        for attribute in ("xformOp:translate", "xformOp:orient", "xformOp:scale", "xformOpOrder"):
            assert not stage.GetSessionLayer().GetAttributeAtPath(prim.GetPath().AppendProperty(attribute))


def test_unknown_asset_fails_before_partial_resize():
    stage = Usd.Stage.CreateInMemory()
    root = stage.DefinePrim("/Robot", "Xform")
    before = stage.GetSessionLayer().ExportToString()
    with pytest.raises(ValueError, match="Carter"):
        resize_carter(root, .5)
    assert stage.GetSessionLayer().ExportToString() == before


@pytest.mark.parametrize('profile,scale', [('original', 1.), ('half_xy', .5)])
def test_stability_overrides_are_finite_local_and_preserve_geometry(carter, profile, scale):
    from isaac_web.robot_profiles import apply_robot_profile
    from isaac_web.stability import configure_carter_dynamics
    stage, root = carter
    cfg = apply_robot_profile({}, profile)
    resize_carter(root, scale)
    before = visual_size(root)
    disk_hash = hashlib.sha256(ASSET.read_bytes()).hexdigest()
    configure_carter_dynamics(root, cfg['stability'])
    np.testing.assert_allclose(visual_size(root), before)
    assert hashlib.sha256(ASSET.read_bytes()).hexdigest() == disk_hash
    assert root.GetAttribute('physxArticulation:solverVelocityIterationCount').Get() == 4
    for p in Usd.PrimRange(root):
        if p.GetName() in ('left_wheel', 'right_wheel'):
            drive = UsdPhysics.DriveAPI(p, 'angular')
            assert drive.GetMaxForceAttr().Get() == cfg['stability']['wheel_max_torque']
            assert drive.GetDampingAttr().Get() == cfg['stability']['wheel_damping']
        if p.GetName() in ('chassis_link', 'com_offset'):
            com = UsdPhysics.MassAPI(p).GetCenterOfMassAttr().Get()
            assert com[2] == pytest.approx(-cfg['stability']['com_lowering'])
    with pytest.raises(ValueError, match='already applied'):
        configure_carter_dynamics(root, cfg['stability'])
