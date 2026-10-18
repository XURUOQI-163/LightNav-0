"""USD-only checks; run with Isaac's omni.usd.libs extension on PYTHONPATH."""

import pytest

pytest.importorskip("pxr.Usd")
from isaac_web.usd_scene import repair_static_colliders
from pxr import Usd, UsdGeom, UsdPhysics


def triangle(stage, path):
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    return mesh.GetPrim()


def test_empty_parent_collision_reaches_floor_without_colliding_sky_or_disabled_parts():
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    parent = UsdGeom.Mesh.Define(stage, "/World/Floor").GetPrim()
    UsdPhysics.CollisionAPI.Apply(parent)
    floor = triangle(stage, "/World/Floor/Geometry")
    sky = triangle(stage, "/World/Sky")
    disabled = UsdGeom.Xform.Define(stage, "/World/Floor/Disabled").GetPrim()
    UsdPhysics.CollisionAPI.Apply(disabled).CreateCollisionEnabledAttr(False)
    child = triangle(stage, "/World/Floor/Disabled/Geometry")
    assert repair_static_colliders(root) == 1
    assert floor.HasAPI(UsdPhysics.CollisionAPI)
    assert UsdPhysics.MeshCollisionAPI(floor).GetApproximationAttr().Get() == "none"
    assert not parent.HasAPI(UsdPhysics.CollisionAPI)
    assert not sky.HasAPI(UsdPhysics.CollisionAPI)
    assert not child.HasAPI(UsdPhysics.CollisionAPI)


def test_existing_mesh_collider_is_preserved_and_repair_is_idempotent():
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    floor = triangle(stage, "/World/Floor")
    UsdPhysics.CollisionAPI.Apply(floor)
    assert repair_static_colliders(root) == 0
    assert repair_static_colliders(root) == 0
    assert floor.HasAPI(UsdPhysics.CollisionAPI)


def test_mdl_aliases_normalize_in_session_without_modifying_source_or_parameters(tmp_path):
    from isaac_web.usd_scene import normalize_mdl_source_paths
    from pxr import Sdf

    materials = tmp_path / "Materials"
    materials.mkdir()
    mdl = materials / "Shared.mdl"
    mdl.write_text("mdl 1.0;\n")
    (tmp_path / "AliasA").symlink_to(materials, target_is_directory=True)
    (tmp_path / "AliasB").symlink_to(materials, target_is_directory=True)
    source_file = tmp_path / "source.usda"
    source = Usd.Stage.CreateNew(str(source_file))
    root = source.DefinePrim("/Root", "Xform")
    source.SetDefaultPrim(root)
    for name, alias, roughness in (("A", "AliasA", .2), ("B", "AliasB", .8)):
        shader = source.DefinePrim(f"/Root/{name}", "Shader")
        shader.CreateAttribute("info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath(f"{alias}/Shared.mdl"))
        shader.CreateAttribute("inputs:roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        shader.CreateAttribute("inputs:texture", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath("texture.png"))
    source.GetRootLayer().Save()
    original = source_file.read_bytes()
    stage = Usd.Stage.CreateInMemory()
    referenced = stage.DefinePrim("/Environment")
    referenced.GetReferences().AddReference(str(source_file))
    root_layer = stage.GetRootLayer().ExportToString()
    stats = normalize_mdl_source_paths(referenced)
    assert stats["canonical_files"] == 1
    assert stats["rewritten"] == 2
    for name, roughness in (("A", .2), ("B", .8)):
        shader = stage.GetPrimAtPath(f"/Environment/{name}")
        assert shader.GetAttribute("info:mdl:sourceAsset").Get().path == str(mdl)
        assert shader.GetAttribute("inputs:roughness").Get() == pytest.approx(roughness)
        assert shader.GetAttribute("inputs:texture").Get().path == "texture.png"
    assert normalize_mdl_source_paths(referenced)["rewritten"] == 0
    assert stage.GetRootLayer().ExportToString() == root_layer
    assert source_file.read_bytes() == original


def test_mdl_missing_and_remote_assets_are_not_rewritten():
    from isaac_web.usd_scene import normalize_mdl_source_paths
    from pxr import Sdf

    stage = Usd.Stage.CreateInMemory()
    root = stage.DefinePrim("/Environment")
    for name, path in (("Missing", "missing.mdl"), ("Remote", "https://example.com/a.mdl")):
        shader = stage.DefinePrim(f"/Environment/{name}", "Shader")
        shader.CreateAttribute("info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(path))
    before = stage.GetSessionLayer().ExportToString()
    assert normalize_mdl_source_paths(root)["rewritten"] == 0
    assert stage.GetSessionLayer().ExportToString() == before
