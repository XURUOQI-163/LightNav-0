"""Runtime-only repairs for GRScenes' collision tags on empty grouping prims."""


def repair_static_colliders(root):
    """Copy enabled group collision to concrete meshes, preserving triangle openings.

    USD CollisionAPI is not inherited by children. GRScenes navigation assets put
    it on empty Mesh/Xform parents. Only descend through authored collision scopes;
    unrelated visual geometry (e.g. the sky) must not acquire collision. This helper
    is for the static environment, never the Carter articulation.
    """
    from pxr import UsdGeom, UsdPhysics

    count = 0

    def visit(prim, enabled=False):
        nonlocal count
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError(f"Expected a static navigation scene, found rigid body: {prim.GetPath()}")
        tagged = prim.HasAPI(UsdPhysics.CollisionAPI)
        if tagged:
            enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False
        is_mesh = prim.IsA(UsdGeom.Mesh)
        has_points = is_mesh and bool(UsdGeom.Mesh(prim).GetPointsAttr().Get())
        if enabled and has_points and not tagged:
            UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("none")
            count += 1
        for child in prim.GetChildren():
            visit(child, enabled)
        if tagged and not has_points and (is_mesh or prim.IsA(UsdGeom.Xform)):
            prim.RemoveAPI(UsdPhysics.CollisionAPI)
            prim.RemoveAPI(UsdPhysics.MeshCollisionAPI)

    visit(root)
    return count


def normalize_mdl_source_paths(root):
    """Unify local MDL symlink aliases in an anonymous session layer only.

    Keep separate shader prims, parameters, textures and bindings. A resolved
    asset path is required: resolving a relative USD path against cwd is wrong.
    Remote assets, package paths and time-varying sources are left untouched.
    """
    from pathlib import Path

    from pxr import Sdf, Usd

    stage = root.GetStage()
    layer = stage.GetSessionLayer()
    if layer is None or not layer.anonymous:
        raise ValueError("MDL normalization requires an anonymous session layer")
    cache, aliases, targets = {}, set(), set()
    stats = dict(rewritten=0, skipped=0, aliases=0, canonical_files=0)
    with Usd.EditContext(stage, layer):
        for prim in Usd.PrimRange(root):
            attr = prim.GetAttribute("info:mdl:sourceAsset")
            if not attr:
                continue
            asset = attr.Get()
            if (prim.IsInstanceProxy() or attr.GetNumTimeSamples() or
                    not isinstance(asset, Sdf.AssetPath)):
                stats["skipped"] += 1
                continue
            raw, resolved = asset.path, asset.resolvedPath
            if (not resolved or not Path(resolved).is_absolute() or
                    not resolved.lower().endswith(".mdl") or "://" in raw or
                    "[" in resolved or "[" in raw):
                stats["skipped"] += 1
                continue
            if resolved not in cache:
                try:
                    target = Path(resolved).resolve(strict=True)
                    cache[resolved] = str(target) if target.is_file() else None
                except (OSError, RuntimeError):
                    cache[resolved] = None
            canonical = cache[resolved]
            if canonical is None:
                stats["skipped"] += 1
                continue
            aliases.add(resolved)
            targets.add(canonical)
            if raw != canonical:
                if not attr.Set(Sdf.AssetPath(canonical)):
                    raise RuntimeError(f"Cannot normalize MDL source: {attr.GetPath()}")
                stats["rewritten"] += 1
    stats.update(aliases=len(aliases), canonical_files=len(targets))
    return stats
