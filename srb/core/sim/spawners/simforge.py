"""SRB-specific SimForge spawner compatibility helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.sim import PreviewSurfaceCfg
from isaaclab.sim.schemas import modify_collision_properties
from isaaclab.sim.spawners.wrappers import MultiAssetSpawnerCfg, spawn_multi_asset
from isaaclab.sim.utils import get_current_stage
from pxr import Usd, UsdGeom, UsdPhysics
from simforge import ModelFileFormat
from simforge.integrations.isaaclab.spawner.from_files import UsdFileCfg
from simforge.utils import logging
from simforge.utils.color import color_palette_hue

if TYPE_CHECKING:
    from simforge.integrations.isaaclab.spawner.simforge_asset.cfg import (
        SimforgeAssetCfg,
    )


_IGNORED_SPAWN_ATTRIBUTES = (
    "func",
    "assets",
    "export_kwargs",
    "num_assets",
    "seed",
    "use_cache",
    "random_choice",
    # Collision schemas have to be applied after the generated USD reference is
    # composed.  The upstream file spawner applies these fields too early.
    "collision_props",
    "mesh_collision_props",
    # The planned source path is supplied directly to ``spawn_multi_asset``.
    "spawn_path",
)


def _ensure_mesh_collision_api(
    prim_path: str, collision_props: object | None
) -> list[Usd.Prim]:
    """Apply writable collision schemas to every generated terrain mesh."""
    stage = get_current_stage()
    root_prim = stage.GetPrimAtPath(prim_path)
    if not root_prim.IsValid():
        raise RuntimeError(f"Generated SimForge prim does not exist: {prim_path}")

    predicate = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
    meshes = [
        prim
        for prim in Usd.PrimRange(root_prim, predicate)
        if prim.IsA(UsdGeom.Mesh) and not prim.IsInstance() and not prim.IsInstanceProxy()
    ]
    if not meshes:
        raise RuntimeError(
            f"Generated SimForge prim has no editable mesh collider: {prim_path}"
        )

    collision_enabled = getattr(collision_props, "collision_enabled", None)
    if collision_enabled is None:
        collision_enabled = True
    for mesh in meshes:
        collision_api = UsdPhysics.CollisionAPI.Apply(mesh)
        collision_api.CreateCollisionEnabledAttr().Set(collision_enabled)
    return meshes


def spawn_simforge_static_asset(
    prim_path: str,
    cfg: "SimforgeAssetCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn one static SimForge asset at its clone-planned path.

    SimForge 0.4 wraps every generated file in a fresh ``MultiAssetSpawnerCfg``
    and appends ``.*`` to the requested path.  Isaac Lab 6.1 has already assigned
    a concrete clone source path to the enclosing single-asset spawner, so that
    extra wildcard creates ``scenery0`` beside the planned ``scenery`` prim.
    Besides breaking path lookup, it causes collision properties to be authored
    before the generated mesh exists.  This adapter retains the exact planned
    path and applies collision schemas only after the USD reference is composed.

    It intentionally supports one generated file, which is the contract for the
    static Moon and Mars terrain assets that use it.  Multi-variant SimForge
    assets must use a clone-aware multi-asset configuration instead.
    """
    if cfg.num_assets != 1 or len(cfg.assets) != 1:
        raise ValueError(
            "spawn_simforge_static_asset supports exactly one configured SimForge asset"
        )

    logging.debug(f'Spawning static SimForge asset for "{prim_path}"')
    asset = cfg.assets[0]
    generator = asset.generator_type(
        num_assets=1,
        seed=cfg.seed,
        file_format=ModelFileFormat.USDZ,
        use_cache=cfg.use_cache,
    )
    output = generator.generate_subprocess(asset, export_kwargs=cfg.export_kwargs)
    if len(output) != 1:
        raise RuntimeError(
            "Static SimForge terrain generation must produce exactly one USD file, "
            f"but generated {len(output)} files for {prim_path}"
        )
    usd_path, _metadata = output[0]

    spawn_kwargs = {
        attr_name: attr_value
        for attr_name, attr_value in cfg.__dict__.items()
        if attr_name not in _IGNORED_SPAWN_ATTRIBUTES
    }
    proto_cfg = UsdFileCfg(**spawn_kwargs)
    proto_cfg.usd_path = usd_path.as_posix()
    # Delay both collision writers until the mesh is visible on the stage.
    proto_cfg.collision_props = None
    proto_cfg.mesh_collision_props = None
    if getattr(cfg, "visual_material", None) is None and not generator.BAKER.enabled:
        proto_cfg.visual_material = PreviewSurfaceCfg(
            diffuse_color=color_palette_hue(1)[0]
        )

    spawned_prim = spawn_multi_asset(
        prim_path=prim_path,
        cfg=MultiAssetSpawnerCfg(
            assets_cfg=[proto_cfg],
            spawn_paths=[prim_path],
        ),
        translation=translation,
        orientation=orientation,
    )

    meshes = _ensure_mesh_collision_api(prim_path, cfg.collision_props)
    if cfg.mesh_collision_props is not None:
        cfg.mesh_collision_props.func(prim_path, cfg.mesh_collision_props)
    if cfg.collision_props is not None:
        stage = get_current_stage()
        for mesh in meshes:
            modify_collision_properties(
                mesh.GetPath().pathString,
                cfg.collision_props,
                stage=stage,
            )

    return spawned_prim
