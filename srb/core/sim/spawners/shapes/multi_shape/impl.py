from typing import TYPE_CHECKING, Tuple

from isaaclab.sim.spawners.wrappers import MultiAssetSpawnerCfg, spawn_multi_asset
from pxr import Usd

from srb.core.sim import CapsuleCfg, ConeCfg, CuboidCfg, CylinderCfg, SphereCfg

if TYPE_CHECKING:
    from .cfg import MultiShapeSpawnerCfg

IGNORED_SPAWN_ATTRIBUTES = (
    "func",
    "shapes",
    "scale",
    "radius",
    "height",
    "axis",
    "random_choice",
    "spawn_paths",
)


def spawn_multi_shape(
    prim_path: str,
    cfg: "MultiShapeSpawnerCfg",
    translation: Tuple[float, float, float] | None = None,
    orientation: Tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    # Collect shape spawner kwargs
    shape_cfg_kwargs = {
        attr_name: attr_value
        for attr_name, attr_value in cfg.__dict__.items()
        if attr_name not in IGNORED_SPAWN_ATTRIBUTES
    }

    # Collect shape spawner configs
    assets_cfg = []
    if not cfg.shapes or "cuboid" in cfg.shapes:
        assets_cfg.append(CuboidCfg(size=cfg.scale, **shape_cfg_kwargs))
    if not cfg.shapes or "sphere" in cfg.shapes:
        assets_cfg.append(
            SphereCfg(radius=cfg.radius or cfg.scale[0], **shape_cfg_kwargs)
        )
    if not cfg.shapes or "cylinder" in cfg.shapes:
        assets_cfg.append(
            CylinderCfg(
                radius=cfg.radius or cfg.scale[0],
                height=cfg.height or cfg.scale[1],
                axis=cfg.axis,
                **shape_cfg_kwargs,
            )
        )
    if not cfg.shapes or "capsule" in cfg.shapes:
        assets_cfg.append(
            CapsuleCfg(
                radius=cfg.radius or cfg.scale[0],
                height=cfg.height or cfg.scale[1],
                axis=cfg.axis,
                **shape_cfg_kwargs,
            )
        )
    if not cfg.shapes or "cone" in cfg.shapes:
        assets_cfg.append(
            ConeCfg(
                radius=cfg.radius or cfg.scale[0],
                height=cfg.height or cfg.scale[1],
                axis=cfg.axis,
                **shape_cfg_kwargs,
            )
        )

    if cfg.spawn_paths is not None:
        # Clone planner has assigned concrete paths — pass them through
        pass
    elif ".*" not in prim_path.split("/")[-1]:
        prim_path = prim_path + ".*"

    # Create and spawn multi-asset configuration
    return spawn_multi_asset(
        prim_path=prim_path,
        cfg=MultiAssetSpawnerCfg(
            assets_cfg=assets_cfg,
            random_choice=cfg.random_choice,
            spawn_paths=cfg.spawn_paths,
        ),
        translation=translation,
        orientation=orientation,
    )
