from typing import TYPE_CHECKING, Tuple

from isaaclab.sim.spawners.wrappers import MultiAssetSpawnerCfg, spawn_multi_asset
from pxr import Usd

if TYPE_CHECKING:
    from .cfg import MultiShapeSpawnerCfg


def spawn_multi_shape(
    prim_path: str,
    cfg: "MultiShapeSpawnerCfg",
    translation: Tuple[float, float, float] | None = None,
    orientation: Tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    # Keep the variants in sync with the list exposed to Isaac Lab's clone
    # planner.  In particular, the planner may assign only one concrete path
    # in ``spawn_paths`` for the current environment.
    assets_cfg = cfg._build_assets_cfg()
    cfg.assets_cfg = assets_cfg

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
            mass_props=cfg.mass_props,
            rigid_props=cfg.rigid_props,
            collision_props=cfg.collision_props,
            activate_contact_sensors=cfg.activate_contact_sensors,
        ),
        translation=translation,
        orientation=orientation,
    )
