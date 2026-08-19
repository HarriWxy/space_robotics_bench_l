from srb.core.asset import AssetBaseCfg, Frame, Pedestal, Transform
from srb.core.sim import CollisionPropertiesCfg, UsdFileCfg
from srb.utils.path import SRB_ASSETS_DIR_SRB_OBJECT


def spawn_pedestal_with_collision(
    prim_path: str,
    cfg,
    translation=None,
    orientation=None,
    **kwargs,
):
    from isaaclab.sim.schemas import modify_collision_properties
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Usd, UsdGeom, UsdPhysics
    from simforge.integrations.isaaclab.spawner.from_files.impl import spawn_from_usd

    prim = spawn_from_usd(
        prim_path,
        cfg.replace(collision_props=None),
        translation=translation,
        orientation=orientation,
        **kwargs,
    )
    stage = get_current_stage()
    for mesh_prim in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if not mesh_prim.IsA(UsdGeom.Mesh):
            continue
        if not mesh_prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(mesh_prim)
        mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
        mesh_collision_api.GetApproximationAttr().Set("convexHull")
        if cfg.collision_props is not None:
            modify_collision_properties(
                mesh_prim.GetPath().pathString,
                cfg.collision_props,
                stage,
            )
    return prim


class IndustrialPedestal25(Pedestal):
    asset_cfg: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/pedestal",
        spawn=UsdFileCfg(
            usd_path=SRB_ASSETS_DIR_SRB_OBJECT.joinpath(
                "industrial_pedestal_25cm.usdz"
            ).as_posix(),
            func=spawn_pedestal_with_collision,
            collision_props=CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.005,
                rest_offset=0.0,
            ),
        ),
    )

    frame_manipulator_mount: Frame = Frame(
        prim_relpath="pedestal", offset=Transform(pos=(0.0, 0.0, 0.25))
    )


class IndustrialPedestal50(Pedestal):
    asset_cfg: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/pedestal",
        spawn=UsdFileCfg(
            usd_path=SRB_ASSETS_DIR_SRB_OBJECT.joinpath(
                "industrial_pedestal_50cm.usdz"
            ).as_posix(),
            func=spawn_pedestal_with_collision,
            collision_props=CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.005,
                rest_offset=0.0,
            ),
        ),
    )

    frame_manipulator_mount: Frame = Frame(
        prim_relpath="pedestal", offset=Transform(pos=(0.0, 0.0, 0.5))
    )


class IndustrialPedestal100(Pedestal):
    asset_cfg: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/pedestal",
        spawn=UsdFileCfg(
            usd_path=SRB_ASSETS_DIR_SRB_OBJECT.joinpath(
                "industrial_pedestal_100cm.usdz"
            ).as_posix(),
            func=spawn_pedestal_with_collision,
            collision_props=CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.005,
                rest_offset=0.0,
            ),
        ),
    )

    frame_manipulator_mount: Frame = Frame(
        prim_relpath="pedestal", offset=Transform(pos=(0.0, 0.0, 1.0))
    )
