from simforge.integrations.isaaclab.schemas import (  # noqa: F401
    MeshCollisionPropertiesCfg,
)
from simforge.integrations.isaaclab.schemas.impl import (
    set_mesh_collision_properties as _set_mesh_collision_properties,
)
from isaaclab.sim.utils import get_current_stage
from pxr import Usd, UsdGeom, UsdPhysics


def set_mesh_collision_properties_with_collision(
    prim_path: str, cfg: MeshCollisionPropertiesCfg
):
    stage = get_current_stage()
    root_prim = stage.GetPrimAtPath(prim_path)
    if not root_prim.IsValid():
        return

    for prim in Usd.PrimRange(root_prim):
        if prim.IsInstance() or not prim.IsA(UsdGeom.Mesh):
            continue
        collision_api = UsdPhysics.CollisionAPI.Apply(prim)
        collision_api.CreateCollisionEnabledAttr().Set(True)

    _set_mesh_collision_properties(prim_path, cfg)
