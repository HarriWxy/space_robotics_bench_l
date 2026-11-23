from isaaclab.sim.schemas import *  # noqa: F403

# Isaac Lab develop (3.0) relocated the PhysX schema property configs to
# ``isaaclab_physx.sim.schemas`` and re-exposes them on ``isaaclab.sim.schemas``
# through a lazy module ``__getattr__`` forwarding shim. Those forwarded names
# are intentionally NOT listed in ``isaaclab.sim.schemas.__all__``, so the
# wildcard ``import *`` above does not bind them. Import them explicitly (an
# explicit ``from ... import`` triggers the module ``__getattr__``) so they
# re-export through ``srb.core.sim`` for the asset/robot/scenery modules that
# construct ``collision_props``/``rigid_props``/etc.
from isaaclab.sim.schemas import (  # noqa: F401
    ArticulationRootPropertiesCfg,
    CollisionPropertiesCfg,
    FixedTendonPropertiesCfg,
    JointDrivePropertiesCfg,
    MassPropertiesCfg,
    RigidBodyPropertiesCfg,
)
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
