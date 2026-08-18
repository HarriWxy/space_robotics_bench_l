import simforge_foundry

from srb.core.asset import DeformableObjectCfg, Object
from srb.core.sim import (
    DeformableBodyMaterialCfg,
    DeformableBodyPropertiesCfg,
    SimforgeAssetCfg,
)


class Bag(Object):
    asset_cfg: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/bag",
        spawn=SimforgeAssetCfg(
            assets=[simforge_foundry.BagRandom()],
            deformable_props=DeformableBodyPropertiesCfg(
                solver_position_iteration_count=64,
                # simulation_hexahedral_resolution=32,
                self_collision=False,
                # self_collision_filter_distance=None,
                settling_threshold=0.0,
                settling_damping=0.0,
                sleep_threshold=0.0,
                linear_damping=0.0,
                # collision_simplification_remeshing=True,
                # collision_simplification_remeshing_resolution=0,
                # collision_simplification_target_triangle_count=0,
                # collision_simplification_force_conforming=True,
                # contact_offset=None,
                # contact_offset=0.005,
                # rest_offset=None,
                # rest_offset=0.0,
                # max_depenetration_velocity=None,
            ),
            physics_material=DeformableBodyMaterialCfg(
                density=1500.0,
                dynamic_friction=0.5,
                youngs_modulus=250000.0,
                poissons_ratio=0.475,
                elasticity_damping=0.001,
                # damping_scale=1.0,
            ),
        ),
    )
