from srb.core.action.action_group import ActionGroup
from srb.core.action.group.common import JointPositionBoundedActionGroup
from srb.core.action import JointPositionToLimitsActionCfg
from srb.core.actuator import ImplicitActuatorCfg
from srb.core.asset import ActiveTool, ArticulationCfg, Frame, Transform
from srb.core.sim import (
    ArticulationRootPropertiesCfg,
    CollisionPropertiesCfg,
    RigidBodyPropertiesCfg,
    UsdFileCfg,
    MeshCollisionPropertiesCfg,
)
from srb.utils.math import rpy_to_quat
from srb.utils.path import SRB_ASSETS_DIR_SRB_ROBOT


class RobotiqHandE(ActiveTool):
    ## Model
    asset_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/robotiq_hand_e",
        spawn=UsdFileCfg(
            usd_path=SRB_ASSETS_DIR_SRB_ROBOT.joinpath("gripper")
            .joinpath("robotiq_hand_e.usdz")
            .as_posix(),
            activate_contact_sensors=True,
            collision_props=CollisionPropertiesCfg(
                contact_offset=0.005, rest_offset=0.0
            ),
            rigid_props=RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
            mesh_collision_props=MeshCollisionPropertiesCfg(mesh_approximation="convexDecomposition"),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "Slider_[1-2]": 0.0,
            },
        ),
        actuators={
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["Slider_[1-2]"],
                velocity_limit_sim=4.0,
                effort_limit_sim=20.0,
                stiffness=400.0,
                damping=50.0,
            ),
        },
    )

    ## Actions
    actions: ActionGroup = JointPositionBoundedActionGroup(
        JointPositionToLimitsActionCfg(
            asset_name="robot",
            joint_names=["Slider_[1-2]"],
            scale=1.0,
            rescale_to_limits=True,
        ),
    )

    ## Frames
    frame_mount: Frame = Frame(
        prim_relpath="base_link",
        offset=Transform(pos=(0.0, 0.0, 0.08609), rot=rpy_to_quat((90.0, 0.0, 0.0))),
    )
    frame_tool_centre_point: Frame = Frame(
        prim_relpath="base_link", offset=Transform(pos=(0.0, 0.0, 0.135715))
    )
