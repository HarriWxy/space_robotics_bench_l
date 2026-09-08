from srb.core.action import ActionGroup, WheeledDriveActionCfg, WheeledDriveActionGroup
from srb.core.actuator import ImplicitActuatorCfg
from srb.core.asset import ArticulationCfg, Frame, Transform, WheeledRobot
from srb.core.sim import (
    ArticulationRootPropertiesCfg,
    CollisionPropertiesCfg,
    RigidBodyPropertiesCfg,
    UsdFileCfg,
)
from srb.utils.math import deg_to_rad, rpy_to_quat
from srb.utils.path import SRB_ASSETS_DIR_SRB_ROBOT


class OpenSourceRover(WheeledRobot):
    ## Model
    asset_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/open_source_rover",
        spawn=UsdFileCfg(
            usd_path=SRB_ASSETS_DIR_SRB_ROBOT.joinpath("rover")
            .joinpath("osr.usdz")
            .as_posix(),
            activate_contact_sensors=True,
            collision_props=CollisionPropertiesCfg(
                contact_offset=0.02, rest_offset=0.005
            ),
            rigid_props=RigidBodyPropertiesCfg(
                max_linear_velocity=1.5,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
            ),
        ),
        actuators={
            "joint_wheel": ImplicitActuatorCfg(
                joint_names_expr=["joint_wheel_.*"],
                joint_velocity_limit=40.0,
                joint_effort_limit=50.0,
                damping=2000.0,
                stiffness=0.0,
            ),
            "joints_steer": ImplicitActuatorCfg(
                joint_names_expr=["joint_steer_.*"],
                joint_velocity_limit=0.5,
                joint_effort_limit=500.0,
                damping=80.0,
                stiffness=100.0,
            ),
            "joints_rocker": ImplicitActuatorCfg(
                joint_names_expr=["joint_rocker_.*"],
                joint_velocity_limit=0.3,
                joint_effort_limit=1500.0,
                damping=2.0,
                stiffness=25.0,
            ),
            "joints_bogie": ImplicitActuatorCfg(
                joint_names_expr=["joint_bogie_.*"],
                joint_velocity_limit=0.6,
                joint_effort_limit=750.0,
                damping=4.0,
                stiffness=1.0,
            ),
            "joint_linkage": ImplicitActuatorCfg(
                joint_names_expr=["joint_linkage"],
                joint_velocity_limit=0.2,
                joint_effort_limit=25.0,
                damping=0.1,
                stiffness=0.05,
            ),
            # 500 / 50 tendon
        },
    )

    ## Actions
    actions: ActionGroup = WheeledDriveActionGroup(
        WheeledDriveActionCfg(
            asset_name="robot",
            wheelbase=(0.5835, 0.39),
            wheelbase_mid=0.52,
            wheel_radius=0.072,
            steering_joint_names=[
                "joint_steer_rr",
                "joint_steer_lf",
                "joint_steer_rf",
                "joint_steer_lr",
            ],
            drive_joint_names=[
                "joint_wheel_rf",
                "joint_wheel_lf",
                "joint_wheel_rm",
                "joint_wheel_lm",
                "joint_wheel_rr",
                "joint_wheel_lr",
            ],
            scale_linear=0.2,
            scale_angular=deg_to_rad(20),
        )
    )

    ## Frames
    frame_base: Frame = Frame(prim_relpath="chassis")
    frame_payload_mount: Frame = Frame(
        prim_relpath="chassis",
        offset=Transform(
            pos=(-0.05, 0.0, 0.121),
            rot=rpy_to_quat(0.0, 0.0, 180.0),
        ),
    )
    frame_manipulator_mount: Frame = Frame(
        prim_relpath="chassis",
        offset=Transform(
            pos=(-0.1, 0.0, 0.121),
            rot=rpy_to_quat(0.0, 0.0, 180.0),
        ),
    )
    frame_front_camera: Frame = Frame(
        prim_relpath="chassis/camera_front",
        offset=Transform(
            pos=(-0.17, 0.0, 0.075),
            rot=rpy_to_quat(0.0, 15.0, 180.0),
        ),
    )
