from srb.core.action import (
    ActionGroup,
    JointPositionActionCfg,
    JointPositionActionGroup,
)
from srb.core.actuator import ImplicitActuatorCfg
from srb.core.asset import ArticulationCfg, Frame, Humanoid, Transform
from srb.core.sim import (
    ArticulationRootPropertiesCfg,
    CollisionPropertiesCfg,
    RigidBodyPropertiesCfg,
    UsdFileCfg,
)
from srb.utils.math import rpy_to_quat
from srb.utils.nucleus import ISAACLAB_NUCLEUS_DIR


class UnitreeH1(Humanoid):
    ## Model
    asset_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/unitree_h1",
        spawn=UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/H1/h1.usd",
            # H1's USD contains instanceable collision meshes.  Disable USD
            # instancing before applying collision overrides so that
            # contact_offset and rest_offset are authored on the colliders.
            make_uninstanceable=True,
            activate_contact_sensors=True,
            collision_props=CollisionPropertiesCfg(
                contact_offset=0.005, rest_offset=0.0
            ),
            rigid_props=RigidBodyPropertiesCfg(
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.05),
            joint_pos={
                ".*_hip_yaw": 0.0,
                ".*_hip_roll": 0.0,
                ".*_hip_pitch": -0.28,
                ".*_knee": 0.79,
                ".*_ankle": -0.52,
                "torso": 0.0,
                ".*_shoulder_pitch": 0.28,
                ".*_shoulder_roll": 0.0,
                ".*_shoulder_yaw": 0.0,
                ".*_elbow": 0.52,
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[
                    ".*_hip_yaw",
                    ".*_hip_roll",
                    ".*_hip_pitch",
                    ".*_knee",
                    "torso",
                ],
                effort_limit_sim=300.0,
                velocity_limit_sim=100.0,
                stiffness={
                    ".*_hip_yaw": 150.0,
                    ".*_hip_roll": 150.0,
                    ".*_hip_pitch": 200.0,
                    ".*_knee": 200.0,
                    "torso": 200.0,
                },
                damping={
                    ".*_hip_yaw": 5.0,
                    ".*_hip_roll": 5.0,
                    ".*_hip_pitch": 5.0,
                    ".*_knee": 5.0,
                    "torso": 5.0,
                },
                armature={
                    ".*_hip_.*": 0.01,
                    ".*_knee": 0.01,
                    "torso": 0.01,
                },
            ),
            "feet": ImplicitActuatorCfg(
                joint_names_expr=[".*_ankle"],
                effort_limit_sim=100.0,
                velocity_limit_sim=100.0,
                stiffness={".*_ankle": 20.0},
                damping={".*_ankle": 4.0},
                armature={".*_ankle": 0.01},
            ),
            "arms": ImplicitActuatorCfg(
                joint_names_expr=[
                    ".*_shoulder_pitch",
                    ".*_shoulder_roll",
                    ".*_shoulder_yaw",
                    ".*_elbow",
                ],
                effort_limit_sim=300.0,
                velocity_limit_sim=100.0,
                stiffness={
                    ".*_shoulder_pitch": 40.0,
                    ".*_shoulder_roll": 40.0,
                    ".*_shoulder_yaw": 40.0,
                    ".*_elbow": 40.0,
                },
                damping={
                    ".*_shoulder_pitch": 10.0,
                    ".*_shoulder_roll": 10.0,
                    ".*_shoulder_yaw": 10.0,
                    ".*_elbow": 10.0,
                },
                armature={
                    ".*_shoulder_.*": 0.01,
                    ".*_elbow": 0.01,
                },
            ),
        },
    )

    ## Actions
    actions: ActionGroup = JointPositionActionGroup(
        JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5)
    )

    ## Frames
    frame_base: Frame = Frame(prim_relpath="pelvis")
    frame_front_camera: Frame = Frame(
        prim_relpath="pelvis/camera_front",
        offset=Transform(
            pos=(-0.7675, 0.0, 1.9793),
            rot=rpy_to_quat(0.0, 15.0, -90.0),
        ),
    )

    ## Links
    regex_feet_links: str = "(left|right)_ankle_link"


class UnitreeG1(Humanoid):
    ## Model
    asset_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/unitree_g1",
        spawn=UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/G1/g1.usd",
            activate_contact_sensors=True,
            collision_props=CollisionPropertiesCfg(
                contact_offset=0.005, rest_offset=0.0
            ),
            rigid_props=RigidBodyPropertiesCfg(
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.74),
            joint_pos={
                ".*_hip_pitch_joint": -0.20,
                ".*_knee_joint": 0.42,
                ".*_ankle_pitch_joint": -0.23,
                ".*_elbow_pitch_joint": 0.87,
                "left_shoulder_roll_joint": 0.16,
                "left_shoulder_pitch_joint": 0.35,
                "right_shoulder_roll_joint": -0.16,
                "right_shoulder_pitch_joint": 0.35,
                "left_one_joint": 1.0,
                "right_one_joint": -1.0,
                "left_two_joint": 0.52,
                "right_two_joint": -0.52,
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[
                    ".*_hip_yaw_joint",
                    ".*_hip_roll_joint",
                    ".*_hip_pitch_joint",
                    ".*_knee_joint",
                    "torso_joint",
                ],
                effort_limit_sim=300.0,
                velocity_limit_sim=100.0,
                stiffness={
                    ".*_hip_yaw_joint": 150.0,
                    ".*_hip_roll_joint": 150.0,
                    ".*_hip_pitch_joint": 200.0,
                    ".*_knee_joint": 200.0,
                    "torso_joint": 200.0,
                },
                damping={
                    ".*_hip_yaw_joint": 5.0,
                    ".*_hip_roll_joint": 5.0,
                    ".*_hip_pitch_joint": 5.0,
                    ".*_knee_joint": 5.0,
                    "torso_joint": 5.0,
                },
                armature={
                    ".*_hip_.*": 0.01,
                    ".*_knee_joint": 0.01,
                    "torso_joint": 0.01,
                },
            ),
            "feet": ImplicitActuatorCfg(
                effort_limit_sim=20.0,
                joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
                stiffness=20.0,
                damping=2.0,
                armature=0.01,
            ),
            "arms": ImplicitActuatorCfg(
                joint_names_expr=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_pitch_joint",
                    ".*_elbow_roll_joint",
                    ".*_five_joint",
                    ".*_three_joint",
                    ".*_six_joint",
                    ".*_four_joint",
                    ".*_zero_joint",
                    ".*_one_joint",
                    ".*_two_joint",
                ],
                effort_limit_sim=300.0,
                velocity_limit_sim=100.0,
                stiffness=40.0,
                damping=10.0,
                armature={
                    ".*_shoulder_.*": 0.01,
                    ".*_elbow_.*": 0.01,
                    ".*_five_joint": 0.001,
                    ".*_three_joint": 0.001,
                    ".*_six_joint": 0.001,
                    ".*_four_joint": 0.001,
                    ".*_zero_joint": 0.001,
                    ".*_one_joint": 0.001,
                    ".*_two_joint": 0.001,
                },
            ),
        },
    )

    ## Actions
    actions: ActionGroup = JointPositionActionGroup(
        JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5)
    )

    ## Frames
    frame_base: Frame = Frame(prim_relpath="pelvis")
    frame_imu: Frame | None = Frame(prim_relpath="imu_link")
    frame_front_camera: Frame = Frame(
        prim_relpath="pelvis/camera_front",
        offset=Transform(
            pos=(-0.7675, 0.0, 1.9793),
            rot=rpy_to_quat(0.0, 15.0, -90.0),
        ),
    )

    ## Links
    regex_feet_links: str = "(left|right)_ankle_roll_link"
