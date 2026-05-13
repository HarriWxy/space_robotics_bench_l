from dataclasses import MISSING
from typing import Sequence, Tuple

import torch

from srb._typing import StepReturn
from srb.core.asset import (
    Articulation,
    AssetVariant,
    Object,
    RigidObject,
    RigidObjectCfg,
)
from srb.core.domain import Domain
from srb.core.env import (
    ManipulationEnv,
    ManipulationEnvCfg,
    ManipulationEventCfg,
    ManipulationSceneCfg,
)
from srb.core.manager import EventTermCfg, SceneEntityCfg
from srb.core.marker import VisualizationMarkers, VisualizationMarkersCfg
from srb.core.mdp import reset_root_state_uniform
from srb.core.sensor import ContactSensor, ContactSensorCfg
from srb.core.sim import PreviewSurfaceCfg, SphereCfg
from srb.utils.cfg import configclass
from srb.utils.math import (
    matrix_from_quat,
    rotmat_to_rot6d,
    scale_transform,
    subtract_frame_transforms,
)

from .asset import select_sample
from .terrain import terrain_surface_heights

##############
### Config ###
##############


@configclass
class SceneCfg(ManipulationSceneCfg):
    sample: RigidObjectCfg = MISSING  # type: ignore


@configclass
class EventCfg(ManipulationEventCfg):
    randomize_object_state: EventTermCfg = EventTermCfg(
        func=reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("sample"),
            "pose_range": MISSING,
            "velocity_range": {},
        },
    )


@configclass
class TaskCfg(ManipulationEnvCfg):
    ## Scenario
    domain: Domain = Domain.MARS

    ## Assets
    sample: Object | AssetVariant = AssetVariant.DATASET

    ## Scene
    scene: SceneCfg = SceneCfg()

    ## Events
    events: EventCfg = EventCfg()

    ## Time
    episode_length_s: float = 15 # 7.5
    is_finite_horizon: bool = True

    ## Curriculum
    stage: int = 1

    ## Target
    tf_pos_target: Tuple[float, float, float] = (0.5, 0.0, 0.75)
    tf_quat_target: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    target_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/target",
        markers={
            "target": SphereCfg(
                radius=0.02,
                visual_material=PreviewSurfaceCfg(emissive_color=(0.2, 0.2, 0.8)),
            )
        },
    )

    def __post_init__(self):
        super().__post_init__()

        # Scene: Sample
        sample = select_sample(
            self,
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0)),
            activate_contact_sensors=True,
        )
        self.scene.sample = sample.asset_cfg

        # Event: Randomize object state
        self.events.randomize_object_state.params["pose_range"] = (
            sample.state_randomizer.params["pose_range"]
        )

        # Sensor: End-effector contacts
        if isinstance(self.scene.contacts_end_effector, ContactSensorCfg):
            self.scene.contacts_end_effector.filter_prim_paths_expr = [
                self.scene.sample.prim_path
            ]

        # Update seed & number of variants for procedural assets
        self._update_procedural_assets()


############
### Task ###
############


class Task(ManipulationEnv):
    cfg: TaskCfg

    def __init__(self, cfg: TaskCfg, **kwargs):
        super().__init__(cfg, **kwargs)

        ## Get scene assets
        self._obj: RigidObject = self.scene["sample"]
        self._target_marker: VisualizationMarkers = VisualizationMarkers(
            self.cfg.target_marker_cfg
        )

        ## Initialize buffers
        self._tf_pos_obj_initial = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.device
        )
        self._tf_pos_target = self.scene.env_origins + torch.tensor(
            self.cfg.tf_pos_target, dtype=torch.float32, device=self.device
        ).repeat(self.num_envs, 1)
        self._tf_quat_target = torch.tensor(
            self.cfg.tf_quat_target, dtype=torch.float32, device=self.device
        ).repeat(self.num_envs, 1)

        ## Visualize target
        self._target_marker.visualize(self._tf_pos_target, self._tf_quat_target)

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        root_pose = self._obj.data.root_pos_w[env_ids_tensor].clone()
        root_quat = self._obj.data.root_quat_w[env_ids_tensor].clone()
        baseline_pose = root_pose.clone()
        terrain_prim_path = self._scenery.prim_paths[0] if self._scenery is not None else None

        if terrain_prim_path is not None:
            terrain_heights = terrain_surface_heights(
                terrain_prim_path,
                root_pose,
                self.scene.env_origins,
                env_ids_tensor,
            )
            baseline_pose[:, 2] = terrain_heights + 0.08
            root_pose[:, 2] = baseline_pose[:, 2]

        if self.cfg.stage > 1 and len(env_ids_tensor) > 0: # 仅在第二阶段及以上进行预抓取位置的随机化
            tf_pos_end_effector = self._tf_end_effector.data.target_pos_w[env_ids_tensor, 0, :]
            curriculum_mix = torch.rand(len(env_ids_tensor), device=self.device)
            pregrasp_mask = curriculum_mix < 0.45
            transport_mask = (curriculum_mix >= 0.45) & (curriculum_mix < 0.75)

            if pregrasp_mask.any():
                pregrasp_pose = root_pose[pregrasp_mask].clone()
                pregrasp_pose[:, :2] = (
                    tf_pos_end_effector[pregrasp_mask, :2]
                    + 0.012 * torch.randn_like(tf_pos_end_effector[pregrasp_mask, :2])
                )
                if terrain_prim_path is not None:
                    pregrasp_heights = terrain_surface_heights(
                        terrain_prim_path,
                        pregrasp_pose,
                        self.scene.env_origins,
                        env_ids_tensor[pregrasp_mask],
                    )
                    pregrasp_pose[:, 2] = torch.maximum(
                        pregrasp_heights + 0.05,
                        tf_pos_end_effector[pregrasp_mask, 2] - 0.07,
                    )
                else:
                    pregrasp_pose[:, 2] = tf_pos_end_effector[pregrasp_mask, 2] - 0.07
                root_pose[pregrasp_mask] = pregrasp_pose

            if transport_mask.any():
                transport_pose = root_pose[transport_mask].clone()
                transport_pose[:, :2] = (
                    tf_pos_end_effector[transport_mask, :2]
                    + 0.006 * torch.randn_like(tf_pos_end_effector[transport_mask, :2])
                )
                if terrain_prim_path is not None:
                    transport_heights = terrain_surface_heights(
                        terrain_prim_path,
                        transport_pose,
                        self.scene.env_origins,
                        env_ids_tensor[transport_mask],
                    )
                    transport_pose[:, 2] = torch.maximum(
                        transport_heights + 0.14,
                        tf_pos_end_effector[transport_mask, 2] - 0.04,
                    )
                else:
                    transport_pose[:, 2] = tf_pos_end_effector[transport_mask, 2] - 0.04
                root_pose[transport_mask] = transport_pose

            if isinstance(self._end_effector, Articulation):
                end_effector_joint_pos = self._end_effector.data.default_joint_pos[env_ids_tensor].clone()
                end_effector_joint_vel = self._end_effector.data.default_joint_vel[env_ids_tensor].clone()
                if pregrasp_mask.any():
                    end_effector_joint_pos[pregrasp_mask] = -0.004
                if transport_mask.any():
                    end_effector_joint_pos[transport_mask] = -0.014
                self._end_effector.write_joint_state_to_sim(
                    end_effector_joint_pos,
                    end_effector_joint_vel,
                    env_ids=env_ids_tensor,
                )

        self._obj.write_root_pose_to_sim(
            torch.cat([root_pose, root_quat], dim=-1),
            env_ids=env_ids_tensor,
        )
        self._obj.write_root_velocity_to_sim(
            torch.zeros((len(env_ids_tensor), 6), dtype=root_pose.dtype, device=self.device),
            env_ids=env_ids_tensor,
        )
        self._tf_pos_obj_initial[env_ids_tensor] = baseline_pose
        self._robot.data.joint_acc[env_ids] = 0.0
        self._robot.data.applied_torque[env_ids] = 0.0
        self._robot.data.joint_vel[env_ids] = 0.0

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        processed_actions = actions
        if actions.size(1) > 6:
            processed_actions = actions.clone()
            gripper_actions = processed_actions[:, 6:]
            if self.cfg.stage <= 1:
                processed_actions[:, 6:] = torch.ones_like(gripper_actions)
            else:
                tf_pos_end_effector = self._tf_end_effector.data.target_pos_w[:, 0, :]
                tf_pos_obj = self._obj.data.root_com_pos_w
                distance_xy_to_obj = torch.norm(
                    tf_pos_obj[:, :2] - tf_pos_end_effector[:, :2], dim=1
                )
                height_above_obj = tf_pos_end_effector[:, 2] - tf_pos_obj[:, 2]
                lateral_gate = torch.clamp((0.09 - distance_xy_to_obj) / 0.09, 0.0, 1.0)
                lower_gate = torch.clamp((height_above_obj + 0.03) / 0.06, 0.0, 1.0)
                upper_gate = torch.clamp((0.12 - height_above_obj) / 0.12, 0.0, 1.0)
                close_scale = 0.15 + 0.85 * lateral_gate * lower_gate * upper_gate
                close_action = torch.clamp_min(-gripper_actions, 0.0)
                open_action = torch.clamp_min(gripper_actions, 0.0)
                processed_actions[:, 6:] = open_action - close_action * close_scale.unsqueeze(-1)
        super()._pre_physics_step(processed_actions)

    def extract_step_return(self) -> StepReturn:
        return _compute_step_return(
            ## Time
            episode_length=self.episode_length_buf,
            max_episode_length=self.max_episode_length,
            truncate_episodes=self.cfg.truncate_episodes,
            stage=self.cfg.stage,
            ## Actions
            act_current=self.action_manager.action,
            act_previous=self.action_manager.prev_action,
            ## States
            # Joints
            joint_pos_robot=self._robot.data.joint_pos,
            joint_pos_limits_robot=(
                self._robot.data.soft_joint_pos_limits
                if torch.all(torch.isfinite(self._robot.data.soft_joint_pos_limits))
                else None
            ),
            joint_pos_end_effector=self._end_effector.data.joint_pos
            if isinstance(self._end_effector, Articulation)
            else None,
            joint_pos_limits_end_effector=(
                self._end_effector.data.soft_joint_pos_limits
                if isinstance(self._end_effector, Articulation)
                and torch.all(
                    torch.isfinite(self._end_effector.data.soft_joint_pos_limits)
                )
                else None
            ),
            joint_vel_robot=self._robot.data.joint_vel,
            joint_vel_end_effector=self._end_effector.data.joint_vel
            if isinstance(self._end_effector, Articulation)
            else None,
            joint_acc_robot=self._robot.data.joint_acc,
            joint_applied_torque_robot=self._robot.data.applied_torque,
            # Kinematics
            fk_pos_end_effector=self._tf_end_effector.data.target_pos_source[:, 0, :],
            fk_quat_end_effector=self._tf_end_effector.data.target_quat_source[:, 0, :],
            # Transforms (world frame)
            tf_pos_end_effector=self._tf_end_effector.data.target_pos_w[:, 0, :],
            tf_quat_end_effector=self._tf_end_effector.data.target_quat_w[:, 0, :],
            tf_pos_obj_initial=self._tf_pos_obj_initial,
            tf_pos_obj=self._obj.data.root_com_pos_w,
            tf_quat_obj=self._obj.data.root_com_quat_w,
            vel_lin_obj=self._obj.data.root_lin_vel_w,
            vel_ang_obj=self._obj.data.root_ang_vel_w,
            tf_pos_target=self._tf_pos_target,
            tf_quat_target=self._tf_quat_target,
            terrain_height_end_effector=(
                terrain_surface_heights(
                    self._scenery.prim_paths[0],
                    self._tf_end_effector.data.target_pos_w[:, 0, :],
                    self.scene.env_origins,
                    torch.arange(self.num_envs, device=self.device, dtype=torch.long),
                )
                if self._scenery is not None
                else torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            ),
            # Contacts
            contact_forces_robot=self._contacts_robot.data.net_forces_w,  # type: ignore
            contact_forces_end_effector=self._contacts_end_effector.data.net_forces_w
            if isinstance(self._contacts_end_effector, ContactSensor)
            else None,
            contact_forces_end_effector_collision=(
                self._contacts_end_effector_collision.data.net_forces_w
                if isinstance(self._contacts_end_effector_collision, ContactSensor)
                else None
            ),
            contact_force_matrix_end_effector=self._contacts_end_effector.data.force_matrix_w
            if isinstance(self._contacts_end_effector, ContactSensor)
            else None,
        )

    # def _is_gripper_closed(self) -> torch.Tensor:
    #     """返回夹爪是否闭合的布尔张量（基于关节位置或动作）"""
    #     # 假设夹爪关节名称为 "Slider_[1-2]"，闭合位置为 -0.025
    #     if hasattr(self._end_effector, 'data') and self._end_effector.data is not None:
    #         joint_pos = self._end_effector.data.joint_pos
    #         # 取平均值作为闭合程度
    #         gripper_closed = joint_pos.mean(dim=1) < -0.02
    #         return gripper_closed
    #     return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    # def _pre_physics_step(self, actions: torch.Tensor) -> torch.Tensor:
    #     # 假设动作空间的结构：[arm_actions (6维), gripper_action (1维)]
    #     # 如果使用 InverseKinematics，前6维是末端位姿增量，最后一维是夹爪
    #     arm_actions = actions[:, :-1]
    #     gripper_action = actions[:, -1]

    #     # 获取末端执行器到物体的距离（已在 extract_step_return 中计算，但这里需要实时获取）
    #     # 最简单的方式：从当前观测中提取距离。但在此方法中，可以访问 self._obj 和 self._tf_end_effector
    #     if hasattr(self, '_obj') and hasattr(self, '_tf_end_effector'):
    #         ee_pos = self._tf_end_effector.data.target_pos_w[:, 0, :]  # (num_envs, 3)
    #         obj_pos = self._obj.data.root_com_pos_w
    #         distance_to_obj = torch.norm(ee_pos - obj_pos, dim=1)
    #         # 也可以检查接触力
    #         if self._contacts_end_effector is not None:
    #             contact_force = self._contacts_end_effector.data.force_matrix_w
    #             contact_exist = torch.norm(contact_force, dim=-1).max(dim=1)[0] > 0.1
    #         else:
    #             contact_exist = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    #     else:
    #         distance_to_obj = torch.full((self.num_envs,), 1.0, device=self.device)
    #         contact_exist = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    #     # 只有距离小于阈值（例如 0.05 米）或有接触时，才允许闭合动作；否则强制设为打开（0.0）
    #     ALLOW_CLOSE_DISTANCE = 0.05
    #     allow_close = (distance_to_obj < ALLOW_CLOSE_DISTANCE) | contact_exist
    #     # 注意：夹爪闭合通常对应负值（例如 -0.025），打开为 0.0
    #     # 如果动作是二进制的（0/1），则闭合为 1，打开为 0，需相应调整。
    #     # 这里假设闭合为负，张开为 0。
    #     gripper_action = torch.where(allow_close, gripper_action, torch.zeros_like(gripper_action))

    #     # 重新组合动作
    #     modified_actions = torch.cat([arm_actions, gripper_action.unsqueeze(1)], dim=1)
    #     return super()._pre_physics_step(modified_actions)


@torch.jit.script
def _compute_step_return(
    *,
    ## Time
    episode_length: torch.Tensor,
    max_episode_length: int,
    truncate_episodes: bool,
    stage: int,
    ## Actions
    act_current: torch.Tensor,
    act_previous: torch.Tensor,
    ## States
    # Joints
    joint_pos_robot: torch.Tensor,
    joint_pos_limits_robot: torch.Tensor | None,
    joint_pos_end_effector: torch.Tensor | None,
    joint_pos_limits_end_effector: torch.Tensor | None,
    joint_vel_robot: torch.Tensor,
    joint_vel_end_effector: torch.Tensor | None,
    joint_acc_robot: torch.Tensor,
    joint_applied_torque_robot: torch.Tensor,
    # Kinematics
    fk_pos_end_effector: torch.Tensor,
    fk_quat_end_effector: torch.Tensor,
    # Transforms (world frame)
    tf_pos_end_effector: torch.Tensor, # 末端执行器在世界坐标系中的位置
    tf_quat_end_effector: torch.Tensor,
    tf_pos_obj_initial: torch.Tensor,
    tf_pos_obj: torch.Tensor,
    tf_quat_obj: torch.Tensor,
    vel_lin_obj: torch.Tensor,
    vel_ang_obj: torch.Tensor,
    tf_pos_target: torch.Tensor,
    tf_quat_target: torch.Tensor,
    terrain_height_end_effector: torch.Tensor,
    # Contacts
    contact_forces_robot: torch.Tensor,
    contact_forces_end_effector: torch.Tensor | None,
    contact_forces_end_effector_collision: torch.Tensor | None,
    contact_force_matrix_end_effector: torch.Tensor | None,
) -> StepReturn:
    num_envs = episode_length.size(0)
    dtype = tf_pos_end_effector.dtype
    device = episode_length.device

    # # ====================== 稳定化：NaN/inf 清理 ======================
    # joint_pos_robot = torch.nan_to_num(joint_pos_robot, nan=0.0, posinf=0.0, neginf=0.0)
    # if joint_pos_end_effector is not None:
    #     joint_pos_end_effector = torch.nan_to_num(joint_pos_end_effector, nan=0.0, posinf=0.0, neginf=0.0)
    # joint_acc_robot = torch.nan_to_num(joint_acc_robot, nan=0.0, posinf=0.0, neginf=0.0)
    # joint_applied_torque_robot = torch.nan_to_num(joint_applied_torque_robot, nan=0.0, posinf=0.0, neginf=0.0)
    # fk_pos_end_effector = torch.nan_to_num(fk_pos_end_effector, nan=0.0, posinf=0.0, neginf=0.0) 
    # fk_quat_end_effector = torch.nan_to_num(fk_quat_end_effector, nan=0.0, posinf=0.0, neginf=0.0)
    # tf_pos_end_effector = torch.nan_to_num(tf_pos_end_effector, nan=0.0, posinf=0.0, neginf=0.0)
    # tf_quat_end_effector = torch.nan_to_num(tf_quat_end_effector, nan=0.0, posinf=0.0, neginf=0.0)
    # tf_pos_obj = torch.nan_to_num(tf_pos_obj, nan=0.0, posinf=0.0, neginf=0.0)
    # tf_quat_obj = torch.nan_to_num(tf_quat_obj, nan=0.0, posinf=0.0, neginf=0.0)

    # 强制钳位
    joint_acc_robot = torch.clamp(joint_acc_robot, -10.0, 10.0)
    joint_applied_torque_robot = torch.clamp(joint_applied_torque_robot, -50.0, 50.0)

    ############
    ## States ##
    ############
    ## Joints
    # Robot joints
    joint_pos_robot_normalized = (
        scale_transform(
            joint_pos_robot,
            joint_pos_limits_robot[:, :, 0],
            joint_pos_limits_robot[:, :, 1] + 1e-6,
        )
        if joint_pos_limits_robot is not None
        else joint_pos_robot
    )
    # End-effector joints
    joint_pos_end_effector_normalized = (
        scale_transform(
            joint_pos_end_effector,
            joint_pos_limits_end_effector[:, :, 0],
            joint_pos_limits_end_effector[:, :, 1] + 1e-6,
        )
        if joint_pos_end_effector is not None
        and joint_pos_limits_end_effector is not None
        else (
            joint_pos_end_effector
            if joint_pos_end_effector is not None
            else torch.empty((num_envs, 0), dtype=dtype, device=device)
        )
    )

    # joint_pos_robot_normalized = torch.clamp(joint_pos_robot_normalized, -5.0, 5.0)
    # joint_pos_end_effector_normalized = torch.clamp(joint_pos_end_effector_normalized, -5.0, 5.0)
    joint_vel_end_effector_obs = (
        joint_vel_end_effector
        if joint_vel_end_effector is not None
        else torch.empty((num_envs, 0), dtype=dtype, device=device)
    )

    ## Kinematics
    fk_rotmat_end_effector = matrix_from_quat(fk_quat_end_effector)
    # fk_rotmat_end_effector = torch.nan_to_num(fk_rotmat_end_effector, 0.0, 0.0, 0.0)
    fk_rot6d_end_effector = rotmat_to_rot6d(fk_rotmat_end_effector)

    ## Transforms (world frame)
    # End-effector -> Object
    tf_pos_end_effector_to_obj, tf_quat_end_effector_to_obj = subtract_frame_transforms(
        t01=tf_pos_end_effector,
        q01=tf_quat_end_effector,
        t02=tf_pos_obj,
        q02=tf_quat_obj,
    )
    tf_rotmat_end_effector_to_obj = matrix_from_quat(tf_quat_end_effector_to_obj)
    # tf_rotmat_end_effector_to_obj = torch.nan_to_num(tf_rotmat_end_effector_to_obj, 0.0, 0.0, 0.0)
    tf_rot6d_end_effector_to_obj = rotmat_to_rot6d(tf_rotmat_end_effector_to_obj)
    # Object -> Target
    tf_pos_obj_to_target, tf_quat_obj_to_target = subtract_frame_transforms(
        t01=tf_pos_obj,
        q01=tf_quat_obj,
        t02=tf_pos_target,
        q02=tf_quat_target,
    )
    tf_rotmat_obj_to_target = matrix_from_quat(tf_quat_obj_to_target)
    tf_rot6d_obj_to_target = rotmat_to_rot6d(tf_rotmat_obj_to_target)

    height_above_terrain = tf_pos_end_effector[:, 2] - terrain_height_end_effector
    obj_lift_height = torch.clamp_min(tf_pos_obj[:, 2] - tf_pos_obj_initial[:, 2], 0.0)
    distance_obj_to_target = torch.norm(tf_pos_obj_to_target, dim=-1)
    obj_lin_speed = torch.norm(vel_lin_obj, dim=1)
    obj_ang_speed = torch.norm(vel_ang_obj, dim=1)
    gripper_aperture = (
        torch.mean(joint_pos_end_effector_normalized, dim=1)
        if joint_pos_end_effector_normalized.size(1) > 0
        else torch.zeros(num_envs, dtype=dtype, device=device)
    ) # 夹爪张开程度（假设所有夹爪关节的平均位置可以代表整体张开程度）

    ## Contacts
    contact_forces_mean_robot = contact_forces_robot.mean(dim=1)
    contact_forces_mean_end_effector = (
        contact_forces_end_effector.mean(dim=1)
        if contact_forces_end_effector is not None
        else torch.empty((num_envs, 0), dtype=dtype, device=device)
    )
    contact_forces_mean_end_effector_collision = (
        contact_forces_end_effector_collision.mean(dim=1)
        if contact_forces_end_effector_collision is not None
        else torch.empty((num_envs, 0), dtype=dtype, device=device)
    )
    contact_forces_end_effector = (
        contact_forces_end_effector
        if contact_forces_end_effector is not None
        else torch.empty((num_envs, 0), dtype=dtype, device=device)
    )
    contact_forces_end_effector_collision = (
        contact_forces_end_effector_collision
        if contact_forces_end_effector_collision is not None
        else torch.empty((num_envs, 0), dtype=dtype, device=device)
    )

    #############
    ## Rewards ##
    #############
    # Penalty: Action rate
    WEIGHT_ACTION_RATE = -0.5
    penalty_action_rate = WEIGHT_ACTION_RATE * torch.mean(
        torch.square(act_current - act_previous), dim=1
    ) # 惩罚动作变化率，鼓励平滑动作输出

    # Penalty: Joint torque
    WEIGHT_JOINT_TORQUE = -0.000025
    MAX_JOINT_TORQUE_PENALTY = -4.0
    # joint_torque_clipped = torch.clamp(joint_applied_torque_robot, -50.0, 50.0)
    penalty_joint_torque = torch.clamp_min(
        WEIGHT_JOINT_TORQUE
        * torch.sum(torch.square(joint_applied_torque_robot), dim=1),
        min=MAX_JOINT_TORQUE_PENALTY,
    ) # 加大关节扭矩惩罚权重，并设置最大惩罚值，鼓励更节能的动作

    # Penalty: Joint acceleration
    WEIGHT_JOINT_ACCELERATION = -0.0005
    MAX_JOINT_ACCELERATION_PENALTY = -4.0
    penalty_joint_acceleration = torch.clamp_min(
        WEIGHT_JOINT_ACCELERATION * torch.sum(torch.square(joint_acc_robot), dim=1),
        min=MAX_JOINT_ACCELERATION_PENALTY,
    ) # 加速度惩罚项,

    # Penalty: Undesired robot contacts
    WEIGHT_UNDESIRED_ROBOT_CONTACTS = -1.0
    THRESHOLD_UNDESIRED_ROBOT_CONTACTS = 10.0
    penalty_undesired_robot_contacts = WEIGHT_UNDESIRED_ROBOT_CONTACTS * (
        torch.max(torch.norm(contact_forces_robot, dim=-1), dim=1)[0]
        > THRESHOLD_UNDESIRED_ROBOT_CONTACTS
    ) # 惩罚机器人与环境的过大接触力，鼓励更轻柔的操作

    # Penalty: End-effector too close to / below terrain surface
    GROUND_CLEARANCE_MARGIN = 0.015
    ground_clearance_violation = height_above_terrain < GROUND_CLEARANCE_MARGIN
    penalty_end_effector_ground_clearance = -2.0 * ground_clearance_violation.to(dtype=dtype)

    # Penalty: Time (鼓励快速完成任务)
    WEIGHT_TIME_PENALTY = -0.005
    penalty_time = WEIGHT_TIME_PENALTY * torch.ones(num_envs, dtype=dtype, device=device)

    # Reward: End-effector top-down orientation
    WEIGHT_TOP_DOWN_ORIENTATION = 1.0 / 2
    TANH_STD_TOP_DOWN_ORIENTATION = 0.15
    top_down_alignment = torch.sum(
        fk_rotmat_end_effector[:, :, 2] * torch.tensor((0.0, 0.0, -1.0), device=device)
        .unsqueeze(0).expand(num_envs, 3), dim=1,
    ) # 计算末端执行器 Z 轴与世界坐标系向下方向的对齐程度，值越接近 1.0 表示越向下，鼓励末端执行器保持向下的姿态，便于抓取物体
    # top_down_alignment = torch.nan_to_num(top_down_alignment, 0.0, 1.0, -1.0)  # 处理 NaN 和 inf，确保在 [-1, 1] 范围内
    reward_top_down_orientation = WEIGHT_TOP_DOWN_ORIENTATION * (
        1.0 - torch.tanh((1.0 - top_down_alignment) / TANH_STD_TOP_DOWN_ORIENTATION)
    ) # 鼓励末端执行器保持向下的姿态，便于抓取物体

    # Reward: Pre-grasp alignment
    PREGRASP_HEIGHT_OFFSET = 0.08
    PREGRASP_ALIGNMENT_THRESHOLD = 0.85
    PREGRASP_XY_THRESHOLD = 0.03
    PREGRASP_HEIGHT_THRESHOLD = 0.03
    delta_pos_end_effector_to_obj_world = tf_pos_obj - tf_pos_end_effector
    distance_xy_to_obj = torch.norm(delta_pos_end_effector_to_obj_world[:, :2], dim=-1)
    height_above_obj = tf_pos_end_effector[:, 2] - tf_pos_obj[:, 2]
    pregrasp_height_error = torch.abs(height_above_obj - PREGRASP_HEIGHT_OFFSET)
    pregrasp_ready = (
        (distance_xy_to_obj < PREGRASP_XY_THRESHOLD)
        & (pregrasp_height_error < PREGRASP_HEIGHT_THRESHOLD)
        & (top_down_alignment > PREGRASP_ALIGNMENT_THRESHOLD)
    )

    WEIGHT_LATERAL_ALIGNMENT = 4.0 * 10  / stage
    TANH_STD_LATERAL_ALIGNMENT = 0.08
    reward_lateral_alignment = WEIGHT_LATERAL_ALIGNMENT * (
        1.0 - torch.tanh(distance_xy_to_obj / TANH_STD_LATERAL_ALIGNMENT)
    ) # 鼓励末端执行器与物体在 XY 平面上的对齐

    WEIGHT_PREGRASP_HEIGHT = 3.0
    TANH_STD_PREGRASP_HEIGHT = 0.04
    reward_pregrasp_height = WEIGHT_PREGRASP_HEIGHT * (
        1.0 - torch.tanh(pregrasp_height_error / TANH_STD_PREGRASP_HEIGHT)
    )

    WEIGHT_PREGRASP_READY = 2.0
    reward_pregrasp_ready = WEIGHT_PREGRASP_READY * pregrasp_ready.to(dtype=dtype)

    # Reward: Distance | End-effector <--> Object
    WEIGHT_DISTANCE_END_EFFECTOR_TO_OBJ = 2.5 * 40
    TANH_STD_DISTANCE_END_EFFECTOR_TO_OBJ = 0.2
    # dist_ee_obj = torch.norm(tf_pos_end_effector_to_obj, dim=-1)
    # dist_ee_obj = torch.clamp(dist_ee_obj, 0.0, 10.0)
    distance_to_obj = torch.norm(tf_pos_end_effector_to_obj, dim=-1)
    reward_distance_end_effector_to_obj = WEIGHT_DISTANCE_END_EFFECTOR_TO_OBJ * (
        0.8 - torch.tanh(
            distance_to_obj / TANH_STD_DISTANCE_END_EFFECTOR_TO_OBJ
        )) # 鼓励末端执行器接近物体

    # Reward: Grasp object
    WEIGHT_GRASP = 4.0 * 40
    THRESHOLD_GRASP = 2.5
    contact_force_sample_mean = (
        torch.mean(
            torch.max(torch.norm(contact_force_matrix_end_effector, dim=-1), dim=-1)[0],
            dim=1,
        )
        if contact_force_matrix_end_effector is not None
        else torch.zeros(num_envs, dtype=dtype, device=device)
    )
    stable_grasp = (
        (contact_force_sample_mean > THRESHOLD_GRASP)
        & (distance_xy_to_obj < 0.05)
        & (height_above_obj > -0.03)
        & (height_above_obj < 0.10)
    )
    transport_ready = stable_grasp & (obj_lift_height > 0.06)
    reward_grasp = WEIGHT_GRASP * stable_grasp.to(dtype=dtype)

    WEIGHT_GRASP_STABILITY = 3.0 * 40
    reward_grasp_stability = (
        WEIGHT_GRASP_STABILITY
        * stable_grasp.to(dtype=dtype)
        * (1.0 - torch.tanh(obj_lin_speed / 0.15))
        * (1.0 - torch.tanh(obj_ang_speed / 3.0))
    )

    # Reward / Penalty: End-effector collisions with ground or other scene objects
    THRESHOLD_END_EFFECTOR_COLLISION = 6.0
    collision_force_max_end_effector = (
        torch.max(torch.norm(contact_forces_end_effector_collision, dim=-1), dim=1)[0]
        if contact_forces_end_effector_collision is not None
        else torch.zeros(num_envs, dtype=dtype, device=device)
    )
    undesired_end_effector_collision = (
        (collision_force_max_end_effector > THRESHOLD_END_EFFECTOR_COLLISION)
        & ~stable_grasp
    )
    penalty_end_effector_collision = -2.0 * undesired_end_effector_collision.to(dtype=dtype)

    # ========== 稀疏成功奖励（新增） ==========
    WEIGHT_SUCCESS = 20.0 * 40               # 成功奖励的权重
    LIFT_HEIGHT_SUCCESS = 0.18           # 提起 18 cm 即视为成功，便于第二阶段更早收到稀疏正反馈
    
    success = stable_grasp & (obj_lift_height > LIFT_HEIGHT_SUCCESS)
    reward_success = WEIGHT_SUCCESS * success.to(dtype=dtype)


    # Reward: Lift object
    WEIGHT_LIFT = 6.0 * 40
    reward_lift = (
        WEIGHT_LIFT
        * stable_grasp.to(dtype=dtype)
        * torch.tanh(obj_lift_height / 0.12)
    )


    # Reward: Distance | Object <--> Target
    WEIGHT_DISTANCE_OBJ_TO_TARGET = 32.0
    TANH_STD_DISTANCE_OBJ_TO_TARGET = 0.2
    reward_distance_obj_to_target = transport_ready.to(dtype=dtype) * WEIGHT_DISTANCE_OBJ_TO_TARGET * (
        1.0 - torch.tanh(
            distance_obj_to_target / TANH_STD_DISTANCE_OBJ_TO_TARGET
        )
    )

    if stage > 1:
        pregrasp_focus = (~stable_grasp).to(dtype=dtype)
        reward_lateral_alignment = reward_lateral_alignment * pregrasp_focus
        reward_pregrasp_height = reward_pregrasp_height * pregrasp_focus
        reward_pregrasp_ready = reward_pregrasp_ready * pregrasp_focus
        reward_distance_end_effector_to_obj = (
            reward_distance_end_effector_to_obj * pregrasp_focus
        ) # 第二阶段开始后，稳定抓取的环境不再获得末端执行器与物体距离的奖励，鼓励它们专注于保持稳定抓取和完成运输任务

    # ========== 新增惩罚项 ==========
    # 获取夹爪动作（假设动作向量最后一维为夹爪，维度 > 6）
    if act_current.size(1) > 6:
        gripper_action = act_current[:, 6:]
        gripper_closed = torch.mean(gripper_action, dim=1) < -0.1
    else:
        gripper_closed = torch.zeros(num_envs, dtype=torch.bool, device=device)

    # 计算末端执行器到物体的距离（已在先前定义，若没有则重新计算）
    # distance_to_obj = torch.norm(tf_pos_end_effector_to_obj, dim=-1)

    # 1. 远距离闭合惩罚
    far_close_penalty_weight = -1.0
    far_close_penalty = far_close_penalty_weight * (
        gripper_closed
        & ((distance_xy_to_obj > 0.08) | (height_above_obj > 0.12))
    ).to(dtype=dtype)

    # 2. 虚抓惩罚：闭合但没有有效接触力（接触力 < 2.0 N）
    # 注意：contact_force_matrix_end_effector 可能为 None，需判空
    bad_grasp = gripper_closed & (contact_force_sample_mean < THRESHOLD_GRASP)
    fake_grasp_penalty_weight = -0.5
    fake_grasp_penalty = fake_grasp_penalty_weight * bad_grasp.to(dtype=dtype)

    if stage <= 1:
        success = pregrasp_ready
        reward_success = 8.0 * success.to(dtype=dtype)
        reward_grasp = torch.zeros_like(reward_grasp)
        reward_grasp_stability = torch.zeros_like(reward_grasp_stability)
        reward_lift = torch.zeros_like(reward_lift)
        reward_distance_obj_to_target = torch.zeros_like(reward_distance_obj_to_target)
        far_close_penalty = torch.zeros_like(far_close_penalty)
        fake_grasp_penalty = torch.zeros_like(fake_grasp_penalty)



    # 将新惩罚项加入到总奖励中（原有奖励变量名请根据实际情况调整）
    # 通常原有奖励已经汇总为一个变量（例如 reward_total），如果尚未汇总，你需要将以下项加到返回的字典中
    # 为了简单，我们在返回的奖励字典中直接添加这两个键值对。

    ##################
    ## Terminations ##
    ##################
    # No termination condition
    termination = undesired_end_effector_collision | (height_above_terrain < 0.0)
    # Truncation
    truncation = (
        episode_length >= max_episode_length
        if truncate_episodes
        else torch.zeros(num_envs, dtype=torch.bool, device=device)
    )

    return StepReturn(
        {
            "state": {
                "contact_forces_mean_robot": contact_forces_mean_robot,
                "contact_forces_mean_end_effector": contact_forces_mean_end_effector,
                "contact_forces_mean_end_effector_collision": contact_forces_mean_end_effector_collision,
                "height_above_terrain": height_above_terrain.unsqueeze(-1),
                "distance_xy_end_effector_to_obj": distance_xy_to_obj.unsqueeze(-1),
                "height_above_obj": height_above_obj.unsqueeze(-1),
                "sample_lift_height": obj_lift_height.unsqueeze(-1),
                "distance_obj_to_target": distance_obj_to_target.unsqueeze(-1),
                "sample_lin_speed": obj_lin_speed.unsqueeze(-1),
                "sample_ang_speed": obj_ang_speed.unsqueeze(-1),
                "gripper_aperture": gripper_aperture.unsqueeze(-1),
                "tf_pos_end_effector_to_obj": tf_pos_end_effector_to_obj,
                "tf_rot6d_end_effector_to_obj": tf_rot6d_end_effector_to_obj,
                "tf_pos_obj_to_target": tf_pos_obj_to_target,
                "tf_rot6d_obj_to_target": tf_rot6d_obj_to_target,
                "stable_grasp": stable_grasp.float().unsqueeze(-1),
                "transport_ready": transport_ready.float().unsqueeze(-1),
                "end_effector_collision_force_max": collision_force_max_end_effector.unsqueeze(-1),
                "end_effector_collision_undesired": undesired_end_effector_collision.float().unsqueeze(-1),
                "pregrasp_ready": pregrasp_ready.float().unsqueeze(-1),
                "success": success.float().unsqueeze(-1),
            },
            "state_dyn": {
                "sample_lin_vel": vel_lin_obj,
                "sample_ang_vel": vel_ang_obj,
                "contact_forces_robot": contact_forces_robot,
                "contact_forces_end_effector": contact_forces_end_effector,
            },
            "proprio": {
                "fk_pos_end_effector": fk_pos_end_effector,
                "fk_rot6d_end_effector": fk_rot6d_end_effector,
            },
            "proprio_dyn": {
                "joint_pos_robot_normalized": joint_pos_robot_normalized,
                "joint_pos_end_effector_normalized": joint_pos_end_effector_normalized,
                "joint_vel_robot": joint_vel_robot,
                "joint_vel_end_effector": joint_vel_end_effector_obs,
                "joint_acc_robot": joint_acc_robot,
                "joint_applied_torque_robot": joint_applied_torque_robot,
            },
        },
        {
            "penalty_action_rate": penalty_action_rate,
            "penalty_joint_torque": penalty_joint_torque,
            "penalty_joint_acceleration": penalty_joint_acceleration,
            "penalty_undesired_robot_contacts": penalty_undesired_robot_contacts,
            "penalty_time": penalty_time,
            "penalty_end_e_g_c": penalty_end_effector_ground_clearance,
            "reward_top_down_orientation": reward_top_down_orientation,
            "reward_lateral_alignment": reward_lateral_alignment,
            "reward_pregrasp_height": reward_pregrasp_height,
            "reward_pregrasp_ready": reward_pregrasp_ready,
            "reward_distance_end_effector_to_obj": reward_distance_end_effector_to_obj,
            "reward_grasp": reward_grasp,
            "reward_grasp_stability": reward_grasp_stability,
            "reward_lift": reward_lift,
            "reward_distance_obj_to_target": reward_distance_obj_to_target,
            "reward_success": reward_success,
            "far_close_penalty": far_close_penalty,
            "fake_grasp_penalty": fake_grasp_penalty,
            "penalty_end_effector_collision": penalty_end_effector_collision,
        },
        termination,
        truncation,
        {
            "sample_lift_height": obj_lift_height,
            "stable_grasp": stable_grasp.to(dtype=dtype),
            "transport_ready": transport_ready.to(dtype=dtype),
            "sample_lin_speed": obj_lin_speed,
            "sample_ang_speed": obj_ang_speed,
        },
    )
