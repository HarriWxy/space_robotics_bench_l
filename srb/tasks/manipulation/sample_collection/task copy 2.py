from dataclasses import MISSING
from typing import Sequence, Tuple

import torch

from srb._typing import StepReturn
from srb.core.asset import (
    AssetVariant,
    BaseArticulation,
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
from srb.core.sensor import ContactSensorCfg
from srb.core.sim import PreviewSurfaceCfg, SphereCfg
from srb.utils.cfg import configclass
from srb.utils.math import (
    matrix_from_quat,
    rotmat_to_rot6d,
    scale_transform,
    subtract_frame_transforms,
)

from .asset import select_sample

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

    ## Target
    tf_pos_target: Tuple[float, float, float] = (0.5, 0.0, 0.75)
    tf_quat_target: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
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
        self._tf_pos_obj_initial[env_ids] = self._obj.data.root_com_pos_w.torch[env_ids]
        self._robot.data.joint_acc.torch[env_ids] = 0.0
        self._robot.actuators.applied_effort.torch[env_ids] = 0.0
        self._robot.data.joint_vel.torch[env_ids] = 0.0

    def extract_step_return(self) -> StepReturn:
        return _compute_step_return(
            ## Time
            episode_length=self.episode_length_buf,
            max_episode_length=self.max_episode_length,
            truncate_episodes=self.cfg.truncate_episodes,
            ## Actions
            act_current=self.action_manager.action,
            act_previous=self.action_manager.prev_action,
            ## States
            # Joints
            joint_pos_robot=self._robot.data.joint_pos.torch,
            joint_pos_limits_robot=(
                self._robot.data.soft_joint_pos_limits.torch
                if torch.all(torch.isfinite(self._robot.data.soft_joint_pos_limits.torch))
                else None
            ),
            joint_pos_end_effector=self._end_effector.data.joint_pos.torch
            if isinstance(self._end_effector, BaseArticulation)
            else None,
            joint_pos_limits_end_effector=(
                self._end_effector.data.soft_joint_pos_limits.torch
                if isinstance(self._end_effector, BaseArticulation)
                and torch.all(
                    torch.isfinite(self._end_effector.data.soft_joint_pos_limits.torch)
                )
                else None
            ),
            joint_acc_robot=self._robot.data.joint_acc.torch,
            joint_applied_torque_robot=self._robot.actuators.applied_effort.torch,
            # Kinematics
            fk_pos_end_effector=self._tf_end_effector.data.target_pos_source.torch[:, 0, :],
            fk_quat_end_effector=self._tf_end_effector.data.target_quat_source.torch[:, 0, :],
            # Transforms (world frame)
            tf_pos_end_effector=self._tf_end_effector.data.target_pos_w.torch[:, 0, :],
            tf_quat_end_effector=self._tf_end_effector.data.target_quat_w.torch[:, 0, :],
            tf_pos_obj_initial=self._tf_pos_obj_initial,
            tf_pos_obj=self._obj.data.root_com_pos_w.torch,
            tf_quat_obj=self._obj.data.root_com_quat_w.torch,
            tf_pos_target=self._tf_pos_target,
            tf_quat_target=self._tf_quat_target,
            # Contacts
            contact_forces_robot=self._contacts_robot.data.net_normal_forces_w.torch,  # type: ignore
            contact_forces_end_effector=self._contacts_end_effector.data.net_normal_forces_w.torch
            if self._contacts_end_effector is not None
            else None,
            contact_force_matrix_end_effector=self._contacts_end_effector.data.force_matrix_w.torch
            if self._contacts_end_effector is not None
            else None,
        )

    # def _is_gripper_closed(self) -> torch.Tensor:
    #     """返回夹爪是否闭合的布尔张量（基于关节位置或动作）"""
    #     # 假设夹爪关节名称为 "Slider_[1-2]"，闭合位置为 -0.025
    #     if hasattr(self._end_effector, 'data') and self._end_effector.data is not None:
    #         joint_pos = self._end_effector.data.joint_pos.torch
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
    #         ee_pos = self._tf_end_effector.data.target_pos_w.torch[:, 0, :]  # (num_envs, 3)
    #         obj_pos = self._obj.data.root_com_pos_w.torch
    #         distance_to_obj = torch.norm(ee_pos - obj_pos, dim=1)
    #         # 也可以检查接触力
    #         if self._contacts_end_effector is not None:
    #             contact_force = self._contacts_end_effector.data.force_matrix_w.torch
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
    ## Actions
    act_current: torch.Tensor,
    act_previous: torch.Tensor,
    ## States
    # Joints
    joint_pos_robot: torch.Tensor,
    joint_pos_limits_robot: torch.Tensor | None,
    joint_pos_end_effector: torch.Tensor | None,
    joint_pos_limits_end_effector: torch.Tensor | None,
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
    tf_pos_target: torch.Tensor,
    tf_quat_target: torch.Tensor,
    # Contacts
    contact_forces_robot: torch.Tensor,
    contact_forces_end_effector: torch.Tensor | None,
    contact_force_matrix_end_effector: torch.Tensor | None,
) -> StepReturn:
    num_envs = episode_length.size(0)
    dtype = episode_length.dtype
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

    ## Contacts
    contact_forces_mean_robot = contact_forces_robot.mean(dim=1)
    contact_forces_mean_end_effector = (
        contact_forces_end_effector.mean(dim=1)
        if contact_forces_end_effector is not None
        else torch.empty((num_envs, 0), dtype=dtype, device=device)
    )
    contact_forces_end_effector = (
        contact_forces_end_effector
        if contact_forces_end_effector is not None
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

    # Penalty: Time (鼓励快速完成任务)
    WEIGHT_TIME_PENALTY = -0.005
    penalty_time = WEIGHT_TIME_PENALTY * torch.ones(num_envs, dtype=dtype, device=device)

    # Reward: End-effector top-down orientation
    WEIGHT_TOP_DOWN_ORIENTATION = 1.0 / 2
    TANH_STD_TOP_DOWN_ORIENTATION = 0.15
    top_down_alignment = torch.sum(
        fk_rotmat_end_effector[:, :, 2] * torch.tensor((0.0, 0.0, -1.0), device=device)
        .unsqueeze(0).expand(num_envs, 3), dim=1,
    )
    # top_down_alignment = torch.nan_to_num(top_down_alignment, 0.0, 1.0, -1.0)  # 处理 NaN 和 inf，确保在 [-1, 1] 范围内
    reward_top_down_orientation = WEIGHT_TOP_DOWN_ORIENTATION * (
        1.0 - torch.tanh((1.0 - top_down_alignment) / TANH_STD_TOP_DOWN_ORIENTATION)
    ) # 鼓励末端执行器保持向下的姿态，便于抓取物体

    # Reward: Distance | End-effector <--> Object
    WEIGHT_DISTANCE_END_EFFECTOR_TO_OBJ = 2.5 * 2 * 10
    TANH_STD_DISTANCE_END_EFFECTOR_TO_OBJ = 0.2
    # dist_ee_obj = torch.norm(tf_pos_end_effector_to_obj, dim=-1)
    # dist_ee_obj = torch.clamp(dist_ee_obj, 0.0, 10.0)
    distance_to_obj = torch.norm(tf_pos_end_effector_to_obj, dim=-1)
    reward_distance_end_effector_to_obj = WEIGHT_DISTANCE_END_EFFECTOR_TO_OBJ * (
        1.0 - torch.tanh(
            distance_to_obj / TANH_STD_DISTANCE_END_EFFECTOR_TO_OBJ
        )) # 鼓励末端执行器接近物体

    # Reward: Grasp object
    WEIGHT_GRASP = 4.0 * 2
    THRESHOLD_GRASP = 2.5
    reward_grasp = (
        WEIGHT_GRASP * ( 
            torch.mean(torch.max(torch.norm(contact_force_matrix_end_effector, dim=-1), dim=-1)[0],
                dim=1,)  > THRESHOLD_GRASP
        )
        if contact_force_matrix_end_effector is not None
        else torch.zeros(num_envs, dtype=dtype, device=device)
    ) # 鼓励成功抓取物体，基于末端执行器的接触力矩阵中最大接触力的平均值是否超过阈值

    # ========== 稀疏成功奖励（新增） ==========
    WEIGHT_SUCCESS = 20.0                # 成功奖励的权重
    LIFT_HEIGHT_SUCCESS = 0.35           # 提起 35 cm 即视为成功 (原提升门槛是 0.5 米，太高)
    
    success = reward_grasp > 0.0  # 已经抓住物体
    success = success & ((tf_pos_obj[:, 2] - tf_pos_obj_initial[:, 2]) > LIFT_HEIGHT_SUCCESS)
    reward_success = WEIGHT_SUCCESS * success.to(dtype=dtype)


    # Reward: Lift object
    WEIGHT_LIFT = 4.0
    HEIGHT_OFFSET_LIFT = 0.25
    HEIGHT_SPAN_LIFT = 0.10
    TANH_STD_HEIGHT_LIFT = 0.05
    reward_lift = success * WEIGHT_LIFT * (
        1.0- torch.tanh((
                torch.abs(tf_pos_obj[:, 2] - tf_pos_obj_initial[:, 2] - HEIGHT_OFFSET_LIFT)
                - HEIGHT_SPAN_LIFT
            ).clamp(min=0.0)
            / TANH_STD_HEIGHT_LIFT
        )
    )


    # Reward: Distance | Object <--> Target
    WEIGHT_DISTANCE_OBJ_TO_TARGET = 32.0
    TANH_STD_DISTANCE_OBJ_TO_TARGET = 0.2
    # dist_obj_target = torch.norm(tf_pos_obj_to_target, dim=-1)
    # dist_obj_target = torch.clamp(dist_obj_target, 0.0, 10.0)
    reward_distance_obj_to_target = WEIGHT_DISTANCE_OBJ_TO_TARGET * (
        1.0 - torch.tanh(
            torch.norm(tf_pos_obj_to_target, dim=-1) / TANH_STD_DISTANCE_OBJ_TO_TARGET
        )
    )

    # ========== 新增惩罚项 ==========
    # 获取夹爪动作（假设动作向量最后一维为夹爪，维度 > 6）
    if act_current.size(1) > 6:
        gripper_action = act_current[:, -1]
        gripper_closed = gripper_action < -0.01          # 闭合阈值
    else:
        gripper_closed = torch.zeros(num_envs, dtype=torch.bool, device=device)

    # 计算末端执行器到物体的距离（已在先前定义，若没有则重新计算）
    # distance_to_obj = torch.norm(tf_pos_end_effector_to_obj, dim=-1)

    # 1. 远距离闭合惩罚
    far_close_penalty_weight = -1.0
    far_close_penalty = far_close_penalty_weight * (gripper_closed & (distance_to_obj > 0.15)).to(dtype=dtype)

    # 2. 虚抓惩罚：闭合但没有有效接触力（接触力 < 2.0 N）
    # 注意：contact_force_matrix_end_effector 可能为 None，需判空
    if contact_force_matrix_end_effector is not None:
        contact_force_max = torch.norm(contact_force_matrix_end_effector, dim=-1).max(dim=1)[0]
        bad_grasp = gripper_closed & (contact_force_max < 2.0)
    else:
        bad_grasp = torch.zeros(num_envs, dtype=torch.bool, device=device)
    fake_grasp_penalty_weight = -0.5
    fake_grasp_penalty = fake_grasp_penalty_weight * bad_grasp.to(dtype=dtype)

    # 将新惩罚项加入到总奖励中（原有奖励变量名请根据实际情况调整）
    # 通常原有奖励已经汇总为一个变量（例如 reward_total），如果尚未汇总，你需要将以下项加到返回的字典中
    # 为了简单，我们在返回的奖励字典中直接添加这两个键值对。

    ##################
    ## Terminations ##
    ##################
    # No termination condition
    termination = torch.zeros(num_envs, dtype=torch.bool, device=device)
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
                "tf_pos_end_effector_to_obj": tf_pos_end_effector_to_obj,
                "tf_rot6d_end_effector_to_obj": tf_rot6d_end_effector_to_obj,
                "tf_pos_obj_to_target": tf_pos_obj_to_target,
                "tf_rot6d_obj_to_target": tf_rot6d_obj_to_target,
                "success": success.float().unsqueeze(-1),
            },
            "state_dyn": {
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
            "reward_top_down_orientation": reward_top_down_orientation,
            "reward_distance_end_effector_to_obj": reward_distance_end_effector_to_obj,
            "reward_grasp": reward_grasp,
            "reward_lift": reward_lift,
            "reward_distance_obj_to_target": reward_distance_obj_to_target,
            "reward_success": reward_success,
            "far_close_penalty": far_close_penalty,
            "fake_grasp_penalty": fake_grasp_penalty,
        },
        termination,
        truncation,
    )
