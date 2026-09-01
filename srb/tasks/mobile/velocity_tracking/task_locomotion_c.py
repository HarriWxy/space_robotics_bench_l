from dataclasses import MISSING
from typing import List, Sequence

import torch

from srb import assets
from srb._typing import StepReturn
from srb.core.asset import AssetVariant, Humanoid, LeggedRobot
from srb.core.manager import EventTermCfg, SceneEntityCfg
from srb.core.mdp import (
    push_by_setting_velocity,  # noqa: F401
    reset_joints_by_scale,
)
from srb.core.sensor import ContactSensor, ContactSensorCfg
from srb.utils.cfg import configclass
from srb.utils.math import matrix_from_quat, rotmat_to_rot6d, scale_transform

from .task import EventCfg, SceneCfg, Task, TaskCfg

##############
### Config ###
##############


@configclass
class LocomotionCurriculumCfg:
    """Training stages expressed in total simulated environment transitions."""

    enabled: bool = True
    fixed_stage: int = 2
    stage_env_steps: tuple[int, int] = (20_000_000, 60_000_000)
    linear_velocity_magnitudes: tuple[float, float, float] = (0.35, 0.6, 1.0)
    lateral_velocity_scales: tuple[float, float, float] = (0.0, 0.5, 1.0)
    angular_velocity_magnitudes: tuple[float, float, float] = (0.35, 0.6, 1.0)
    zero_command_probabilities: tuple[float, float, float] = (0.25, 0.1, 0.05)
    command_interval_ranges: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ] = ((2.0, 4.0), (1.0, 3.0), (0.5, 5.0))
    joint_position_ranges: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ] = ((0.9, 1.1), (0.75, 1.25), (0.5, 1.5))
    joint_velocity_range: tuple[float, float] = (0.0, 0.0)


@configclass
class LocomotionRewardCfg:
    action_rate_weight: float = -0.25
    joint_torque_weight: float = -0.000025
    joint_torque_max_penalty: float = -4.0
    joint_acceleration_weight: float = -0.00000025
    joint_acceleration_max_penalty: float = -2.0
    undesired_contact_weight: float = -1.0
    undesired_contact_force_threshold: float = 5.0
    command_linear_weight: float = 3.0
    command_linear_exp_std: float = 0.5
    command_angular_weight: float = 1.5
    command_angular_exp_std: float = 0.25
    feet_air_time_weight: float = 0.2
    feet_air_time_target: float = 0.35
    moving_command_threshold: float = 0.1
    foot_slip_weight: float = -0.1
    foot_slip_contact_force_threshold: float = 1.0
    foot_slip_max_penalty: float = -2.0
    undesired_linear_velocity_z_weight: float = -0.5
    undesired_angular_velocity_xy_weight: float = -0.1
    gravity_alignment_weight: float = -2.0


@configclass
class LocomotionTerminationCfg:
    min_base_height: float = 0.3
    min_body_up_z: float = 0.3
    tracking_linear_error_threshold: float = 0.25
    tracking_angular_error_threshold: float = 0.25
    tracking_min_body_up_z: float = 0.8


@configclass
class LocomotionSceneCfg(SceneCfg):
    contacts_robot: ContactSensorCfg = ContactSensorCfg(
        prim_path=MISSING,  # type: ignore
        update_period=0.0,
        history_length=3,
        track_air_time=True,
    )


#############################
### Curriculum event terms ###
#############################


def _resolve_curriculum_stage(
    env: Task,
    *,
    curriculum_enabled: bool,
    fixed_stage: int,
    stage_env_steps: tuple[int, ...],
) -> int:
    if not curriculum_enabled:
        return fixed_stage

    total_env_steps = int(env.common_step_counter) * int(env.num_envs)
    stage = 0
    for threshold in stage_env_steps:
        if total_env_steps < threshold:
            break
        stage += 1
    return stage


def randomize_velocity_command_curriculum(
    env: Task,
    env_ids: torch.Tensor | None,
    *,
    env_attr_name: str,
    curriculum_enabled: bool,
    fixed_stage: int,
    stage_env_steps: tuple[int, ...],
    linear_velocity_magnitudes: tuple[float, ...],
    lateral_velocity_scales: tuple[float, ...],
    angular_velocity_magnitudes: tuple[float, ...],
    zero_command_probabilities: tuple[float, ...],
    command_interval_ranges: tuple[tuple[float, float], ...],
):
    """Sample body-frame commands from the active curriculum stage."""

    unwrapped_env = env.unwrapped  # type: ignore
    if env_ids is None:
        env_ids = torch.arange(unwrapped_env.num_envs, device=unwrapped_env.device)

    stage = _resolve_curriculum_stage(
        unwrapped_env,
        curriculum_enabled=curriculum_enabled,
        fixed_stage=fixed_stage,
        stage_env_steps=stage_env_steps,
    )
    command = getattr(unwrapped_env, env_attr_name)
    num_commands = len(env_ids)
    heading = 2.0 * torch.pi * torch.rand(
        num_commands, device=unwrapped_env.device
    )
    speed = linear_velocity_magnitudes[stage] * torch.rand(
        num_commands, device=unwrapped_env.device
    )

    command[env_ids, 0] = speed * torch.cos(heading)
    command[env_ids, 1] = (
        lateral_velocity_scales[stage] * speed * torch.sin(heading)
    )
    command[env_ids, 2] = angular_velocity_magnitudes[stage] * (
        2.0 * torch.rand(num_commands, device=unwrapped_env.device) - 1.0
    )

    zero_command_mask = torch.rand(num_commands, device=unwrapped_env.device) < (
        zero_command_probabilities[stage]
    )
    command[env_ids[zero_command_mask]] = 0.0

    # EventManager reads this object again before it samples the next interval.
    unwrapped_env.cfg.events.command.interval_range_s = command_interval_ranges[stage]


def reset_joints_by_scale_curriculum(
    env: Task,
    env_ids: torch.Tensor,
    *,
    asset_cfg: SceneEntityCfg,
    curriculum_enabled: bool,
    fixed_stage: int,
    stage_env_steps: tuple[int, ...],
    joint_position_ranges: tuple[tuple[float, float], ...],
    velocity_range: tuple[float, float],
):
    """Reset around the nominal pose with progressively wider joint perturbations."""

    unwrapped_env = env.unwrapped  # type: ignore
    stage = _resolve_curriculum_stage(
        unwrapped_env,
        curriculum_enabled=curriculum_enabled,
        fixed_stage=fixed_stage,
        stage_env_steps=stage_env_steps,
    )
    reset_joints_by_scale(
        env,
        env_ids,
        position_range=joint_position_ranges[stage],
        velocity_range=velocity_range,
        asset_cfg=asset_cfg,
    )


@configclass
class LocomotionEventCfg(EventCfg):
    command: EventTermCfg = EventTermCfg(
        func=randomize_velocity_command_curriculum,
        mode="interval",
        interval_range_s=(2.0, 4.0),
        params={
            "env_attr_name": "_command",
            "curriculum_enabled": True,
            "fixed_stage": 2,
            "stage_env_steps": (20_000_000, 60_000_000),  # total 100_000_000
            "linear_velocity_magnitudes": (0.35, 0.6, 1.0),
            "lateral_velocity_scales": (0.0, 0.5, 1.0),
            "angular_velocity_magnitudes": (0.35, 0.6, 1.0),
            "zero_command_probabilities": (0.25, 0.1, 0.05),
            "command_interval_ranges": ((2.0, 4.0), (1.0, 3.0), (0.5, 5.0)),
        },
    )
    randomize_robot_joints: EventTermCfg = EventTermCfg(
        func=reset_joints_by_scale_curriculum,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "curriculum_enabled": True,
            "fixed_stage": 2,
            "stage_env_steps": (20_000_000, 60_000_000),
            "joint_position_ranges": ((0.9, 1.1), (0.75, 1.25), (0.5, 1.5)),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class LocomotionTaskCfg(TaskCfg):
    ## Assets
    robot: LeggedRobot | Humanoid | AssetVariant = assets.Spot()
    _robot: LeggedRobot = MISSING  # type: ignore

    ## Scene
    scene: LocomotionSceneCfg = LocomotionSceneCfg()

    ## Events
    events: LocomotionEventCfg = LocomotionEventCfg()

    ## Training
    curriculum: LocomotionCurriculumCfg = LocomotionCurriculumCfg()
    rewards: LocomotionRewardCfg = LocomotionRewardCfg()
    terminations: LocomotionTerminationCfg = LocomotionTerminationCfg()

    ## Time
    env_rate: float = 1.0 / 125.0

    ## Visualization
    command_vis: bool = True  

    def __post_init__(self):
        super().__post_init__()

        self._validate_curriculum()

        # Sensor: Robot contacts
        self.scene.contacts_robot.prim_path = f"{self.scene.robot.prim_path}/.*"

        initial_stage = (
            0 if self.curriculum.enabled else self.curriculum.fixed_stage
        )
        self.events.command.interval_range_s = self.curriculum.command_interval_ranges[
            initial_stage
        ]
        self.events.command.params.update(
            {
                "curriculum_enabled": self.curriculum.enabled,
                "fixed_stage": self.curriculum.fixed_stage,
                "stage_env_steps": self.curriculum.stage_env_steps,
                "linear_velocity_magnitudes": (
                    self.curriculum.linear_velocity_magnitudes
                ),
                "lateral_velocity_scales": self.curriculum.lateral_velocity_scales,
                "angular_velocity_magnitudes": (
                    self.curriculum.angular_velocity_magnitudes
                ),
                "zero_command_probabilities": (
                    self.curriculum.zero_command_probabilities
                ),
                "command_interval_ranges": self.curriculum.command_interval_ranges,
            }
        )
        self.events.randomize_robot_joints.params.update(
            {
                "curriculum_enabled": self.curriculum.enabled,
                "fixed_stage": self.curriculum.fixed_stage,
                "stage_env_steps": self.curriculum.stage_env_steps,
                "joint_position_ranges": self.curriculum.joint_position_ranges,
                "velocity_range": self.curriculum.joint_velocity_range,
            }
        )

    def _validate_curriculum(self):
        num_stages = len(self.curriculum.linear_velocity_magnitudes)
        stage_fields = (
            self.curriculum.lateral_velocity_scales,
            self.curriculum.angular_velocity_magnitudes,
            self.curriculum.zero_command_probabilities,
            self.curriculum.command_interval_ranges,
            self.curriculum.joint_position_ranges,
        )
        if num_stages < 1 or any(len(field) != num_stages for field in stage_fields):
            raise ValueError(
                "All locomotion curriculum stage fields must have equal length."
            )
        if len(self.curriculum.stage_env_steps) != num_stages - 1:
            raise ValueError(
                "stage_env_steps must contain exactly one less entry than the "
                "number of stages."
            )
        if not 0 <= self.curriculum.fixed_stage < num_stages:
            raise ValueError("fixed_stage must refer to an existing curriculum stage.")
        if any(
            previous >= current
            for previous, current in zip(
                self.curriculum.stage_env_steps,
                self.curriculum.stage_env_steps[1:],
            )
        ):
            raise ValueError("stage_env_steps must be strictly increasing.")
        if any(
            magnitude < 0.0
            for magnitude in self.curriculum.linear_velocity_magnitudes
        ):
            raise ValueError("linear_velocity_magnitudes must be non-negative.")
        if any(
            magnitude < 0.0
            for magnitude in self.curriculum.angular_velocity_magnitudes
        ):
            raise ValueError("angular_velocity_magnitudes must be non-negative.")
        if any(
            not 0.0 <= probability <= 1.0
            for probability in self.curriculum.zero_command_probabilities
        ):
            raise ValueError("zero_command_probabilities must be in [0, 1].")
        if any(
            lower <= 0.0 or upper < lower
            for lower, upper in self.curriculum.command_interval_ranges
        ):
            raise ValueError(
                "Each command_interval_range must be positive and ordered as "
                "(min, max)."
            )


############
### Task ###
############


class LocomotionTask(Task):
    cfg: LocomotionTaskCfg

    def __init__(self, cfg: LocomotionTaskCfg, **kwargs):
        super().__init__(cfg, **kwargs)

        ## Get scene assets
        self._contacts_robot: ContactSensor = self.scene["contacts_robot"]

        ## Cache metrics
        self._feet_indices, _ = self._robot.find_bodies(
            self.cfg._robot.regex_feet_links
        )
        if not self._feet_indices:
            raise RuntimeError(
                "No feet bodies matched cfg._robot.regex_feet_links; "
                "foot-contact rewards cannot be computed."
            )
        all_body_indices, _ = self._robot.find_bodies(".*")
        self._undesired_contact_body_indices = [
            idx for idx in all_body_indices if idx not in self._feet_indices
        ]
        self._resample_commands(torch.arange(self.num_envs, device=self.device))

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)

        # The base constructor may reset before Task initializes _command.
        if hasattr(self, "_command"):
            self._resample_commands(env_ids)

    def _resample_commands(self, env_ids: Sequence[int] | torch.Tensor):
        if isinstance(env_ids, torch.Tensor):
            command_env_ids = env_ids.to(device=self.device, dtype=torch.long)
        else:
            command_env_ids = torch.as_tensor(
                env_ids,
                device=self.device,
                dtype=torch.long,
            )
        randomize_velocity_command_curriculum(
            self,
            command_env_ids,
            **self.cfg.events.command.params,
        )

    def extract_step_return(self) -> StepReturn:
        if self.cfg.command_vis or self.cfg.debug_vis:
            self._update_visualization_markers()

        tf_quat_robot = torch.nan_to_num(
            self._robot.data.root_quat_w.torch,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        default_quat = torch.zeros_like(tf_quat_robot)
        default_quat[:, 3] = 1.0
        tf_quat_robot = torch.where(
            torch.norm(tf_quat_robot, dim=-1, keepdim=True) < 1.0e-6,
            default_quat,
            tf_quat_robot,
        )
        tf_pos_robot = torch.nan_to_num(
            self._robot.data.root_pos_w.torch,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        vel_lin_robot = torch.nan_to_num(
            self._robot.data.root_lin_vel_b.torch,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        vel_ang_robot = torch.nan_to_num(
            self._robot.data.root_ang_vel_b.torch,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        projected_gravity_robot = torch.nan_to_num(
            self._robot.data.projected_gravity_b.torch,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        joint_pos_robot = torch.nan_to_num(
            self._robot.data.joint_pos.torch,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        joint_vel_robot = torch.nan_to_num(
            self._robot.data.joint_vel.torch,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        joint_acc_robot = torch.nan_to_num(
            self._robot.data.joint_acc.torch,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        joint_applied_torque_robot = torch.nan_to_num(
            self._robot.actuators.applied_effort.torch,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        joint_pos_limits_robot = self._robot.data.soft_joint_pos_limits.torch
        if not torch.all(torch.isfinite(joint_pos_limits_robot)):
            joint_pos_limits_robot = None

        contact_forces_robot = torch.nan_to_num(
            self._contacts_robot.data.net_forces_w.torch,  # type: ignore
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        contact_robot = self._contacts_robot.compute_first_contact(self.step_dt).torch
        contact_last_air_time = torch.nan_to_num(
            self._contacts_robot.data.last_air_time.torch,  # type: ignore
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        foot_link_vel_w = torch.nan_to_num(
            self._robot.data.body_link_vel_w.torch,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        imu_lin_acc = torch.nan_to_num(
            self._imu_robot.data.lin_acc_b.torch,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        imu_ang_vel = torch.nan_to_num(
            self._imu_robot.data.ang_vel_b.torch,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        command = torch.nan_to_num(
            self._command,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        step_return = _compute_step_return(
            ## Time
            episode_length=self.episode_length_buf,
            max_episode_length=self.max_episode_length,
            truncate_episodes=self.cfg.truncate_episodes,
            ## Actions
            act_current=self.action_manager.action,
            act_previous=self.action_manager.prev_action,
            ## States
            tf_quat_robot=tf_quat_robot,
            tf_pos_robot=tf_pos_robot,
            vel_lin_robot=vel_lin_robot,
            vel_ang_robot=vel_ang_robot,
            projected_gravity_robot=projected_gravity_robot,
            joint_pos_robot=joint_pos_robot,
            joint_pos_limits_robot=joint_pos_limits_robot,
            joint_vel_robot=joint_vel_robot,
            joint_acc_robot=joint_acc_robot,
            joint_applied_torque_robot=joint_applied_torque_robot,
            contact_forces_robot=contact_forces_robot,
            contact_robot=contact_robot,
            contact_last_air_time=contact_last_air_time,
            foot_link_vel_w=foot_link_vel_w,
            imu_lin_acc=imu_lin_acc,
            imu_ang_vel=imu_ang_vel,
            ## Robot descriptors
            robot_feet_indices=self._feet_indices,
            robot_undesired_contact_body_indices=self._undesired_contact_body_indices,
            ## Command
            command=command,
            ## Rewards
            action_rate_weight=self.cfg.rewards.action_rate_weight,
            joint_torque_weight=self.cfg.rewards.joint_torque_weight,
            joint_torque_max_penalty=self.cfg.rewards.joint_torque_max_penalty,
            joint_acceleration_weight=self.cfg.rewards.joint_acceleration_weight,
            joint_acceleration_max_penalty=(
                self.cfg.rewards.joint_acceleration_max_penalty
            ),
            undesired_contact_weight=self.cfg.rewards.undesired_contact_weight,
            undesired_contact_force_threshold=(
                self.cfg.rewards.undesired_contact_force_threshold
            ),
            command_linear_weight=self.cfg.rewards.command_linear_weight,
            command_linear_exp_std=self.cfg.rewards.command_linear_exp_std,
            command_angular_weight=self.cfg.rewards.command_angular_weight,
            command_angular_exp_std=self.cfg.rewards.command_angular_exp_std,
            feet_air_time_weight=self.cfg.rewards.feet_air_time_weight,
            feet_air_time_target=self.cfg.rewards.feet_air_time_target,
            moving_command_threshold=self.cfg.rewards.moving_command_threshold,
            foot_slip_weight=self.cfg.rewards.foot_slip_weight,
            foot_slip_contact_force_threshold=(
                self.cfg.rewards.foot_slip_contact_force_threshold
            ),
            foot_slip_max_penalty=self.cfg.rewards.foot_slip_max_penalty,
            undesired_linear_velocity_z_weight=(
                self.cfg.rewards.undesired_linear_velocity_z_weight
            ),
            undesired_angular_velocity_xy_weight=(
                self.cfg.rewards.undesired_angular_velocity_xy_weight
            ),
            gravity_alignment_weight=self.cfg.rewards.gravity_alignment_weight,
            ## Terminations
            min_base_height=self.cfg.terminations.min_base_height,
            min_body_up_z=self.cfg.terminations.min_body_up_z,
        )

        tf_rotmat_robot = matrix_from_quat(tf_quat_robot)
        body_up_z = tf_rotmat_robot[:, 2, 2]
        linear_tracking_error = torch.norm(
            command[:, :2] - vel_lin_robot[:, :2],
            dim=1,
        )
        angular_tracking_error = torch.abs(command[:, 2] - vel_ang_robot[:, 2])
        tracking_success = (
            (
                linear_tracking_error
                < self.cfg.terminations.tracking_linear_error_threshold
            )
            & (
                angular_tracking_error
                < self.cfg.terminations.tracking_angular_error_threshold
            )
            & (body_up_z >= self.cfg.terminations.tracking_min_body_up_z)
            & ~step_return.termination
        )
        current_stage = _resolve_curriculum_stage(
            self,
            curriculum_enabled=self.cfg.curriculum.enabled,
            fixed_stage=self.cfg.curriculum.fixed_stage,
            stage_env_steps=self.cfg.curriculum.stage_env_steps,
        )
        if self._undesired_contact_body_indices:
            undesired_contact = torch.max(
                torch.norm(
                    contact_forces_robot[
                        :, self._undesired_contact_body_indices, :
                    ],
                    dim=-1,
                ),
                dim=1,
            )[0] > self.cfg.rewards.undesired_contact_force_threshold
        else:
            undesired_contact = torch.zeros(
                self.num_envs,
                dtype=torch.bool,
                device=self.device,
            )

        return StepReturn(
            step_return.observation,
            step_return.reward,
            step_return.termination,
            step_return.truncation,
            {
                "metrics/curriculum_stage": torch.full_like(
                    linear_tracking_error,
                    float(current_stage),
                ),
                "metrics/command_lin_error": linear_tracking_error,
                "metrics/command_ang_error": angular_tracking_error,
                "metrics/tracking_success": tracking_success.float(),
                "metrics/body_up_z": body_up_z,
                "metrics/fallen": step_return.termination.float(),
                "metrics/undesired_contact": undesired_contact.float(),
            },
        )


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
    tf_quat_robot: torch.Tensor,
    tf_pos_robot: torch.Tensor,
    vel_lin_robot: torch.Tensor,
    vel_ang_robot: torch.Tensor,
    projected_gravity_robot: torch.Tensor,
    joint_pos_robot: torch.Tensor,
    joint_pos_limits_robot: torch.Tensor | None,
    joint_vel_robot: torch.Tensor,
    joint_acc_robot: torch.Tensor,
    joint_applied_torque_robot: torch.Tensor,
    contact_forces_robot: torch.Tensor,
    contact_robot: torch.Tensor,
    contact_last_air_time: torch.Tensor,
    foot_link_vel_w: torch.Tensor,
    imu_lin_acc: torch.Tensor,
    imu_ang_vel: torch.Tensor,
    ## Robot descriptors
    robot_feet_indices: List[int],
    robot_undesired_contact_body_indices: List[int],
    ## Command
    command: torch.Tensor,
    ## Rewards
    action_rate_weight: float,
    joint_torque_weight: float,
    joint_torque_max_penalty: float,
    joint_acceleration_weight: float,
    joint_acceleration_max_penalty: float,
    undesired_contact_weight: float,
    undesired_contact_force_threshold: float,
    command_linear_weight: float,
    command_linear_exp_std: float,
    command_angular_weight: float,
    command_angular_exp_std: float,
    feet_air_time_weight: float,
    feet_air_time_target: float,
    moving_command_threshold: float,
    foot_slip_weight: float,
    foot_slip_contact_force_threshold: float,
    foot_slip_max_penalty: float,
    undesired_linear_velocity_z_weight: float,
    undesired_angular_velocity_xy_weight: float,
    gravity_alignment_weight: float,
    ## Terminations
    min_base_height: float,
    min_body_up_z: float,
) -> StepReturn:
    num_envs = episode_length.size(0)
    device = episode_length.device

    ############
    ## States ##
    ############
    default_quat = torch.zeros_like(tf_quat_robot)
    default_quat[:, 3] = 1.0
    tf_quat_robot = torch.where(
        torch.norm(tf_quat_robot, dim=-1, keepdim=True) < 1.0e-6,
        default_quat,
        tf_quat_robot,
    )
    tf_rotmat_robot = matrix_from_quat(tf_quat_robot)
    tf_rot6d_robot = rotmat_to_rot6d(tf_rotmat_robot)

    joint_pos_robot_normalized = (
        scale_transform(
            joint_pos_robot,
            joint_pos_limits_robot[:, :, 0],
            joint_pos_limits_robot[:, :, 1],
        )
        if joint_pos_limits_robot is not None
        else joint_pos_robot
    )
    contact_forces_mean_robot = contact_forces_robot.mean(dim=1)

    #############
    ## Rewards ##
    #############
    penalty_action_rate = action_rate_weight * torch.mean(
        torch.square(act_current - act_previous),
        dim=1,
    )
    penalty_joint_torque = torch.clamp_min(
        joint_torque_weight
        * torch.sum(torch.square(joint_applied_torque_robot), dim=1),
        min=joint_torque_max_penalty,
    )
    penalty_joint_acceleration = torch.clamp_min(
        joint_acceleration_weight * torch.sum(torch.square(joint_acc_robot), dim=1),
        min=joint_acceleration_max_penalty,
    )

    penalty_undesired_robot_contacts = torch.zeros(
        num_envs,
        dtype=vel_lin_robot.dtype,
        device=device,
    )
    if len(robot_undesired_contact_body_indices) > 0:
        undesired_contact_force = torch.max(
            torch.norm(
                contact_forces_robot[
                    :, robot_undesired_contact_body_indices, :
                ],
                dim=-1,
            ),
            dim=1,
        )[0]
        penalty_undesired_robot_contacts = undesired_contact_weight * (
            undesired_contact_force > undesired_contact_force_threshold
        )

    reward_cmd_lin_vel_xy = command_linear_weight * torch.exp(
        -torch.sum(torch.square(command[:, :2] - vel_lin_robot[:, :2]), dim=1)
        / command_linear_exp_std
    )
    reward_cmd_ang_vel_z = command_angular_weight * torch.exp(
        -torch.square(command[:, 2] - vel_ang_robot[:, 2])
        / command_angular_exp_std
    )

    moving_command = torch.norm(command[:, :2], dim=1) > moving_command_threshold
    reward_feet_air_time = (
        feet_air_time_weight
        * moving_command
        * torch.sum(
            torch.clamp_min(
                contact_last_air_time[:, robot_feet_indices] - feet_air_time_target,
                min=0.0,
            )
            * contact_robot[:, robot_feet_indices],
            dim=1,
        )
    )

    feet_contact_force = torch.norm(
        contact_forces_robot[:, robot_feet_indices, :],
        dim=-1,
    )
    feet_in_contact = feet_contact_force > foot_slip_contact_force_threshold
    foot_slip_speed = torch.norm(
        foot_link_vel_w[:, robot_feet_indices, :2],
        dim=-1,
    )
    penalty_foot_slip = torch.clamp_min(
        foot_slip_weight
        * torch.sum(feet_in_contact * foot_slip_speed, dim=1),
        min=foot_slip_max_penalty,
    )

    penalty_undesired_lin_vel_z = undesired_linear_velocity_z_weight * torch.square(
        vel_lin_robot[:, 2]
    )
    penalty_undesired_ang_vel_xy = undesired_angular_velocity_xy_weight * torch.sum(
        torch.square(vel_ang_robot[:, :2]),
        dim=1,
    )
    penalty_gravity_rotation_alignment = gravity_alignment_weight * (
        torch.sum(torch.square(projected_gravity_robot[:, :2]), dim=1)
        + torch.square(projected_gravity_robot[:, 2] + 1.0)
    )

    ##################
    ## Terminations ##
    ##################
    termination_fallen = tf_pos_robot[:, 2] < min_base_height
    body_up_z = tf_rotmat_robot[:, 2, 2]
    termination_bad_orientation = body_up_z < min_body_up_z
    termination = termination_fallen | termination_bad_orientation
    truncation = (
        episode_length >= max_episode_length
        if truncate_episodes
        else torch.zeros(num_envs, dtype=torch.bool, device=device)
    )

    return StepReturn(
        {
            "state": {
                "contact_forces_mean_robot": contact_forces_mean_robot,
                "tf_rot6d_robot": tf_rot6d_robot,
                "vel_lin_robot": vel_lin_robot,
                "vel_ang_robot": vel_ang_robot,
                "projected_gravity_robot": projected_gravity_robot,
            },
            "state_dyn": {
                "contact_forces_robot": contact_forces_robot,
            },
            "proprio": {
                "vel_lin_robot": vel_lin_robot,
                "vel_ang_robot": vel_ang_robot,
                "projected_gravity_robot": projected_gravity_robot,
                "imu_lin_acc": imu_lin_acc,
                "imu_ang_vel": imu_ang_vel,
            },
            "proprio_dyn": {
                "joint_pos_robot_normalized": joint_pos_robot_normalized,
                "joint_vel_robot": joint_vel_robot,
                "joint_acc_robot": joint_acc_robot,
                "joint_applied_torque_robot": joint_applied_torque_robot,
                "act_previous": act_previous,
            },
            "command": {
                "cmd_vel": command,
            },
        },
        {
            "pen_action_rate": penalty_action_rate,
            "pen_joint_torque": penalty_joint_torque,
            "pen_joint_acceleration": penalty_joint_acceleration,
            "pen_und_robot_contacts": penalty_undesired_robot_contacts,
            "reward_cmd_lin_vel_xy": reward_cmd_lin_vel_xy,
            "reward_cmd_ang_vel_z": reward_cmd_ang_vel_z,
            "reward_feet_air_time": reward_feet_air_time,
            "pen_foot_slip": penalty_foot_slip,
            "pen_und_lin_vel_z": penalty_undesired_lin_vel_z,
            "pen_und_ang_vel_xy": penalty_undesired_ang_vel_xy,
            "pen_gravity_rot_ali": penalty_gravity_rotation_alignment,
        },
        termination,
        truncation,
    )
