"""Lunar obstacle crossing base task.

A ground locomotion task on the Moon surface with procedurally generated
rock obstacles. The robot must navigate forward while avoiding collisions
with obstacles placed on the terrain.
"""

from dataclasses import MISSING
from typing import Sequence, Tuple

import torch

from srb import assets
from srb._typing import StepReturn
from srb.core.asset import RigidObjectCollection, RigidObjectCollectionCfg
from srb.core.domain import Domain
from srb.core.env import GroundEnv, GroundEnvCfg, GroundEventCfg, GroundSceneCfg
from srb.core.manager import EventTermCfg, SceneEntityCfg
from srb.core.mdp import (
    randomize_command,
    reset_collection_root_state_uniform_poisson_disk_2d,
)
from srb.utils.cfg import configclass
from srb.utils.math import matrix_from_quat, rotmat_to_rot6d

from .asset import select_obstacle

##############
### Config ###
##############


@configclass
class SceneCfg(GroundSceneCfg):
    env_spacing: float = 32.0

    ## Obstacles (populated in TaskCfg.__post_init__)
    objs: RigidObjectCollectionCfg = RigidObjectCollectionCfg(
        rigid_objects=MISSING,  # type: ignore
    )


@configclass
class EventCfg(GroundEventCfg):
    randomize_object_state: EventTermCfg = EventTermCfg(
        func=reset_collection_root_state_uniform_poisson_disk_2d,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("objs"),
            "pose_range": {
                "x": (-10.0, 10.0),
                "y": (-10.0, 10.0),
                "z": (-0.2, 0.0),
                "roll": (-torch.pi, torch.pi),
                "pitch": (-torch.pi, torch.pi),
                "yaw": (-torch.pi, torch.pi),
            },
            "velocity_range": {},
            "radius": (2.0),
        },
    )


@configclass
class TaskCfg(GroundEnvCfg):
    ## Domain: Moon (gravity = 1.625 m/s²)
    domain: Domain = Domain.MOON

    ## Scene
    scene: SceneCfg = SceneCfg()

    ## Events
    events: EventCfg = EventCfg()

    ## Obstacles
    num_obstacles: int = 12

    ## Time
    episode_length_s: float = 30.0
    is_finite_horizon_task: bool = False

    def __post_init__(self):
        super().__post_init__()

        # Scene: Moon rock obstacles
        self.scene.objs.rigid_objects = {
            f"obstacle{i}": select_obstacle(
                self,
                prim_path=f"{{ENV_REGEX_NS}}/obstacle{i}",
                seed=self.seed + (i * self.scene.num_envs),
                scale=(0.3 + 0.1 * (i % 3), 0.3 + 0.1 * (i % 3), 0.3 + 0.1 * (i % 3)),
                activate_contact_sensors=True,
            )
            for i in range(self.num_obstacles)
        }

        # Update seed & number of variants for procedural assets
        self._update_procedural_assets()


############
### Task ###
############


class Task(GroundEnv):
    cfg: TaskCfg

    def __init__(self, cfg: TaskCfg, **kwargs):
        super().__init__(cfg, **kwargs)

        ## Get scene assets
        self._objs: RigidObjectCollection = self.scene["objs"]

        ## Initialize command buffer (vx, vy, wz)
        self._command = torch.zeros(self.num_envs, 3, device=self.device)

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)

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
            # Root
            tf_pos_robot=self._robot.data.root_pos_w,
            tf_quat_robot=self._robot.data.root_quat_w,
            vel_lin_robot=self._robot.data.root_lin_vel_b,
            vel_ang_robot=self._robot.data.root_ang_vel_b,
            # Obstacles
            tf_pos_objs=self._objs.data.object_com_pos_w,
            # IMU
            imu_lin_acc=self._imu_robot.data.lin_acc_b,
            imu_ang_vel=self._imu_robot.data.ang_vel_b,
            ## Command
            command=self._command,
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
    # Root
    tf_pos_robot: torch.Tensor,
    tf_quat_robot: torch.Tensor,
    vel_lin_robot: torch.Tensor,
    vel_ang_robot: torch.Tensor,
    # Obstacles
    tf_pos_objs: torch.Tensor,
    # IMU
    imu_lin_acc: torch.Tensor,
    imu_ang_vel: torch.Tensor,
    ## Command
    command: torch.Tensor,
) -> StepReturn:
    num_envs = episode_length.size(0)
    device = episode_length.device

    ############
    ## States ##
    ############
    tf_rotmat_robot = matrix_from_quat(tf_quat_robot)
    tf_rot6d_robot = rotmat_to_rot6d(tf_rotmat_robot)

    # Relative positions to obstacles
    pos_robot_to_objs = tf_pos_objs - tf_pos_robot.unsqueeze(1)  # (N, K, 3)
    dist_to_objs = torch.norm(pos_robot_to_objs, dim=-1)  # (N, K)
    min_dist_to_obj, _ = dist_to_objs.min(dim=1)  # (N,)

    #############
    ## Rewards ##
    #############
    # Penalty: Action rate
    WEIGHT_ACTION_RATE = -0.5
    penalty_action_rate = WEIGHT_ACTION_RATE * torch.mean(
        torch.square(act_current - act_previous), dim=1
    )

    # Reward: Forward progress (positive x-direction)
    WEIGHT_FORWARD_PROGRESS = 2.0
    reward_forward_progress = WEIGHT_FORWARD_PROGRESS * vel_lin_robot[:, 0]

    # Reward: Command tracking (linear XY)
    WEIGHT_CMD_LIN_VEL_XY = 1.5
    EXP_STD_CMD_LIN_VEL_XY = 0.5
    reward_cmd_lin_vel_xy = WEIGHT_CMD_LIN_VEL_XY * torch.exp(
        -torch.sum(torch.square(command[:, :2] - vel_lin_robot[:, :2]), dim=1)
        / EXP_STD_CMD_LIN_VEL_XY
    )

    # Reward: Command tracking (angular Z)
    WEIGHT_CMD_ANG_VEL_Z = 0.75
    EXP_STD_CMD_ANG_VEL_Z = 0.25
    reward_cmd_ang_vel_z = WEIGHT_CMD_ANG_VEL_Z * torch.exp(
        -torch.square(command[:, 2] - vel_ang_robot[:, 2]) / EXP_STD_CMD_ANG_VEL_Z
    )

    # Penalty: Undesired linear velocity (z)
    WEIGHT_UNDESIRED_LIN_VEL_Z = -0.5
    penalty_undesired_lin_vel_z = WEIGHT_UNDESIRED_LIN_VEL_Z * torch.square(
        vel_lin_robot[:, 2]
    )

    # Penalty: Undesired angular velocity (xy)
    WEIGHT_UNDESIRED_ANG_VEL_XY = -0.1
    penalty_undesired_ang_vel_xy = WEIGHT_UNDESIRED_ANG_VEL_XY * torch.sum(
        torch.square(vel_ang_robot[:, :2]), dim=-1
    )

    # Penalty: Obstacle proximity
    WEIGHT_OBSTACLE_PROXIMITY = -1.0
    MIN_SAFE_DIST = 0.5
    penalty_obstacle_proximity = WEIGHT_OBSTACLE_PROXIMITY * torch.clamp(
        MIN_SAFE_DIST - min_dist_to_obj, min=0.0
    )

    # Penalty: Gravity alignment (stay upright)
    WEIGHT_GRAVITY_ROTATION_ALIGNMENT = -2.0
    projected_gravity = tf_rotmat_robot[:, :, 2]  # z-column of rotation matrix
    penalty_gravity_rotation_alignment = WEIGHT_GRAVITY_ROTATION_ALIGNMENT * (
        torch.sum(torch.square(projected_gravity[:, :2]), dim=1)
        + torch.square(projected_gravity[:, 2] + 1.0)
    )

    ##################
    ## Terminations ##
    ##################
    # Terminate if robot falls (base height too low or tilted too much)
    termination = (tf_pos_robot[:, 2] < 0.3) | (projected_gravity[:, 2] > -0.3)

    # Truncation
    truncation = (
        episode_length >= max_episode_length
        if truncate_episodes
        else torch.zeros(num_envs, dtype=torch.bool, device=device)
    )

    return StepReturn(
        {
            "state": {
                "tf_rot6d_robot": tf_rot6d_robot,
                "vel_lin_robot": vel_lin_robot,
                "vel_ang_robot": vel_ang_robot,
                "min_dist_to_obj": min_dist_to_obj.unsqueeze(-1),
            },
            "proprio": {
                "imu_lin_acc": imu_lin_acc,
                "imu_ang_vel": imu_ang_vel,
            },
            "command": {
                "cmd_vel": command,
            },
        },
        {
            "penalty_action_rate": penalty_action_rate,
            "reward_forward_progress": reward_forward_progress,
            "reward_cmd_lin_vel_xy": reward_cmd_lin_vel_xy,
            "reward_cmd_ang_vel_z": reward_cmd_ang_vel_z,
            "penalty_undesired_lin_vel_z": penalty_undesired_lin_vel_z,
            "penalty_undesired_ang_vel_xy": penalty_undesired_ang_vel_xy,
            "penalty_obstacle_proximity": penalty_obstacle_proximity,
            "penalty_gravity_rotation_alignment": penalty_gravity_rotation_alignment,
        },
        termination,
        truncation,
    )
