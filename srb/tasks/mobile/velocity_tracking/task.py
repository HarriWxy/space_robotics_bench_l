from typing import Sequence

import torch

from srb._typing import StepReturn
from srb.core.env import GroundEnv, GroundEnvCfg, GroundEventCfg, GroundSceneCfg
from srb.core.manager import EventTermCfg, SceneEntityCfg
from srb.core.marker import ARROW_CFG, VisualizationMarkers
from srb.core.mdp import randomize_command, reset_root_state_uniform
from srb.core.sim import PreviewSurfaceCfg
from srb.utils.cfg import configclass
from srb.utils.math import deg_to_rad, matrix_from_quat, rotmat_to_rot6d

##############
### Config ###
##############


@configclass
class SceneCfg(GroundSceneCfg):
    pass


@configclass
class EventCfg(GroundEventCfg):
    randomize_robot_state: EventTermCfg = EventTermCfg(
        func=reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x": (-0.25, 0.25),
                "y": (-0.25, 0.25),
                "z": (0.0, 0.1),
                "yaw": (-torch.pi, torch.pi),
            },
            "velocity_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "z": (0.0, 0.1),
                "roll": (-deg_to_rad(2.0), deg_to_rad(2.0)),
                "pitch": (-deg_to_rad(2.0), deg_to_rad(2.0)),
                "yaw": (-deg_to_rad(5.0), deg_to_rad(5.0)),
            },
        },
    )

    command = EventTermCfg(
        func=randomize_command,
        mode="interval",
        interval_range_s=(0.5, 5.0),
        params={
            "env_attr_name": "_command",
            # "magnitude": 1.0,
        },
    )


@configclass
class TaskCfg(GroundEnvCfg):
    ## Scene
    scene: SceneCfg = SceneCfg()
    stack: bool = True

    ## Events
    events: EventCfg = EventCfg()

    ## Time
    episode_length_s: float = 200.0  # steps = episode_length_s / agent_rate
    is_finite_horizon: bool = False

    ## Visualization
    command_vis: bool = True

    def __post_init__(self):
        super().__post_init__()


############
### Task ###
############


class Task(GroundEnv):
    cfg: TaskCfg

    def __init__(self, cfg: TaskCfg, **kwargs):
        super().__init__(cfg, **kwargs)

        ## Initialize buffers
        self._command = torch.zeros(self.num_envs, 3, device=self.device)

        ## Visualization
        if self.cfg.command_vis:
            self._setup_visualization_markers()

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)

    def _setup_visualization_markers(self):
        ## Linear velocity
        cfg = ARROW_CFG.copy().replace(  # type: ignore
            prim_path="/Visuals/command/target_linvel"
        )
        cfg.markers["arrow"].tail_radius = 0.01
        cfg.markers["arrow"].tail_length = 0.5
        cfg.markers["arrow"].head_radius = 0.02
        cfg.markers["arrow"].head_length = 0.1
        cfg.markers["arrow"].visual_material = PreviewSurfaceCfg(
            emissive_color=(0.0, 1.0, 0.0)
        )
        self._marker_target_linvel = VisualizationMarkers(cfg)
        cfg = ARROW_CFG.copy().replace(  # type: ignore
            prim_path="/Visuals/command/robot_linvel"
        )
        cfg.markers["arrow"].tail_radius = 0.01
        cfg.markers["arrow"].tail_length = 0.5
        cfg.markers["arrow"].head_radius = 0.02
        cfg.markers["arrow"].head_length = 0.1
        cfg.markers["arrow"].visual_material = PreviewSurfaceCfg(
            emissive_color=(0.2, 0.8, 0.2)
        )
        self._marker_robot_linvel = VisualizationMarkers(cfg)

        ## Angular velocity
        cfg = ARROW_CFG.copy().replace(  # type: ignore
            prim_path="/Visuals/command/target_angvel"
        )
        cfg.markers["arrow"].tail_length = 0.0
        cfg.markers["arrow"].tail_radius = 0.0
        cfg.markers["arrow"].head_radius = 0.025
        cfg.markers["arrow"].head_length = 0.15
        cfg.markers["arrow"].visual_material = PreviewSurfaceCfg(
            emissive_color=(0.2, 0.2, 0.8)
        )
        self._marker_target_angvel = VisualizationMarkers(cfg)
        cfg = ARROW_CFG.copy().replace(  # type: ignore
            prim_path="/Visuals/command/robot_angvel"
        )
        cfg.markers["arrow"].tail_length = 0.0
        cfg.markers["arrow"].tail_radius = 0.0
        cfg.markers["arrow"].head_radius = 0.025
        cfg.markers["arrow"].head_length = 0.15
        cfg.markers["arrow"].visual_material = PreviewSurfaceCfg(
            emissive_color=(0.2, 0.2, 0.8)
        )
        self._marker_robot_angvel = VisualizationMarkers(cfg)

    def _update_visualization_markers(self):
        MARKER_OFFSET_Z_LINVEL = 0.2
        MARKER_OFFSET_Z_ANGVEL = 0.175

        ## Common
        robot_pos_w = self._robot.data.root_link_pos_w
        marker_pos = torch.zeros(
            (self.cfg.scene.num_envs, 3), dtype=torch.float32, device=self.device
        )
        marker_orientation = torch.zeros(
            (self.cfg.scene.num_envs, 4), dtype=torch.float32, device=self.device
        )
        marker_scale = torch.ones(
            (self.cfg.scene.num_envs, 3), dtype=torch.float32, device=self.device
        )
        marker_pos[:, :2] = robot_pos_w[:, :2]

        ## Target linear velocity
        marker_pos[:, 2] = robot_pos_w[:, 2] + MARKER_OFFSET_Z_LINVEL
        marker_heading = self._robot.data.heading_w + torch.atan2(
            self._command[:, 1], self._command[:, 0]
        )
        marker_orientation[:, 0] = torch.cos(marker_heading * 0.5)
        marker_orientation[:, 3] = torch.sin(marker_heading * 0.5)
        marker_scale[:, 0] = torch.norm(
            torch.stack(
                (self._command[:, 0], self._command[:, 1]),
                dim=-1,
            ),
            dim=-1,
        )
        self._marker_target_linvel.visualize(
            marker_pos, marker_orientation, marker_scale
        )

        ## Robot linear velocity
        marker_heading = self._robot.data.heading_w + torch.atan2(
            self._robot.data.root_lin_vel_b[:, 1],
            self._robot.data.root_lin_vel_b[:, 0],
        )
        marker_orientation[:, 0] = torch.cos(marker_heading * 0.5)
        marker_orientation[:, 3] = torch.sin(marker_heading * 0.5)
        marker_scale[:, 0] = torch.norm(self._robot.data.root_lin_vel_b[:, :2], dim=-1)
        self._marker_robot_linvel.visualize(
            marker_pos, marker_orientation, marker_scale
        )

        ## Target angular velocity
        _target_angvel_abs = self._command[:, 2].abs()
        normalization_fac = torch.where(
            self._command[:, 2] != 0.0,
            (torch.pi / 2.0) / self._command[:, 2].abs(),
            torch.ones_like(self._command[:, 2]),
        ).clamp(max=1.0)
        marker_pos[:, 2] = robot_pos_w[:, 2] + MARKER_OFFSET_Z_ANGVEL
        marker_heading = (
            self._robot.data.heading_w + normalization_fac * self._command[:, 2]
        )
        marker_orientation[:, 0] = torch.cos(marker_heading * 0.5)
        marker_orientation[:, 3] = torch.sin(marker_heading * 0.5)
        marker_scale[:, 0] = 1.0
        self._marker_target_angvel.visualize(
            marker_pos, marker_orientation, marker_scale
        )

        ## Robot angular velocity
        marker_heading = (
            self._robot.data.heading_w
            + normalization_fac * self._robot.data.root_ang_vel_w[:, -1]
        )
        marker_orientation[:, 0] = torch.cos(marker_heading * 0.5)
        marker_orientation[:, 3] = torch.sin(marker_heading * 0.5)
        marker_scale[:, 0] = 1.0
        self._marker_robot_angvel.visualize(
            marker_pos, marker_orientation, marker_scale
        )

    def extract_step_return(self) -> StepReturn:
        if self.cfg.command_vis or self.cfg.debug_vis:
            self._update_visualization_markers()

        # Sanitize sensor data to prevent NaN/Inf propagation into the
        # observation pipeline and camera rendering (Warp CUDA kernels).
        imu_lin_acc = torch.nan_to_num(self._imu_robot.data.lin_acc_b, nan=0.0, posinf=0.0, neginf=0.0)
        imu_ang_vel = torch.nan_to_num(self._imu_robot.data.ang_vel_b, nan=0.0, posinf=0.0, neginf=0.0)
        vel_lin_robot = torch.nan_to_num(self._robot.data.root_lin_vel_b, nan=0.0, posinf=0.0, neginf=0.0)
        vel_ang_robot = torch.nan_to_num(self._robot.data.root_ang_vel_b, nan=0.0, posinf=0.0, neginf=0.0)

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
            tf_quat_robot=self._robot.data.root_quat_w,
            tf_pos_robot=self._robot.data.root_pos_w,
            vel_lin_robot=vel_lin_robot,
            vel_ang_robot=vel_ang_robot,
            # IMU
            imu_lin_acc=imu_lin_acc,
            imu_ang_vel=imu_ang_vel,
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
    tf_quat_robot: torch.Tensor,
    tf_pos_robot: torch.Tensor,
    vel_lin_robot: torch.Tensor,
    vel_ang_robot: torch.Tensor,
    # IMU
    imu_lin_acc: torch.Tensor,
    imu_ang_vel: torch.Tensor,
    ## Command
    command: torch.Tensor,
) -> StepReturn:
    num_envs = episode_length.size(0)
    # dtype = episode_length.dtype
    device = episode_length.device

    ############
    ## States ##
    ############
    ## Root
    # Sanitize quaternion before conversion to prevent NaN propagation
    tf_quat_robot = torch.nan_to_num(tf_quat_robot, nan=0.0)
    tf_quat_robot = torch.where(
        torch.norm(tf_quat_robot, dim=-1, keepdim=True) < 1e-6,
        torch.tensor([1.0, 0.0, 0.0, 0.0], device=device),
        tf_quat_robot,
    )
    tf_rotmat_robot = matrix_from_quat(tf_quat_robot)
    tf_rot6d_robot = rotmat_to_rot6d(tf_rotmat_robot)

    #############
    ## Rewards ##
    #############
    # Penalty: Action rate
    WEIGHT_ACTION_RATE = -0.5
    penalty_action_rate = WEIGHT_ACTION_RATE * torch.mean(
        torch.square(act_current - act_previous), dim=1
    )

    # Reward: Command tracking (linear)
    WEIGHT_CMD_LIN_VEL_XY = 3.0
    EXP_STD_CMD_LIN_VEL_XY = 0.5
    reward_cmd_lin_vel_xy = WEIGHT_CMD_LIN_VEL_XY * torch.exp(
        -torch.sum(torch.square(command[:, :2] - vel_lin_robot[:, :2]), dim=1)
        / EXP_STD_CMD_LIN_VEL_XY
    )

    # Reward: Command tracking (angular)
    WEIGHT_CMD_ANG_VEL_Z = 1.5
    EXP_STD_CMD_ANG_VEL_Z = 0.25
    reward_cmd_ang_vel_z = WEIGHT_CMD_ANG_VEL_Z * torch.exp(
        -torch.square(command[:, 2] - vel_ang_robot[:, 2]) / EXP_STD_CMD_ANG_VEL_Z
    )

    # Penalty: Minimize non-command motion (linear)
    WEIGHT_UNDESIRED_LIN_VEL_Z = -0.5
    penalty_undesired_lin_vel_z = WEIGHT_UNDESIRED_LIN_VEL_Z * torch.square(
        vel_lin_robot[:, 2]
    )

    # Penalty: Minimize non-command motion (angular)
    WEIGHT_UNDESIRED_ANG_VEL_XY = -0.1
    penalty_undesired_ang_vel_xy = WEIGHT_UNDESIRED_ANG_VEL_XY * torch.sum(
        torch.square(vel_ang_robot[:, :2]), dim=-1
    )

    ##################
    ## Terminations ##
    ##################
    # Termination: Robot has fallen (body height too low)
    # G1 init height is ~0.74m, terminate if below 0.3m
    termination_fallen = tf_pos_robot[:, 2] < 0.3
    # Termination: Bad orientation (robot flipped or severely tilted)
    # z-axis of rotation matrix is the body's up direction in world frame
    body_up_z = tf_rotmat_robot[:, 2, 2]  # cos(tilt angle)
    termination_bad_orientation = body_up_z < 0.3  # tilted more than ~70 degrees

    termination = termination_fallen | termination_bad_orientation
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
            "pen_action_rate": penalty_action_rate,
            "reward_cmd_lin_vel_xy": reward_cmd_lin_vel_xy,
            "reward_cmd_ang_vel_z": reward_cmd_ang_vel_z,
            "pen_und_lin_vel_z": penalty_undesired_lin_vel_z,
            "pen_und_ang_vel_xy": penalty_undesired_ang_vel_xy,
        },
        termination,
        truncation,
    )
