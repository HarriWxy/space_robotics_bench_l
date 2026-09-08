from typing import Sequence

import torch

from srb import assets
from srb._typing import StepReturn
from srb.core.action import ThrustAction
from srb.core.asset import AssetVariant, ExtravehicularScenery, MobileRobot
from srb.core.env import (
    KitVisualizerCfg,
    OrbitalEnv,
    OrbitalEnvCfg,
    OrbitalEventCfg,
    OrbitalSceneCfg,
)
from srb.core.manager import EventTermCfg, SceneEntityCfg
from srb.core.marker import VisualizationMarkers, VisualizationMarkersCfg
from srb.core.mdp import apply_external_force_torque, offset_pose_natural
from srb.core.sim import ArrowCfg, PreviewSurfaceCfg
from srb.utils import logging
from srb.utils.cfg import configclass
from srb.utils.math import (
    matrix_from_quat,
    rotmat_to_rot6d,
    rpy_to_quat,
    subtract_frame_transforms,
)

##############
### Config ###
##############


@configclass
class SceneCfg(OrbitalSceneCfg):
    pass


@configclass
class EventCfg(OrbitalEventCfg):
    target_pose_evolution: EventTermCfg = EventTermCfg(
        func=offset_pose_natural,
        mode="interval",
        interval_range_s=(0.1, 0.1),
        is_global_time=True,
        params={
            "env_attr_name": "_goal",
            "pos_axes": ("x", "y", "z"),
            "pos_step_range": (0.01, 0.1),
            "pos_smoothness": 0.96,
            "pos_step_smoothness": 0.8,
            "pos_bounds": {
                "x": (-32.0, 32.0),
                "y": (-32.0, 32.0),
                "z": (-32.0, 32.0),
            },
            "orient_yaw_only": False,
            "orient_smoothness": 0.8,
        },
    )
    push_robot: EventTermCfg = EventTermCfg(
        func=apply_external_force_torque,
        mode="interval",
        interval_range_s=(5.0, 20.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "force_range": (-2.0, 2.0),
            "torque_range": (-0.08727, 0.08727),
        },
    )


@configclass
class TaskCfg(OrbitalEnvCfg):
    ## Scene
    scene: SceneCfg = SceneCfg()
    stack: bool = True

    ## Assets
    scenery: ExtravehicularScenery | MobileRobot | AssetVariant | None = (
        assets.Gateway()
    )
    scenery.asset_cfg.init_state.pos = (0.0, 7.0, -10.0)
    scenery.asset_cfg.init_state.rot = rpy_to_quat(0.0, 0.0, 90.0)
    scenery.asset_cfg.spawn.collision_props.collision_enabled = False  # type: ignore
    scenery.asset_cfg.spawn.mesh_collision_props.mesh_approximation = None  # type: ignore

    ## Events
    events: EventCfg = EventCfg()

    ## Time
    env_rate: float = 1.0 / 40.0
    agent_rate: float = 1.0 / 10.0
    episode_length_s: float = 30.0
    is_finite_horizon: bool = True

    ## Target
    target_pos_range_ratio: float = 0.9
    target_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/target",
        markers={
            "target": ArrowCfg(
                tail_radius=0.02,
                tail_length=0.3,
                head_radius=0.1,
                head_length=0.15,
                visual_material=PreviewSurfaceCfg(emissive_color=(0.2, 0.2, 0.8)),
            )
        },
    )

    ## Viewer
    visualizer: KitVisualizerCfg = KitVisualizerCfg(
        eye=(3.0, -3.0, 3.0),
        lookat=(0.0, 0.0, 0.0),
        origin_type="asset",
        origin_track_path="robot",
    )


############
### Task ###
############


class Task(OrbitalEnv):
    cfg: TaskCfg

    def __init__(self, cfg: TaskCfg, **kwargs):
        super().__init__(cfg, **kwargs)

        ## Get scene assets
        self._target_marker: VisualizationMarkers = VisualizationMarkers(
            self.cfg.target_marker_cfg
        )

        ## Initialize buffers
        self._goal = torch.zeros(self.num_envs, 7, device=self.device)
        self._goal[:, 0:3] = self.scene.env_origins
        self._goal[:, 6] = 1.0

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)

        ## Reset goal position
        self._goal[env_ids, 0:3] = self.scene.env_origins[env_ids]
        self._goal[env_ids, 3:7] = torch.tensor(
            [0.0, 0.0, 0.0, 1.0], device=self.device
        )

        ## Setup TF listener for ROS-defined target (if applicable)
        if hasattr(self.unwrapped, "ros_node"):
            from tf2_ros import Buffer, TransformListener

            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(
                self.tf_buffer, self.unwrapped.ros_node
            )
            logging.info("TF listener initialized.")
            # Disable target evolution event if using ROS TF
            self.cfg.events.target_pose_evolution.interval_range_s = (1.0e9, 1.0e9)
            self.event_manager.set_term_cfg(
                "target_pose_evolution", self.cfg.events.target_pose_evolution
            )

            self.event_manager.active_terms

    def extract_step_return(self) -> StepReturn:
        ## Get target pose from ROS TF (if applicable)
        if hasattr(self, "tf_buffer"):
            from rclpy.time import Duration, Time

            try:
                tf_stamped = self.tf_buffer.lookup_transform(
                    "world",
                    "target",
                    Time(),
                    timeout=Duration(
                        seconds=1,
                        nanoseconds=0,
                    ),
                )
                logging.debug(f"Got transform from 'world' to 'target': {tf_stamped}")
                self._goal[:, 0] = tf_stamped.transform.translation.x
                self._goal[:, 1] = tf_stamped.transform.translation.y
                self._goal[:, 2] = tf_stamped.transform.translation.z
                self._goal[:, 3] = tf_stamped.transform.rotation.w
                self._goal[:, 4] = tf_stamped.transform.rotation.x
                self._goal[:, 5] = tf_stamped.transform.rotation.y
                self._goal[:, 6] = tf_stamped.transform.rotation.z
            except Exception as e:
                logging.warning(
                    f"Failed to get transform from 'world' to 'target': {e}"
                )

        ## Visualize target
        self._target_marker.visualize(self._goal[:, 0:3], self._goal[:, 3:7])

        ## Get remaining fuel (if applicable)
        if self._thrust_action_term_key:
            thrust_action_term: ThrustAction = self.action_manager._terms[  # type: ignore
                self._thrust_action_term_key
            ]
            remaining_fuel = (
                thrust_action_term.remaining_fuel / thrust_action_term.cfg.fuel_capacity
            ).unsqueeze(-1)
        else:
            remaining_fuel = None

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
            tf_pos_robot=self._robot.data.root_pos_w.torch,
            tf_quat_robot=self._robot.data.root_quat_w.torch,
            vel_lin_robot=self._robot.data.root_lin_vel_b.torch,
            vel_ang_robot=self._robot.data.root_ang_vel_b.torch,
            # Transforms (world frame)
            tf_pos_target=self._goal[:, 0:3],
            tf_quat_target=self._goal[:, 3:7],
            # IMU
            imu_lin_acc=self._imu_robot.data.lin_acc_b.torch,
            imu_ang_vel=self._imu_robot.data.ang_vel_b.torch,
            # Fuel
            remaining_fuel=remaining_fuel,
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
    # Transforms (world frame)
    tf_pos_target: torch.Tensor,
    tf_quat_target: torch.Tensor,
    # IMU
    imu_lin_acc: torch.Tensor,
    imu_ang_vel: torch.Tensor,
    # Fuel
    remaining_fuel: torch.Tensor | None,
) -> StepReturn:
    num_envs = episode_length.size(0)
    dtype = episode_length.dtype
    device = episode_length.device

    ############
    ## States ##
    ############
    ## Root
    tf_rotmat_robot = matrix_from_quat(tf_quat_robot)
    tf_rot6d_robot = rotmat_to_rot6d(tf_rotmat_robot)

    ## Transforms (world frame)
    # Robot -> Target
    tf_pos_robot_to_target, tf_quat_robot_to_target = subtract_frame_transforms(
        t01=tf_pos_robot,
        q01=tf_quat_robot,
        t02=tf_pos_target,
        q02=tf_quat_target,
    )
    tf_rotmat_robot_to_target = matrix_from_quat(tf_quat_robot_to_target)
    tf_rot6d_robot_to_target = rotmat_to_rot6d(tf_rotmat_robot_to_target)
    dist_robot_to_target = torch.norm(tf_pos_robot_to_target, dim=-1)

    ## Fuel
    remaining_fuel = (
        remaining_fuel
        if remaining_fuel is not None
        else torch.ones((num_envs, 1), dtype=dtype, device=device)
    )

    #############
    ## Rewards ##
    #############
    # Penalty: Action rate
    WEIGHT_ACTION_RATE = -16.0
    _action_rate = torch.mean(torch.square(act_current - act_previous), dim=1)
    penalty_action_rate = WEIGHT_ACTION_RATE * _action_rate

    # Penalty: Action magnitude
    WEIGHT_ACTION_MAGNITUDE = -16.0
    _action_magnitude = torch.mean(torch.square(act_current), dim=1)
    penalty_action_magnitude = WEIGHT_ACTION_MAGNITUDE * _action_magnitude

    # Penalty: Fuel consumption
    WEIGHT_FUEL_CONSUMPTION = -8.0
    penalty_fuel_consumption = WEIGHT_FUEL_CONSUMPTION * torch.square(
        1.0 - remaining_fuel.squeeze(-1)
    )

    # Penalty: Position tracking | Robot <--> Target
    WEIGHT_POSITION_TRACKING = -2.0
    MAX_POSITION_TRACKING_PENALTY = -16.0
    penalty_position_tracking = torch.clamp_min(
        WEIGHT_POSITION_TRACKING * torch.square(dist_robot_to_target),
        min=MAX_POSITION_TRACKING_PENALTY,
    )

    # Reward: Position tracking | Robot <--> Target (precision)
    WEIGHT_POSITION_TRACKING_PRECISION = 16.0
    TANH_STD_POSITION_TRACKING_PRECISION = 0.1
    _position_tracking_precision = 1.0 - torch.tanh(
        dist_robot_to_target / TANH_STD_POSITION_TRACKING_PRECISION
    )
    reward_position_tracking_precision = (
        WEIGHT_POSITION_TRACKING_PRECISION * _position_tracking_precision
    )

    # Reward: Target orientation tracking once position is reached | Robot <--> Target
    WEIGHT_ORIENTATION_TRACKING = 64.0
    TANH_STD_ORIENTATION_TRACKING = 0.25
    orientation_error = torch.linalg.matrix_norm(
        tf_rotmat_robot_to_target
        - torch.eye(3, device=device).unsqueeze(0).expand_as(tf_rotmat_robot_to_target),
        ord="fro",
    )
    _orientation_tracking_precision = _position_tracking_precision * (
        1.0 - torch.tanh(orientation_error / TANH_STD_ORIENTATION_TRACKING)
    )
    reward_orientation_tracking = (
        WEIGHT_ORIENTATION_TRACKING * _orientation_tracking_precision
    )

    # Reward: Action rate at target
    WEIGHT_ACTION_RATE_AT_TARGET = 128.0
    TANH_STD_ACTION_RATE_AT_TARGET = 0.2
    reward_action_rate_at_target = (
        WEIGHT_ACTION_RATE_AT_TARGET
        * _orientation_tracking_precision
        * (1.0 - torch.tanh(_action_rate / TANH_STD_ACTION_RATE_AT_TARGET))
    )

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
                "act_previous": act_previous,
                "tf_rot6d_robot": tf_rot6d_robot,
                "vel_lin_robot": vel_lin_robot,
                "vel_ang_robot": vel_ang_robot,
                "tf_pos_robot_to_target": tf_pos_robot_to_target,
                "tf_rot6d_robot_to_target": tf_rot6d_robot_to_target,
            },
            "proprio": {
                "imu_lin_acc": imu_lin_acc,
                "imu_ang_vel": imu_ang_vel,
                "remaining_fuel": remaining_fuel,
            },
        },
        {
            "penalty_action_rate": penalty_action_rate,
            "penalty_action_magnitude": penalty_action_magnitude,
            "penalty_fuel_consumption": penalty_fuel_consumption,
            "penalty_position_tracking": penalty_position_tracking,
            "reward_position_tracking_precision": reward_position_tracking_precision,
            "reward_orientation_tracking": reward_orientation_tracking,
            "reward_action_rate_at_target": reward_action_rate_at_target,
        },
        termination,
        truncation,
    )
