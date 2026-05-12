from __future__ import annotations

import re
from typing import TYPE_CHECKING, Type

import torch
from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointActionCfg

from srb.core.manager import ActionTerm
from srb.utils.cfg import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from srb.core.asset import Articulation


class ScalarJointPositionAction(ActionTerm):
    """Map a single scalar action in [-1, 1] to a joint position range.

    A value of -1.0 corresponds to the configured close command and +1.0 to the
    open command. Intermediate values linearly interpolate between the two.
    """

    cfg: "ScalarJointPositionActionCfg"
    _asset: "Articulation"

    def __init__(self, cfg: "ScalarJointPositionActionCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)

        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)
        self._num_joints = len(self._joint_ids)

        self._raw_actions = torch.zeros(self.num_envs, 1, device=self.device)
        self._processed_actions = torch.zeros(
            self.num_envs, self._num_joints, device=self.device
        )

        self._open_command = torch.zeros(self._num_joints, device=self.device)
        self._close_command = torch.zeros(self._num_joints, device=self.device)

        for joint_id, joint_name in enumerate(self._joint_names):
            open_value = None
            close_value = None
            for expr, value in self.cfg.open_command_expr.items():
                if re.fullmatch(expr, joint_name):
                    open_value = value
                    break
            for expr, value in self.cfg.close_command_expr.items():
                if re.fullmatch(expr, joint_name):
                    close_value = value
                    break
            if open_value is None or close_value is None:
                raise ValueError(
                    f"Could not resolve open/close commands for joint '{joint_name}'"
                )
            self._open_command[joint_id] = open_value
            self._close_command[joint_id] = close_value

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        alpha = torch.clamp((actions + 1.0) * 0.5, 0.0, 1.0)
        self._processed_actions = torch.lerp(
            self._close_command.unsqueeze(0),
            self._open_command.unsqueeze(0),
            alpha,
        )

    def apply_actions(self):
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)


@configclass
class ScalarJointPositionActionCfg(BinaryJointActionCfg):
    class_type: Type[ActionTerm] = ScalarJointPositionAction
    asset_name: str = "robot"