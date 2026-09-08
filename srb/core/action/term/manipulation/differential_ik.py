from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.envs.mdp.actions.actions_cfg import (
    DifferentialInverseKinematicsActionCfg as __DifferentialInverseKinematicsActionCfg,
)
from isaaclab.envs.mdp.actions.task_space_actions import (
    DifferentialInverseKinematicsAction as __DifferentialInverseKinematicsAction,
)
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from srb._typing import AnyEnv
    from srb.core.asset import Articulation


class _StableDifferentialIKController(DifferentialIKController):
    """Differential IK controller with a numerically stable DLS solve."""

    _MIN_DLS_DAMPING = 1.0e-4

    def _compute_delta_joint_pos(
        self, delta_pose: torch.Tensor, jacobian: torch.Tensor
    ) -> torch.Tensor:
        """Compute DLS updates without explicitly inverting the normal matrix."""
        if self.cfg.ik_method != "dls":
            return super()._compute_delta_joint_pos(delta_pose, jacobian)

        if self.cfg.ik_params is None:
            raise RuntimeError(
                "Inverse-kinematics parameters for method 'dls' are not defined!"
            )

        # A positive damping term makes J J^T + lambda^2 I positive definite for
        # finite Jacobians.  The floor also protects old/overridden configs with
        # a zero damping coefficient.
        damping = max(
            abs(float(self.cfg.ik_params["lambda_val"])), self._MIN_DLS_DAMPING
        )
        jacobian_t = torch.transpose(jacobian, dim0=1, dim1=2)
        normal_matrix = torch.bmm(jacobian, jacobian_t)
        normal_matrix = normal_matrix + (damping**2) * torch.eye(
            jacobian.shape[1], device=jacobian.device, dtype=jacobian.dtype
        )
        return torch.bmm(
            jacobian_t,
            torch.linalg.solve(normal_matrix, delta_pose.unsqueeze(-1)),
        ).squeeze(-1)


class DifferentialInverseKinematicsAction(__DifferentialInverseKinematicsAction):
    cfg: "DifferentialInverseKinematicsActionCfg"
    _env: "AnyEnv"
    _asset: "Articulation"

    def __init__(self, cfg: "DifferentialInverseKinematicsActionCfg", env: "AnyEnv"):
        super().__init__(
            cfg,
            env,  # type: ignore
        )

        # The upstream action constructs the controller directly.  Keep its
        # command/state buffers but use the local DLS implementation above.
        self._ik_controller = _StableDifferentialIKController(
            cfg=self.cfg.controller,
            num_envs=self.num_envs,
            device=self.device,
        )
        self._skip_ik_once = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._fallback_jacobian = torch.eye(6, self._num_joints, device=self.device)

        if self.cfg.base_name:
            base_ids, base_names = self._asset.find_bodies(self.cfg.base_name)
            if len(base_ids) != 1:
                raise ValueError(
                    f"Expected one match for the base name: {self.cfg.base_name}. Found {len(base_ids)}: {base_names}."
                )
            self._base_idx = base_ids[0]
        else:
            self._base_idx = None

    @property
    def jacobian_b(self) -> torch.Tensor:
        # Do not transform the data-layer Jacobian in place.  It is reused by
        # the articulation state and is stale for the first step after a reset.
        jacobian = self.jacobian_w.clone()

        if self._base_idx is not None:
            base_quat_w = self._asset.data.body_quat_w.torch[:, self._base_idx]
        else:
            base_quat_w = self._asset.data.root_quat_w.torch

        base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(base_quat_w))
        jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])
        return jacobian

    def _compute_frame_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        ee_pos_w = self._asset.data.body_pos_w.torch[:, self._body_idx]
        ee_quat_w = self._asset.data.body_quat_w.torch[:, self._body_idx]

        if self._base_idx is not None:
            base_pos_w = self._asset.data.body_pos_w.torch[:, self._base_idx]
            base_quat_w = self._asset.data.body_quat_w.torch[:, self._base_idx]
        else:
            base_pos_w = self._asset.data.root_pos_w.torch
            base_quat_w = self._asset.data.root_quat_w.torch

        ee_pose_b, ee_quat_b = math_utils.subtract_frame_transforms(
            base_pos_w, base_quat_w, ee_pos_w, ee_quat_w
        )

        if self.cfg.body_offset is not None:
            ee_pose_b, ee_quat_b = math_utils.combine_frame_transforms(
                ee_pose_b, ee_quat_b, self._offset_pos, self._offset_rot
            )

        return ee_pose_b, ee_quat_b

    def _compute_frame_jacobian(self) -> torch.Tensor:
        self._jacobian_b[:] = self.jacobian_b
        if self.cfg.body_offset is not None:
            self._jacobian_b[:, :3, :] += torch.bmm(
                -math_utils.skew_symmetric_matrix(self._offset_pos),
                self._jacobian_b[:, 3:, :],
            )
            self._jacobian_b[:, 3:, :] = torch.bmm(
                math_utils.matrix_from_quat(self._offset_rot),
                self._jacobian_b[:, 3:, :],
            )
        return self._jacobian_b

    def apply_actions(self):
        """Apply IK targets only when the state and Jacobian are usable."""
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        joint_pos = self._asset.data.joint_pos.torch[:, self._joint_ids]
        jacobian = self._compute_frame_jacobian()

        quat_norm = torch.linalg.vector_norm(ee_quat_curr, dim=-1)
        jacobian_norm = torch.linalg.vector_norm(jacobian, dim=(-2, -1))
        usable = (
            torch.isfinite(ee_pos_curr).all(dim=-1)
            & torch.isfinite(ee_quat_curr).all(dim=-1)
            & torch.isfinite(joint_pos).all(dim=-1)
            & torch.isfinite(jacobian).all(dim=(-2, -1))
            & torch.isfinite(quat_norm)
            & torch.isfinite(jacobian_norm)
            & (quat_norm > 1.0e-6)
            & (jacobian_norm > 1.0e-6)
        )

        # IsaacLab's controller test explicitly skips the first post-reset
        # step: articulation Jacobians are not refreshed until one simulation
        # step has completed.  Substitute a harmless full-rank Jacobian for
        # skipped/invalid rows so batched IK remains safe for the other rows.
        safe_joint_pos = torch.nan_to_num(joint_pos, nan=0.0, posinf=0.0, neginf=0.0)
        safe_ee_pos = torch.nan_to_num(ee_pos_curr, nan=0.0, posinf=0.0, neginf=0.0)
        safe_ee_quat = torch.where(
            usable.unsqueeze(-1),
            ee_quat_curr,
            torch.tensor(
                (0.0, 0.0, 0.0, 1.0), device=self.device, dtype=ee_quat_curr.dtype
            ),
        )
        safe_jacobian = torch.where(
            usable[:, None, None],
            jacobian,
            self._fallback_jacobian.to(dtype=jacobian.dtype).expand_as(jacobian),
        )
        joint_pos_candidate = self._ik_controller.compute(
            safe_ee_pos,
            safe_ee_quat,
            safe_jacobian,
            safe_joint_pos,
        )

        solve_mask = (
            usable
            & ~self._skip_ik_once
            & torch.isfinite(joint_pos_candidate).all(dim=-1)
        )
        joint_pos_des = torch.where(
            solve_mask.unsqueeze(-1), joint_pos_candidate, safe_joint_pos
        )
        self._skip_ik_once.zero_()
        self._asset.actuators.target_command.set_position_index(
            value=joint_pos_des,
            joint_ids=self._joint_ids,
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            self._skip_ik_once.fill_(True)
        else:
            self._skip_ik_once[env_ids] = True


@configclass
class DifferentialInverseKinematicsActionCfg(__DifferentialInverseKinematicsActionCfg):
    class_type: type[ActionTerm] = DifferentialInverseKinematicsAction
    base_name: str = ""
