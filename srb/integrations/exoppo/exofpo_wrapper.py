"""SRB adapter with FPO's direct, clipped flow-action contract."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from srb.integrations.exoppo.wrapper import SrbExoPpoEnvWrapper


class SrbExoFpoEnvWrapper(SrbExoPpoEnvWrapper):
    """Expose flat observations while sending flow actions directly to SRB.

    ``SrbExoPpoEnvWrapper`` maps an unconstrained policy output through ``tanh``.
    FPO instead integrates its flow directly in the bounded action space and
    clips the generated action before ``env.step``. ExO-FPO keeps that latter
    contract so the CFM loss is evaluated on exactly the action seen by SRB.
    """

    def __init__(
        self,
        env: Any,
        *,
        actor_keys: Sequence[str] | None = None,
        critic_keys: Sequence[str] | None = None,
        clip_actions: float | None = 1.0,
        validate: bool = True,
    ) -> None:
        self.clip_actions = clip_actions
        super().__init__(
            env,
            actor_keys=actor_keys,
            critic_keys=critic_keys,
            validate=validate,
        )

    def action_from_pre_tanh(self, action: torch.Tensor) -> torch.Tensor:
        """Clip a generated flow action to the configured SRB action bounds."""

        action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        expected_shape = (self.num_envs, self.num_actions)
        if tuple(action.shape) != expected_shape:
            raise ValueError(
                f"ExO-FPO action must have shape {expected_shape}, "
                f"got {tuple(action.shape)}"
            )
        if self.clip_actions is not None:
            clip = float(self.clip_actions)
            if clip <= 0.0:
                raise ValueError(f"clip_actions must be positive or None, got {clip}")
            action = action.clamp(-clip, clip)
        return torch.maximum(torch.minimum(action, self.action_high), self.action_low)
