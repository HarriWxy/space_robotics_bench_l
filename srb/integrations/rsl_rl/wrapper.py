"""Adapt SRB environments to Isaac Lab's RSL-RL vectorized contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch


class SrbRslRlVecEnvWrapper:
    """Validate SRB observations and delegate the RSL-RL contract to Isaac Lab.

    Isaac Lab owns the canonical ``RslRlVecEnvWrapper`` implementation. SRB
    keeps this thin adapter so the integration uses the same reset, timeout,
    action-clipping, and ``TensorDict`` semantics as native Isaac Lab tasks
    while checking the SRB observation-group contract at runtime.
    """

    def __init__(
        self,
        env: Any,
        *,
        clip_actions: float | None = None,
        obs_groups: Mapping[str, Sequence[str]] | None = None,
        validate: bool = True,
    ) -> None:
        try:
            from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        except ImportError as exc:
            raise ImportError(
                "The RSL-RL integration requires Isaac Lab's isaaclab_rl "
                "package and rsl-rl-lib. Install the SRB rsl_rl extra."
            ) from exc

        self._env = RslRlVecEnvWrapper(env, clip_actions=clip_actions)
        self.env = self._env
        self.obs_groups = {
            name: tuple(groups) for name, groups in (obs_groups or {}).items()
        }
        self.validate = validate
        self._validated_after_reset = False

        # RslRlVecEnvWrapper resets in its constructor because OnPolicyRunner
        # expects observations to be available immediately.
        self._validate_observations(
            self._env.get_observations(), context="initial reset"
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    @property
    def unwrapped(self) -> Any:
        return self._env.unwrapped

    def reset(self):
        observations, extras = self._env.reset()
        self._validate_observations(observations, context="reset")
        return observations, extras

    def get_observations(self):
        observations = self._env.get_observations()
        self._validate_observations(observations, context="get_observations")
        return observations

    def step(self, actions: torch.Tensor):
        result = self._env.step(actions)
        if not self._validated_after_reset:
            self._validate_observations(result[0], context="first step")
        return result

    def close(self):
        return self._env.close()

    def _validate_observations(self, observations: Any, *, context: str) -> None:
        if not self.validate or self._validated_after_reset:
            return

        for obs_set, groups in self.obs_groups.items():
            if not groups:
                raise ValueError(
                    f"RSL-RL observation set '{obs_set}' must contain at least one group."
                )
            for group in groups:
                if group not in observations:
                    available = ", ".join(str(key) for key in observations.keys())
                    raise KeyError(
                        f"RSL-RL observation group '{group}' from '{obs_set}' is "
                        f"missing during {context}. Available groups: {available}."
                    )
                value = observations[group]
                if not isinstance(value, torch.Tensor):
                    value = torch.as_tensor(value)
                if value.ndim != 2:
                    raise ValueError(
                        f"RSL-RL observation group '{group}' must be a 2-D tensor "
                        f"[num_envs, features], got shape {tuple(value.shape)}."
                    )
                if value.shape[0] != self._env.num_envs:
                    raise ValueError(
                        f"RSL-RL observation group '{group}' has batch size "
                        f"{value.shape[0]}, expected {self._env.num_envs}."
                    )
                if not torch.isfinite(value).all():
                    raise ValueError(
                        f"RSL-RL observation group '{group}' contains non-finite values "
                        f"during {context}."
                    )

        self._validated_after_reset = True

