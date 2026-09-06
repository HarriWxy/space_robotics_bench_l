"""Adapt SRB observations and actions to the PyTorch ExO-PPO flow contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import gymnasium
import torch


class SrbExoPpoEnvWrapper:
    """Keep SRB's vector environment on-device and expose flat actor/critic data."""

    def __init__(
        self,
        env: Any,
        *,
        actor_keys: Sequence[str] | None = None,
        critic_keys: Sequence[str] | None = None,
        validate: bool = True,
    ) -> None:
        self.env = env
        self.unwrapped = env.unwrapped
        self.device = torch.device(
            getattr(self.unwrapped, "device", getattr(env, "device", "cpu"))
        )
        self.num_envs = int(self.unwrapped.num_envs)
        self.actor_keys = tuple(actor_keys) if actor_keys else None
        self.critic_keys = tuple(critic_keys) if critic_keys else None
        self.validate = bool(validate)
        self._validated = False

        action_space = getattr(self.unwrapped, "single_action_space", None)
        if action_space is None:
            action_space = getattr(self.unwrapped, "action_space", None)
        if not isinstance(action_space, gymnasium.spaces.Box):
            raise TypeError("ExO-PPO flow requires a continuous Box action space")
        if len(action_space.shape) != 1:
            raise TypeError(
                "ExO-PPO flow requires a flat Box action space, got "
                f"shape={action_space.shape}"
            )
        action_low = torch.as_tensor(
            action_space.low, dtype=torch.float32, device=self.device
        ).reshape(1, -1)
        action_high = torch.as_tensor(
            action_space.high, dtype=torch.float32, device=self.device
        ).reshape(1, -1)
        if (
            not torch.isfinite(action_low).all()
            or not torch.isfinite(action_high).all()
        ):
            raise ValueError("ExO-PPO flow requires finite action bounds")
        if not torch.all(action_high > action_low):
            raise ValueError("every action upper bound must exceed its lower bound")
        self.action_low = action_low
        self.action_high = action_high
        self.action_center = (action_high + action_low) * 0.5
        self.action_scale = (action_high - action_low) * 0.5
        self.num_actions = int(action_low.shape[1])

        action_manager = getattr(self.unwrapped, "action_manager", None)
        if action_manager is not None:
            managed_actions = int(action_manager.total_action_dim)
            if managed_actions != self.num_actions:
                raise ValueError(
                    f"SRB action manager exposes {managed_actions} actions but the "
                    f"Box space exposes {self.num_actions}"
                )

        cfg = getattr(self.unwrapped, "cfg", None)
        self.bootstrap_truncated = getattr(cfg, "is_finite_horizon", None) is not True
        if self.bootstrap_truncated and hasattr(cfg, "compute_final_obs"):
            cfg.compute_final_obs = True

    @staticmethod
    def _as_batched_tensor(
        value: Any,
        *,
        num_envs: int,
        device: torch.device,
        name: str,
    ) -> torch.Tensor:
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        tensor = tensor.to(device=device, dtype=torch.float32)
        if tensor.numel() % num_envs != 0:
            raise ValueError(
                f"{name} has {tensor.numel()} values and cannot be batched over "
                f"{num_envs} environments: shape={tuple(tensor.shape)}"
            )
        return tensor.reshape(num_envs, -1)

    def _concat(
        self,
        observations: Mapping[str, Any],
        keys: Sequence[str],
        *,
        name: str,
    ) -> torch.Tensor:
        missing = [key for key in keys if key not in observations]
        if missing:
            available = ", ".join(sorted(observations))
            raise KeyError(
                f"ExO-PPO {name} observation keys {missing} are missing; "
                f"available keys: [{available}]"
            )
        return torch.cat(
            [
                self._as_batched_tensor(
                    observations[key],
                    num_envs=self.num_envs,
                    device=self.device,
                    name=f"{name}.{key}",
                )
                for key in keys
            ],
            dim=-1,
        )

    def encode_observations(
        self, observations: Mapping[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Flatten configured SRB observation categories without host copies."""

        if not isinstance(observations, Mapping):
            raise TypeError(
                "SRB ExO-PPO integration expects mapping observations, got "
                f"{type(observations).__name__}"
            )
        if self.actor_keys is not None:
            actor = self._concat(observations, self.actor_keys, name="actor")
        elif "policy" in observations:
            actor = self._as_batched_tensor(
                observations["policy"],
                num_envs=self.num_envs,
                device=self.device,
                name="actor.policy",
            )
        else:
            inferred_keys = tuple(
                key
                for key in ("proprio", "proprio_dyn", "command")
                if key in observations
            )
            if not inferred_keys:
                raise KeyError(
                    "Cannot infer ExO-PPO actor observations. Configure "
                    "agent.obs.actor_keys or expose observation['policy']."
                )
            actor = self._concat(observations, inferred_keys, name="actor")

        if self.critic_keys is None:
            critic = actor
        else:
            critic = self._concat(observations, self.critic_keys, name="critic")

        if self.validate and not self._validated:
            if not torch.isfinite(actor).all() or not torch.isfinite(critic).all():
                raise FloatingPointError(
                    "Non-finite observation reached the ExO-PPO adapter"
                )
            self._validated = True
        return actor, critic

    @staticmethod
    def _vector(
        value: Any,
        *,
        num_envs: int,
        device: torch.device,
        dtype: torch.dtype,
        name: str,
    ) -> torch.Tensor:
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        tensor = tensor.to(device=device, dtype=dtype).reshape(-1)
        if tensor.shape != (num_envs,):
            raise ValueError(
                f"ExO-PPO {name} must have shape [{num_envs}], "
                f"got {tuple(tensor.shape)}"
            )
        return tensor

    def action_from_pre_tanh(self, pre_tanh_action: torch.Tensor) -> torch.Tensor:
        """Apply the exact bounded transform assumed by the augmented ratio."""

        pre_tanh_action = torch.as_tensor(
            pre_tanh_action, dtype=torch.float32, device=self.device
        )
        expected_shape = (self.num_envs, self.num_actions)
        if pre_tanh_action.shape != expected_shape:
            raise ValueError(
                f"ExO-PPO action must have shape {expected_shape}, "
                f"got {tuple(pre_tanh_action.shape)}"
            )
        return self.action_center + self.action_scale * torch.tanh(pre_tanh_action)

    def reset(self) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        observations, info = self.env.reset()
        actor, critic = self.encode_observations(observations)
        return actor, critic, dict(info) if isinstance(info, Mapping) else {}

    def step(
        self, pre_tanh_action: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        env_action = self.action_from_pre_tanh(pre_tanh_action)
        observations, reward, terminated, truncated, info = self.env.step(env_action)
        actor, critic = self.encode_observations(observations)
        reward = self._vector(
            reward,
            num_envs=self.num_envs,
            device=self.device,
            dtype=torch.float32,
            name="reward",
        )
        terminated = self._vector(
            terminated,
            num_envs=self.num_envs,
            device=self.device,
            dtype=torch.bool,
            name="terminated",
        )
        truncated = self._vector(
            truncated,
            num_envs=self.num_envs,
            device=self.device,
            dtype=torch.bool,
            name="truncated",
        )
        extras = dict(info) if isinstance(info, Mapping) else {}
        return actor, critic, reward, terminated, truncated, extras

    def final_observations(
        self, extras: Mapping[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Encode Isaac Lab's pre-reset terminal observation when available."""

        final_obs = extras.get("final_obs")
        if not isinstance(final_obs, Mapping):
            return None
        return self.encode_observations(final_obs)

    @property
    def max_episode_length(self) -> int:
        return int(self.unwrapped.max_episode_length)

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.unwrapped.episode_length_buf

    def close(self):
        return self.env.close()
