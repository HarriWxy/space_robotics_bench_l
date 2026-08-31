"""Adapt an SRB Gymnasium environment to the FPO vectorized-env contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import gymnasium
import torch


class SrbFpoEnvWrapper:
    """Convert SRB step-return observations into FPO tensors.

    SRB DirectEnv exposes observation categories such as state, proprio and
    command. FPO expects one batched actor tensor and optionally one batched
    privileged critic tensor. The wrapper deliberately does not import
    isaaclab_fpo so that SRB's other algorithms remain importable when the
    optional FPO package has not been installed yet.
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
        self.env = env
        self._unwrapped = env.unwrapped
        self.unwrapped = self._unwrapped

        self.device = torch.device(
            getattr(self._unwrapped, "device", getattr(env, "device", "cpu"))
        )
        self.num_envs = int(self._unwrapped.num_envs)
        self.num_actions = self._resolve_num_actions()
        self.max_episode_length = self._unwrapped.max_episode_length
        self.cfg = getattr(self._unwrapped, "cfg", None)

        self.actor_keys = tuple(actor_keys) if actor_keys is not None else None
        self.critic_keys = tuple(critic_keys) if critic_keys else None
        self.clip_actions = clip_actions
        self.validate = validate
        self._validated = False
        self._obs: Mapping[str, Any] | None = None

        # FPO's runner reads the current observation immediately in __init__.
        self.reset()

    def _resolve_num_actions(self) -> int:
        action_manager = getattr(self._unwrapped, "action_manager", None)
        if action_manager is not None:
            return int(action_manager.total_action_dim)

        action_space = getattr(self._unwrapped, "single_action_space", None)
        if action_space is None:
            action_space = self._unwrapped.action_space
        return int(gymnasium.spaces.flatdim(action_space))

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self._unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        self._unwrapped.episode_length_buf = value

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return getattr(self.env, "observation_space", None)

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
                f"{name} has {tensor.numel()} values, which cannot be batched "
                f"for {num_envs} environments: shape={tuple(tensor.shape)}"
            )
        return tensor.reshape(num_envs, -1)

    def _concat_categories(
        self,
        observations: Mapping[str, Any],
        keys: Sequence[str],
        *,
        name: str,
    ) -> torch.Tensor:
        missing = [key for key in keys if key not in observations]
        if missing:
            available = ", ".join(sorted(observations.keys()))
            raise KeyError(
                f"FPO {name} observation keys {missing} are missing from SRB "
                f"observation; available keys: [{available}]"
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

    def _encode_observations(
        self, observations: Mapping[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.actor_keys is None:
            if "policy" in observations:
                actor = self._as_batched_tensor(
                    observations["policy"],
                    num_envs=self.num_envs,
                    device=self.device,
                    name="actor.policy",
                )
            else:
                # This is the SRB DirectEnv default. The command is essential
                # for velocity tracking and is omitted automatically by tasks
                # that do not expose it.
                default_keys = tuple(
                    key
                    for key in ("state", "proprio", "command")
                    if key in observations
                )
                if not default_keys:
                    raise KeyError(
                        "Cannot infer FPO actor observations. Configure "
                        "agent.obs.actor_keys or expose an observation['policy'] tensor."
                    )
                actor = self._concat_categories(
                    observations, default_keys, name="actor"
                )
        else:
            actor = self._concat_categories(
                observations, self.actor_keys, name="actor"
            )

        critic = None
        if self.critic_keys is not None:
            critic = self._concat_categories(
                observations, self.critic_keys, name="critic"
            )

        if actor.ndim != 2 or actor.shape[0] != self.num_envs:
            raise ValueError(
                f"FPO actor observation must have shape [N, D], got {tuple(actor.shape)}"
            )
        if critic is not None and (
            critic.ndim != 2 or critic.shape[0] != self.num_envs
        ):
            raise ValueError(
                f"FPO critic observation must have shape [N, C], got {tuple(critic.shape)}"
            )

        if self.validate and not self._validated:
            tensors = [actor] + ([critic] if critic is not None else [])
            if any(not torch.isfinite(tensor).all().item() for tensor in tensors):
                raise FloatingPointError("Non-finite observation reached the FPO adapter")
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
                f"FPO {name} must have shape [{num_envs}], got {tuple(tensor.shape)}"
            )
        return tensor

    def _make_extras(
        self,
        info: Any,
        critic_observations: torch.Tensor | None,
    ) -> dict[str, Any]:
        extras = dict(info) if isinstance(info, Mapping) else {}
        observation_extras = extras.get("observations", {})
        observation_extras = (
            dict(observation_extras)
            if isinstance(observation_extras, Mapping)
            else {}
        )
        if critic_observations is None:
            observation_extras.pop("critic", None)
        else:
            observation_extras["critic"] = critic_observations
        extras["observations"] = observation_extras
        return extras

    def get_observations(self) -> tuple[torch.Tensor, dict[str, Any]]:
        if self._obs is None:
            raise RuntimeError("FPO adapter has no current observation; call reset()")
        actor, critic = self._encode_observations(self._obs)
        return actor, self._make_extras({}, critic)

    def reset(self) -> tuple[torch.Tensor, dict[str, Any]]:
        observations, info = self.env.reset()
        if not isinstance(observations, Mapping):
            raise TypeError(
                "SRB FPO integration expects a mapping observation from the environment, "
                f"got {type(observations).__name__}"
            )
        self._obs = observations
        actor, critic = self._encode_observations(observations)
        return actor, self._make_extras(info, critic)

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        actions = actions.to(device=self.device, dtype=torch.float32)
        expected_shape = (self.num_envs, self.num_actions)
        if tuple(actions.shape) != expected_shape:
            raise ValueError(
                f"FPO action must have shape {expected_shape}, got {tuple(actions.shape)}"
            )
        if self.clip_actions is not None:
            clip = float(self.clip_actions)
            if clip <= 0.0:
                raise ValueError(f"clip_actions must be positive or None, got {clip}")
            actions = actions.clamp(-clip, clip)

        observations, reward, terminated, truncated, info = self.env.step(actions)
        if not isinstance(observations, Mapping):
            raise TypeError(
                "SRB FPO integration expects a mapping observation from env.step(), "
                f"got {type(observations).__name__}"
            )
        self._obs = observations
        actor, critic = self._encode_observations(observations)

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
        dones = (terminated | truncated).to(dtype=torch.long)

        extras = self._make_extras(info, critic)
        finite_horizon = getattr(self.cfg, "is_finite_horizon", None)
        if finite_horizon is not True:
            extras["time_outs"] = truncated

        return actor, reward, dones, extras

    def close(self):
        return self.env.close()

