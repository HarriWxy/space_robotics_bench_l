"""Adapt an SRB Gymnasium environment to the FPO vectorized-env contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import gymnasium
import torch

from dyn.dyn_encoder import (
    DynamicsEncoder,
    DynamicsEncoderRuntime,
    DynamicsEncoderTrainer,
)


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
        self._dynamics_encoder: DynamicsEncoder | None = None
        self._dynamics_runtime: DynamicsEncoderRuntime | None = None
        self._dynamics_trainer: DynamicsEncoderTrainer | None = None
        self._dynamics_state_keys: tuple[str, ...] = ()
        self._dynamics_target_key: str | None = None
        self._dynamics_target_scale: float | None = None
        self._dynamics_update_interval = 1
        self._dynamics_step_count = 0
        self._dynamics_last_metrics: dict[str, float] | None = None

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

    def observation_tensor(
        self,
        keys: Sequence[str],
        *,
        observations: Mapping[str, Any] | None = None,
        name: str = "observation",
    ) -> torch.Tensor:
        """Flatten selected SRB observation categories from the current step."""

        if observations is None:
            if self._obs is None:
                raise RuntimeError("the wrapper has no current observation")
            observations = self._obs
        return self._concat_categories(observations, keys, name=name)

    def attach_dynamics_encoder(
        self,
        encoder: DynamicsEncoder,
        *,
        state_keys: Sequence[str] = ("proprio", "proprio_dyn"),
        trainer: DynamicsEncoderTrainer | None = None,
        update_interval: int = 1,
        physics_target_key: str | None = None,
        physics_target_scale: float | None = None,
    ) -> None:
        """Append an online dynamics latent to the actor observation.

        The wrapper records the transition from the previous observation to
        the current observation.  The auxiliary optimizer runs in
        ``env.step`` under ``torch.enable_grad`` so it remains compatible with
        FPO's no-gradient rollout context.  The encoder is not part of FPO's
        policy optimizer; its weights are registered on the policy by the
        FPO integration so the normal checkpoint contains the encoder state.

        ``physics_target_key`` is read from the raw SRB ``info`` mapping.  A
        scalar target is expanded over environments, which matches the current
        scene-global gravity implementation.
        """

        if not state_keys:
            raise ValueError("state_keys must contain at least one observation category")
        if update_interval <= 0:
            raise ValueError(f"update_interval must be positive, got {update_interval}")
        if physics_target_key is not None and encoder.physics_dim <= 0:
            raise ValueError(
                "physics_target_key requires DynamicsEncoderConfig.physics_dim > 0"
            )
        if physics_target_scale is not None and physics_target_scale <= 0.0:
            raise ValueError("physics_target_scale must be positive or None")
        if physics_target_key is None and physics_target_scale is not None:
            raise ValueError(
                "physics_target_scale is only valid with physics_target_key"
            )
        if self._obs is None:
            raise RuntimeError("attach_dynamics_encoder() requires an initialized environment")

        state_keys = tuple(state_keys)
        dynamics_state = self.observation_tensor(
            state_keys,
            name="dynamics_state",
        )
        if dynamics_state.shape[1] != encoder.state_dim:
            raise ValueError(
                "dynamics encoder state_dim does not match configured observation "
                f"categories {state_keys}: expected {dynamics_state.shape[1]}, "
                f"got {encoder.state_dim}"
            )
        if encoder.action_dim != self.num_actions:
            raise ValueError(
                "dynamics encoder action_dim does not match the environment: "
                f"expected {self.num_actions}, got {encoder.action_dim}"
            )

        encoder.to(device=self.device)
        self._dynamics_encoder = encoder
        self._dynamics_state_keys = state_keys
        self._dynamics_target_key = physics_target_key
        self._dynamics_target_scale = physics_target_scale
        self._dynamics_update_interval = update_interval
        self._dynamics_step_count = 0
        self._dynamics_last_metrics = None
        self._validated = False
        self._dynamics_runtime = DynamicsEncoderRuntime(
            encoder,
            num_envs=self.num_envs,
            device=self.device,
        )
        self._dynamics_runtime.reset(dynamics_state)
        self._dynamics_trainer = trainer

    @property
    def dynamics_encoder(self) -> DynamicsEncoder | None:
        """Return the optional encoder for checkpoint registration."""

        return self._dynamics_encoder

    def _dynamics_state(self, observations: Mapping[str, Any]) -> torch.Tensor:
        if self._dynamics_runtime is None:
            raise RuntimeError("dynamics encoder is not attached")
        return self.observation_tensor(
            self._dynamics_state_keys,
            observations=observations,
            name="dynamics_state",
        )

    def _physics_target(self, info: Any) -> torch.Tensor | None:
        if self._dynamics_target_key is None:
            return None
        if not isinstance(info, Mapping) or self._dynamics_target_key not in info:
            raise KeyError(
                "configured dynamics physics_target_key is missing from SRB info: "
                f"{self._dynamics_target_key!r}"
            )
        value = torch.as_tensor(
            info[self._dynamics_target_key],
            device=self.device,
            dtype=torch.float32,
        )
        if value.numel() == 1:
            value = value.expand(self.num_envs)
        if value.numel() % self.num_envs != 0:
            raise ValueError(
                "dynamics physics target cannot be batched over environments: "
                f"shape={tuple(value.shape)}, num_envs={self.num_envs}"
            )
        value = value.reshape(self.num_envs, -1)
        if self._dynamics_target_scale is not None:
            value = value / self._dynamics_target_scale
        return value

    def _append_dynamics_latent(
        self,
        actor: torch.Tensor,
    ) -> torch.Tensor:
        if self._dynamics_runtime is None:
            return actor
        latent = self._dynamics_runtime.begin_step()
        return torch.cat((actor, latent), dim=-1)

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
                    for key in ("proprio", "proprio_dyn", "command")
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
            actor = self._concat_categories(observations, self.actor_keys, name="actor")

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

        actor = self._append_dynamics_latent(actor)

        if self.validate and not self._validated:
            tensors = [actor] + ([critic] if critic is not None else [])
            if any(not torch.isfinite(tensor).all().item() for tensor in tensors):
                raise FloatingPointError(
                    "Non-finite observation reached the FPO adapter"
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
            dict(observation_extras) if isinstance(observation_extras, Mapping) else {}
        )
        if critic_observations is None:
            observation_extras.pop("critic", None)
        else:
            observation_extras["critic"] = critic_observations
        extras["observations"] = observation_extras

        # The upstream FPO runner consumes task logging through ``episode`` or
        # ``log`` rather than the raw SRB info mapping.  Keep the values on the
        # simulator device here; the runner can aggregate them once per
        # collection instead of forcing one host transfer per environment.
        log_container = (
            "episode" if isinstance(extras.get("episode"), Mapping) else "log"
        )
        logs = dict(extras.get(log_container, {}))
        for key, value in extras.items():
            if (
                isinstance(key, str)
                and key.startswith("metrics/")
                and key != "metrics/"
            ):
                metric = torch.as_tensor(
                    value, device=self.device, dtype=torch.float32
                ).detach()
                logs[f"rollout/{key}"] = metric
        reward_terms = extras.get("reward_terms")
        if isinstance(reward_terms, Mapping):
            for name, value in reward_terms.items():
                metric = torch.as_tensor(
                    value, device=self.device, dtype=torch.float32
                ).detach()
                logs[f"rollout/reward_terms/{name}"] = metric
        if logs:
            extras[log_container] = logs
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
        if self._dynamics_runtime is not None:
            self._dynamics_runtime.reset(self._dynamics_state(observations))
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
        dones = terminated | truncated
        if self._dynamics_runtime is not None:
            self._dynamics_runtime.observe(
                actions,
                self._dynamics_state(observations),
                dones,
                physics_target=self._physics_target(info),
            )
            self._dynamics_step_count += 1
            if (
                self._dynamics_trainer is not None
                and self._dynamics_encoder is not None
                and self._dynamics_encoder.training
                and self._dynamics_step_count % self._dynamics_update_interval == 0
            ):
                dynamics_batch = self._dynamics_runtime.drain_batch()
                if dynamics_batch is not None:
                    # FPO collects rollouts inside torch.no_grad().
                    with torch.enable_grad():
                        self._dynamics_last_metrics = self._dynamics_trainer.update(
                            dynamics_batch
                        )
            else:
                # A frozen/pretrained encoder, and FPO's post-training eval,
                # still need history but must not retain auxiliary samples.
                self._dynamics_runtime.drain_batch()

        self._obs = observations
        actor, critic = self._encode_observations(observations)

        reward = self._vector(
            reward,
            num_envs=self.num_envs,
            device=self.device,
            dtype=torch.float32,
            name="reward",
        )
        dones = dones.to(dtype=torch.long)

        extras = self._make_extras(info, critic)
        if self._dynamics_last_metrics is not None:
            log_container = (
                "episode" if isinstance(extras.get("episode"), Mapping) else "log"
            )
            logs = dict(extras.get(log_container, {}))
            logs.update(
                {
                    f"train/dynamics/{name}": value
                    for name, value in self._dynamics_last_metrics.items()
                }
            )
            extras[log_container] = logs
            self._dynamics_last_metrics = None
        finite_horizon = getattr(self.cfg, "is_finite_horizon", None)
        if finite_horizon is not True:
            extras["time_outs"] = truncated

        return actor, reward, dones, extras

    def close(self):
        return self.env.close()
