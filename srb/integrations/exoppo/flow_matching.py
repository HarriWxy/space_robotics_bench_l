"""FPO-style conditional flow matching for the ExO-FPO SRB adapter.

The policy follows the FPO action path: it predicts a velocity field in the
``u`` parameterization and integrates from Gaussian noise at ``t=1`` to an
action at ``t=0``. The trainer uses the same Monte-Carlo CFM loss for the
behavior/current policy ratio, while ExO's recent-policy smooth ratio and OFP
flow-consistency losses remain available around that ratio.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


def _batch_column(
    value: torch.Tensor | float,
    batch_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Convert a scalar or one value per batch item to ``[B, 1]``."""

    tensor = torch.as_tensor(value, device=device, dtype=dtype)
    if tensor.numel() == 1:
        return tensor.reshape(1, 1).expand(batch_size, 1)
    if tensor.numel() != batch_size:
        raise ValueError(
            f"time tensor has {tensor.numel()} values; expected {batch_size}"
        )
    return tensor.reshape(batch_size, 1)


def _activation(name: str) -> nn.Module:
    activations = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
    }
    try:
        return activations[str(name).lower()]()
    except KeyError as error:
        choices = ", ".join(sorted(activations))
        raise ValueError(
            f"unknown flow activation {name!r}; choose from {choices}"
        ) from error


class FlowMatchingPolicy(nn.Module):
    """FPO-compatible conditional velocity field and Euler sampler."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (256, 256, 256),
        *,
        activation: str = "elu",
        actor_scale: float = 1.0,
        mlp_output_scale: float = 1.0,
        timestep_embed_dim: int = 8,
        sampling_steps: int = 8,
        cfm_loss_reduction: str = "sqrt",
        action_perturb_std: float = 0.02,
        actor_final_layer_weight_scale: float | None = None,
    ) -> None:
        super().__init__()
        if obs_dim <= 0 or action_dim <= 0:
            raise ValueError("obs_dim and action_dim must be positive")
        if not hidden_sizes or any(int(size) <= 0 for size in hidden_sizes):
            raise ValueError("hidden_sizes must contain positive integers")
        if timestep_embed_dim <= 0 or timestep_embed_dim % 2:
            raise ValueError("timestep_embed_dim must be a positive even integer")
        if actor_scale <= 0.0:
            raise ValueError("actor_scale must be positive")
        if sampling_steps <= 0:
            raise ValueError("sampling_steps must be positive")
        if cfm_loss_reduction not in {"mean", "sum", "sqrt"}:
            raise ValueError("cfm_loss_reduction must be 'mean', 'sum', or 'sqrt'")
        if action_perturb_std < 0.0:
            raise ValueError("action_perturb_std cannot be negative")

        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.num_actions = self.action_dim
        self.hidden_sizes = tuple(int(size) for size in hidden_sizes)
        self.actor_scale = float(actor_scale)
        self.mlp_output_scale = float(mlp_output_scale)
        self.timestep_embed_dim = int(timestep_embed_dim)
        self.sampling_steps = int(sampling_steps)
        self.cfm_loss_reduction = str(cfm_loss_reduction)
        self.action_perturb_std = float(action_perturb_std)
        self.activation_name = str(activation).lower()

        input_dim = self.obs_dim + self.timestep_embed_dim + self.action_dim
        layers: list[nn.Module] = [nn.Linear(input_dim, self.hidden_sizes[0])]
        layers.append(_activation(activation))
        for index, size in enumerate(self.hidden_sizes):
            if index == len(self.hidden_sizes) - 1:
                layers.append(nn.Linear(size, self.action_dim))
            else:
                layers.append(nn.Linear(size, self.hidden_sizes[index + 1]))
                layers.append(_activation(activation))
        self.actor = nn.Sequential(*layers)

        if (
            actor_final_layer_weight_scale is not None
            and actor_final_layer_weight_scale != 1.0
        ):
            final_layer = self.actor[-1]
            if not isinstance(
                final_layer, nn.Linear
            ):  # pragma: no cover - construction invariant
                raise TypeError("flow actor final layer must be Linear")
            with torch.no_grad():
                final_layer.weight.mul_(float(actor_final_layer_weight_scale))
                if final_layer.bias is not None:
                    final_layer.bias.mul_(float(actor_final_layer_weight_scale))

    def _embed_timestep(self, timestep: torch.Tensor) -> torch.Tensor:
        if timestep.shape[-1] != 1:
            raise ValueError(
                f"timestep must end in one feature, got {tuple(timestep.shape)}"
            )
        frequencies = 2 ** torch.arange(
            self.timestep_embed_dim // 2,
            device=timestep.device,
            dtype=timestep.dtype,
        )
        scaled = timestep * frequencies
        return torch.cat([torch.cos(scaled), torch.sin(scaled)], dim=-1)

    def _actor_velocity(
        self,
        observations: torch.Tensor,
        state: torch.Tensor,
        timestep: torch.Tensor | float,
    ) -> torch.Tensor:
        observations = torch.as_tensor(
            observations, dtype=torch.float32, device=state.device
        )
        state = torch.as_tensor(state, dtype=torch.float32, device=state.device)
        if observations.ndim == state.ndim - 1:
            observations = observations.unsqueeze(-2).expand(
                *state.shape[:-1], observations.shape[-1]
            )
        if observations.shape[:-1] != state.shape[:-1]:
            raise ValueError(
                "observations and flow state have incompatible leading shapes: "
                f"{tuple(observations.shape)} vs {tuple(state.shape)}"
            )

        timestep = torch.as_tensor(timestep, dtype=state.dtype, device=state.device)
        if timestep.numel() == 1:
            timestep = timestep.reshape((1,) * (state.ndim - 1) + (1,)).expand(
                *state.shape[:-1], 1
            )
        elif timestep.shape[-1] != 1:
            timestep = timestep.unsqueeze(-1)
        if timestep.shape[:-1] != state.shape[:-1]:
            timestep = torch.broadcast_to(timestep, (*state.shape[:-1], 1))

        embedded = self._embed_timestep(timestep)
        output = self.actor(torch.cat([observations, embedded, state], dim=-1))
        return self.mlp_output_scale * output

    def cfm_velocity(
        self,
        observations: torch.Tensor,
        state: torch.Tensor,
        timestep: torch.Tensor | float,
    ) -> torch.Tensor:
        """Return FPO's ``u`` velocity in its ``t=1`` noise convention."""

        return self._actor_velocity(observations, state, timestep)

    def velocity(
        self,
        observations: torch.Tensor,
        state: torch.Tensor,
        start_time: torch.Tensor | float,
        end_time: torch.Tensor | float,
        condition_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return velocity in ExO/OFP's ``t=0`` noise to ``t=1`` action frame.

        FPO parameterizes the same straight path in the reverse time direction.
        Negating its velocity makes the optional ExO OFP consistency loss use
        the same physical flow as the FPO CFM objective.
        """

        del end_time, condition_mask
        state = torch.as_tensor(state, dtype=torch.float32)
        start = _batch_column(
            start_time,
            state.shape[0],
            device=state.device,
            dtype=state.dtype,
        )
        return -self.cfm_velocity(observations, state, 1.0 - start)

    def _integrate(
        self, observations: torch.Tensor, initial_state: torch.Tensor
    ) -> torch.Tensor:
        observations = torch.as_tensor(observations, dtype=torch.float32)
        state = torch.as_tensor(
            initial_state, dtype=torch.float32, device=observations.device
        )
        time_path = torch.linspace(
            1.0,
            0.0,
            self.sampling_steps + 1,
            dtype=state.dtype,
            device=state.device,
        )
        for index in range(self.sampling_steps):
            timestep = time_path[index]
            delta = time_path[index + 1] - timestep
            state = state + self.cfm_velocity(observations, state, timestep) * delta
        return state

    def one_step_mean(
        self, observations: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """Generate a flow action from the supplied initial noise."""

        return self.actor_scale * self._integrate(observations, noise)

    def generate(
        self,
        observations: torch.Tensor,
        *,
        deterministic: bool,
        noise: torch.Tensor | None = None,
        perturb: bool = True,
    ) -> torch.Tensor:
        observations = torch.as_tensor(observations, dtype=torch.float32)
        if noise is None:
            if deterministic:
                noise = torch.zeros(
                    (observations.shape[0], self.action_dim),
                    dtype=observations.dtype,
                    device=observations.device,
                )
            else:
                noise = torch.randn(
                    (observations.shape[0], self.action_dim),
                    dtype=observations.dtype,
                    device=observations.device,
                )
        action = self.one_step_mean(observations, noise)
        if not deterministic and perturb and self.action_perturb_std > 0.0:
            action = action + self.action_perturb_std * torch.randn_like(action)
        return action

    def get_cfm_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        eps: torch.Tensor,
        timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute FPO's per-noise/per-time conditional flow-matching loss."""

        observations = torch.as_tensor(observations, dtype=torch.float32)
        actions = torch.as_tensor(
            actions, dtype=torch.float32, device=observations.device
        )
        eps = torch.as_tensor(eps, dtype=torch.float32, device=observations.device)
        timestep = torch.as_tensor(
            timestep, dtype=torch.float32, device=observations.device
        )
        if observations.ndim != 2 or actions.ndim != 2:
            raise ValueError("observations and actions must be two-dimensional")
        if actions.shape[0] != observations.shape[0]:
            raise ValueError("observations and actions must have the same batch size")
        if (
            eps.ndim != 3
            or eps.shape[0] != actions.shape[0]
            or eps.shape[2] != self.action_dim
        ):
            raise ValueError("eps must have shape [batch, samples, action_dim]")
        if timestep.shape != (*eps.shape[:2], 1):
            raise ValueError(
                f"timestep must have shape {(eps.shape[0], eps.shape[1], 1)}, "
                f"got {tuple(timestep.shape)}"
            )

        scaled_actions = actions / self.actor_scale
        state = timestep * eps + (1.0 - timestep) * scaled_actions[:, None, :]
        expanded_observations = observations[:, None, :].expand(
            observations.shape[0], eps.shape[1], observations.shape[1]
        )
        velocity_prediction = self.cfm_velocity(expanded_observations, state, timestep)
        x0_prediction = state - timestep * velocity_prediction
        x1_prediction = x0_prediction + velocity_prediction
        target_velocity = eps - scaled_actions[:, None, :]
        squared_error = (velocity_prediction - target_velocity).square()
        if self.cfm_loss_reduction == "mean":
            loss = squared_error.mean(dim=-1)
        elif self.cfm_loss_reduction == "sum":
            loss = squared_error.sum(dim=-1)
        else:
            loss = squared_error.sum(dim=-1) / math.sqrt(self.action_dim)
        return loss, x1_prediction, x0_prediction


class ValueNetwork(nn.Module):
    """Value MLP for the asymmetric critic observation contract."""

    def __init__(self, obs_dim: int, hidden_sizes: Sequence[int]) -> None:
        super().__init__()
        if obs_dim <= 0 or not hidden_sizes:
            raise ValueError("obs_dim and hidden_sizes must be positive")
        layers: list[nn.Module] = []
        input_dim = int(obs_dim)
        for size in hidden_sizes:
            layers.extend((nn.Linear(input_dim, int(size)), nn.SiLU()))
            input_dim = int(size)
        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(torch.as_tensor(observations, dtype=torch.float32)).squeeze(
            -1
        )


@dataclass
class FlowMatchingSample:
    """One behavior action and the CFM variables needed to replay its ratio."""

    action: torch.Tensor
    cfm_loss_eps: torch.Tensor
    cfm_loss_t: torch.Tensor
    initial_cfm_loss: torch.Tensor


@dataclass
class FlowMatchingRollout:
    actor_observations: torch.Tensor
    critic_observations: torch.Tensor
    actions: torch.Tensor
    cfm_loss_eps: torch.Tensor
    cfm_loss_t: torch.Tensor
    initial_cfm_loss: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor

    def __post_init__(self) -> None:
        sample_count = self.actor_observations.shape[0]
        for name, value in vars(self).items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if value.shape[0] != sample_count:
                raise ValueError(
                    f"{name} has {value.shape[0]} samples; expected {sample_count}"
                )

    def __len__(self) -> int:
        return int(self.actor_observations.shape[0])

    def detached(self) -> FlowMatchingRollout:
        return FlowMatchingRollout(
            **{name: value.detach() for name, value in vars(self).items()}
        )


def flatten_flow_matching_rollout(
    actor_observations: torch.Tensor,
    critic_observations: torch.Tensor,
    actions: torch.Tensor,
    cfm_loss_eps: torch.Tensor,
    cfm_loss_t: torch.Tensor,
    initial_cfm_loss: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
) -> FlowMatchingRollout:
    """Flatten ``[time, environments, ...]`` into replay samples."""

    if actor_observations.ndim < 3:
        raise ValueError("actor_observations must have shape [time, envs, ...]")
    time_steps, num_envs = actor_observations.shape[:2]
    values = {
        "actor_observations": actor_observations,
        "critic_observations": critic_observations,
        "actions": actions,
        "cfm_loss_eps": cfm_loss_eps,
        "cfm_loss_t": cfm_loss_t,
        "initial_cfm_loss": initial_cfm_loss,
        "advantages": advantages,
        "returns": returns,
    }
    for name, value in values.items():
        if value.shape[:2] != (time_steps, num_envs):
            raise ValueError(
                f"{name} has leading shape {tuple(value.shape[:2])}; "
                f"expected {(time_steps, num_envs)}"
            )
    count = time_steps * num_envs
    return FlowMatchingRollout(
        **{
            name: value.reshape(count, *value.shape[2:])
            for name, value in values.items()
        }
    )


class FlowMatchingReplayWindow:
    """Bounded device-resident replay for CFM variables."""

    _FIELDS = tuple(FlowMatchingRollout.__dataclass_fields__)

    def __init__(self, max_rollouts: int) -> None:
        if max_rollouts < 1:
            raise ValueError("max_rollouts must be positive")
        self._rollouts: deque[FlowMatchingRollout] = deque(maxlen=max_rollouts)

    @property
    def rollout_count(self) -> int:
        return len(self._rollouts)

    def __len__(self) -> int:
        return sum(len(rollout) for rollout in self._rollouts)

    def append(self, rollout: FlowMatchingRollout) -> None:
        self._rollouts.append(rollout.detached())

    def tensors(self) -> dict[str, torch.Tensor]:
        if not self._rollouts:
            raise RuntimeError("cannot concatenate an empty replay window")
        return {
            field: torch.cat(
                [getattr(rollout, field) for rollout in self._rollouts], dim=0
            )
            for field in self._FIELDS
        }


def _clamp_ste(
    value: torch.Tensor, *, minimum: float | None = None, maximum: float | None = None
) -> torch.Tensor:
    clamped = value.clamp(min=minimum, max=maximum)
    return value + (clamped - value).detach()


def _smooth_exo_ratio(
    ratio: torch.Tensor,
    center: torch.Tensor,
    clip_radius: float,
    beta: float,
) -> torch.Tensor:
    if clip_radius <= 0.0 or beta <= 0.0:
        raise ValueError("clip_radius and beta must be positive")
    radius = torch.as_tensor(clip_radius, dtype=ratio.dtype, device=ratio.device)
    beta_tensor = torch.as_tensor(beta, dtype=ratio.dtype, device=ratio.device)
    upper = center.detach() + radius
    lower = torch.maximum(center.detach() - radius, torch.zeros_like(center))
    safe_upper_ratio = torch.maximum(ratio, upper)
    safe_lower_ratio = torch.minimum(ratio, lower)
    upper_tail = (
        upper
        + 1.0 / beta_tensor
        - torch.exp(beta_tensor * (upper - safe_upper_ratio)) / beta_tensor
    )
    lower_tail = (
        lower
        - 1.0 / beta_tensor
        + torch.exp(beta_tensor * (safe_lower_ratio - lower)) / beta_tensor
    )
    transformed = torch.where(ratio > upper, upper_tail, ratio)
    transformed = torch.where(ratio < lower, lower_tail, transformed)
    return transformed.clamp_min(0.0)


def _contracting_factor(
    update_step: int,
    contraction_steps: int,
    minimum: float,
    device: torch.device,
) -> torch.Tensor:
    progress = min(max(update_step / float(contraction_steps), 0.0), 1.0)
    return torch.as_tensor(
        minimum + (1.0 - minimum) * (1.0 - progress) ** 2,
        dtype=torch.float32,
        device=device,
    )


def _ofp_losses(
    policy: FlowMatchingPolicy,
    ema_teacher: FlowMatchingPolicy,
    recent_policy: FlowMatchingPolicy,
    observations: torch.Tensor,
    *,
    update_step: int,
    contraction_steps: int,
    minimum_contraction: float,
) -> Mapping[str, torch.Tensor]:
    """Apply ExO's online flow/consistency distillation in FPO time coordinates."""

    batch_size = observations.shape[0]
    action_dim = policy.action_dim
    device = observations.device
    shape = (batch_size, action_dim)
    time_shape = (batch_size, 1)
    recent_noise = torch.randn(shape, dtype=observations.dtype, device=device)
    with torch.no_grad():
        target_action = recent_policy.one_step_mean(observations, recent_noise)

    flow_time = torch.rand(time_shape, dtype=observations.dtype, device=device)
    flow_state = (1.0 - flow_time) * recent_noise + flow_time * target_action
    flow_prediction = policy.velocity(observations, flow_state, flow_time, flow_time)
    flow_loss = (
        (flow_prediction - (target_action - recent_noise)).square().mean(dim=-1).mean()
    )

    minimum_interval = 1e-3
    t = torch.rand(time_shape, dtype=observations.dtype, device=device) * (
        1.0 - minimum_interval
    )
    remaining = 1.0 - t - minimum_interval
    r = t + minimum_interval + remaining * torch.rand_like(t)
    contraction = _contracting_factor(
        update_step, contraction_steps, minimum_contraction, device
    )
    m = t + (r - t) * contraction * torch.rand_like(t)
    state_t = (1.0 - t) * recent_noise + t * target_action
    state_m = (1.0 - m) * recent_noise + m * target_action
    with torch.no_grad():
        teacher_velocity = ema_teacher.velocity(observations, state_m, m, r)
    predicted_state_r = state_m + (r - m) * teacher_velocity
    consistency_target = ((predicted_state_r - state_t) / (r - t)).detach()
    consistency_prediction = policy.velocity(observations, state_t, t, r)
    consistency_loss = (
        (consistency_prediction - consistency_target).square().mean(dim=-1).mean()
    )

    zero = torch.zeros((), dtype=observations.dtype, device=device)
    return {
        "flow_loss": flow_loss,
        "consistency_loss": consistency_loss,
        "guidance_loss": zero,
        "contraction": contraction,
        "target_action_rms": target_action.square().mean().sqrt(),
    }


def _copy_policy(
    source: FlowMatchingPolicy, *, trainable: bool = False
) -> FlowMatchingPolicy:
    copied = FlowMatchingPolicy(
        source.obs_dim,
        source.action_dim,
        source.hidden_sizes,
        activation=source.activation_name,
        actor_scale=source.actor_scale,
        mlp_output_scale=source.mlp_output_scale,
        timestep_embed_dim=source.timestep_embed_dim,
        sampling_steps=source.sampling_steps,
        cfm_loss_reduction=source.cfm_loss_reduction,
        action_perturb_std=source.action_perturb_std,
    ).to(next(source.parameters()).device)
    copied.load_state_dict(source.state_dict())
    copied.train(mode=trainable)
    for parameter in copied.parameters():
        parameter.requires_grad_(trainable)
    return copied


def _update_ema(
    target: FlowMatchingPolicy, source: FlowMatchingPolicy, decay: float
) -> None:
    with torch.no_grad():
        for target_parameter, source_parameter in zip(
            target.parameters(), source.parameters()
        ):
            target_parameter.mul_(decay).add_(source_parameter, alpha=1.0 - decay)
        for target_buffer, source_buffer in zip(target.buffers(), source.buffers()):
            target_buffer.copy_(decay * target_buffer + (1.0 - decay) * source_buffer)


class ExoFpoTrainer:
    """ExO recent-policy optimization with an FPO CFM policy."""

    algorithm_name = "ExO-FPO"

    def __init__(
        self,
        config: Any,
        obs_dim: int,
        action_dim: int,
        critic_obs_dim: int,
        *,
        device: torch.device,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
    ) -> None:
        self.config = config
        self.device = device
        self.policy = FlowMatchingPolicy(
            obs_dim,
            action_dim,
            config.hidden_sizes,
            activation=config.activation,
            actor_scale=config.actor_scale,
            mlp_output_scale=config.mlp_output_scale,
            timestep_embed_dim=config.timestep_embed_dim,
            sampling_steps=config.sampling_steps,
            cfm_loss_reduction=config.cfm_loss_reduction,
            action_perturb_std=config.action_perturb_std,
            actor_final_layer_weight_scale=config.actor_final_layer_weight_scale,
        ).to(device)
        self.recent_policy = _copy_policy(self.policy)
        self.ema_teacher = _copy_policy(self.policy)
        self.critic_obs_dim = int(critic_obs_dim)
        self.value = ValueNetwork(self.critic_obs_dim, config.hidden_sizes).to(device)
        self.action_low = action_low.to(device=device, dtype=torch.float32)
        self.action_high = action_high.to(device=device, dtype=torch.float32)
        self.actor_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=config.actor_learning_rate, eps=1e-5
        )
        self.critic_optimizer = torch.optim.Adam(
            self.value.parameters(), lr=config.critic_learning_rate, eps=1e-5
        )
        self.update_step = 0

    def clip_action(self, action: torch.Tensor) -> torch.Tensor:
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        if self.config.clip_actions is not None:
            action = action.clamp(
                -float(self.config.clip_actions), float(self.config.clip_actions)
            )
        return torch.maximum(torch.minimum(action, self.action_high), self.action_low)

    def _sample_cfm_targets(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        eps = torch.randn(
            (
                actions.shape[0],
                self.config.n_samples_per_action,
                actions.shape[1],
            ),
            dtype=actions.dtype,
            device=actions.device,
        )
        uniform = torch.rand(
            (actions.shape[0], self.config.n_samples_per_action, 1),
            dtype=actions.dtype,
            device=actions.device,
        )
        beta = float(self.config.cfm_loss_t_inverse_cdf_beta)
        timestep = 0.005 + 0.99 * (1.0 - (1.0 - uniform) ** (1.0 / beta))
        return eps, timestep

    @torch.no_grad()
    def sample_rollout(self, observations: torch.Tensor) -> FlowMatchingSample:
        raw_action = self.policy.generate(observations, deterministic=False)
        action = self.clip_action(raw_action)
        eps, timestep = self._sample_cfm_targets(action)
        initial_loss, _, _ = self.policy.get_cfm_loss(
            observations, action, eps, timestep
        )
        return FlowMatchingSample(action, eps, timestep, initial_loss.detach())

    @torch.no_grad()
    def sample_evaluation(
        self, observations: torch.Tensor, *, stochastic: bool
    ) -> torch.Tensor:
        raw_action = self.policy.generate(observations, deterministic=not stochastic)
        return self.clip_action(raw_action)

    def snapshot_recent_policy(self) -> None:
        self.recent_policy.load_state_dict(self.policy.state_dict())
        self.recent_policy.eval()

    def _ema_decay(self) -> float:
        progress = self.update_step / float(self.config.ema_ramp_steps)
        progress = progress / (1.0 + progress)
        return min(
            self.config.ema_initial_decay
            + progress * (self.config.ema_max_decay - self.config.ema_initial_decay),
            self.config.ema_max_decay,
        )

    def _clamp_loss(self, loss: torch.Tensor) -> torch.Tensor:
        if self.config.cfm_loss_clamp > 0.0:
            return loss.clamp(max=self.config.cfm_loss_clamp)
        return loss

    def actor_train_step(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        cfm_loss_eps: torch.Tensor,
        cfm_loss_t: torch.Tensor,
        initial_cfm_loss: torch.Tensor,
        advantages: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        self.policy.train()
        self.actor_optimizer.zero_grad(set_to_none=True)
        current_loss, x1_prediction, _ = self.policy.get_cfm_loss(
            observations, actions, cfm_loss_eps, cfm_loss_t
        )
        with torch.no_grad():
            recent_loss, _, _ = self.recent_policy.get_cfm_loss(
                observations, actions, cfm_loss_eps, cfm_loss_t
            )

        old_raw = initial_cfm_loss.detach()
        current_raw = current_loss
        old_loss = self._clamp_loss(old_raw)
        current_ratio_loss = self._clamp_loss(current_raw)
        if self.config.cfm_loss_clamp_negative_advantages:
            current_ratio_loss = torch.where(
                advantages.reshape(-1, 1) < 0.0,
                current_ratio_loss.clamp(
                    max=self.config.cfm_loss_clamp_negative_advantages_max
                ),
                current_ratio_loss,
            )

        raw_log_ratio = old_loss - current_ratio_loss
        log_ratio = _clamp_ste(
            raw_log_ratio,
            minimum=-self.config.max_log_ratio,
            maximum=self.config.cfm_diff_clamp_max,
        )
        ratio = log_ratio.exp()
        recent_log_ratio = (recent_loss - old_loss).clamp(
            -self.config.max_log_ratio, self.config.max_log_ratio
        )
        recent_ratio = recent_log_ratio.exp().detach()
        exo_ratio = _smooth_exo_ratio(
            ratio,
            recent_ratio,
            self.config.exo_clip_radius,
            self.config.exo_beta,
        )

        advantage = advantages.reshape(-1, 1)
        if self.config.advantage_clamp is not None:
            positive, negative = self.config.advantage_clamp
            advantage = advantage.clamp(-negative, positive)
        direct_surrogate = ratio * advantage
        exo_surrogate = exo_ratio * advantage
        policy_gradient_loss = -torch.minimum(direct_surrogate, exo_surrogate).mean()

        if self.config.ofp_coefficient > 0.0:
            ofp = _ofp_losses(
                self.policy,
                self.ema_teacher,
                self.recent_policy,
                observations,
                update_step=self.update_step,
                contraction_steps=self.config.contraction_steps,
                minimum_contraction=self.config.minimum_contraction,
            )
            self_distill_loss = self.config.ofp_coefficient * (
                self.config.flow_mix * ofp["flow_loss"]
                + self.config.consistency_mix * ofp["consistency_loss"]
            )
        else:
            zero = torch.zeros((), dtype=observations.dtype, device=self.device)
            ofp = {
                "flow_loss": zero,
                "consistency_loss": zero,
                "guidance_loss": zero,
                "contraction": zero,
                "target_action_rms": zero,
            }
            self_distill_loss = zero

        actor_loss = policy_gradient_loss + self_distill_loss
        if not torch.isfinite(actor_loss):
            raise FloatingPointError("non-finite ExO-FPO actor loss")
        actor_loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), self.config.max_grad_norm
        )
        self.actor_optimizer.step()
        ema_decay = self._ema_decay()
        _update_ema(self.ema_teacher, self.policy, ema_decay)
        self.update_step += 1

        return {
            "actor_loss": actor_loss.detach(),
            "policy_loss": policy_gradient_loss.detach(),
            "cfm_loss": current_raw.detach().mean(),
            "old_cfm_loss": old_raw.detach().mean(),
            "cfm_loss_clamp_fraction": (
                (current_raw.detach() >= self.config.cfm_loss_clamp).float().mean()
                if self.config.cfm_loss_clamp > 0.0
                else torch.zeros((), device=self.device)
            ),
            "ratio": ratio.detach().mean(),
            "recent_ratio": recent_ratio.detach().mean(),
            "ratio_abs_log": raw_log_ratio.detach().abs().mean(),
            "approx_kl": 0.5 * raw_log_ratio.detach().square().clamp(max=1e6).mean(),
            "clip_fraction": (
                (ratio.detach() - recent_ratio.detach()).abs()
                > self.config.exo_clip_radius
            )
            .float()
            .mean(),
            "recent_log_shift": (recent_loss.detach() - current_loss.detach())
            .abs()
            .mean(),
            "flow_loss": ofp["flow_loss"].detach(),
            "consistency_loss": ofp["consistency_loss"].detach(),
            "self_distill_loss": self_distill_loss.detach(),
            "contraction": ofp["contraction"].detach(),
            "target_action_rms": ofp["target_action_rms"].detach(),
            "x1_rms": x1_prediction.detach().square().mean().sqrt(),
            "actor_grad_norm": torch.as_tensor(
                gradient_norm, dtype=torch.float32, device=self.device
            ).detach(),
            "ema_decay": torch.as_tensor(ema_decay, device=self.device),
        }

    def critic_train_step(
        self, observations: torch.Tensor, returns: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        self.value.train()
        self.critic_optimizer.zero_grad(set_to_none=True)
        prediction = self.value(observations)
        critic_loss = F.smooth_l1_loss(prediction, returns.reshape(-1))
        if not torch.isfinite(critic_loss):
            raise FloatingPointError("non-finite ExO-FPO critic loss")
        critic_loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.value.parameters(), self.config.max_grad_norm
        )
        self.critic_optimizer.step()
        return {
            "critic_loss": critic_loss.detach(),
            "critic_grad_norm": torch.as_tensor(
                gradient_norm, dtype=torch.float32, device=self.device
            ).detach(),
            "value_mean": prediction.detach().mean(),
        }

    def train_torch_replay(self, replay: FlowMatchingReplayWindow) -> dict[str, float]:
        tensors = replay.tensors()
        advantages = tensors["advantages"]
        if self.config.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-8
            )
        if self.config.advantage_clamp is not None:
            positive, negative = self.config.advantage_clamp
            advantages = advantages.clamp(-negative, positive)
        tensors["advantages"] = advantages

        self.snapshot_recent_policy()
        metric_sums: dict[str, float] = {}
        metric_count = 0
        sample_count = len(replay)
        stop_early = False
        for _ in range(self.config.update_epochs):
            order = torch.randperm(sample_count, device=self.device)
            for offset in range(0, sample_count, self.config.batch_size):
                indices = order[offset : offset + self.config.batch_size]
                actor_metrics = self.actor_train_step(
                    tensors["actor_observations"][indices],
                    tensors["actions"][indices],
                    tensors["cfm_loss_eps"][indices],
                    tensors["cfm_loss_t"][indices],
                    tensors["initial_cfm_loss"][indices],
                    tensors["advantages"][indices],
                )
                critic_metrics = self.critic_train_step(
                    tensors["critic_observations"][indices],
                    tensors["returns"][indices],
                )
                for key, value in {**actor_metrics, **critic_metrics}.items():
                    metric_sums[key] = metric_sums.get(key, 0.0) + float(
                        value.detach().cpu()
                    )
                metric_count += 1
                if (
                    self.config.target_kl > 0.0
                    and float(actor_metrics["recent_log_shift"].detach().cpu())
                    > self.config.target_kl
                ):
                    stop_early = True
                    break
            if stop_early:
                break

        averaged = {
            key: value / max(metric_count, 1) for key, value in metric_sums.items()
        }
        averaged["learning_rate"] = float(self.actor_optimizer.param_groups[0]["lr"])
        averaged["critic_learning_rate"] = float(
            self.critic_optimizer.param_groups[0]["lr"]
        )
        averaged["value_loss"] = averaged.get("critic_loss", 0.0)
        averaged["loss"] = averaged.get("actor_loss", 0.0) + averaged.get(
            "critic_loss", 0.0
        )
        averaged["early_stop"] = float(stop_early)
        averaged["minibatches"] = float(metric_count)
        return averaged
