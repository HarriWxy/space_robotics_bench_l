"""Dynamics representation learning from short robot-state histories.

The module deliberately separates two concepts:

* ``latent`` is inferred from *past* transitions and can be appended to a
  policy observation as a deployable dynamics context.
* ``physics`` is an optional privileged prediction head.  In simulation it
  can be supervised with parameters such as normalized gravity or friction;
  on hardware it is omitted and the transition objective remains usable.

The transition target is kept outside the context window.  This is important:
feeding the transition being predicted into the encoder would let it copy the
answer instead of identifying persistent dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class DynamicsEncoderConfig:
    """Shape and optimization-independent settings for the encoder.

    ``history_length`` is the number of observed transitions used to infer the
    latent. A model call expects ``context_states`` with shape
    ``[batch, history_length + 1, state_dim]`` and ``context_actions`` with
    shape ``[batch, history_length, action_dim]``. The temporal encoder uses a
    Transformer with learned positional embeddings.
    """

    state_dim: int
    action_dim: int
    latent_dim: int = 32
    hidden_dim: int = 128
    history_length: int = 16
    num_heads: int = 4
    num_layers: int = 2
    transformer_ff_dim: int | None = None
    physics_dim: int = 0
    dropout: float = 0.0
    transition_loss_weight: float = 1.0
    physics_loss_weight: float = 1.0


@dataclass(slots=True)
class DynamicsTransitionBatch:
    """A batch with a held-out query transition.

    The context contains only transitions before the query:

    ``context_states[:, i] --context_actions[:, i]--> context_states[:, i+1]``

    The encoder must infer the dynamics from that context before predicting
    ``query_state --query_action--> target_next_state``.
    """

    context_states: Tensor
    context_actions: Tensor
    query_state: Tensor
    query_action: Tensor
    target_next_state: Tensor | None = None
    physics_target: Tensor | None = None
    sample_mask: Tensor | None = None
    history_mask: Tensor | None = None


@dataclass(slots=True)
class DynamicsEncoderOutput:
    """Outputs produced for one query transition."""

    latent: Tensor
    predicted_delta: Tensor
    predicted_next_state: Tensor
    physics: Tensor | None


def _mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    dropout: float,
) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.SiLU(),
    ]
    if dropout > 0.0:
        layers.append(nn.Dropout(dropout))
    layers.extend((nn.Linear(hidden_dim, hidden_dim), nn.SiLU()))
    if dropout > 0.0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


class DynamicsEncoder(nn.Module):
    """Infer a compact dynamics context and predict a held-out transition.

    The per-step input is ``[state_t, action_t, state_{t+1}-state_t]``. A
    Transformer aggregates the context and a bottleneck produces ``latent``.
    The query predictor then receives ``[query_state, query_action, latent]``
    and outputs ``delta_state``. This makes the representation useful even
    when no simulator-only physical parameter is available.

    For a policy that must run on a real robot, pass sensor-derived values such
    as ``proprio`` and ``proprio_dyn`` as ``state``. Do not use simulator truth
    such as root-state or raw contact-force tensors in that deployable input.
    """

    def __init__(self, config: DynamicsEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self._validate_config(config)

        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        self.latent_dim = config.latent_dim
        self.history_length = config.history_length
        self.physics_dim = config.physics_dim

        # state_t, action_t, delta_state_t, and a valid-transition indicator.
        step_input_dim = 2 * config.state_dim + config.action_dim + 1
        self.step_encoder = _mlp(
            step_input_dim,
            config.hidden_dim,
            config.hidden_dim,
            config.dropout,
        )
        self.position_embedding = nn.Parameter(
            torch.empty(1, config.history_length, config.hidden_dim)
        )
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.transformer_ff_dim or 4 * config.hidden_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            transformer_layer,
            num_layers=config.num_layers,
            norm=nn.LayerNorm(config.hidden_dim),
            # The histories are short and fixed-size; dense attention is more
            # predictable than the prototype nested-tensor fast path here.
            enable_nested_tensor=False,
        )
        self.latent_head = _mlp(
            config.hidden_dim,
            config.hidden_dim,
            config.latent_dim,
            config.dropout,
        )
        self.transition_head = _mlp(
            config.state_dim + config.action_dim + config.latent_dim,
            config.hidden_dim,
            config.state_dim,
            config.dropout,
        )
        self.physics_head = (
            _mlp(
                config.latent_dim,
                config.hidden_dim,
                config.physics_dim,
                config.dropout,
            )
            if config.physics_dim > 0
            else None
        )

    @staticmethod
    def _validate_config(config: DynamicsEncoderConfig) -> None:
        positive_fields = (
            ("state_dim", config.state_dim),
            ("action_dim", config.action_dim),
            ("latent_dim", config.latent_dim),
            ("hidden_dim", config.hidden_dim),
            ("history_length", config.history_length),
            ("num_heads", config.num_heads),
            ("num_layers", config.num_layers),
        )
        for name, value in positive_fields:
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if config.physics_dim < 0:
            raise ValueError(f"physics_dim must be non-negative, got {config.physics_dim}")
        if not 0.0 <= config.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {config.dropout}")
        if config.transition_loss_weight < 0.0:
            raise ValueError("transition_loss_weight must be non-negative")
        if config.physics_loss_weight < 0.0:
            raise ValueError("physics_loss_weight must be non-negative")
        if config.hidden_dim % config.num_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by num_heads, got "
                f"{config.hidden_dim} and {config.num_heads}"
            )
        if config.transformer_ff_dim is not None and config.transformer_ff_dim <= 0:
            raise ValueError("transformer_ff_dim must be positive when provided")

    @staticmethod
    def _check_tensor(
        value: Tensor,
        *,
        name: str,
        ndim: int,
        last_dim: int | None = None,
    ) -> None:
        if not isinstance(value, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}")
        if value.ndim != ndim:
            raise ValueError(f"{name} must have {ndim} dimensions, got {tuple(value.shape)}")
        if last_dim is not None and value.shape[-1] != last_dim:
            raise ValueError(
                f"{name} last dimension must be {last_dim}, got {tuple(value.shape)}"
            )
        if not value.is_floating_point():
            raise TypeError(f"{name} must be floating point, got {value.dtype}")

    def _prepare_history_mask(
        self,
        history_mask: Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        if history_mask is None:
            return torch.ones(
                batch_size,
                self.history_length,
                1,
                device=device,
                dtype=dtype,
            )
        if history_mask.ndim == 2:
            history_mask = history_mask.unsqueeze(-1)
        self._check_tensor(
            history_mask,
            name="history_mask",
            ndim=3,
            last_dim=1,
        )
        expected = (batch_size, self.history_length, 1)
        if tuple(history_mask.shape) != expected:
            raise ValueError(
                f"history_mask must have shape {expected}, got {tuple(history_mask.shape)}"
            )
        return history_mask.to(device=device, dtype=dtype).clamp(0.0, 1.0).detach()

    def encode(
        self,
        context_states: Tensor,
        context_actions: Tensor,
        history_mask: Tensor | None = None,
    ) -> Tensor:
        """Encode past transitions into ``[batch, latent_dim]``.

        ``context_states`` has shape ``[B, H + 1, S]`` and
        ``context_actions`` has shape ``[B, H, A]``. The final transition to
        be predicted must not be included in these tensors.
        """

        self._check_tensor(
            context_states,
            name="context_states",
            ndim=3,
            last_dim=self.state_dim,
        )
        self._check_tensor(
            context_actions,
            name="context_actions",
            ndim=3,
            last_dim=self.action_dim,
        )
        expected_states = self.history_length + 1
        if context_states.shape[1] != expected_states:
            raise ValueError(
                "context_states must contain exactly history_length + 1 states: "
                f"expected {expected_states}, got {context_states.shape[1]}"
            )
        expected_actions = (
            context_states.shape[0],
            self.history_length,
            self.action_dim,
        )
        if tuple(context_actions.shape) != expected_actions:
            raise ValueError(
                f"context_actions must have shape {expected_actions}, "
                f"got {tuple(context_actions.shape)}"
            )

        mask = self._prepare_history_mask(
            history_mask,
            batch_size=context_states.shape[0],
            device=context_states.device,
            dtype=context_states.dtype,
        )
        deltas = context_states[:, 1:] - context_states[:, :-1]
        # Padding is explicitly masked, and the mask itself is supplied so the
        # step encoder can distinguish an actual zero transition from padding.
        step_features = torch.cat(
            (
                context_states[:, :-1] * mask,
                context_actions * mask,
                deltas * mask,
                mask,
            ),
            dim=-1,
        )
        step_features = self.step_encoder(step_features)
        step_features = step_features + self.position_embedding
        padding_mask = ~mask.squeeze(-1).bool()
        # TransformerEncoder produces NaNs when every key/value position is
        # masked. This occurs immediately after reset, before one transition
        # has been collected. Keep one zero-information dummy position active
        # for those rows; the masked mean below still returns a zero summary.
        all_padding = padding_mask.all(dim=1)
        if all_padding.any():
            padding_mask = padding_mask.clone()
            padding_mask[all_padding, -1] = False
        temporal_features = self.temporal_encoder(
            step_features,
            src_key_padding_mask=padding_mask,
        )
        valid_count = mask.sum(dim=1).clamp_min(1.0)
        summary = (temporal_features * mask).sum(dim=1) / valid_count
        return self.latent_head(summary)

    def forward(
        self,
        context_states: Tensor,
        context_actions: Tensor,
        query_state: Tensor,
        query_action: Tensor,
        history_mask: Tensor | None = None,
    ) -> DynamicsEncoderOutput:
        """Infer the latent and predict the next state for a query action."""

        self._check_tensor(
            query_state,
            name="query_state",
            ndim=2,
            last_dim=self.state_dim,
        )
        self._check_tensor(
            query_action,
            name="query_action",
            ndim=2,
            last_dim=self.action_dim,
        )
        expected_batch = context_states.shape[0]
        if query_state.shape[0] != expected_batch or query_action.shape[0] != expected_batch:
            raise ValueError(
                "context and query tensors must have the same batch size, got "
                f"{expected_batch}, {query_state.shape[0]}, and {query_action.shape[0]}"
            )

        latent = self.encode(context_states, context_actions, history_mask)
        transition_input = torch.cat((query_state, query_action, latent), dim=-1)
        predicted_delta = self.transition_head(transition_input)
        predicted_next_state = query_state + predicted_delta
        physics = self.physics_head(latent) if self.physics_head is not None else None
        return DynamicsEncoderOutput(
            latent=latent,
            predicted_delta=predicted_delta,
            predicted_next_state=predicted_next_state,
            physics=physics,
        )

    @staticmethod
    def _masked_mean(values: Tensor, sample_mask: Tensor | None) -> Tensor:
        if sample_mask is None:
            return values.mean()
        if sample_mask.ndim == 2 and sample_mask.shape[-1] == 1:
            sample_mask = sample_mask.squeeze(-1)
        if sample_mask.ndim != 1 or sample_mask.shape[0] != values.shape[0]:
            raise ValueError(
                "sample_mask must have shape [batch] or [batch, 1], got "
                f"{tuple(sample_mask.shape)}"
            )
        weights = sample_mask.to(device=values.device, dtype=values.dtype).clamp(0.0, 1.0)
        return (values * weights).sum() / weights.sum().clamp_min(1.0)

    def loss(
        self,
        batch: DynamicsTransitionBatch,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute transition and optional privileged-physics losses.

        Inputs should normally be normalized before entering the model. In
        particular, scale joint positions, velocities, IMU values, and the
        physics targets so that one physical quantity does not dominate the
        loss merely because of units.
        """

        output = self(
            batch.context_states,
            batch.context_actions,
            batch.query_state,
            batch.query_action,
            batch.history_mask,
        )
        total = output.latent.sum() * 0.0
        metrics: dict[str, Tensor] = {}

        if batch.target_next_state is not None:
            self._check_tensor(
                batch.target_next_state,
                name="target_next_state",
                ndim=2,
                last_dim=self.state_dim,
            )
            if batch.target_next_state.shape != output.predicted_next_state.shape:
                raise ValueError(
                    "target_next_state must have shape "
                    f"{tuple(output.predicted_next_state.shape)}, "
                    f"got {tuple(batch.target_next_state.shape)}"
                )
            transition_error = F.smooth_l1_loss(
                output.predicted_next_state,
                batch.target_next_state,
                reduction="none",
            ).mean(dim=-1)
            transition_loss = self._masked_mean(transition_error, batch.sample_mask)
            total = total + self.config.transition_loss_weight * transition_loss
            metrics["transition_loss"] = transition_loss.detach()
            metrics["transition_rmse"] = torch.sqrt(
                self._masked_mean(
                    (output.predicted_next_state - batch.target_next_state).square().mean(-1),
                    batch.sample_mask,
                )
            ).detach()

        if batch.physics_target is not None:
            if output.physics is None:
                raise ValueError(
                    "physics_target was provided but config.physics_dim is zero"
                )
            self._check_tensor(
                batch.physics_target,
                name="physics_target",
                ndim=2,
                last_dim=self.physics_dim,
            )
            if batch.physics_target.shape != output.physics.shape:
                raise ValueError(
                    f"physics_target must have shape {tuple(output.physics.shape)}, "
                    f"got {tuple(batch.physics_target.shape)}"
                )
            physics_error = F.smooth_l1_loss(
                output.physics,
                batch.physics_target,
                reduction="none",
            ).mean(dim=-1)
            physics_loss = self._masked_mean(physics_error, batch.sample_mask)
            total = total + self.config.physics_loss_weight * physics_loss
            metrics["physics_loss"] = physics_loss.detach()
            metrics["physics_rmse"] = torch.sqrt(
                self._masked_mean(
                    (output.physics - batch.physics_target).square().mean(-1),
                    batch.sample_mask,
                )
            ).detach()

        if not metrics:
            raise ValueError(
                "at least one of target_next_state or physics_target is required"
            )
        metrics["loss"] = total.detach()
        if batch.sample_mask is None:
            metrics["valid_samples"] = torch.tensor(
                batch.context_states.shape[0], device=total.device, dtype=total.dtype
            )
        else:
            metrics["valid_samples"] = batch.sample_mask.to(total).sum().detach()
        return total, metrics


class DynamicsEncoderTrainer:
    """Small optimizer wrapper for auxiliary encoder updates."""

    def __init__(
        self,
        encoder: DynamicsEncoder,
        *,
        learning_rate: float = 3.0e-4,
        weight_decay: float = 1.0e-4,
        max_grad_norm: float = 1.0,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive")
        self.encoder = encoder
        self.optimizer = optimizer or torch.optim.AdamW(
            encoder.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.max_grad_norm = max_grad_norm

    def update(self, batch: DynamicsTransitionBatch) -> dict[str, float]:
        """Run one gradient update and return detached scalar metrics."""

        self.encoder.train()
        loss, metrics = self.encoder.loss(batch)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.encoder.parameters(), self.max_grad_norm
        )
        self.optimizer.step()
        metrics["grad_norm"] = torch.as_tensor(grad_norm).detach()
        return {name: float(value.cpu()) for name, value in metrics.items()}


class DynamicsHistory:
    """Fixed-size per-environment transition history for vectorized rollouts."""

    def __init__(
        self,
        num_envs: int,
        *,
        state_dim: int,
        action_dim: int,
        history_length: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if state_dim <= 0 or action_dim <= 0:
            raise ValueError("state_dim and action_dim must be positive")
        if history_length <= 0:
            raise ValueError("history_length must be positive")
        self.num_envs = num_envs
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.history_length = history_length
        self.device = torch.device(device)
        self.states = torch.zeros(
            num_envs, history_length + 1, state_dim, device=self.device, dtype=dtype
        )
        self.actions = torch.zeros(
            num_envs, history_length, action_dim, device=self.device, dtype=dtype
        )
        self.mask = torch.zeros(
            num_envs, history_length, 1, device=self.device, dtype=dtype
        )

    def _check_state(self, state: Tensor, *, name: str) -> Tensor:
        if not isinstance(state, Tensor) or state.shape != (
            self.num_envs,
            self.state_dim,
        ):
            shape = getattr(state, "shape", None)
            raise ValueError(
                f"{name} must have shape [{self.num_envs}, {self.state_dim}], got {shape}"
            )
        return state.to(device=self.device, dtype=self.states.dtype).detach()

    def _check_action(self, action: Tensor) -> Tensor:
        if not isinstance(action, Tensor) or action.shape != (
            self.num_envs,
            self.action_dim,
        ):
            shape = getattr(action, "shape", None)
            raise ValueError(
                f"action must have shape [{self.num_envs}, {self.action_dim}], got {shape}"
            )
        return action.to(device=self.device, dtype=self.actions.dtype).detach()

    def reset(self, initial_state: Tensor, env_ids: Tensor | None = None) -> None:
        """Reset all or selected environments with a post-reset state."""

        initial_state = initial_state.to(device=self.device, dtype=self.states.dtype).detach()
        if env_ids is None:
            if initial_state.shape != (self.num_envs, self.state_dim):
                raise ValueError(
                    "initial_state must have shape "
                    f"[{self.num_envs}, {self.state_dim}], got {tuple(initial_state.shape)}"
                )
            ids = torch.arange(self.num_envs, device=self.device)
        else:
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).reshape(-1)
            if initial_state.shape != (ids.numel(), self.state_dim):
                raise ValueError(
                    "selected reset initial_state must have shape "
                    f"[{ids.numel()}, {self.state_dim}], got {tuple(initial_state.shape)}"
                )
        self.states[ids] = initial_state.unsqueeze(1)
        self.actions[ids].zero_()
        self.mask[ids].zero_()

    def snapshot(self) -> tuple[Tensor, Tensor, Tensor]:
        """Return cloned tensors safe to retain as a training sample."""

        return self.states.clone(), self.actions.clone(), self.mask.clone()

    @property
    def current_state(self) -> Tensor:
        return self.states[:, -1]

    def append(
        self,
        action: Tensor,
        next_state: Tensor,
        done: Tensor | None = None,
    ) -> None:
        """Append a transition and clear history for environments that reset.

        Isaac Lab commonly returns the post-reset observation for a done
        environment. Such an observation must initialize a fresh history, not
        be treated as the successor of the terminal state.
        """

        action = self._check_action(action)
        next_state = self._check_state(next_state, name="next_state")
        if done is None:
            done = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        else:
            done = torch.as_tensor(done, device=self.device, dtype=torch.bool).reshape(-1)
            if done.shape != (self.num_envs,):
                raise ValueError(
                    f"done must have shape [{self.num_envs}], got {tuple(done.shape)}"
                )

        keep = ~done
        if keep.any():
            old_states = self.states[keep].clone()
            old_actions = self.actions[keep].clone()
            old_mask = self.mask[keep].clone()
            self.states[keep] = torch.cat((old_states[:, 1:], next_state[keep, None]), dim=1)
            self.actions[keep] = torch.cat((old_actions[:, 1:], action[keep, None]), dim=1)
            self.mask[keep] = torch.cat(
                (old_mask[:, 1:], torch.ones_like(old_mask[:, :1])),
                dim=1,
            )

        if done.any():
            done_ids = done.nonzero(as_tuple=False).squeeze(-1)
            self.reset(next_state[done_ids], done_ids)


class DynamicsEncoderRuntime:
    """Rollout-side adapter that collects held-out transitions.

    Typical usage in a vectorized loop is:

    .. code-block:: python

        runtime.reset(sensor_state_after_reset)
        latent = runtime.begin_step()
        # append ``latent`` to the policy observation, then obtain action
        runtime.observe(action, next_sensor_state, done, physics_target=gravity)
        batch = runtime.drain_batch()
        if batch is not None:
            trainer.update(batch)

    ``begin_step`` is inference-only. ``drain_batch`` returns detached rollout
    data so the auxiliary update is independent from the RL graph.
    """

    def __init__(
        self,
        encoder: DynamicsEncoder,
        *,
        num_envs: int,
        device: torch.device | str,
    ) -> None:
        self.encoder = encoder
        self.history = DynamicsHistory(
            num_envs,
            state_dim=encoder.state_dim,
            action_dim=encoder.action_dim,
            history_length=encoder.history_length,
            device=device,
        )
        self._pending: tuple[Tensor, Tensor, Tensor, Tensor] | None = None
        self._samples: list[DynamicsTransitionBatch] = []

    def reset(self, initial_state: Tensor, *, clear_samples: bool = True) -> None:
        self.history.reset(initial_state)
        self._pending = None
        if clear_samples:
            self._samples.clear()

    @torch.no_grad()
    def begin_step(self) -> Tensor:
        """Encode the history available before the next policy action."""

        context_states, context_actions, history_mask = self.history.snapshot()
        query_state = context_states[:, -1].clone()
        was_training = self.encoder.training
        self.encoder.eval()
        latent = self.encoder.encode(context_states, context_actions, history_mask)
        if was_training:
            self.encoder.train()
        self._pending = (context_states, context_actions, history_mask, query_state)
        return latent.detach()

    @torch.no_grad()
    def observe(
        self,
        action: Tensor,
        next_state: Tensor,
        done: Tensor | None = None,
        *,
        physics_target: Tensor | None = None,
    ) -> None:
        """Record the query transition and advance/reset the history."""

        if self._pending is None:
            raise RuntimeError("observe() must follow begin_step()")
        context_states, context_actions, history_mask, query_state = self._pending
        action = action.detach().to(self.history.device, dtype=self.history.actions.dtype)
        next_state = next_state.detach().to(
            self.history.device, dtype=self.history.states.dtype
        )
        done_tensor = (
            torch.zeros(self.history.num_envs, device=self.history.device, dtype=torch.bool)
            if done is None
            else torch.as_tensor(done, device=self.history.device, dtype=torch.bool).reshape(-1)
        )
        if done_tensor.shape != (self.history.num_envs,):
            raise ValueError(
                f"done must have shape [{self.history.num_envs}], got {tuple(done_tensor.shape)}"
            )
        sample_mask = (~done_tensor).to(dtype=context_states.dtype)
        target = physics_target
        if target is not None:
            target = target.detach().to(self.history.device, dtype=context_states.dtype)
        self._samples.append(
            DynamicsTransitionBatch(
                context_states=context_states,
                context_actions=context_actions,
                query_state=query_state,
                query_action=action.clone(),
                target_next_state=next_state.clone(),
                physics_target=None if target is None else target.clone(),
                sample_mask=sample_mask,
                history_mask=history_mask,
            )
        )
        self.history.append(action, next_state, done_tensor)
        self._pending = None

    def drain_batch(self) -> DynamicsTransitionBatch | None:
        """Concatenate and clear all samples collected since the last drain."""

        if not self._samples:
            return None
        samples = self._samples
        self._samples = []

        def cat(name: str) -> Tensor:
            values = [getattr(sample, name) for sample in samples]
            if any(value is None for value in values):
                raise RuntimeError(f"runtime samples have inconsistent {name} values")
            return torch.cat([value for value in values if value is not None], dim=0)

        physics_values = [sample.physics_target for sample in samples]
        if all(value is None for value in physics_values):
            physics_target = None
        elif any(value is None for value in physics_values):
            raise RuntimeError(
                "physics_target must be provided for every runtime sample or none"
            )
        else:
            physics_target = torch.cat(
                [value for value in physics_values if value is not None], dim=0
            )
        return DynamicsTransitionBatch(
            context_states=cat("context_states"),
            context_actions=cat("context_actions"),
            query_state=cat("query_state"),
            query_action=cat("query_action"),
            target_next_state=cat("target_next_state"),
            physics_target=physics_target,
            sample_mask=cat("sample_mask"),
            history_mask=cat("history_mask"),
        )


__all__ = [
    "DynamicsEncoder",
    "DynamicsEncoderConfig",
    "DynamicsEncoderOutput",
    "DynamicsEncoderRuntime",
    "DynamicsEncoderTrainer",
    "DynamicsHistory",
    "DynamicsTransitionBatch",
]
