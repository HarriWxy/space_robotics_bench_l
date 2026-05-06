from __future__ import annotations

from collections.abc import Callable
from typing import Any

import flax.linen as nn
import gymnasium as gym
import jax.numpy as jnp
import optax
import tensorflow_probability.substrates.jax as tfp
from jax.nn.initializers import constant
from stable_baselines3.common.type_aliases import Schedule

from sbx.common.policies import Flatten
from sbx.ppo.policies import Actor, Critic, PPOPolicy
from sbx.common.jax_layers import NatureCNN

tfd = tfp.distributions


class ResidualBlock(nn.Module):
    hidden_dim: int
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        residual = x
        if residual.shape[-1] != self.hidden_dim:
            residual = nn.Dense(self.hidden_dim)(residual)

        x = nn.LayerNorm()(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = self.activation_fn(x)
        x = nn.Dense(self.hidden_dim)(x)
        return self.activation_fn(x + residual)


class ResidualActor(Actor):
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> tfd.Distribution:  # type: ignore[name-defined]
        if self.features_extractor is not None:
            x = self.features_extractor(self.features_dim, self.activation_fn)(x)

        x = Flatten()(x)

        for n_units in self.net_arch:
            x = ResidualBlock(n_units, self.activation_fn)(x)

        if self.ortho_init:
            orthogonal_init = nn.initializers.orthogonal(scale=0.01)
            bias_init = nn.initializers.zeros
            action_logits = nn.Dense(
                self.action_dim, kernel_init=orthogonal_init, bias_init=bias_init
            )(x)
        else:
            action_logits = nn.Dense(self.action_dim)(x)

        log_std = jnp.zeros(1)
        if self.num_discrete_choices is None:
            log_std = self.param("log_std", constant(self.log_std_init), (self.action_dim,))
            dist = tfd.MultivariateNormalDiag(loc=action_logits, scale_diag=jnp.exp(log_std))
        elif isinstance(self.num_discrete_choices, int):
            dist = tfd.Categorical(logits=action_logits)
        else:
            # Split action_logits = (batch_size, total_choices=sum(self.num_discrete_choices))
            action_logits = jnp.split(action_logits, self.split_indices, axis=1)
            # Pad to the maximum number of choices (required by tfp.distributions.Categorical).
            # Pad by -inf, so that the probability of these invalid actions is 0.
            logits_padded = jnp.stack(
                [
                    jnp.pad(
                        logit,
                        ((0, 0), (0, self.max_num_choices - logit.shape[1])),
                        constant_values=-jnp.inf,
                    )
                    for logit in action_logits
                ],
                axis=1,
            )
            dist = tfd.Categorical(logits=logits_padded)

        return dist


class ResidualCritic(Critic):
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.features_extractor is not None:
            x = self.features_extractor(self.features_dim, self.activation_fn)(x)

        x = Flatten()(x)
        for n_units in self.net_arch:
            x = ResidualBlock(n_units, self.activation_fn)(x)

        x = nn.Dense(1)(x)
        return x


class ResidualPolicy(PPOPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: list[int] | dict[str, list[int]] | None = None,
        ortho_init: bool = False,
        log_std_init: float = 0.0,
        activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu,
        use_sde: bool = False,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: type[NatureCNN] | None = None,
        features_extractor_kwargs: dict[str, Any] | None = None,
        normalize_images: bool = True,
        optimizer_class: Callable[..., optax.GradientTransformation] = optax.adam,
        optimizer_kwargs: dict[str, Any] | None = None,
        share_features_extractor: bool = False,
        actor_class: type[nn.Module] = ResidualActor,
        critic_class: type[nn.Module] = ResidualCritic,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            ortho_init,
            log_std_init,
            activation_fn,
            use_sde,
            use_expln,
            clip_mean,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            share_features_extractor,
            actor_class,
            critic_class,
        )