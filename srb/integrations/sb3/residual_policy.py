from __future__ import annotations

from typing import Any

import torch as th
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import get_device


class ResidualBlock(nn.Module):
    """A simple residual block for same-dimension MLP layers."""

    def __init__(self, dim: int, activation_fn: type[nn.Module]) -> None:
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.activation = activation_fn()

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.activation(self.fc(x) + x)


class ResidualMlpExtractor(nn.Module):
    """MLP extractor with optional residual connections for same-size hidden layers."""

    def __init__(
        self,
        feature_dim: int,
        net_arch: list[int] | dict[str, list[int]],
        activation_fn: type[nn.Module],
        device: th.device | str = "auto",
    ) -> None:
        super().__init__()
        device = get_device(device)

        policy_net: list[nn.Module] = []
        value_net: list[nn.Module] = []

        last_layer_dim_pi = feature_dim
        last_layer_dim_vf = feature_dim

        if isinstance(net_arch, dict):
            pi_layers_dims = net_arch.get("pi", [])
            vf_layers_dims = net_arch.get("vf", [])
        else:
            pi_layers_dims = vf_layers_dims = net_arch or []

        for curr_layer_dim in pi_layers_dims:
            if curr_layer_dim == last_layer_dim_pi:
                policy_net.append(ResidualBlock(curr_layer_dim, activation_fn))
            else:
                policy_net.append(nn.Linear(last_layer_dim_pi, curr_layer_dim))
                policy_net.append(activation_fn())
            last_layer_dim_pi = curr_layer_dim

        for curr_layer_dim in vf_layers_dims:
            if curr_layer_dim == last_layer_dim_vf:
                value_net.append(ResidualBlock(curr_layer_dim, activation_fn))
            else:
                value_net.append(nn.Linear(last_layer_dim_vf, curr_layer_dim))
                value_net.append(activation_fn())
            last_layer_dim_vf = curr_layer_dim

        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        self.policy_net = nn.Sequential(*policy_net).to(device)
        self.value_net = nn.Sequential(*value_net).to(device)

    def forward(self, features: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class ResidualPolicy(ActorCriticPolicy):
    """ActorCriticPolicy with a residual MLP extractor."""

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = ResidualMlpExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )
