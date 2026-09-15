"""CPU contract test for optional FPO dynamics conditioning."""

from types import SimpleNamespace

import gymnasium
import torch

from dyn.dyn_encoder import (
    DynamicsEncoder,
    DynamicsEncoderConfig,
    DynamicsEncoderTrainer,
)
from srb.integrations.fpo.wrapper import SrbFpoEnvWrapper


class _DummyFpoEnv:
    def __init__(self) -> None:
        self.unwrapped = self
        self.device = "cpu"
        self.num_envs = 2
        self.single_action_space = gymnasium.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
        )
        self.action_space = self.single_action_space
        self.max_episode_length = 20
        self.episode_length_buf = torch.zeros(2, dtype=torch.long)
        self.cfg = SimpleNamespace(is_finite_horizon=False)
        self.last_action: torch.Tensor | None = None
        self._offset = 0.0

    def _observation(self) -> dict[str, torch.Tensor]:
        value = torch.full((2, 1), self._offset)
        return {
            "proprio": value,
            "proprio_dyn": value + 1.0,
            "command": value + 2.0,
        }

    def reset(self):
        self._offset = 0.0
        return self._observation(), {}

    def step(self, action: torch.Tensor):
        self.last_action = action.clone()
        self._offset += 0.1
        return (
            self._observation(),
            torch.ones(2),
            torch.zeros(2, dtype=torch.bool),
            torch.zeros(2, dtype=torch.bool),
            {"metrics/gravity_magnitude": 9.80665},
        )


def test_fpo_wrapper_appends_latent_and_runs_auxiliary_update() -> None:
    env = _DummyFpoEnv()
    wrapper = SrbFpoEnvWrapper(
        env,
        actor_keys=("proprio", "proprio_dyn", "command"),
    )
    encoder = DynamicsEncoder(
        DynamicsEncoderConfig(
            state_dim=2,
            action_dim=2,
            latent_dim=2,
            hidden_dim=8,
            history_length=2,
            physics_dim=1,
        )
    )
    wrapper.attach_dynamics_encoder(
        encoder,
        trainer=DynamicsEncoderTrainer(encoder, learning_rate=1.0e-3),
        update_interval=1,
        physics_target_key="metrics/gravity_magnitude",
        physics_target_scale=9.80665,
    )

    actor, _ = wrapper.get_observations()
    assert actor.shape == (2, 5)
    next_actor, _, dones, extras = wrapper.step(torch.zeros(2, 2))
    assert next_actor.shape == (2, 5)
    assert torch.equal(dones, torch.zeros(2, dtype=torch.long))
    assert env.last_action is not None
    assert "train/dynamics/loss" in extras["log"]
