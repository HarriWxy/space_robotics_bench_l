from __future__ import annotations

from types import SimpleNamespace

import gymnasium
import torch

from srb.integrations.exoppo.exofpo_wrapper import SrbExoFpoEnvWrapper
from srb.integrations.exoppo.flow_matching import FlowMatchingPolicy


class _DummyEnv:
    def __init__(self) -> None:
        self.unwrapped = self
        self.device = "cpu"
        self.num_envs = 2
        self.single_action_space = gymnasium.spaces.Box(
            low=-2.0, high=2.0, shape=(2,), dtype=float
        )
        self.cfg = SimpleNamespace(
            is_finite_horizon=False,
            compute_final_obs=False,
        )
        self.max_episode_length = 10
        self.episode_length_buf = torch.zeros(2, dtype=torch.long)
        self.last_action: torch.Tensor | None = None

    @staticmethod
    def _observation() -> dict[str, torch.Tensor]:
        return {"proprio": torch.ones(2, 1), "command": torch.zeros(2, 1)}

    def reset(self):
        return self._observation(), {}

    def step(self, action: torch.Tensor):
        self.last_action = action
        return (
            self._observation(),
            torch.zeros(2),
            torch.zeros(2, dtype=torch.bool),
            torch.zeros(2, dtype=torch.bool),
            {},
        )


def test_exofpo_wrapper_clips_direct_flow_actions() -> None:
    env = _DummyEnv()
    wrapper = SrbExoFpoEnvWrapper(
        env,
        actor_keys=("proprio", "command"),
        clip_actions=1.0,
    )
    wrapper.reset()
    wrapper.step(torch.tensor([[2.0, -2.0], [0.25, -0.5]]))
    assert env.last_action is not None
    assert torch.equal(
        env.last_action,
        torch.tensor([[1.0, -1.0], [0.25, -0.5]]),
    )


def test_fpo_flow_matching_shapes_and_reverse_time_path() -> None:
    torch.manual_seed(0)
    policy = FlowMatchingPolicy(4, 3, (8, 8), sampling_steps=2)
    observations = torch.randn(5, 4)
    actions = policy.generate(observations, deterministic=False)
    eps = torch.randn(5, 4, 3)
    timestep = torch.rand(5, 4, 1)
    loss, x1_prediction, x0_prediction = policy.get_cfm_loss(
        observations, actions, eps, timestep
    )
    assert actions.shape == (5, 3)
    assert loss.shape == (5, 4)
    assert x1_prediction.shape == x0_prediction.shape == (5, 4, 3)
    assert torch.isfinite(loss).all()
