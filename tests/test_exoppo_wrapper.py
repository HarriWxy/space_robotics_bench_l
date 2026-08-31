from __future__ import annotations

from types import SimpleNamespace

import gymnasium
import torch

from srb.integrations.exoppo.wrapper import SrbExoPpoEnvWrapper


class _DummySrbEnv:
    def __init__(self) -> None:
        self.unwrapped = self
        self.device = "cpu"
        self.num_envs = 2
        self.single_action_space = gymnasium.spaces.Box(low=-2.0, high=4.0, shape=(2,))
        self.cfg = SimpleNamespace(
            is_finite_horizon=False,
            compute_final_obs=False,
        )
        self.max_episode_length = 10
        self.episode_length_buf = torch.zeros(2, dtype=torch.long)
        self.last_action = None

    @staticmethod
    def _observation(offset: float = 0.0) -> dict[str, torch.Tensor]:
        return {
            "state": torch.tensor([[1.0, 2.0], [3.0, 4.0]]) + offset,
            "proprio": torch.tensor([[5.0], [6.0]]) + offset,
            "command": torch.tensor([[7.0], [8.0]]) + offset,
        }

    def reset(self):
        return self._observation(), {}

    def step(self, action: torch.Tensor):
        self.last_action = action
        terminated = torch.tensor([False, True])
        truncated = torch.tensor([True, False])
        return (
            self._observation(1.0),
            torch.tensor([1.0, 2.0]),
            terminated,
            truncated,
            {"final_obs": self._observation(2.0)},
        )


def test_wrapper_keeps_flow_variables_separate_from_bounded_action() -> None:
    env = _DummySrbEnv()
    wrapper = SrbExoPpoEnvWrapper(
        env,
        actor_keys=("state", "proprio", "command"),
    )
    actor, critic, _ = wrapper.reset()
    assert actor.shape == (2, 4)
    assert torch.equal(actor, critic)
    assert env.cfg.compute_final_obs is True

    pre_tanh = torch.zeros((2, 2))
    _, _, reward, terminated, truncated, extras = wrapper.step(pre_tanh)
    assert torch.equal(env.last_action, torch.ones((2, 2)))
    assert torch.equal(reward, torch.tensor([1.0, 2.0]))
    assert torch.equal(terminated, torch.tensor([False, True]))
    assert torch.equal(truncated, torch.tensor([True, False]))
    final_actor, final_critic = wrapper.final_observations(extras)
    assert final_actor.shape == (2, 4)
    assert torch.equal(final_actor, final_critic)
