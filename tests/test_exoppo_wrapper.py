from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace

import gymnasium
import torch

from srb.integrations.exoppo.main import _write_run_manifest
from srb.integrations.exoppo.wrapper import SrbExoPpoEnvWrapper
from train_physics_conditioned_flow import build_srb_argv


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
            "physics": torch.tensor([[0.25], [0.5]]) + offset,
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


def test_wrapper_keeps_physics_context_out_of_actor_input() -> None:
    env = _DummySrbEnv()
    wrapper = SrbExoPpoEnvWrapper(
        env,
        actor_keys=("state", "proprio", "command"),
        physics_keys=("physics",),
    )
    actor, critic, physics, _ = wrapper.reset_with_physics()
    assert actor.shape == (2, 4)
    assert torch.equal(actor, critic)
    assert physics is not None
    assert physics.shape == (2, 1)
    assert torch.equal(physics, torch.tensor([[0.25], [0.5]]))

    _, _, next_physics, _, _, _, extras = wrapper.step_with_physics(
        torch.zeros((2, 2))
    )
    assert next_physics is not None
    assert torch.equal(next_physics, torch.tensor([[1.25], [1.5]]))
    final_encoded = wrapper.final_observations_with_physics(extras)
    assert final_encoded is not None
    _, _, final_physics = final_encoded
    assert final_physics is not None
    assert torch.equal(final_physics, torch.tensor([[2.25], [2.5]]))


def test_wrapper_reports_missing_physics_group() -> None:
    env = _DummySrbEnv()
    env.reset = lambda: ({"state": torch.zeros((2, 2))}, {})
    wrapper = SrbExoPpoEnvWrapper(
        env,
        actor_keys=("state",),
        physics_keys=("physics",),
    )
    try:
        wrapper.reset_with_physics()
    except KeyError as error:
        assert "physics" in str(error)
    else:
        raise AssertionError("missing physics observation must raise KeyError")


@dataclass
class _ManifestFlowConfig:
    rollout_steps: int = 64
    replay_N: int = 4
    warmup_rollouts: int = 4


@dataclass
class _PhysicsConditioning:
    gravity_magnitude_range: tuple[float, float] = (1.62496, 3.72076)
    gravity_interval_s: tuple[float, float] = (30.0, 30.0)


def test_run_manifest_records_resolved_gravity_schedule(tmp_path) -> None:
    env_cfg = SimpleNamespace(
        seed=7,
        domain="moon",
        gravity=None,
        num_envs=2,
        include_gravity_magnitude=True,
        include_physics_context=False,
        gravity_magnitude_reference=9.80665,
        physics_conditioning=_PhysicsConditioning(),
        events=SimpleNamespace(
            randomize_gravity=SimpleNamespace(
                mode="interval",
                is_global_time=True,
                interval_range_s=(30.0, 30.0),
                params={
                    "distribution_params": (
                        (0.0, 0.0, -1.62496),
                        (0.0, 0.0, -3.72076),
                    )
                },
            )
        ),
    )
    wrapped_env = SimpleNamespace(
        num_envs=2,
        actor_keys=("proprio", "proprio_dyn", "command"),
        critic_keys=None,
        physics_keys=None,
        num_actions=2,
        action_low=torch.tensor([[-1.0, -1.0]]),
        action_high=torch.tensor([[1.0, 1.0]]),
    )

    _write_run_manifest(
        logdir=tmp_path,
        workflow="train",
        algorithm="ExO-PPO",
        env_id="srb/locomotion_velocity_tracking_c",
        env_cfg=env_cfg,
        raw_cfg={"seed": 7},
        flow_config=_ManifestFlowConfig(),
        wrapped_env=wrapped_env,
        actor_observation_dim=37,
        critic_observation_dim=37,
    )

    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    environment = manifest["environment"]
    assert environment["physics_conditioning"] == {
        "gravity_interval_s": [30.0, 30.0],
        "gravity_magnitude_range": [1.62496, 3.72076],
    }
    assert environment["gravity_randomization"] == {
        "mode": "interval",
        "is_global_time": True,
        "interval_range_s": [30.0, 30.0],
        "distribution_params": [
            [0.0, 0.0, -1.62496],
            [0.0, 0.0, -3.72076],
        ],
    }


def test_physics_conditioned_launcher_enables_the_full_actor_contract() -> None:
    argv = build_srb_argv(
        SimpleNamespace(
            algo="flowppo",
            seed=3,
            num_envs=8,
            iterations=11,
            no_gravity_context=False,
        )
    )

    assert argv[:6] == [
        "agent",
        "train",
        "--algo",
        "flowppo",
        "--env",
        "locomotion_velocity_tracking_c",
    ]
    assert "env.curriculum.command_mode=omnidirectional" in argv
    assert "env.include_gravity_magnitude=true" in argv
    assert (
        "env.physics_conditioning.gravity_magnitude_range=[1.62496,3.72076]"
        in argv
    )
    assert "env.physics_conditioning.gravity_interval_s=[30.0,30.0]" in argv
    assert "agent.max_iterations=11" in argv

    no_context_argv = build_srb_argv(
        SimpleNamespace(
            algo="flowppo",
            seed=3,
            num_envs=8,
            iterations=11,
            no_gravity_context=True,
        )
    )
    assert "env.include_gravity_magnitude=false" in no_context_argv


def test_film_launcher_enables_separate_context_and_stratified_replay() -> None:
    argv = build_srb_argv(
        SimpleNamespace(
            algo="exoppo",
            seed=3,
            num_envs=8,
            iterations=11,
            no_gravity_context=False,
            conditioning="film",
            stratified_replay=True,
        )
    )
    assert "env.include_gravity_magnitude=false" in argv
    assert "env.include_physics_context=true" in argv
    assert "agent.obs.physics_keys=[physics]" in argv
    assert "agent.physics_film=true" in argv
    assert "agent.physics_stratified_replay=true" in argv
    assert "env.physics_conditioning.schedule_mode=stratified_cycle" in argv
    assert "env.physics_conditioning.gravity_interval_s=[2.56,2.56]" in argv
