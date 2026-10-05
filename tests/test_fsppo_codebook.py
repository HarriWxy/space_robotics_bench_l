"""CPU contracts for exact fixed-codebook joint and marginal PPO likelihoods."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.distributions import Categorical, Independent, MixtureSameFamily, Normal

pytest.importorskip("isaaclab_fpo")

from isaaclab_fpo.algorithms import FSPPOCodebook
from isaaclab_fpo.algorithms.codebook_diagnostics import codebook_diagnostics
from isaaclab_fpo.modules import CodebookPMFActorCritic
from isaaclab_fpo.rl_cfg import (
    FpoRslRlOnPolicyRunnerCfg,
    FpoRslRlPpoActorCriticCfg,
    FpoRslRlPpoAlgorithmCfg,
)
from isaaclab_fpo.runners import OnPolicyRunner


@pytest.fixture(autouse=True)
def cpu_execution(monkeypatch: pytest.MonkeyPatch):
    """Use eager CPU execution without leaking precision, thread, or RNG state."""
    previous_threads = torch.get_num_threads()
    previous_precision = torch.get_float32_matmul_precision()
    previous_rng = torch.get_rng_state()
    monkeypatch.setattr(torch, "compile", lambda fn, *args, **kwargs: fn)
    torch.set_num_threads(1)
    torch.manual_seed(31)
    yield
    torch.set_num_threads(previous_threads)
    torch.set_float32_matmul_precision(previous_precision)
    torch.set_rng_state(previous_rng)


def make_policy_cfg(**overrides) -> FpoRslRlPpoActorCriticCfg:
    settings = {
        "class_name": "CodebookPMFActorCritic",
        "actor_hidden_dims": [12, 8],
        "critic_hidden_dims": [8, 8],
        "activation": "elu",
        "sampling_steps": 1,
        "action_perturb_std": 0.25,
        "actor_scale": 0.5,
        "codebook_size": 3,
        "codebook_seed": 12345,
    }
    settings.update(overrides)
    return FpoRslRlPpoActorCriticCfg(**settings)


def make_policy(**overrides) -> CodebookPMFActorCritic:
    return CodebookPMFActorCritic(3, 3, 2, make_policy_cfg(**overrides))


@pytest.mark.parametrize(
    ("sigma", "action_values"),
    [
        (0.25, [[0.10, -0.20], [0.45, 0.25]]),
        (0.03125, [[10.0, -8.0], [-9.0, 13.0]]),
    ],
    ids=["overlapping-components", "small-noise-far-tail"],
)
def test_log_densities_and_gradients_match_independent_distribution_oracle(
    sigma: float, action_values: list[list[float]]
) -> None:
    """Protect mixture normalization, dimension reduction, and all-component gradients."""
    policy = make_policy(action_perturb_std=sigma).double()
    actions = torch.tensor(action_values, dtype=torch.float64, requires_grad=True)
    means = torch.tensor(
        [
            [[-0.20, 0.10], [0.10, -0.10], [0.35, 0.20]],
            [[0.30, 0.00], [0.55, 0.40], [0.20, 0.35]],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    actual_components = policy.component_action_log_probs(actions, means)
    actual_marginal = policy.marginal_action_log_prob(actions, means)

    reference_actions = actions.detach().clone().requires_grad_()
    reference_means = means.detach().clone().requires_grad_()
    components = Independent(Normal(reference_means, sigma), 1)
    mixture = MixtureSameFamily(
        Categorical(probs=torch.full((2, 3), 1.0 / 3.0, dtype=torch.float64)),
        components,
    )
    expected_components = components.log_prob(reference_actions[:, None, :])
    expected_marginal = mixture.log_prob(reference_actions)
    torch.testing.assert_close(actual_components, expected_components)
    torch.testing.assert_close(actual_marginal[:, 0], expected_marginal)
    assert torch.isfinite(actual_marginal).all()

    actual_gradients = torch.autograd.grad(actual_marginal.sum(), (actions, means))
    expected_gradients = torch.autograd.grad(expected_marginal.sum(), (reference_actions, reference_means))
    for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, expected)
        assert torch.isfinite(actual).all()


def test_component_permutation_changes_joint_ratio_but_preserves_marginal_ratio() -> None:
    """Relabelling identical action distributions must affect only the joint objective."""
    policy = make_policy(codebook_size=2).double()
    actions = torch.tensor([[-0.55, 0.20]], dtype=torch.float64)
    old_means = torch.tensor([[[-0.60, 0.25], [0.45, -0.30]]], dtype=torch.float64)
    new_means = old_means[:, [1, 0]]
    selected_component = 0

    old_component_logp = policy.component_action_log_probs(actions, old_means)
    new_component_logp = policy.component_action_log_probs(actions, new_means)
    joint_ratio = (new_component_logp[:, selected_component] - old_component_logp[:, selected_component]).exp()
    marginal_ratio = (
        policy.marginal_action_log_prob(actions, new_means) - policy.marginal_action_log_prob(actions, old_means)
    ).exp()

    torch.testing.assert_close(marginal_ratio, torch.ones_like(marginal_ratio))
    assert not torch.allclose(joint_ratio, torch.ones_like(joint_ratio))
    reference_old = Independent(Normal(old_means[:, 0], 0.25), 1)
    reference_new = Independent(Normal(new_means[:, 0], 0.25), 1)
    torch.testing.assert_close(joint_ratio, (reference_new.log_prob(actions) - reference_old.log_prob(actions)).exp())


@pytest.mark.parametrize("density_offset", [0.0, -1.0e6])
def test_posterior_and_clipping_diagnostics_resolve_a_component_permutation(
    density_offset: float,
) -> None:
    """A fixed marginal can have positive posterior KL and clipping cost for either sign."""
    old_weights = torch.tensor([[0.9, 0.1], [0.1, 0.9]], dtype=torch.float64)
    new_weights = old_weights[:, [1, 0]]
    old_logp = old_weights.log() + density_offset
    new_logp = new_weights.log() + density_offset
    diagnostics = codebook_diagnostics(old_logp, new_logp, torch.tensor([[1.0], [-2.0]], dtype=torch.float64), 0.2)
    expected_posterior_kl = torch.distributions.kl_divergence(
        Categorical(probs=old_weights), Categorical(probs=new_weights)
    ).mean()
    torch.testing.assert_close(diagnostics["codebook/posterior_kl"], expected_posterior_kl, atol=1.0e-9, rtol=1.0e-9)
    torch.testing.assert_close(
        diagnostics["codebook/posterior_entropy"],
        Categorical(probs=old_weights).entropy().mean(),
        atol=1.0e-9,
        rtol=1.0e-9,
    )
    # Positive row: marginal=1, posterior-averaged clipped joint=0.22.
    # Negative row: marginal=-2, posterior-averaged clipped joint=-3.24.
    torch.testing.assert_close(
        diagnostics["codebook/clip_jensen_gap"],
        torch.tensor(1.01, dtype=torch.float64),
        atol=1.0e-9,
        rtol=1.0e-9,
    )
    assert diagnostics["codebook/marginal_kl_k3"] == 0.0
    assert diagnostics["codebook/marginal_kl_mc"] == 0.0
    assert all(torch.isfinite(value) for value in diagnostics.values())


def test_single_component_has_no_posterior_or_clipping_gap_even_after_an_update() -> None:
    """Changes to a Gaussian density cannot create a latent-assignment penalty."""
    old_logp = torch.tensor([[-1.0], [-1000.0]], dtype=torch.float64)
    new_logp = torch.tensor([[-1.5], [-999.75]], dtype=torch.float64)
    result = codebook_diagnostics(old_logp, new_logp, torch.tensor([[2.0], [-3.0]], dtype=torch.float64), 0.2)
    assert result["codebook/posterior_entropy"] == 0.0
    assert result["codebook/posterior_kl"] == 0.0
    assert result["codebook/clip_jensen_gap"] == 0.0
    assert result["codebook/posterior_max_probability"] == 1.0
    assert result["codebook/effective_components"] == 1.0
    assert result["codebook/marginal_kl_k3"] > 0.0


def test_single_component_matches_gaussian_likelihood_and_gradient() -> None:
    """K=1 has the same objective in both modes, including its policy gradient."""
    policy = make_policy(codebook_size=1).double()
    actions = torch.tensor([[0.20, -0.15], [-0.10, 0.35]], dtype=torch.float64)
    means = torch.tensor([[[0.0, 0.10]], [[0.20, -0.10]]], dtype=torch.float64, requires_grad=True)
    conditional = policy.component_action_log_probs(actions, means)[:, 0]
    marginal = policy.marginal_action_log_prob(actions, means)[:, 0]
    reference = Independent(Normal(means[:, 0], 0.25), 1).log_prob(actions)
    torch.testing.assert_close(conditional, reference)
    torch.testing.assert_close(marginal, reference)
    conditional_gradient = torch.autograd.grad(conditional.sum(), means, retain_graph=True)[0]
    marginal_gradient = torch.autograd.grad(marginal.sum(), means, retain_graph=True)[0]
    reference_gradient = torch.autograd.grad(reference.sum(), means)[0]
    torch.testing.assert_close(conditional_gradient, reference_gradient)
    torch.testing.assert_close(marginal_gradient, reference_gradient)


def test_codebook_initialization_has_an_independent_rng_and_persistent_identity() -> None:
    """Changing K or its seed must not change network initialization or rollout RNG."""
    torch.manual_seed(73)
    first = make_policy(codebook_size=3, codebook_seed=11)
    first_rng = torch.get_rng_state().clone()
    torch.manual_seed(73)
    second = make_policy(codebook_size=7, codebook_seed=17)
    assert torch.equal(torch.get_rng_state(), first_rng)
    for name, value in first.state_dict().items():
        if name.startswith(("actor.", "critic.")):
            assert torch.equal(value, second.state_dict()[name])

    torch.manual_seed(999)
    repeated = make_policy(codebook_size=3, codebook_seed=11)
    assert torch.equal(first.latent_codebook, repeated.latent_codebook)
    assert not first.latent_codebook.requires_grad
    assert "latent_codebook" in first.state_dict()
    assert "codebook_noise_std" in first.state_dict()


class SmallEnv:
    """Provide observations and action-dependent rewards without a simulator."""

    num_actions = 2
    num_envs = 3
    device = "cpu"
    max_episode_length = 10

    def __init__(self) -> None:
        self.unwrapped = self
        self.cfg = SimpleNamespace()
        self.common_step_counter = 0
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)

    def get_observations(self):
        obs = torch.tensor([[0.10, -0.20, 0.30], [0.20, 0.10, -0.10], [-0.30, 0.25, 0.15]])
        obs = obs + self.common_step_counter * 0.01
        return obs, {"observations": {"critic": obs}}

    def step(self, actions: torch.Tensor):
        self.common_step_counter += 1
        self.episode_length_buf += 1
        obs, infos = self.get_observations()
        target = actions.new_tensor([0.20, -0.15])
        reward = -(actions - target).square().sum(dim=-1)
        return obs, reward, torch.zeros(self.num_envs, dtype=torch.bool), infos


def make_algorithm_cfg(mode: str, **overrides) -> FpoRslRlPpoAlgorithmCfg:
    settings = {
        "class_name": "FSPPOCodebook",
        "fsppo_codebook_ratio_mode": mode,
        "num_learning_epochs": 2,
        "num_mini_batches": 3,
        "learning_rate": 1.0e-3,
        "weight_decay": 0.0,
        "schedule": "fixed",
        "trust_region_mode": "ppo",
        "clip_param": 0.2,
        "normalize_advantage": False,
        "knn_entropy_coef": 0.0,
        "storage_action_noise_std": 0.0,
        "fsppo_joint_enable_budget": False,
        "fsppo_joint_aux_loss_coef": 0.0,
        "action_perturb_std_final": 0.125,
        "action_perturb_std_decay_env_steps": 4096,
        "ema_decay": 0.0,
    }
    settings.update(overrides)
    return FpoRslRlPpoAlgorithmCfg(**settings)


def test_algorithm_uses_the_selected_ratio_for_clipped_loss_and_actor_gradient() -> None:
    """Catch a mode switch ignored by the learner even when density helpers are correct."""
    policy = make_policy(actor_hidden_dims=[2], activation="identity", codebook_size=2, codebook_seed=11).double()
    with torch.no_grad():
        for parameter in policy.actor.parameters():
            parameter.zero_()
        # Both components overlap: their public means differ only through a
        # small projection of the actual fixed code vectors.
        policy.actor[0].weight[0, -2] = 1.0
        policy.actor[-1].weight[0, 0] = 0.20
        policy.actor[-1].weight[1, 0] = -0.10
    observations = torch.zeros(2, 3, dtype=torch.float64)
    indices = torch.tensor([0, 1])
    old_means = policy.component_means(observations).detach()
    actions = old_means[torch.arange(2), indices] + torch.tensor([[0.10, -0.12], [-0.15, 0.10]], dtype=torch.float64)
    old_components = Independent(Normal(old_means, 0.25), 1).log_prob(actions[:, None, :])
    old_marginal = MixtureSameFamily(
        Categorical(probs=torch.full((2, 2), 0.5, dtype=torch.float64)),
        Independent(Normal(old_means, 0.25), 1),
    ).log_prob(actions)
    batch = {
        "obs": observations,
        "critic_obs": observations,
        "actions": actions,
        "component_indices": indices,
        "action_log_probs": old_components.gather(1, indices[:, None]),
        "old_component_log_probs": old_components,
        "advantages": torch.tensor([[1.0], [-0.8]], dtype=torch.float64),
        "returns": torch.zeros(2, 1, dtype=torch.float64),
        "values": torch.zeros(2, 1, dtype=torch.float64),
    }
    losses, gradients = {}, {}
    for mode in ("joint", "marginal"):
        algorithm = FSPPOCodebook(copy.deepcopy(policy), make_algorithm_cfg(mode, value_loss_coef=0.0))
        with torch.no_grad():
            algorithm.policy.actor[-1].weight[0, 0] = 0.28
            algorithm.policy.actor[-1].weight[1, 0] = -0.12
            algorithm.policy.actor[-1].bias[:2] = torch.tensor([0.08, -0.04])
        actual_loss, _ = algorithm._batch_loss(batch)
        means = algorithm.policy.component_means(observations)
        if mode == "joint":
            new_distribution = Independent(Normal(means[torch.arange(2), indices], 0.25), 1)
            old_density = batch["action_log_probs"][:, 0]
        else:
            new_distribution = MixtureSameFamily(
                Categorical(probs=torch.full((2, 2), 0.5, dtype=torch.float64)),
                Independent(Normal(means, 0.25), 1),
            )
            old_density = old_marginal
        oracle_ratio = (new_distribution.log_prob(actions) - old_density).exp()
        advantage = batch["advantages"][:, 0]
        expected_loss = -torch.minimum(
            oracle_ratio * advantage,
            oracle_ratio.clamp(0.8, 1.2) * advantage,
        ).mean()
        torch.testing.assert_close(actual_loss, expected_loss)
        parameters = tuple(algorithm.policy.actor.parameters())
        actual_gradient = torch.autograd.grad(actual_loss, parameters)
        expected_gradient = torch.autograd.grad(expected_loss, parameters)
        for actual, expected in zip(actual_gradient, expected_gradient, strict=True):
            torch.testing.assert_close(actual, expected)
        losses[mode] = actual_loss.detach()
        gradients[mode] = torch.cat([gradient.flatten() for gradient in actual_gradient])

    assert not torch.allclose(losses["joint"], losses["marginal"])
    assert not torch.allclose(gradients["joint"], gradients["marginal"])


def make_runner(
    mode: str,
    *,
    policy_overrides: dict | None = None,
    algorithm_overrides: dict | None = None,
) -> OnPolicyRunner:
    cfg = FpoRslRlOnPolicyRunnerCfg(
        policy=make_policy_cfg(**(policy_overrides or {})),
        algorithm=make_algorithm_cfg(mode, **(algorithm_overrides or {})),
        num_steps_per_env=3,
        max_iterations=1,
        save_interval=1,
        empirical_normalization=False,
        enable_post_training_eval=False,
        logger="tensorboard",
    )
    runner = OnPolicyRunner(SmallEnv(), cfg, device="cpu")
    runner.logger_type = "tensorboard"
    return runner


def collect_rollout(runner: OnPolicyRunner) -> None:
    obs, _ = runner.env.get_observations()
    runner.alg.begin_rollout(runner.tot_timesteps)
    for _ in range(runner.num_steps_per_env):
        actions = runner.alg.act(obs, obs)
        obs, rewards, dones, infos = runner.env.step(actions)
        runner.alg.process_env_step(rewards, dones, infos)
    runner.alg.compute_returns(obs)


def assert_tree_equal(actual, expected) -> None:
    if isinstance(expected, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_tree_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            assert_tree_equal(a, b)
    else:
        assert actual == expected


@pytest.mark.parametrize("mode", ["joint", "marginal"])
def test_real_runner_update_preserves_behavior_density_and_checkpoint_contract(mode: str, tmp_path: Path) -> None:
    """Exercise both objective branches and resume their actual sampling state."""
    torch.manual_seed(47)
    runner = make_runner(mode)
    rng_before_rollout = torch.get_rng_state().clone()
    collect_rollout(runner)
    torch.manual_seed(47)
    other_mode = make_runner("marginal" if mode == "joint" else "joint")
    assert_tree_equal(other_mode.alg.policy.state_dict(), runner.alg.policy.state_dict())
    torch.set_rng_state(rng_before_rollout)
    collect_rollout(other_mode)
    assert torch.equal(other_mode.alg.storage.actions, runner.alg.storage.actions)
    assert torch.equal(other_mode.alg.storage.component_indices, runner.alg.storage.component_indices)
    storage = runner.alg.storage
    observations = storage.observations.flatten(0, 1).clone()
    actions = storage.actions.flatten(0, 1).clone()
    indices = storage.component_indices.flatten().clone()
    old_logp = storage.action_log_probs.clone()
    old_means = runner.alg.policy.component_means(observations)
    oracle = MixtureSameFamily(
        Categorical(probs=torch.full((9, 3), 1.0 / 3.0)),
        Independent(Normal(old_means, 0.25), 1),
    )
    expected_old_components = Independent(Normal(old_means, 0.25), 1).log_prob(actions[:, None, :]).detach().clone()
    selected_means = old_means[torch.arange(9), indices]
    expected_old_logp = Independent(Normal(selected_means, 0.25), 1).log_prob(actions)
    torch.testing.assert_close(old_logp.flatten(), expected_old_logp)
    expected_latents = runner.alg.policy.latent_codebook[indices]
    assert torch.equal(storage.action_latents.flatten(0, 1), expected_latents)

    actor_before = copy.deepcopy(runner.alg.policy.actor.state_dict())
    codebook_before = runner.alg.policy.latent_codebook.clone()
    results = runner.alg.update()
    assert results["metrics"]["joint/accepted_updates"] == 6
    assert storage.step == 0
    assert torch.equal(storage.action_log_probs, old_logp)
    torch.testing.assert_close(storage.old_component_log_probs.flatten(0, 1), expected_old_components)
    torch.testing.assert_close(
        torch.logsumexp(storage.old_component_log_probs.flatten(0, 1), dim=-1) - math.log(3),
        oracle.log_prob(actions),
    )
    assert torch.equal(runner.alg.policy.latent_codebook, codebook_before)
    assert any(
        not torch.equal(value, actor_before[name]) for name, value in runner.alg.policy.actor.state_dict().items()
    )
    assert all(torch.isfinite(parameter).all() for parameter in runner.alg.policy.parameters())
    gradients = [p.grad for p in runner.alg.policy.actor.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    for group in ({k: v for k, v in results.items() if k != "metrics"}, results["metrics"]):
        assert all(math.isfinite(float(value)) for value in group.values())

    runner.current_learning_iteration = 1
    # Save a noninitial noise scale at the next rollout boundary.
    runner.alg.begin_rollout(runner.env.num_envs * runner.num_steps_per_env)
    assert runner.alg.sigma < 0.25
    path = tmp_path / "codebook.pt"
    runner.save(str(path))
    restored = make_runner(mode)
    restored.load(str(path))
    assert_tree_equal(restored.alg.policy.state_dict(), runner.alg.policy.state_dict())
    assert_tree_equal(restored.alg.optimizer.state_dict(), runner.alg.optimizer.state_dict())
    assert_tree_equal(restored.alg.state_dict(), runner.alg.state_dict())
    assert restored.current_learning_iteration == 1
    assert restored.alg.sigma == runner.alg.sigma
    assert restored.alg.policy.action_perturb_std == runner.alg.policy.action_perturb_std

    model_only = make_runner(mode)
    model_only.current_learning_iteration = 7
    model_only.load(str(path), load_optimizer=False)
    assert model_only.current_learning_iteration == 7
    assert model_only.alg.policy.action_perturb_std == runner.alg.sigma
    torch.testing.assert_close(
        model_only.alg.policy.act_inference(
            observations, eval_mode="fixed_seed", eval_fixed_seed=91
        ),
        runner.alg.policy.act_inference(
            observations, eval_mode="fixed_seed", eval_fixed_seed=91
        ),
    )

    before_rejected_load = copy.deepcopy(other_mode.alg.policy.state_dict())
    with pytest.raises(ValueError):
        other_mode.load(str(path))
    assert_tree_equal(other_mode.alg.policy.state_dict(), before_rejected_load)
    for policy_overrides, algorithm_overrides in (
        ({"codebook_size": 4}, {}),
        ({"codebook_seed": 7}, {}),
        ({}, {"action_perturb_std_decay_env_steps": 8192}),
    ):
        incompatible = make_runner(mode, policy_overrides=policy_overrides, algorithm_overrides=algorithm_overrides)
        before_rejected_load = copy.deepcopy(incompatible.alg.policy.state_dict())
        with pytest.raises(ValueError):
            incompatible.load(str(path))
        assert_tree_equal(incompatible.alg.policy.state_dict(), before_rejected_load)


def test_evaluation_samples_the_actual_codebook_mixture_and_fixed_seed_repeats() -> None:
    """Evaluation must include the same components and Gaussian noise as training."""
    policy = make_policy().eval()
    observations = torch.tensor([[0.10, -0.20, 0.30], [-0.40, 0.50, 0.60]])
    torch.manual_seed(83)
    reference_actions, indices, latents, means, conditional_logp = policy.sample_codebook_transport(observations)
    assert torch.equal(latents, policy.latent_codebook[indices])
    torch.testing.assert_close(means, policy.component_means(observations)[torch.arange(2), indices])
    torch.testing.assert_close(conditional_logp[:, 0], Independent(Normal(means, 0.25), 1).log_prob(reference_actions))
    torch.manual_seed(83)
    torch.testing.assert_close(policy.act_inference(observations, eval_mode="random"), reference_actions)

    seed = 29
    generator = torch.Generator(device=observations.device).manual_seed(seed)
    expected_fixed = policy.sample_codebook_transport(observations, generator=generator)[0]
    first = policy.act_inference(observations, eval_mode="fixed_seed", eval_fixed_seed=seed)
    torch.randn(17)
    second = policy.act_inference(observations, eval_mode="fixed_seed", eval_fixed_seed=seed)
    torch.testing.assert_close(first, expected_fixed)
    assert torch.equal(first, second)
    torch.testing.assert_close(
        policy.act_inference(observations, eval_mode="mean"),
        policy.component_means(observations).mean(dim=1),
    )
