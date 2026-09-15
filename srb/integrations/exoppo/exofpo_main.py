"""Run ExO-FPO: ExO recent-policy optimization with FPO flow matching."""

from __future__ import annotations

import random
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import gymnasium
import numpy as np
import torch

from srb.integrations.exoppo.exofpo_wrapper import SrbExoFpoEnvWrapper
from srb.integrations.exoppo.flow_matching import (
    ExoFpoTrainer,
    FlowMatchingReplayWindow,
    flatten_flow_matching_rollout,
)
from srb.integrations.exoppo.main import (
    _accumulate_scalar_batch,
    _as_dict,
    _build_flow_config,
    _iter_task_metrics,
    _load_checkpoint,
    _normalize,
    _resolve_checkpoint,
    _save_checkpoint,
    _summary_writer,
    _write_run_manifest,
)
from srb.integrations.tensorboard import write_scalars
from srb.utils import logging
from srb.wrappers import maybe_wrap_action_smoothing

if TYPE_CHECKING:
    from isaacsim.simulation_app import SimulationApp

    from srb._typing import AnyEnv, AnyEnvCfg


FRAMEWORK_NAME = "exofpo"
ALGORITHM_NAME = "ExO-FPO"


@dataclass(frozen=True)
class ExoFpoConfig:
    """Flat config accepted by the SRB Hydra registry and ExO-FPO trainer."""

    env_id: str = ""
    env_backend: str = "gymnasium"
    seed: int = 0
    total_steps: int = 1_000_000
    num_envs: int = 256
    rollout_steps: int = 32
    replay_N: int = 4
    warmup_rollouts: int = 4
    update_epochs: int = 2
    batch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95

    hidden_sizes: tuple[int, ...] = (256, 256, 256)
    initial_log_std: float = -1.0
    actor_learning_rate: float = 1.0e-4
    critic_learning_rate: float = 2.0e-4
    max_grad_norm: float = 1.0

    exo_clip_radius: float = 0.05
    exo_beta: float = 5.0
    entropy_coefficient: float = 0.0
    max_log_ratio: float = 12.0
    target_kl: float = 0.0

    ofp_coefficient: float = 0.1
    flow_mix: float = 0.8
    consistency_mix: float = 0.2
    guidance_mix: float = 0.0
    guidance_scale: float = 1.0
    condition_dropout: float = 0.0
    contraction_steps: int = 50_000
    minimum_contraction: float = 0.05
    ema_initial_decay: float = 0.75
    ema_max_decay: float = 0.9999
    ema_ramp_steps: int = 10_000

    warm_start_time: float = 0.0
    eval_every_rollouts: int = 20
    eval_episodes: int = 5
    stochastic_eval: bool = False
    device: str = "auto"
    envpool_num_threads: int = 0
    log_dir: str = "logs"
    checkpoint_dir: str = ""

    # FPO-compatible flow generator and CFM objective.
    activation: str = "elu"
    actor_scale: float = 1.0
    mlp_output_scale: float = 1.0
    actor_final_layer_weight_scale: float | None = None
    timestep_embed_dim: int = 8
    training_sampling_steps: int | None = None
    sampling_steps: int = 8
    cfm_loss_t_inverse_cdf_beta: float = 1.0
    cfm_loss_reduction: str = "sqrt"
    action_perturb_std: float = 0.02
    n_samples_per_action: int = 16
    cfm_diff_clamp_max: float = 10.0
    # Disabled by default because a hard CFM-loss ceiling can saturate the
    # ratio. It remains configurable for exact FPO-style ablations.
    cfm_loss_clamp: float = -1.0
    cfm_loss_clamp_negative_advantages: bool = False
    cfm_loss_clamp_negative_advantages_max: float = 20.0
    clip_actions: float | None = 1.0
    normalize_advantage: bool = True
    advantage_clamp: tuple[float, float] | None = (100.0, 100.0)


def validate_config(config: ExoFpoConfig) -> None:
    if config.env_backend not in {"gymnasium", "envpool"}:
        raise ValueError("env_backend must be 'gymnasium' or 'envpool'")
    positive_integer_fields = (
        "total_steps",
        "num_envs",
        "rollout_steps",
        "replay_N",
        "warmup_rollouts",
        "update_epochs",
        "batch_size",
        "contraction_steps",
        "ema_ramp_steps",
        "eval_every_rollouts",
        "eval_episodes",
        "sampling_steps",
        "n_samples_per_action",
    )
    for field_name in positive_integer_fields:
        if getattr(config, field_name) <= 0:
            raise ValueError(f"{field_name} must be positive")
    if config.warmup_rollouts > config.replay_N:
        raise ValueError("warmup_rollouts cannot exceed replay_N")
    if config.envpool_num_threads < 0:
        raise ValueError("envpool_num_threads must be non-negative")
    if not 0.0 < config.gamma <= 1.0 or not 0.0 <= config.gae_lambda <= 1.0:
        raise ValueError("gamma and gae_lambda must be in their valid ranges")
    if config.actor_learning_rate <= 0.0 or config.critic_learning_rate <= 0.0:
        raise ValueError("learning rates must be positive")
    if config.max_grad_norm <= 0.0:
        raise ValueError("max_grad_norm must be positive")
    if config.exo_clip_radius <= 0.0 or config.exo_beta <= 0.0:
        raise ValueError("ExO clip radius and beta must be positive")
    if config.max_log_ratio <= 0.0 or config.cfm_diff_clamp_max <= 0.0:
        raise ValueError("ratio clamps must be positive")
    if not 0.0 <= config.minimum_contraction <= 1.0:
        raise ValueError("minimum_contraction must be in [0, 1]")
    if not 0.0 <= config.ema_initial_decay <= config.ema_max_decay < 1.0:
        raise ValueError("EMA decays must satisfy 0 <= initial <= max < 1")
    if config.cfm_loss_t_inverse_cdf_beta <= 0.0:
        raise ValueError("cfm_loss_t_inverse_cdf_beta must be positive")
    if config.cfm_loss_clamp == 0.0 or config.cfm_loss_clamp < -1.0:
        raise ValueError("cfm_loss_clamp must be -1 or positive")
    if config.cfm_loss_clamp_negative_advantages_max <= 0.0:
        raise ValueError("negative-advantage CFM clamp must be positive")
    if config.clip_actions is not None and config.clip_actions <= 0.0:
        raise ValueError("clip_actions must be positive or null")
    if config.entropy_coefficient != 0.0:
        raise ValueError(
            "ExO-FPO has no Gaussian residual entropy term; set entropy_coefficient=0"
        )
    if config.warm_start_time != 0.0:
        raise ValueError(
            "ExO-FPO follows FPO's fresh-noise sampler; set warm_start_time=0"
        )
    if config.guidance_mix != 0.0 or config.condition_dropout != 0.0:
        raise ValueError(
            "ExO-FPO currently uses FPO's unconditional actor; set "
            "guidance_mix=0 and condition_dropout=0"
        )
    if config.advantage_clamp is not None and (
        len(config.advantage_clamp) != 2
        or any(value <= 0.0 for value in config.advantage_clamp)
    ):
        raise ValueError("advantage_clamp must contain two positive values")


def _collect_rollout(
    *,
    wrapped_env: SrbExoFpoEnvWrapper,
    sim_app: SimulationApp,
    trainer: ExoFpoTrainer,
    actor_observation: torch.Tensor,
    critic_observation: torch.Tensor,
    actor_stats: Any,
    critic_stats: Any,
    empirical_normalization: bool,
    observation_clip: float,
    episode_returns: torch.Tensor,
    episode_lengths: torch.Tensor,
    compute_gae: Any,
) -> tuple[Any | None, torch.Tensor, torch.Tensor, dict[str, float], int]:
    actor_observations: list[torch.Tensor] = []
    critic_observations: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    cfm_loss_eps: list[torch.Tensor] = []
    cfm_loss_t: list[torch.Tensor] = []
    initial_cfm_loss: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []
    terminated_values: list[torch.Tensor] = []
    truncated_values: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    next_values: list[torch.Tensor] = []
    completed_return_sum = torch.zeros((), device=trainer.device)
    completed_length_sum = torch.zeros((), device=trainer.device)
    completed_count = torch.zeros((), device=trainer.device)
    reward_term_sums: dict[str, torch.Tensor] = {}
    reward_term_counts: dict[str, int] = {}
    task_metric_sums: dict[str, torch.Tensor] = {}
    task_metric_counts: dict[str, int] = {}
    episode_event_sums: dict[str, torch.Tensor] = {}

    trainer.policy.eval()
    trainer.value.eval()
    for _ in range(trainer.config.rollout_steps):
        if not sim_app.is_running():
            break
        with torch.no_grad():
            sample = trainer.sample_rollout(actor_observation)
            value = trainer.value(critic_observation)
            (
                next_raw_actor,
                next_raw_critic,
                reward,
                terminated,
                truncated,
                extras,
            ) = wrapped_env.step(sample.action)
            next_actor = _normalize(
                next_raw_actor,
                actor_stats,
                enabled=empirical_normalization,
                clip=observation_clip,
                update=True,
            )
            next_critic = _normalize(
                next_raw_critic,
                critic_stats,
                enabled=empirical_normalization,
                clip=observation_clip,
                update=True,
            )
            next_value = trainer.value(next_critic)

            reward_terms = extras.get("reward_terms")
            if isinstance(reward_terms, Mapping):
                for name, term in reward_terms.items():
                    _accumulate_scalar_batch(
                        reward_term_sums,
                        reward_term_counts,
                        str(name),
                        term,
                        device=trainer.device,
                    )
            for name, metric in _iter_task_metrics(extras):
                metric_name = name.removeprefix("metrics/")
                if metric_name.startswith("episode_"):
                    event_values = torch.as_tensor(
                        metric, dtype=torch.float32, device=trainer.device
                    ).reshape(-1)
                    event_values = event_values[torch.isfinite(event_values)]
                    if event_values.numel() > 0:
                        episode_event_sums[metric_name] = (
                            episode_event_sums.get(
                                metric_name, torch.zeros((), device=trainer.device)
                            )
                            + event_values.sum()
                        )
                    continue
                _accumulate_scalar_batch(
                    task_metric_sums,
                    task_metric_counts,
                    metric_name,
                    metric,
                    device=trainer.device,
                )

            if wrapped_env.bootstrap_truncated:
                next_value = torch.where(truncated, value, next_value)
                final_observations = wrapped_env.final_observations(extras)
                if final_observations is not None:
                    _, final_raw_critic = final_observations
                    final_critic = _normalize(
                        final_raw_critic,
                        critic_stats,
                        enabled=empirical_normalization,
                        clip=observation_clip,
                        update=False,
                    )
                    final_value = trainer.value(final_critic)
                    next_value = torch.where(truncated, final_value, next_value)

        actor_observations.append(actor_observation)
        critic_observations.append(critic_observation)
        actions.append(sample.action)
        cfm_loss_eps.append(sample.cfm_loss_eps)
        cfm_loss_t.append(sample.cfm_loss_t)
        initial_cfm_loss.append(sample.initial_cfm_loss)
        rewards.append(reward)
        terminated_values.append(terminated)
        truncated_values.append(truncated)
        values.append(value)
        next_values.append(next_value)

        episode_returns += reward
        episode_lengths += 1
        done = terminated | truncated
        done_float = done.to(dtype=torch.float32)
        completed_return_sum += (episode_returns * done_float).sum()
        completed_length_sum += (episode_lengths * done_float).sum()
        completed_count += done_float.sum()
        episode_returns.masked_fill_(done, 0.0)
        episode_lengths.masked_fill_(done, 0)
        actor_observation = next_actor
        critic_observation = next_critic

    step_count = len(rewards)
    if step_count == 0:
        return None, actor_observation, critic_observation, {}, 0

    reward_tensor = torch.stack(rewards)
    terminated_tensor = torch.stack(terminated_values)
    truncated_tensor = torch.stack(truncated_values)
    value_tensor = torch.stack(values)
    next_value_tensor = torch.stack(next_values)
    advantages, returns = compute_gae(
        reward_tensor,
        terminated_tensor,
        truncated_tensor,
        value_tensor,
        next_value_tensor,
        gamma=trainer.config.gamma,
        gae_lambda=trainer.config.gae_lambda,
        bootstrap_truncated=wrapped_env.bootstrap_truncated,
    )
    rollout = flatten_flow_matching_rollout(
        torch.stack(actor_observations),
        torch.stack(critic_observations),
        torch.stack(actions),
        torch.stack(cfm_loss_eps),
        torch.stack(cfm_loss_t),
        torch.stack(initial_cfm_loss),
        advantages,
        returns,
    )
    count = float(completed_count.cpu())
    metrics: dict[str, float] = {
        "rollout/cfm_initial_loss": float(torch.stack(initial_cfm_loss).mean().cpu()),
    }
    if trainer.config.cfm_loss_clamp > 0.0:
        metrics["rollout/cfm_initial_clamp_fraction"] = float(
            (torch.stack(initial_cfm_loss) >= trainer.config.cfm_loss_clamp)
            .float()
            .mean()
            .cpu()
        )
    if count > 0.0:
        metrics["rollout/ep_rew_mean"] = float(completed_return_sum.cpu()) / count
        metrics["rollout/ep_len_mean"] = float(completed_length_sum.cpu()) / count

    for name, term_sum in reward_term_sums.items():
        metrics[f"rollout/reward_terms/{name}"] = (
            float(term_sum.cpu()) / reward_term_counts[name]
        )
    for name, metric_sum in task_metric_sums.items():
        metrics[f"rollout/metrics/{name}"] = (
            float(metric_sum.cpu()) / task_metric_counts[name]
        )
    event_zero = torch.zeros((), device=trainer.device)
    completed = float(episode_event_sums.get("episode_completed", event_zero).cpu())
    if completed > 0.0:
        metrics["rollout/metrics/episode_completed"] = completed
        metrics["rollout/episode_success_rate"] = (
            float(episode_event_sums.get("episode_success", event_zero).cpu())
            / completed
        )
        metrics["rollout/episode_failure_rate"] = (
            float(episode_event_sums.get("episode_failed", event_zero).cpu())
            / completed
        )
        metrics["rollout/episode_tracking_fraction"] = (
            float(episode_event_sums.get("episode_tracking_fraction", event_zero).cpu())
            / completed
        )
        metrics["rollout/episode_duration_s"] = (
            float(episode_event_sums.get("episode_duration_s", event_zero).cpu())
            / completed
        )
        metrics["rollout/episode_torso_contact_rate"] = (
            float(
                episode_event_sums.get(
                    "episode_torso_contact_rate", event_zero
                ).cpu()
            )
            / completed
        )
        metrics["rollout/episode_foot_slip_speed"] = (
            float(
                episode_event_sums.get("episode_foot_slip_speed", event_zero).cpu()
            )
            / completed
        )
    return rollout, actor_observation, critic_observation, metrics, step_count


def _train(
    *,
    wrapped_env: SrbExoFpoEnvWrapper,
    sim_app: SimulationApp,
    trainer: ExoFpoTrainer,
    raw_cfg: Mapping[str, Any],
    logdir: Path,
    max_iterations: int,
    checkpoint: Path | None,
    normalizer_class: type,
    compute_gae: Any,
) -> None:
    empirical_normalization = bool(raw_cfg.get("empirical_normalization", True))
    observation_clip = float(raw_cfg.get("observation_clip", 10.0))
    save_interval = int(raw_cfg.get("save_interval", 50))
    log_interval = int(raw_cfg.get("log_interval", 1))
    if observation_clip <= 0.0 or save_interval <= 0 or log_interval <= 0:
        raise ValueError(
            "observation_clip, save_interval, and log_interval must be positive"
        )

    raw_actor, raw_critic, _ = wrapped_env.reset()
    actor_stats = normalizer_class((raw_actor.shape[1],), device=wrapped_env.device)
    critic_stats = normalizer_class((raw_critic.shape[1],), device=wrapped_env.device)
    start_iteration = 0
    environment_steps = 0
    if checkpoint is not None:
        logging.info(f"Loading {ALGORITHM_NAME} checkpoint from {checkpoint}")
        start_iteration, environment_steps = _load_checkpoint(
            checkpoint,
            trainer=trainer,
            actor_stats=actor_stats,
            critic_stats=critic_stats,
            load_optimizers=True,
            expected_algorithm=ALGORITHM_NAME,
        )

    actor_observation = _normalize(
        raw_actor,
        actor_stats,
        enabled=empirical_normalization,
        clip=observation_clip,
        update=True,
    )
    critic_observation = _normalize(
        raw_critic,
        critic_stats,
        enabled=empirical_normalization,
        clip=observation_clip,
        update=True,
    )
    if bool(raw_cfg.get("randomize_reset_episode_progress", True)):
        wrapped_env.episode_length_buf.random_(0, wrapped_env.max_episode_length)

    replay = FlowMatchingReplayWindow(trainer.config.replay_N)
    episode_returns = torch.zeros(
        wrapped_env.num_envs, dtype=torch.float32, device=wrapped_env.device
    )
    episode_lengths = torch.zeros(
        wrapped_env.num_envs, dtype=torch.long, device=wrapped_env.device
    )
    writer = _summary_writer(logdir)
    started = time.monotonic()
    last_iteration = start_iteration - 1
    replay_capacity_steps = (
        trainer.config.replay_N * trainer.config.rollout_steps * wrapped_env.num_envs
    )
    write_scalars(
        writer,
        {
            "config/seed": trainer.config.seed,
            "config/num_envs": wrapped_env.num_envs,
            "config/total_steps": trainer.config.total_steps,
            "config/rollout_steps": trainer.config.rollout_steps,
            "config/replay_n": trainer.config.replay_N,
            "config/replay_capacity_steps": replay_capacity_steps,
            "config/warmup_rollouts": trainer.config.warmup_rollouts,
            "config/gamma": trainer.config.gamma,
            "config/gae_lambda": trainer.config.gae_lambda,
            "config/sampling_steps": trainer.config.sampling_steps,
            "config/n_samples_per_action": trainer.config.n_samples_per_action,
            "config/actor_scale": trainer.config.actor_scale,
            "config/cfm_loss_clamp": trainer.config.cfm_loss_clamp,
        },
        step=0,
    )
    try:
        for iteration in range(start_iteration, max_iterations):
            (
                rollout,
                actor_observation,
                critic_observation,
                metrics,
                collected_steps,
            ) = _collect_rollout(
                wrapped_env=wrapped_env,
                sim_app=sim_app,
                trainer=trainer,
                actor_observation=actor_observation,
                critic_observation=critic_observation,
                actor_stats=actor_stats,
                critic_stats=critic_stats,
                empirical_normalization=empirical_normalization,
                observation_clip=observation_clip,
                episode_returns=episode_returns,
                episode_lengths=episode_lengths,
                compute_gae=compute_gae,
            )
            if rollout is None:
                break
            replay.append(rollout)
            environment_steps += collected_steps * wrapped_env.num_envs
            last_iteration = iteration
            metrics.update(
                {
                    "replay/samples": float(len(replay)),
                    "replay/rollouts": float(replay.rollout_count),
                    "time/fps": environment_steps
                    / max(time.monotonic() - started, 1e-6),
                }
            )
            if replay.rollout_count >= trainer.config.warmup_rollouts:
                metrics.update(
                    {
                        f"train/{key}": value
                        for key, value in trainer.train_torch_replay(replay).items()
                    }
                )
            write_scalars(writer, metrics, environment_steps)
            writer.flush()
            if (iteration + 1) % log_interval == 0:
                concise = {
                    key: round(value, 5)
                    for key, value in metrics.items()
                    if key
                    in {
                        "rollout/ep_rew_mean",
                        "train/actor_loss",
                        "train/critic_loss",
                        "train/cfm_loss",
                        "train/approx_kl",
                        "train/clip_fraction",
                        "train/ratio",
                        "train/flow_loss",
                    }
                }
                logging.info(
                    f"{ALGORITHM_NAME} iteration={iteration} "
                    f"steps={environment_steps} metrics={concise}"
                )
            if (iteration + 1) % save_interval == 0:
                _save_checkpoint(
                    logdir / f"model_{iteration}.pt",
                    trainer=trainer,
                    actor_stats=actor_stats,
                    critic_stats=critic_stats,
                    iteration=iteration,
                    environment_steps=environment_steps,
                )
    finally:
        if last_iteration >= 0:
            _save_checkpoint(
                logdir / f"model_{last_iteration}.pt",
                trainer=trainer,
                actor_stats=actor_stats,
                critic_stats=critic_stats,
                iteration=last_iteration,
                environment_steps=environment_steps,
            )
        writer.close()


def _evaluate(
    *,
    wrapped_env: SrbExoFpoEnvWrapper,
    sim_app: SimulationApp,
    trainer: ExoFpoTrainer,
    raw_cfg: Mapping[str, Any],
    checkpoint: Path | None,
    normalizer_class: type,
    logdir: Path,
) -> None:
    empirical_normalization = bool(raw_cfg.get("empirical_normalization", True))
    observation_clip = float(raw_cfg.get("observation_clip", 10.0))
    eval_steps = int(raw_cfg.get("eval_steps", 0))
    if eval_steps < 0:
        raise ValueError("eval_steps cannot be negative")

    raw_actor, raw_critic, _ = wrapped_env.reset()
    actor_stats = normalizer_class((raw_actor.shape[1],), device=wrapped_env.device)
    critic_stats = normalizer_class((raw_critic.shape[1],), device=wrapped_env.device)
    if checkpoint is not None:
        logging.info(f"Loading {ALGORITHM_NAME} checkpoint from {checkpoint}")
        _load_checkpoint(
            checkpoint,
            trainer=trainer,
            actor_stats=actor_stats,
            critic_stats=critic_stats,
            load_optimizers=False,
            expected_algorithm=ALGORITHM_NAME,
        )
    actor_observation = _normalize(
        raw_actor,
        actor_stats,
        enabled=empirical_normalization,
        clip=observation_clip,
        update=False,
    )
    trainer.policy.eval()
    episode_returns = torch.zeros(
        wrapped_env.num_envs, dtype=torch.float32, device=wrapped_env.device
    )
    episode_lengths = torch.zeros(
        wrapped_env.num_envs, dtype=torch.long, device=wrapped_env.device
    )
    completed_return_sum = torch.zeros((), device=wrapped_env.device)
    completed_length_sum = torch.zeros((), device=wrapped_env.device)
    completed_count = torch.zeros((), device=wrapped_env.device)
    step_reward_sum = torch.zeros((), device=wrapped_env.device)
    task_metric_sums: dict[str, torch.Tensor] = {}
    task_metric_counts: dict[str, int] = {}
    episode_event_sums: dict[str, torch.Tensor] = {}
    step = 0
    writer = _summary_writer(logdir)
    try:
        with torch.inference_mode():
            while sim_app.is_running() and (eval_steps == 0 or step < eval_steps):
                action = trainer.sample_evaluation(
                    actor_observation,
                    stochastic=bool(trainer.config.stochastic_eval),
                )
                (
                    raw_actor,
                    _,
                    reward,
                    terminated,
                    truncated,
                    extras,
                ) = wrapped_env.step(action)
                actor_observation = _normalize(
                    raw_actor,
                    actor_stats,
                    enabled=empirical_normalization,
                    clip=observation_clip,
                    update=False,
                )
                for name, metric in _iter_task_metrics(extras):
                    metric_name = name.removeprefix("metrics/")
                    if metric_name.startswith("episode_"):
                        values = torch.as_tensor(
                            metric, dtype=torch.float32, device=wrapped_env.device
                        ).reshape(-1)
                        values = values[torch.isfinite(values)]
                        if values.numel() > 0:
                            episode_event_sums[metric_name] = (
                                episode_event_sums.get(
                                    metric_name,
                                    torch.zeros((), device=wrapped_env.device),
                                )
                                + values.sum()
                            )
                        continue
                    _accumulate_scalar_batch(
                        task_metric_sums,
                        task_metric_counts,
                        metric_name,
                        metric,
                        device=wrapped_env.device,
                    )
                episode_returns += reward
                episode_lengths += 1
                step_reward_sum += reward.sum()
                done = terminated | truncated
                done_float = done.to(dtype=torch.float32)
                completed_return_sum += (episode_returns * done_float).sum()
                completed_length_sum += (episode_lengths * done_float).sum()
                completed_count += done_float.sum()
                episode_returns.masked_fill_(done, 0.0)
                episode_lengths.masked_fill_(done, 0)
                step += 1
    finally:
        completed = float(completed_count.cpu())
        metrics: dict[str, float] = {}
        if completed > 0.0:
            metrics["eval/ep_rew_mean"] = float(completed_return_sum.cpu()) / completed
            metrics["eval/ep_len_mean"] = float(completed_length_sum.cpu()) / completed
        if step > 0:
            metrics["eval/mean_step_reward"] = float(step_reward_sum.cpu()) / (
                step * wrapped_env.num_envs
            )
        for name, metric_sum in task_metric_sums.items():
            metrics[f"eval/metrics/{name}"] = (
                float(metric_sum.cpu()) / task_metric_counts[name]
            )
        event_zero = torch.zeros((), device=wrapped_env.device)
        completed_events = float(
            episode_event_sums.get("episode_completed", event_zero).cpu()
        )
        if completed_events > 0.0:
            metrics["eval/metrics/episode_completed"] = completed_events
            metrics["eval/episode_success_rate"] = (
                float(episode_event_sums.get("episode_success", event_zero).cpu())
                / completed_events
            )
            metrics["eval/episode_failure_rate"] = (
                float(episode_event_sums.get("episode_failed", event_zero).cpu())
                / completed_events
            )
            metrics["eval/episode_tracking_fraction"] = (
                float(
                    episode_event_sums.get(
                        "episode_tracking_fraction", event_zero
                    ).cpu()
                )
                / completed_events
            )
            metrics["eval/episode_duration_s"] = (
                float(episode_event_sums.get("episode_duration_s", event_zero).cpu())
                / completed_events
            )
            metrics["eval/episode_torso_contact_rate"] = (
                float(
                    episode_event_sums.get(
                        "episode_torso_contact_rate", event_zero
                    ).cpu()
                )
                / completed_events
            )
            metrics["eval/episode_foot_slip_speed"] = (
                float(
                    episode_event_sums.get(
                        "episode_foot_slip_speed", event_zero
                    ).cpu()
                )
                / completed_events
            )
        write_scalars(writer, metrics, step * wrapped_env.num_envs)
        writer.flush()
        writer.close()


def run(
    workflow: Literal["train", "eval"],
    env: AnyEnv | gymnasium.Env,
    sim_app: SimulationApp,
    env_id: str,
    env_cfg: AnyEnvCfg | None,
    agent_cfg: dict,
    logdir: Path,
    model: Path | None = None,
    continue_training: bool | None = None,
    untrained: bool = False,
    **kwargs: Any,
) -> None:
    """Run the ExO-FPO flow-matching pipeline."""

    del kwargs
    try:
        from flow.torch_buffer import (
            TorchRunningMeanStd,
            generalized_advantage_estimate,
        )
    except ImportError as error:
        raise ImportError(
            "The ExO-FPO integration requires the Python 3.12-compatible "
            "editable ExO-PPO package. Install /root/R2A/Algos/ExO-PPO in "
            "the 'srb' environment."
        ) from error

    raw_cfg = _as_dict(agent_cfg)
    smoothing_cfg = raw_cfg.get("smoothing", {}) or {}
    if smoothing_cfg.get("enabled", False):
        logging.warning(
            "Action smoothing changes the action represented by ExO-FPO's CFM "
            "ratio; disable it for strict action consistency."
        )
        env = maybe_wrap_action_smoothing(env, smoothing_cfg)

    obs_cfg = raw_cfg.get("obs", {}) or {}
    wrapped_env = SrbExoFpoEnvWrapper(
        env,
        actor_keys=obs_cfg.get("actor_keys"),
        critic_keys=obs_cfg.get("critic_keys"),
        clip_actions=raw_cfg.get("clip_actions", 1.0),
        validate=bool(raw_cfg.get("validate", True)),
    )
    flow_config, max_iterations = _build_flow_config(
        raw_cfg,
        ExoFpoConfig,
        validate_config,
        wrapped_env=wrapped_env,
        env_id=env_id,
        logdir=Path(logdir),
    )

    random.seed(flow_config.seed)
    np.random.seed(flow_config.seed)
    torch.manual_seed(flow_config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(flow_config.seed)

    initial_actor, initial_critic, _ = wrapped_env.reset()
    trainer = ExoFpoTrainer(
        flow_config,
        obs_dim=int(initial_actor.shape[1]),
        action_dim=wrapped_env.num_actions,
        critic_obs_dim=int(initial_critic.shape[1]),
        device=wrapped_env.device,
        action_low=wrapped_env.action_low,
        action_high=wrapped_env.action_high,
    )
    checkpoint = _resolve_checkpoint(
        workflow=workflow,
        logdir=Path(logdir),
        model=model,
        continue_training=continue_training,
        untrained=untrained,
        algorithm_name=ALGORITHM_NAME,
    )
    if workflow == "train":
        _write_run_manifest(
            logdir=Path(logdir),
            workflow=workflow,
            algorithm=ALGORITHM_NAME,
            env_id=env_id,
            env_cfg=env_cfg,
            raw_cfg=raw_cfg,
            flow_config=flow_config,
            wrapped_env=wrapped_env,
            actor_observation_dim=int(initial_actor.shape[1]),
            critic_observation_dim=int(initial_critic.shape[1]),
            framework=FRAMEWORK_NAME,
        )
        _train(
            wrapped_env=wrapped_env,
            sim_app=sim_app,
            trainer=trainer,
            raw_cfg=raw_cfg,
            logdir=Path(logdir),
            max_iterations=max_iterations,
            checkpoint=checkpoint,
            normalizer_class=TorchRunningMeanStd,
            compute_gae=generalized_advantage_estimate,
        )
        return
    if checkpoint is None and not untrained:
        raise FileNotFoundError(
            f"A {ALGORITHM_NAME} checkpoint is required for evaluation"
        )
    _evaluate(
        wrapped_env=wrapped_env,
        sim_app=sim_app,
        trainer=trainer,
        raw_cfg=raw_cfg,
        checkpoint=checkpoint,
        normalizer_class=TorchRunningMeanStd,
        logdir=Path(logdir),
    )
