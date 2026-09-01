"""Run the PyTorch ExO-PPO one-step flow against an SRB environment."""

from __future__ import annotations

import random
import re
import time
from collections.abc import Mapping
from dataclasses import asdict, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import gymnasium
import numpy as np
import torch

from srb.integrations.exoppo.wrapper import SrbExoPpoEnvWrapper
from srb.integrations.tensorboard import write_scalars
from srb.utils import logging
from srb.wrappers import maybe_wrap_action_smoothing

if TYPE_CHECKING:
    from isaacsim.simulation_app import SimulationApp

    from srb._typing import AnyEnv, AnyEnvCfg


FRAMEWORK_NAME = "exoppo"
_CHECKPOINT_PATTERN = re.compile(r"model_(\d+)\.pt$")
_RUNNER_KEYS = {
    "max_iterations",
    "save_interval",
    "log_interval",
    "eval_steps",
    "empirical_normalization",
    "observation_clip",
    "randomize_reset_episode_progress",
    "obs",
    "smoothing",
    "validate",
}


class _NullSummaryWriter:
    def add_scalar(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return _plain(value)
    if hasattr(value, "to_dict"):
        return _plain(value.to_dict())
    raise TypeError(
        f"ExO-PPO agent config must be a mapping, got {type(value).__name__}"
    )


def _last_checkpoint(logdir: Path) -> Path | None:
    matches: list[tuple[int, Path]] = []
    if logdir.is_dir():
        for path in logdir.iterdir():
            match = _CHECKPOINT_PATTERN.fullmatch(path.name)
            if path.is_file() and match:
                matches.append((int(match.group(1)), path))
    return max(matches, default=(0, None), key=lambda item: item[0])[1]


def _resolve_checkpoint(
    *,
    workflow: Literal["train", "eval"],
    logdir: Path,
    model: Path | None,
    continue_training: bool | None,
    untrained: bool,
) -> Path | None:
    if model:
        return Path(model)
    if workflow == "eval" or continue_training:
        checkpoint = _last_checkpoint(logdir)
        if checkpoint is None and workflow == "eval" and not untrained:
            raise FileNotFoundError(
                f"No ExO-PPO checkpoint matching model_<iteration>.pt in {logdir}"
            )
        return checkpoint
    return None


def _build_flow_config(
    raw_cfg: Mapping[str, Any],
    config_class: type,
    validate_config: Any,
    *,
    wrapped_env: SrbExoPpoEnvWrapper,
    env_id: str,
    logdir: Path,
) -> tuple[Any, int]:
    field_names = {field.name for field in fields(config_class)}
    values = {key: raw_cfg[key] for key in field_names if key in raw_cfg}
    max_iterations = int(raw_cfg.get("max_iterations", 1500))
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")

    configured_device = values.get("device")
    if (
        configured_device not in (None, "auto")
        and torch.device(configured_device) != wrapped_env.device
    ):
        logging.warning(
            f"ExO-PPO device {configured_device!s} does not match SRB environment "
            f"device {wrapped_env.device}; using the environment device."
        )
    values.update(
        {
            "env_id": env_id,
            "num_envs": wrapped_env.num_envs,
            "total_steps": max_iterations
            * int(values.get("rollout_steps", 256))
            * wrapped_env.num_envs,
            "device": str(wrapped_env.device),
            "log_dir": str(logdir),
            "checkpoint_dir": str(logdir),
        }
    )
    if "env_backend" in field_names:
        values["env_backend"] = "gymnasium"
    if "envpool_num_threads" in field_names:
        values["envpool_num_threads"] = 0
    if "hidden_sizes" in values:
        values["hidden_sizes"] = tuple(int(size) for size in values["hidden_sizes"])

    config = config_class(**values)
    validate_config(config)
    unknown = sorted(set(raw_cfg) - field_names - _RUNNER_KEYS)
    if unknown:
        logging.warning(f"Ignoring unknown ExO-PPO config keys: {unknown}")
    return config, max_iterations


def _normalize(
    observation: torch.Tensor,
    normalizer: Any,
    *,
    enabled: bool,
    clip: float,
    update: bool,
) -> torch.Tensor:
    if not enabled:
        return observation.to(dtype=torch.float32)
    if update:
        normalizer.update(observation)
    return normalizer.normalize(observation, clip=clip)


def _policy_sample(
    trainer: Any,
    observation: torch.Tensor,
    previous_action: torch.Tensor,
    has_previous_action: torch.Tensor,
    *,
    warm_start_time: float,
    stochastic: bool,
) -> Any:
    batch_size = observation.shape[0]
    if stochastic:
        pure_noise = torch.randn(
            (batch_size, trainer.policy.action_dim),
            dtype=torch.float32,
            device=trainer.device,
        )
    else:
        pure_noise = torch.zeros(
            (batch_size, trainer.policy.action_dim),
            dtype=torch.float32,
            device=trainer.device,
        )
    if warm_start_time > 0.0:
        start = has_previous_action.to(dtype=torch.float32).reshape(-1, 1)
        start = start * warm_start_time
        flow_init = (1.0 - start) * pure_noise + start * previous_action
    else:
        start = torch.zeros((batch_size, 1), dtype=torch.float32, device=trainer.device)
        flow_init = pure_noise
    return trainer.policy.sample(
        observation,
        flow_init=flow_init,
        flow_start=start,
        deterministic=not stochastic,
    )


def _collect_rollout(
    *,
    wrapped_env: SrbExoPpoEnvWrapper,
    sim_app: SimulationApp,
    trainer: Any,
    actor_observation: torch.Tensor,
    critic_observation: torch.Tensor,
    actor_stats: Any,
    critic_stats: Any,
    empirical_normalization: bool,
    observation_clip: float,
    previous_action: torch.Tensor,
    has_previous_action: torch.Tensor,
    episode_returns: torch.Tensor,
    episode_lengths: torch.Tensor,
    flatten_rollout: Any,
    compute_gae: Any,
) -> tuple[Any | None, torch.Tensor, torch.Tensor, dict[str, float], int]:
    actor_observations: list[torch.Tensor] = []
    critic_observations: list[torch.Tensor] = []
    pre_tanh_actions: list[torch.Tensor] = []
    flow_initializations: list[torch.Tensor] = []
    flow_start_times: list[torch.Tensor] = []
    behavior_log_probs: list[torch.Tensor] = []
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

    trainer.policy.eval()
    trainer.value.eval()
    for _ in range(trainer.config.rollout_steps):
        if not sim_app.is_running():
            break
        with torch.no_grad():
            sample = _policy_sample(
                trainer,
                actor_observation,
                previous_action,
                has_previous_action,
                warm_start_time=trainer.config.warm_start_time,
                stochastic=True,
            )
            value = trainer.value(critic_observation)
            (
                next_raw_actor,
                next_raw_critic,
                reward,
                terminated,
                truncated,
                extras,
            ) = wrapped_env.step(sample.pre_tanh_action)
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
                    term_tensor = torch.as_tensor(
                        term, dtype=torch.float32, device=trainer.device
                    ).reshape(-1)
                    if term_tensor.numel() == 0:
                        continue
                    key = str(name)
                    reward_term_sums[key] = reward_term_sums.get(
                        key, torch.zeros((), device=trainer.device)
                    ) + term_tensor.sum()
                    reward_term_counts[key] = reward_term_counts.get(key, 0) + int(
                        term_tensor.numel()
                    )

            if wrapped_env.bootstrap_truncated:
                # This fallback matches the established Isaac Lab runner behavior.
                # When compute_final_obs is supported, the exact terminal value below
                # replaces it for every truncated environment.
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
        pre_tanh_actions.append(sample.pre_tanh_action)
        flow_initializations.append(sample.flow_init)
        flow_start_times.append(sample.flow_start)
        behavior_log_probs.append(sample.log_prob)
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

        previous_action.copy_(sample.pre_tanh_action)
        has_previous_action.fill_(True)
        previous_action[done] = 0.0
        has_previous_action[done] = False
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
    rollout = flatten_rollout(
        torch.stack(actor_observations),
        torch.stack(critic_observations),
        torch.stack(pre_tanh_actions),
        torch.stack(flow_initializations),
        torch.stack(flow_start_times),
        torch.stack(behavior_log_probs),
        advantages,
        returns,
    )
    count = float(completed_count.cpu())
    metrics: dict[str, float] = {}
    if count > 0.0:
        metrics["rollout/ep_rew_mean"] = float(completed_return_sum.cpu()) / count
        metrics["rollout/ep_len_mean"] = float(completed_length_sum.cpu()) / count

    return_variance = torch.var(returns, unbiased=False)
    if float(return_variance) > 1.0e-8:
        explained_variance = 1.0 - torch.var(
            returns - value_tensor, unbiased=False
        ) / return_variance
        metrics["train/explained_variance"] = float(explained_variance.cpu())
    else:
        metrics["train/explained_variance"] = 0.0

    for name, term_sum in reward_term_sums.items():
        term_count = reward_term_counts[name]
        metrics[f"rollout/reward_terms/{name}"] = float(term_sum.cpu()) / term_count
    return rollout, actor_observation, critic_observation, metrics, step_count


def _checkpoint_payload(
    *,
    trainer: Any,
    actor_stats: Any,
    critic_stats: Any,
    iteration: int,
    environment_steps: int,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "policy": trainer.policy.state_dict(),
        "recent_policy": trainer.recent_policy.state_dict(),
        "ema_teacher": trainer.ema_teacher.state_dict(),
        "value": trainer.value.state_dict(),
        "actor_optimizer": trainer.actor_optimizer.state_dict(),
        "critic_optimizer": trainer.critic_optimizer.state_dict(),
        "update_step": trainer.update_step,
        "iteration": iteration,
        "environment_steps": environment_steps,
        "actor_observation_stats": actor_stats.state_dict(),
        "critic_observation_stats": critic_stats.state_dict(),
        "config": asdict(trainer.config),
    }


def _save_checkpoint(
    path: Path,
    *,
    trainer: Any,
    actor_stats: Any,
    critic_stats: Any,
    iteration: int,
    environment_steps: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        _checkpoint_payload(
            trainer=trainer,
            actor_stats=actor_stats,
            critic_stats=critic_stats,
            iteration=iteration,
            environment_steps=environment_steps,
        ),
        path,
    )


def _load_checkpoint(
    path: Path,
    *,
    trainer: Any,
    actor_stats: Any,
    critic_stats: Any,
    load_optimizers: bool,
) -> tuple[int, int]:
    checkpoint = torch.load(path, map_location=trainer.device, weights_only=False)
    trainer.policy.load_state_dict(checkpoint["policy"])
    trainer.value.load_state_dict(checkpoint["value"])
    trainer.recent_policy.load_state_dict(
        checkpoint.get("recent_policy", checkpoint["policy"])
    )
    trainer.ema_teacher.load_state_dict(
        checkpoint.get("ema_teacher", checkpoint["policy"])
    )
    trainer.update_step = int(checkpoint.get("update_step", 0))
    if load_optimizers:
        if "actor_optimizer" in checkpoint:
            trainer.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        if "critic_optimizer" in checkpoint:
            trainer.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])

    actor_state = checkpoint.get("actor_observation_stats")
    critic_state = checkpoint.get("critic_observation_stats")
    if actor_state is None and "observation_mean" in checkpoint:
        actor_state = {
            "mean": checkpoint["observation_mean"],
            "var": checkpoint["observation_var"],
            "count": checkpoint["observation_count"],
        }
    if actor_state is not None:
        actor_stats.load_state_dict(actor_state)
    if critic_state is not None:
        critic_stats.load_state_dict(critic_state)
    elif actor_state is not None and actor_stats.mean.shape == critic_stats.mean.shape:
        critic_stats.load_state_dict(actor_state)
    return int(checkpoint.get("iteration", -1)) + 1, int(
        checkpoint.get("environment_steps", 0)
    )


def _summary_writer(logdir: Path) -> Any:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        logging.warning(
            "TensorBoard is unavailable; ExO-PPO scalar logging is disabled"
        )
        return _NullSummaryWriter()
    return SummaryWriter(log_dir=str(logdir))


def _train(
    *,
    wrapped_env: SrbExoPpoEnvWrapper,
    sim_app: SimulationApp,
    trainer: Any,
    raw_cfg: Mapping[str, Any],
    logdir: Path,
    max_iterations: int,
    checkpoint: Path | None,
    normalizer_class: type,
    replay_class: type,
    flatten_rollout: Any,
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
        logging.info(f"Loading ExO-PPO checkpoint from {checkpoint}")
        start_iteration, environment_steps = _load_checkpoint(
            checkpoint,
            trainer=trainer,
            actor_stats=actor_stats,
            critic_stats=critic_stats,
            load_optimizers=True,
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

    replay = replay_class(trainer.config.replay_N)
    previous_action = torch.zeros(
        (wrapped_env.num_envs, wrapped_env.num_actions),
        dtype=torch.float32,
        device=wrapped_env.device,
    )
    has_previous_action = torch.zeros(
        wrapped_env.num_envs, dtype=torch.bool, device=wrapped_env.device
    )
    episode_returns = torch.zeros(
        wrapped_env.num_envs, dtype=torch.float32, device=wrapped_env.device
    )
    episode_lengths = torch.zeros(
        wrapped_env.num_envs, dtype=torch.long, device=wrapped_env.device
    )
    writer = _summary_writer(logdir)
    started = time.monotonic()
    last_iteration = start_iteration - 1
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
                previous_action=previous_action,
                has_previous_action=has_previous_action,
                episode_returns=episode_returns,
                episode_lengths=episode_lengths,
                flatten_rollout=flatten_rollout,
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
                train_metrics = trainer.train_torch_replay(replay)
                metrics.update(
                    {f"train/{key}": value for key, value in train_metrics.items()}
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
                        "train/loss",
                        "train/approx_kl",
                        "train/clip_fraction",
                        "train/ratio",
                        "train/recent_ratio",
                        "train/flow_loss",
                        "train/consistency_loss",
                    }
                }
                logging.info(
                    f"ExO-PPO iteration={iteration} steps={environment_steps} "
                    f"metrics={concise}"
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
    wrapped_env: SrbExoPpoEnvWrapper,
    sim_app: SimulationApp,
    trainer: Any,
    raw_cfg: Mapping[str, Any],
    checkpoint: Path | None,
    normalizer_class: type,
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
        logging.info(f"Loading ExO-PPO checkpoint from {checkpoint}")
        _load_checkpoint(
            checkpoint,
            trainer=trainer,
            actor_stats=actor_stats,
            critic_stats=critic_stats,
            load_optimizers=False,
        )
    actor_observation = _normalize(
        raw_actor,
        actor_stats,
        enabled=empirical_normalization,
        clip=observation_clip,
        update=False,
    )
    previous_action = torch.zeros(
        (wrapped_env.num_envs, wrapped_env.num_actions),
        dtype=torch.float32,
        device=wrapped_env.device,
    )
    has_previous_action = torch.zeros(
        wrapped_env.num_envs, dtype=torch.bool, device=wrapped_env.device
    )
    stochastic = bool(trainer.config.stochastic_eval)
    trainer.policy.eval()
    step = 0
    with torch.inference_mode():
        while sim_app.is_running() and (eval_steps == 0 or step < eval_steps):
            sample = _policy_sample(
                trainer,
                actor_observation,
                previous_action,
                has_previous_action,
                warm_start_time=trainer.config.warm_start_time,
                stochastic=stochastic,
            )
            raw_actor, _, _, terminated, truncated, _ = wrapped_env.step(
                sample.pre_tanh_action
            )
            actor_observation = _normalize(
                raw_actor,
                actor_stats,
                enabled=empirical_normalization,
                clip=observation_clip,
                update=False,
            )
            done = terminated | truncated
            previous_action.copy_(sample.pre_tanh_action)
            has_previous_action.fill_(True)
            previous_action[done] = 0.0
            has_previous_action[done] = False
            step += 1


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
    """Run only the PyTorch implementation from ``ExO-PPO/flow``."""

    del env_cfg, kwargs
    try:
        from flow.torch_buffer import (
            TorchReplayWindow,
            TorchRunningMeanStd,
            flatten_torch_rollout,
            generalized_advantage_estimate,
        )
        from flow.torch_train import TorchTrainConfig, Trainer, validate_config
    except ImportError as error:
        raise ImportError(
            "The SRB ExO-PPO integration requires the Python 3.12-compatible "
            "editable ExO-PPO package. In the 'srb' conda environment run: "
            "python -m pip install --no-deps --no-build-isolation -e "
            "/root/R2A/ExO-PPO"
        ) from error

    raw_cfg = _as_dict(agent_cfg)
    smoothing_cfg = raw_cfg.get("smoothing", {}) or {}
    if smoothing_cfg.get("enabled", False):
        logging.warning(
            "Action smoothing changes the action represented by ExO-PPO's exact "
            "augmented-policy ratio; keep it disabled for strict training semantics."
        )
        env = maybe_wrap_action_smoothing(env, smoothing_cfg)

    obs_cfg = raw_cfg.get("obs", {}) or {}
    wrapped_env = SrbExoPpoEnvWrapper(
        env,
        actor_keys=obs_cfg.get("actor_keys"),
        critic_keys=obs_cfg.get("critic_keys"),
        validate=bool(raw_cfg.get("validate", True)),
    )
    flow_config, max_iterations = _build_flow_config(
        raw_cfg,
        TorchTrainConfig,
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
    trainer = Trainer(
        flow_config,
        obs_dim=int(initial_actor.shape[1]),
        action_dim=wrapped_env.num_actions,
        critic_obs_dim=int(initial_critic.shape[1]),
        device=wrapped_env.device,
    )
    checkpoint = _resolve_checkpoint(
        workflow=workflow,
        logdir=Path(logdir),
        model=model,
        continue_training=continue_training,
        untrained=untrained,
    )
    if workflow == "train":
        _train(
            wrapped_env=wrapped_env,
            sim_app=sim_app,
            trainer=trainer,
            raw_cfg=raw_cfg,
            logdir=Path(logdir),
            max_iterations=max_iterations,
            checkpoint=checkpoint,
            normalizer_class=TorchRunningMeanStd,
            replay_class=TorchReplayWindow,
            flatten_rollout=flatten_torch_rollout,
            compute_gae=generalized_advantage_estimate,
        )
        return
    if checkpoint is None and not untrained:
        raise FileNotFoundError("An ExO-PPO checkpoint is required for evaluation")
    _evaluate(
        wrapped_env=wrapped_env,
        sim_app=sim_app,
        trainer=trainer,
        raw_cfg=raw_cfg,
        checkpoint=checkpoint,
        normalizer_class=TorchRunningMeanStd,
    )
