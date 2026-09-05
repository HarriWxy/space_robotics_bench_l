"""TensorBoard conventions shared by SRB's non-SB3 agent integrations.

SRB's SB3 and SBX integrations use the Stable-Baselines3 logger schema.  The
custom PyTorch runners do not depend on that logger, so this module provides a
small adapter which keeps their native diagnostics and adds the same canonical
scalar names.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping
from typing import Any

import torch

PPO_TENSORBOARD_TAGS = frozenset(
    {
        "time/fps",
        "rollout/ep_rew_mean",
        "rollout/ep_len_mean",
        "train/approx_kl",
        "train/clip_fraction",
        "train/clip_range",
        "train/entropy_loss",
        "train/explained_variance",
        "train/learning_rate",
        "train/loss",
        "train/policy_gradient_loss",
        "train/std",
        "train/value_loss",
    }
)

SAC_TENSORBOARD_TAGS = frozenset(
    {
        "train/actor_loss",
        "train/critic_loss",
        "train/ent_coef",
        "train/ent_coef_loss",
    }
)


def scalar_value(value: Any) -> float:
    """Convert a scalar-like value to a Python float for SummaryWriter."""

    if isinstance(value, torch.Tensor):
        value = value.detach()
        if value.numel() != 1:
            value = value.float().mean()
        value = value.cpu().item()
    elif hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu()
        if getattr(value, "numel", lambda: 1)() != 1:
            value = value.float().mean()
        value = value.item()
    elif isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        values = [scalar_value(item) for item in value]
        value = sum(values) / len(values) if values else 0.0
    return float(value)


def write_scalars(writer: Any, values: Mapping[str, Any], step: int) -> None:
    """Write a mapping of scalar values using a single integer x-axis."""

    for tag, value in values.items():
        if value is None:
            continue
        writer.add_scalar(tag, scalar_value(value), int(step))


def _first_value(
    values: Mapping[str, Any], keys: tuple[str, ...], default: Any = 0.0
) -> Any:
    for key in keys:
        if key in values:
            return values[key]
    return default


def _agent_value(agent: Any, names: tuple[str, ...], default: Any = None) -> Any:
    for name in names:
        if hasattr(agent, name):
            return getattr(agent, name)
    cfg = getattr(agent, "cfg", None)
    if isinstance(cfg, Mapping):
        for name in names:
            if name in cfg:
                return cfg[name]
    return default


def policyflow_ppo_scalars(
    training_info: Mapping[str, Any], agent: Any | None = None
) -> dict[str, Any]:
    """Map PolicyFlow's native metrics to the SB3 PPO scalar names.

    PolicyFlow exposes a PPO-like update but historically returns names such as
    ``Loss/policy_loss`` and ``Policy/mean_noise_std``.  The fallback values are
    intentional: a metric that the upstream implementation does not expose is
    still present in the common schema with a neutral value, while all native
    fields remain available in the event file.
    """

    policy_gradient_loss = _first_value(
        training_info,
        ("train/policy_gradient_loss", "Loss/policy_loss"),
    )
    value_loss = _first_value(
        training_info,
        ("train/value_loss", "Loss/value_loss"),
    )
    weighted_entropy_loss = _first_value(
        training_info,
        ("train/weighted_entropy_loss", "Loss/gaussian_entropy_loss"),
    )
    entropy_coefficient = scalar_value(
        _agent_value(
            agent,
            ("_gaussian_entropy_loss_scale", "gaussian_entropy_loss_scale"),
            0.0,
        )
        or 0.0
    )
    if "train/entropy_loss" in training_info:
        entropy_loss = training_info["train/entropy_loss"]
    elif entropy_coefficient:
        # PolicyFlow stores -coefficient * entropy as gaussian_entropy_loss.
        entropy_loss = scalar_value(weighted_entropy_loss) / entropy_coefficient
    else:
        entropy_loss = 0.0

    brownian_loss = _first_value(
        training_info,
        ("train/brownian_reg_loss", "Loss/brownian_reg_loss"),
        0.0,
    )
    total_loss = _first_value(training_info, ("train/loss",), None)
    if total_loss is None:
        total_loss = (
            scalar_value(policy_gradient_loss)
            + scalar_value(weighted_entropy_loss)
            + scalar_value(value_loss)
            + scalar_value(brownian_loss)
        )

    ratio_clip = _agent_value(
        agent,
        ("_ratio_clip", "ratio_clip"),
        0.2,
    )
    return {
        "train/approx_kl": _first_value(
            training_info,
            ("train/approx_kl", "Metrics/approx_kl", "Loss/kl"),
        ),
        "train/clip_fraction": _first_value(
            training_info,
            ("train/clip_fraction", "Metrics/clip_fraction"),
        ),
        "train/clip_range": _first_value(
            training_info,
            ("train/clip_range", "Metrics/clip_range"),
            ratio_clip,
        ),
        "train/entropy_loss": entropy_loss,
        "train/explained_variance": _first_value(
            training_info,
            ("train/explained_variance", "Metrics/explained_variance"),
        ),
        "train/learning_rate": _first_value(
            training_info,
            ("train/learning_rate", "Loss/learning_rate"),
        ),
        "train/loss": total_loss,
        "train/policy_gradient_loss": policy_gradient_loss,
        "train/std": _first_value(
            training_info,
            ("train/std", "Policy/mean_noise_std", "Policy/policy_std"),
        ),
        "train/value_loss": value_loss,
    }


def _value_list(value: Any) -> list[float]:
    if isinstance(value, torch.Tensor):
        return value.detach().float().reshape(-1).cpu().tolist()
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().reshape(-1).tolist()
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        result: list[float] = []
        for item in value:
            result.extend(_value_list(item))
        return result
    return [float(value)]


def _mean_values(values: list[Any]) -> float | None:
    flattened: list[float] = []
    for value in values:
        flattened.extend(_value_list(value))
    if not flattened:
        return None
    return sum(flattened) / len(flattened)


def _environment_step(runner: Any, stat: Mapping[str, Any]) -> int:
    """Use SB3-like environment timesteps instead of update iteration."""

    iteration = int(stat.get("current_iteration", 0))
    cfg = getattr(runner, "_cfg", {}) or {}
    rollouts = int(cfg.get("rollouts", 1)) if isinstance(cfg, Mapping) else 1
    env = getattr(runner, "_env", None)
    num_envs = int(getattr(env, "num_envs", 1))
    return max(1, (iteration + 1) * rollouts * num_envs)


def make_policyflow_tensorboard_cb(directory: str):
    """Create a PolicyFlow callback with native and SB3-compatible tags."""

    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir=directory, flush_secs=10)
    started = time.monotonic()

    def callback(runner: Any, stat: Mapping[str, Any]) -> None:
        step = _environment_step(runner, stat)
        write_scalars(
            writer,
            {"time/fps": step / max(time.monotonic() - started, 1.0e-6)},
            step,
        )
        training_info = stat.get("training_info", {})
        if isinstance(training_info, Mapping):
            write_scalars(writer, training_info, step)
            write_scalars(
                writer,
                policyflow_ppo_scalars(training_info, getattr(runner, "_agent", None)),
                step,
            )

        returns = stat.get("returns", [])
        lengths = stat.get("lengths", [])
        if returns:
            write_scalars(
                writer,
                {
                    "rollout/ep_rew_mean": _mean_values(returns),
                    "rollout/ep_len_mean": _mean_values(lengths),
                },
                step,
            )

        reward_terms = stat.get("reward_terms", [])
        if reward_terms:
            by_name: dict[str, list[Any]] = {}
            for terms in reward_terms:
                if not isinstance(terms, Mapping):
                    continue
                for name, value in terms.items():
                    by_name.setdefault(str(name), []).append(value)
            write_scalars(
                writer,
                {
                    f"rollout/reward_terms/{name}": _mean_values(values)
                    for name, values in sorted(by_name.items())
                },
                step,
            )

        task_metrics: dict[str, list[Any]] = {}
        for episode_info in stat.get("info", []):
            if not isinstance(episode_info, Mapping):
                continue
            for key, value in episode_info.items():
                if isinstance(key, str) and key.startswith("metrics/"):
                    task_metrics.setdefault(key.removeprefix("metrics/"), []).append(
                        value
                    )
            write_scalars(
                writer,
                {
                    key if "/" in key else f"Episode/{key}": value
                    for key, value in episode_info.items()
                },
                step,
            )
        write_scalars(
            writer,
            {
                f"rollout/metrics/{name}": _mean_values(values)
                for name, values in sorted(task_metrics.items())
            },
            step,
        )
        writer.flush()

    return callback
