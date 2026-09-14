"""TensorBoard conventions shared by SRB's agent integrations.

SB3 is the reference for SRB's scalar schema.  The other runners use different
logging implementations and historically emitted tags such as ``Loss/...``
and ``Perf/...``.  This module keeps the public tags lowercase and maps those
legacy names to the same ``time/``, ``rollout/``, ``train/``, ``eval/``,
``config/`` and ``replay/`` namespaces.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping
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


_TRAIN_TAG_ALIASES = {
    "action_std": "std",
    "approx_kl": "approx_kl",
    "clip_param": "clip_range",
    "kl": "approx_kl",
    "mean_noise_std": "std",
    "mean_std": "std",
    "pg_loss": "policy_gradient_loss",
    "policy_loss": "policy_gradient_loss",
    "policy_std": "std",
    "surrogate": "policy_gradient_loss",
    "surrogate_loss": "policy_gradient_loss",
    "value": "value_loss",
    "value_loss": "value_loss",
}


def _normalise_tag(tag: str) -> str:
    """Normalise tag spelling before applying semantic aliases."""

    return str(tag).strip().replace(" ", "_").casefold()


def canonical_tag(tag: str) -> str:
    """Return the lowercase SRB TensorBoard tag for ``tag``.

    The aliases cover the native RSL-RL/FPO logger names as well as the
    historical short names used by the Torch flow runners.  Unknown suffixes
    are retained under a lowercase namespace so algorithm-specific diagnostics
    remain available without creating a second capitalization variant.
    """

    normalised = _normalise_tag(tag)
    prefix, separator, suffix = normalised.partition("/")
    if not separator:
        return normalised

    suffix = suffix.strip("/")
    if prefix == "episode":
        if suffix.startswith(("rollout/", "train/", "eval/", "time/")):
            return canonical_tag(suffix)
        if suffix.startswith("metrics/"):
            suffix = suffix.removeprefix("metrics/")
        elif suffix.startswith("reward_terms/"):
            return f"rollout/reward_terms/{suffix.removeprefix('reward_terms/')}"
        if suffix in {
            "r",
            "reward",
            "episode_reward",
            "mean_reward",
            "mean_reward/time",
            "ep_rew_mean",
        }:
            return "rollout/ep_rew_mean"
        if suffix in {
            "l",
            "episode_length",
            "mean_episode_length",
            "mean_episode_length/time",
            "length",
            "ep_len_mean",
        }:
            return "rollout/ep_len_mean"
        return f"rollout/metrics/{suffix}"

    if prefix == "loss":
        if suffix in {"entropy", "entropy_loss"}:
            return "train/entropy_loss"
        suffix = _TRAIN_TAG_ALIASES.get(suffix, suffix)
        return f"train/{suffix}"

    if prefix in {"metrics", "policy"}:
        suffix = _TRAIN_TAG_ALIASES.get(suffix, suffix)
        return f"train/{suffix}"

    if prefix == "perf":
        if suffix in {"fps", "total_fps"}:
            suffix = "fps"
        return f"time/{suffix}"

    if prefix == "train":
        if suffix in {
            "r",
            "reward",
            "episode_reward",
            "mean_reward",
            "mean_reward/time",
            "ep_rew_mean",
        }:
            return "rollout/ep_rew_mean"
        if suffix in {
            "l",
            "length",
            "episode_length",
            "mean_episode_length",
            "mean_episode_length/time",
            "ep_len_mean",
        }:
            return "rollout/ep_len_mean"
        suffix = _TRAIN_TAG_ALIASES.get(suffix, suffix)
        return f"train/{suffix}"

    if prefix == "eval":
        if suffix in {
            "return",
            "episode_reward",
            "mean_reward",
            "ep_rew_mean",
        }:
            return "eval/ep_rew_mean"
        if suffix in {
            "episode_length",
            "mean_episode_length",
            "ep_len_mean",
        }:
            return "eval/ep_len_mean"
        return f"eval/{suffix}"

    if prefix.startswith("posteval_"):
        mode = prefix.removeprefix("posteval_")
        suffix = {
            "mean_reward": "ep_rew_mean",
            "std_reward": "ep_rew_std",
            "mean_episode_length": "ep_len_mean",
            "std_episode_length": "ep_len_std",
        }.get(suffix, suffix)
        return f"eval/{mode}/{suffix}"

    if prefix == "rnd":
        return f"train/rnd/{suffix}"

    return f"{prefix}/{suffix}"


def _negate_scalar(value: Any) -> Any:
    """Negate a scalar while keeping tensors on their original device."""

    if isinstance(value, torch.Tensor):
        return -value
    try:
        return -value
    except TypeError:
        return -scalar_value(value)


def canonical_scalar(tag: str, value: Any) -> tuple[str, Any]:
    """Map a scalar tag and value to the public schema.

    RSL-RL and FPO expose entropy as a positive bonus, whereas SB3's
    ``train/entropy_loss`` is the negative entropy term used in the objective.
    Apply that sign conversion only to the legacy loss namespace; already
    canonical ``train/entropy_loss`` values are left untouched.
    """

    normalised = _normalise_tag(tag)
    canonical = canonical_tag(normalised)
    prefix, _, suffix = normalised.partition("/")
    if prefix == "loss" and suffix in {"entropy", "entropy_loss"}:
        value = _negate_scalar(value)
    return canonical, value


def canonicalize_scalars(values: Mapping[str, Any]) -> dict[str, Any]:
    """Collapse a scalar mapping after canonicalizing its tags.

    If a mapping contains both a legacy alias and its canonical spelling, the
    canonical spelling wins.  This prevents one training update from emitting
    two points for the same field and step.
    """

    result: dict[str, Any] = {}
    canonical_spelling: dict[str, bool] = {}
    for tag, value in values.items():
        canonical, value = canonical_scalar(str(tag), value)
        is_canonical = str(tag) == canonical
        if canonical not in result or is_canonical or not canonical_spelling[canonical]:
            result[canonical] = value
            canonical_spelling[canonical] = is_canonical
    return result


class CanonicalScalarWriter:
    """Proxy a scalar writer with canonical tags and optional step selection."""

    def __init__(
        self,
        writer: Any,
        *,
        step_fn: Callable[[], int] | None = None,
    ) -> None:
        self._writer = writer
        self._step_fn = step_fn
        self._last_writes: dict[str, tuple[int, str]] = {}

    def add_scalar(
        self,
        tag: str,
        scalar_value: Any,
        global_step: int | None = None,
        walltime: float | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        raw_tag = str(tag).strip().replace(" ", "_")
        canonical, scalar_value = canonical_scalar(tag, scalar_value)
        step = self._step(global_step)
        if step is not None:
            previous = self._last_writes.get(canonical)
            if previous is not None and previous[0] == step and previous[1] != raw_tag:
                return
            self._last_writes[canonical] = (step, raw_tag)

        if walltime is None and not args and not kwargs:
            self._writer.add_scalar(canonical, scalar_value, step)
        else:
            self._writer.add_scalar(
                canonical,
                scalar_value,
                step,
                walltime,
                *args,
                **kwargs,
            )

    def _step(self, global_step: int | None) -> int | None:
        if self._step_fn is None:
            return None if global_step is None else int(global_step)
        step = int(self._step_fn())
        if step <= 0 and global_step is not None:
            return int(global_step)
        return step

    def __getattr__(self, name: str) -> Any:
        return getattr(self._writer, name)


def patch_scalar_writer(
    writer: Any,
    *,
    step_fn: Callable[[], int] | None = None,
) -> Any:
    """Patch a writer in place, preserving optional writer type checks.

    RSL-RL's external W&B/Neptune writers are checked with ``isinstance`` by
    the upstream logger.  Patching ``add_scalar`` in place keeps those checks
    working; a proxy is used only for writers that reject instance attributes.
    """

    if writer is None or isinstance(writer, CanonicalScalarWriter):
        return writer
    if getattr(writer, "_srb_canonical_scalar_writer", False):
        return writer

    try:
        original_add_scalar = writer.add_scalar
        last_writes: dict[str, tuple[int, str]] = {}

        def add_scalar(
            tag: str,
            scalar_value: Any,
            global_step: int | None = None,
            walltime: float | None = None,
            *args: Any,
            **kwargs: Any,
        ) -> None:
            raw_tag = str(tag).strip().replace(" ", "_")
            canonical, scalar_value = canonical_scalar(tag, scalar_value)
            if step_fn is None:
                step = None if global_step is None else int(global_step)
            else:
                step = int(step_fn())
                if step <= 0 and global_step is not None:
                    step = int(global_step)
            if step is not None:
                previous = last_writes.get(canonical)
                if (
                    previous is not None
                    and previous[0] == step
                    and previous[1] != raw_tag
                ):
                    return
                last_writes[canonical] = (step, raw_tag)
            if walltime is None and not args and not kwargs:
                original_add_scalar(canonical, scalar_value, step)
            else:
                original_add_scalar(
                    canonical,
                    scalar_value,
                    step,
                    walltime,
                    *args,
                    **kwargs,
                )

        writer.add_scalar = add_scalar
        writer._srb_canonical_scalar_writer = True
        return writer
    except (AttributeError, TypeError):
        return CanonicalScalarWriter(writer, step_fn=step_fn)


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
    """Write canonical scalar values using a single integer x-axis."""

    for tag, value in canonicalize_scalars(values).items():
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
    still present in the common schema with a neutral value.
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
    """Create a PolicyFlow callback that writes the shared SRB tag schema."""

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
            training_scalars = dict(training_info)
            training_scalars.update(
                policyflow_ppo_scalars(
                    training_info,
                    getattr(runner, "_agent", None),
                )
            )
            write_scalars(
                writer,
                training_scalars,
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
                    tag = canonical_tag(f"rollout/reward_terms/{name}")
                    by_name.setdefault(tag, []).append(value)
            write_scalars(
                writer,
                {
                    name: _mean_values(values)
                    for name, values in sorted(by_name.items())
                },
                step,
            )

        task_metrics: dict[str, list[Any]] = {}
        episode_events: dict[str, float] = {}
        for episode_info in stat.get("info", []):
            if not isinstance(episode_info, Mapping):
                continue
            episode_scalars: dict[str, Any] = {}
            for key, value in episode_info.items():
                if isinstance(key, str) and key.startswith("metrics/"):
                    metric_name = canonical_tag(
                        f"rollout/metrics/{key.removeprefix('metrics/')}"
                    ).removeprefix("rollout/metrics/")
                    if metric_name.startswith("episode_"):
                        episode_events[metric_name] = episode_events.get(
                            metric_name, 0.0
                        ) + sum(_value_list(value))
                    else:
                        task_metrics.setdefault(metric_name, []).append(value)
                    continue
                if key == "reward_terms" or isinstance(value, Mapping):
                    continue
                if isinstance(key, str):
                    episode_scalars[key if "/" in key else f"Episode/{key}"] = value
            if episode_scalars:
                write_scalars(
                    writer,
                    episode_scalars,
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
        completed = episode_events.get("episode_completed", 0.0)
        if completed > 0.0:
            write_scalars(
                writer,
                {
                    "rollout/metrics/episode_completed": completed,
                    "rollout/episode_success_rate": episode_events.get(
                        "episode_success", 0.0
                    )
                    / completed,
                    "rollout/episode_failure_rate": episode_events.get(
                        "episode_failed", 0.0
                    )
                    / completed,
                    "rollout/episode_tracking_fraction": episode_events.get(
                        "episode_tracking_fraction", 0.0
                    )
                    / completed,
                    "rollout/episode_duration_s": episode_events.get(
                        "episode_duration_s", 0.0
                    )
                    / completed,
                },
                step,
            )
        writer.flush()

    return callback
