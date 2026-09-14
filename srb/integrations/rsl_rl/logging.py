"""SRB metric forwarding for the RSL-RL logger."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import torch

from srb.integrations.tensorboard import (
    canonical_tag,
    patch_scalar_writer,
    scalar_value,
)
from srb.utils import logging


class SrbRslRlLogger:
    """Keep RSL-RL logs and add SRB task/reward metrics to TensorBoard."""

    _EPISODE_METRICS: ClassVar[frozenset[str]] = frozenset(
        {
            "episode_success",
            "episode_failed",
            "episode_tracking_fraction",
            "episode_duration_s",
        }
    )

    def __init__(self, logger: Any) -> None:
        self._logger = logger
        self._metric_sums: dict[str, float] = {}
        self._metric_counts: dict[str, int] = {}
        self._reward_term_sums: dict[str, float] = {}
        self._reward_term_counts: dict[str, int] = {}
        self._episode_metric_sums: dict[str, float] = {}
        self._completed_episodes = 0.0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._logger, name)

    def process_env_step(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: Mapping[str, Any],
        intrinsic_rewards: torch.Tensor | None = None,
    ) -> None:
        self._collect_metrics(extras)
        self._logger.process_env_step(rewards, dones, dict(extras), intrinsic_rewards)

    def init_logging_writer(self, *args: Any, **kwargs: Any) -> Any:
        """Initialize the native writer and attach the shared SRB schema."""

        result = self._logger.init_logging_writer(*args, **kwargs)
        self._patch_writer()
        return result

    def log(self, *args: Any, **kwargs: Any) -> Any:
        self._patch_writer()
        try:
            result = self._logger.log(*args, **kwargs)
            self._write_algorithm_metrics(args, kwargs)
            self._write_metrics()
            return result
        finally:
            self._clear_metrics()

    def _patch_writer(self) -> None:
        writer = getattr(self._logger, "writer", None)
        if writer is not None:
            self._logger.writer = patch_scalar_writer(
                writer,
                step_fn=lambda: int(self._logger.tot_timesteps),
            )

    def _algorithm_cfg(self) -> Mapping[str, Any]:
        cfg = getattr(self._logger, "cfg", {})
        algorithm_cfg = cfg.get("algorithm", {}) if isinstance(cfg, Mapping) else {}
        return algorithm_cfg if isinstance(algorithm_cfg, Mapping) else {}

    def _write_algorithm_metrics(
        self, args: tuple[Any, ...], kwargs: Mapping[str, Any]
    ) -> None:
        """Fill common PPO fields not emitted by native RSL-RL."""

        writer = getattr(self._logger, "writer", None)
        if writer is None:
            return

        loss_dict = kwargs.get("loss_dict")
        if loss_dict is None and len(args) > 5:
            loss_dict = args[5]
        if not isinstance(loss_dict, Mapping):
            loss_dict = {}

        metrics = loss_dict.get("metrics", {})
        if not isinstance(metrics, Mapping):
            metrics = {}

        def first_value(names: tuple[str, ...], default: Any = None) -> Any:
            for name in names:
                if name in loss_dict:
                    return loss_dict[name]
                if name in metrics:
                    return metrics[name]
            return default

        def as_float(value: Any) -> float | None:
            if value is None:
                return None
            try:
                return scalar_value(value)
            except (TypeError, ValueError):
                return None

        surrogate_loss = as_float(
            first_value(("surrogate", "surrogate_loss", "policy_gradient_loss"))
        )
        value_loss = as_float(first_value(("value", "value_loss")))
        entropy = as_float(first_value(("entropy", "entropy_loss")))
        if (
            surrogate_loss is not None
            and value_loss is not None
            and entropy is not None
        ):
            algorithm_cfg = self._algorithm_cfg()
            value_coef = float(algorithm_cfg.get("value_loss_coef", 1.0))
            entropy_coef = float(algorithm_cfg.get("entropy_coef", 0.0))
            writer.add_scalar(
                "train/loss",
                surrogate_loss + value_coef * value_loss - entropy_coef * entropy,
                int(self._logger.tot_timesteps),
            )

        clip_range = first_value(("clip_param", "clip_range"))
        if clip_range is None:
            clip_range = self._algorithm_cfg().get("clip_param")
        clip_range = as_float(clip_range)
        if clip_range is not None:
            writer.add_scalar(
                "train/clip_range", clip_range, int(self._logger.tot_timesteps)
            )

        for target, names in {
            "train/approx_kl": ("approx_kl", "kl"),
            "train/clip_fraction": ("clip_fraction",),
            "train/explained_variance": ("explained_variance",),
        }.items():
            value = as_float(first_value(names))
            if value is not None:
                writer.add_scalar(target, value, int(self._logger.tot_timesteps))

    def _collect_metrics(self, extras: Mapping[str, Any]) -> None:
        for key, value in extras.items():
            if key.startswith("metrics/"):
                name = canonical_tag(
                    f"rollout/metrics/{key.removeprefix('metrics/')}"
                ).removeprefix("rollout/metrics/")
                values = self._finite_values(value)
                if values is None:
                    continue
                if name == "episode_completed":
                    self._completed_episodes += float(values.sum().item())
                elif name in self._EPISODE_METRICS:
                    self._episode_metric_sums[name] = self._episode_metric_sums.get(
                        name, 0.0
                    ) + float(values.sum().item())
                else:
                    self._add_mean(self._metric_sums, self._metric_counts, name, values)

            elif key == "reward_terms" and isinstance(value, Mapping):
                for term_name, term_value in value.items():
                    values = self._finite_values(term_value)
                    if values is not None:
                        self._add_mean(
                            self._reward_term_sums,
                            self._reward_term_counts,
                            canonical_tag(
                                f"rollout/reward_terms/{term_name}"
                            ).removeprefix("rollout/reward_terms/"),
                            values,
                        )

    def _write_metrics(self) -> None:
        episode_rates = self._episode_rates()
        if episode_rates:
            logging.info(
                "RSL-RL task metrics: completed=%d success_rate=%.4f "
                "failure_rate=%.4f tracking_fraction=%.4f duration_s=%.4f",
                round(self._completed_episodes),
                episode_rates["episode_success"],
                episode_rates["episode_failed"],
                episode_rates["episode_tracking_fraction"],
                episode_rates["episode_duration_s"],
            )

        writer = getattr(self._logger, "writer", None)
        if writer is None:
            return

        step = int(self._logger.tot_timesteps)
        for name, total in self._reward_term_sums.items():
            count = self._reward_term_counts[name]
            if count:
                writer.add_scalar(f"rollout/reward_terms/{name}", total / count, step)

        for name, total in self._metric_sums.items():
            count = self._metric_counts[name]
            if count:
                writer.add_scalar(f"rollout/metrics/{name}", total / count, step)

        if episode_rates:
            writer.add_scalar(
                "rollout/metrics/episode_completed", self._completed_episodes, step
            )
            rates = {
                "episode_success": "rollout/episode_success_rate",
                "episode_failed": "rollout/episode_failure_rate",
                "episode_tracking_fraction": "rollout/episode_tracking_fraction",
                "episode_duration_s": "rollout/episode_duration_s",
            }
            for name, tag in rates.items():
                writer.add_scalar(tag, episode_rates[name], step)

    def _episode_rates(self) -> dict[str, float]:
        if self._completed_episodes <= 0.0:
            return {}
        return {
            name: self._episode_metric_sums.get(name, 0.0) / self._completed_episodes
            for name in self._EPISODE_METRICS
        }

    def _clear_metrics(self) -> None:
        self._metric_sums.clear()
        self._metric_counts.clear()
        self._reward_term_sums.clear()
        self._reward_term_counts.clear()
        self._episode_metric_sums.clear()
        self._completed_episodes = 0.0

    @staticmethod
    def _add_mean(
        sums: dict[str, float],
        counts: dict[str, int],
        name: str,
        values: torch.Tensor,
    ) -> None:
        sums[name] = sums.get(name, 0.0) + float(values.sum().item())
        counts[name] = counts.get(name, 0) + int(values.numel())

    @staticmethod
    def _finite_values(value: Any) -> torch.Tensor | None:
        if isinstance(value, torch.Tensor):
            values = value.detach().float().reshape(-1)
        else:
            try:
                values = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
            except (TypeError, ValueError):
                return None
        finite = torch.isfinite(values)
        return values[finite] if finite.any() else None
