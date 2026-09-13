"""SRB metric forwarding for the RSL-RL logger."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


class SrbRslRlLogger:
    """Keep RSL-RL logs and add SRB task/reward metrics to TensorBoard."""

    _EPISODE_METRICS = {
        "episode_success",
        "episode_failed",
        "episode_tracking_fraction",
        "episode_duration_s",
    }

    def __init__(self, logger: Any) -> None:
        self._logger = logger
        self._metric_sums: dict[str, float] = {}
        self._metric_counts: dict[str, int] = {}
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

    def log(self, *args: Any, **kwargs: Any) -> Any:
        result = self._logger.log(*args, **kwargs)
        self._write_metrics()
        self._clear_metrics()
        return result

    def _collect_metrics(self, extras: Mapping[str, Any]) -> None:
        for key, value in extras.items():
            if key.startswith("metrics/"):
                name = key.removeprefix("metrics/")
                values = self._finite_values(value)
                if values is None:
                    continue
                if name == "episode_completed":
                    self._completed_episodes += float(values.sum().item())
                elif name in self._EPISODE_METRICS:
                    self._episode_metric_sums[name] = (
                        self._episode_metric_sums.get(name, 0.0)
                        + float(values.sum().item())
                    )
                else:
                    self._add_mean(self._metric_sums, self._metric_counts, name, values)

            elif key == "reward_terms" and isinstance(value, Mapping):
                for term_name, term_value in value.items():
                    values = self._finite_values(term_value)
                    if values is not None:
                        self._add_mean(
                            self._metric_sums,
                            self._metric_counts,
                            f"reward_terms/{term_name}",
                            values,
                        )

    def _write_metrics(self) -> None:
        writer = getattr(self._logger, "writer", None)
        if writer is None:
            return

        step = int(self._logger.tot_timesteps)
        for name, total in self._metric_sums.items():
            count = self._metric_counts[name]
            if count:
                writer.add_scalar(f"rollout/metrics/{name}", total / count, step)

        if self._completed_episodes > 0.0:
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
                if name in self._episode_metric_sums:
                    writer.add_scalar(
                        tag,
                        self._episode_metric_sums[name] / self._completed_episodes,
                        step,
                    )

    def _clear_metrics(self) -> None:
        self._metric_sums.clear()
        self._metric_counts.clear()
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
