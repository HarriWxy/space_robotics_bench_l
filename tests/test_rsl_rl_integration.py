from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from srb.integrations.rsl_rl.logging import SrbRslRlLogger
from srb.integrations.rsl_rl.main import _build_config, _last_checkpoint


class _Writer:
    def __init__(self) -> None:
        self.scalars: list[tuple[str, float, int]] = []

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self.scalars.append((tag, float(value), int(step)))


class _Logger:
    def __init__(self) -> None:
        self.writer = _Writer()
        self.tot_timesteps = 48

    def process_env_step(self, *args, **kwargs) -> None:
        pass

    def log(self, *args, **kwargs) -> None:
        pass


def test_rsl_rl_config_uses_srb_observation_groups() -> None:
    cfg = _build_config(
        {},
        env_cfg=SimpleNamespace(seed=7),
        env_id="srb/locomotion_velocity_tracking_c",
        env_device=torch.device("cpu"),
    )

    assert cfg["seed"] == 7
    assert cfg["obs_groups"] == {
        "actor": ["proprio", "proprio_dyn", "command"],
        "critic": ["proprio", "proprio_dyn", "command"],
    }
    assert cfg["actor"]["hidden_dims"] == [128, 128, 128]
    assert cfg["device"] == "cpu"


def test_last_checkpoint_uses_numeric_iteration(tmp_path) -> None:
    for iteration in (9, 10, 100):
        (tmp_path / f"model_{iteration}.pt").touch()
    (tmp_path / "model_latest.pt").touch()

    assert _last_checkpoint(tmp_path) == tmp_path / "model_100.pt"


def test_rsl_rl_logger_aggregates_completed_episode_metrics() -> None:
    logger = _Logger()
    adapter = SrbRslRlLogger(logger)
    adapter.process_env_step(
        torch.zeros(2),
        torch.ones(2, dtype=torch.long),
        {
            "metrics/episode_completed": torch.tensor([1.0, 0.0]),
            "metrics/episode_success": torch.tensor([1.0, 0.0]),
            "metrics/episode_failed": torch.tensor([0.0, 0.0]),
            "metrics/episode_tracking_fraction": torch.tensor([0.75, 0.0]),
            "metrics/episode_duration_s": torch.tensor([4.0, 0.0]),
            "metrics/body_up_z": torch.tensor([0.9, 0.8]),
        },
    )
    adapter.log()

    scalars = {tag: value for tag, value, _ in logger.writer.scalars}
    assert scalars["rollout/metrics/episode_completed"] == 1.0
    assert scalars["rollout/episode_success_rate"] == 1.0
    assert scalars["rollout/episode_tracking_fraction"] == 0.75
    assert abs(scalars["rollout/metrics/body_up_z"] - 0.85) < 1.0e-6


def test_rsl_rl_logger_reports_success_rate_to_console(monkeypatch) -> None:
    logger = _Logger()
    adapter = SrbRslRlLogger(logger)
    messages: list[str] = []

    import srb.integrations.rsl_rl.logging as rsl_logging

    monkeypatch.setattr(
        rsl_logging.logging,
        "info",
        lambda message, *args: messages.append(message % args),
    )
    adapter.process_env_step(
        torch.zeros(1),
        torch.ones(1, dtype=torch.long),
        {
            "metrics/episode_completed": torch.tensor([1.0]),
            "metrics/episode_success": torch.tensor([0.0]),
            "metrics/episode_failed": torch.tensor([1.0]),
        },
    )
    adapter.log()

    assert any("success_rate=0.0000" in message for message in messages)
