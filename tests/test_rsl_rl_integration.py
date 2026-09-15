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
        self.cfg = {
            "algorithm": {
                "value_loss_coef": 1.0,
                "entropy_coef": 0.01,
                "clip_param": 0.2,
            }
        }

    def process_env_step(self, *args, **kwargs) -> None:
        pass

    def log(self, *args, **kwargs) -> None:
        pass


class _NativeLogger(_Logger):
    def log(self, *args, **kwargs) -> None:
        self.writer.add_scalar("Loss/value", 2.0, 0)
        self.writer.add_scalar("Loss/surrogate", -0.25, 0)
        self.writer.add_scalar("Loss/entropy", 0.5, 0)
        self.writer.add_scalar("Policy/mean_std", 0.7, 0)
        self.writer.add_scalar("Perf/total_fps", 100.0, 0)
        self.writer.add_scalar("Train/mean_reward", 3.0, 0)
        self.writer.add_scalar("Train/mean_episode_length", 4.0, 0)


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
            "metrics/episode_torso_contact_rate": torch.tensor([0.1, 0.0]),
            "metrics/episode_foot_slip_speed": torch.tensor([0.2, 0.0]),
            "metrics/body_up_z": torch.tensor([0.9, 0.8]),
            "reward_terms": {"Reward_Tracking": torch.tensor([1.0, 3.0])},
        },
    )
    adapter.log()

    scalars = {tag: value for tag, value, _ in logger.writer.scalars}
    assert scalars["rollout/metrics/episode_completed"] == 1.0
    assert scalars["rollout/episode_success_rate"] == 1.0
    assert scalars["rollout/episode_tracking_fraction"] == 0.75
    assert scalars["rollout/episode_torso_contact_rate"] == pytest.approx(0.1)
    assert scalars["rollout/episode_foot_slip_speed"] == pytest.approx(0.2)
    assert abs(scalars["rollout/metrics/body_up_z"] - 0.85) < 1.0e-6
    assert scalars["rollout/reward_terms/reward_tracking"] == 2.0


def test_rsl_rl_logger_canonicalizes_native_fields() -> None:
    logger = _NativeLogger()
    adapter = SrbRslRlLogger(logger)
    adapter.log(loss_dict={"value": 2.0, "surrogate": -0.25, "entropy": 0.5})

    scalars = {tag: value for tag, value, _ in logger.writer.scalars}
    assert scalars["train/value_loss"] == 2.0
    assert scalars["train/policy_gradient_loss"] == -0.25
    assert scalars["train/entropy_loss"] == -0.5
    assert scalars["train/loss"] == 1.745
    assert scalars["train/clip_range"] == 0.2
    assert scalars["train/std"] == 0.7
    assert scalars["time/fps"] == 100.0
    assert scalars["rollout/ep_rew_mean"] == 3.0
    assert scalars["rollout/ep_len_mean"] == 4.0


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
