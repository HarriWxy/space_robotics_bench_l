from __future__ import annotations

from types import SimpleNamespace

from srb.integrations.tensorboard import (
    PPO_TENSORBOARD_TAGS,
    SAC_TENSORBOARD_TAGS,
    CanonicalScalarWriter,
    canonical_scalar,
    canonical_tag,
    canonicalize_scalars,
    make_policyflow_tensorboard_cb,
)


class _DummyEnv:
    num_envs = 3


class _ScalarWriter:
    def __init__(self) -> None:
        self.scalars: list[tuple[str, float, int | None]] = []

    def add_scalar(self, tag: str, value: float, step: int | None = None) -> None:
        self.scalars.append((tag, float(value), step))


def test_canonicalizes_legacy_training_tags() -> None:
    assert canonical_tag("Loss/value") == "train/value_loss"
    assert canonical_tag("Loss/surrogate") == "train/policy_gradient_loss"
    assert canonical_tag("Policy/mean_std") == "train/std"
    assert canonical_tag("Perf/total_fps") == "time/fps"
    assert canonical_tag("Train/mean_episode_length") == "rollout/ep_len_mean"
    assert canonical_tag("config/replay_N") == "config/replay_n"
    assert canonical_tag("PostEval_Zero/mean_reward") == "eval/zero/ep_rew_mean"
    assert canonical_scalar("Loss/entropy", 0.5) == (
        "train/entropy_loss",
        -0.5,
    )

    values = canonicalize_scalars(
        {
            "Loss/value": 1.0,
            "train/value_loss": 2.0,
        }
    )
    assert values == {"train/value_loss": 2.0}


def test_canonical_scalar_writer_uses_environment_steps_once() -> None:
    writer = _ScalarWriter()
    wrapped = CanonicalScalarWriter(writer, step_fn=lambda: 48)

    wrapped.add_scalar("Perf/total_fps", 100.0, 0)
    wrapped.add_scalar("time/fps", 200.0, 0)

    assert writer.scalars == [("time/fps", 100.0, 48)]


def test_canonical_scalar_writer_keeps_repeated_same_tag_events() -> None:
    writer = _ScalarWriter()
    wrapped = CanonicalScalarWriter(writer, step_fn=lambda: 48)

    wrapped.add_scalar("PostEval_Zero/mean_reward", 1.0, 0)
    wrapped.add_scalar("PostEval_Zero/mean_reward", 2.0, 0)

    assert writer.scalars == [
        ("eval/zero/ep_rew_mean", 1.0, 48),
        ("eval/zero/ep_rew_mean", 2.0, 48),
    ]


def test_policyflow_tensorboard_uses_srb_ppo_schema(tmp_path) -> None:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    runner = SimpleNamespace(
        _cfg={"rollouts": 2},
        _env=_DummyEnv(),
        _agent=SimpleNamespace(
            _gaussian_entropy_loss_scale=0.01,
            _ratio_clip=0.2,
            cfg={},
        ),
    )
    callback = make_policyflow_tensorboard_cb(str(tmp_path))
    callback(
        runner,
        {
            "current_iteration": 0,
            "training_info": {
                "Loss/policy_loss": -0.25,
                "Loss/gaussian_entropy_loss": -0.01,
                "Loss/value_loss": 0.5,
                "Loss/learning_rate": 1.0e-4,
                "Loss/kl": 0.02,
                "Policy/mean_noise_std": 0.7,
            },
            "returns": [1.0, 3.0],
            "lengths": [4.0, 6.0],
            "info": [
                {
                    "metrics/Tracking_Success": [0.5],
                    "metrics/episode_completed": [1.0],
                    "metrics/episode_success": [1.0],
                    "metrics/episode_failed": [0.0],
                    "metrics/episode_tracking_fraction": [0.75],
                    "metrics/episode_duration_s": [4.0],
                    "metrics/episode_torso_contact_rate": [0.1],
                    "metrics/episode_foot_slip_speed": [0.2],
                }
            ],
            "reward_terms": [{"Reward_Tracking": [1.0, 3.0]}],
        },
    )

    event_file = next(tmp_path.glob("events.*"))
    accumulator = EventAccumulator(str(event_file))
    accumulator.Reload()
    tags = set(accumulator.Tags()["scalars"])
    assert PPO_TENSORBOARD_TAGS <= tags
    assert SAC_TENSORBOARD_TAGS.isdisjoint(PPO_TENSORBOARD_TAGS)
    assert "rollout/metrics/tracking_success" in tags
    assert "rollout/reward_terms/reward_tracking" in tags
    assert "rollout/metrics/episode_completed" in tags
    assert "rollout/episode_success_rate" in tags
    assert "rollout/episode_torso_contact_rate" in tags
    assert "rollout/episode_foot_slip_speed" in tags
    assert "train/tracking_success" not in tags

    for tag in PPO_TENSORBOARD_TAGS:
        events = accumulator.Scalars(tag)
        assert events[-1].step == 6
        assert events[-1].value == events[-1].value
