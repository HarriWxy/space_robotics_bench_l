from __future__ import annotations

from types import SimpleNamespace

from srb.integrations.tensorboard import (
    PPO_TENSORBOARD_TAGS,
    SAC_TENSORBOARD_TAGS,
    make_policyflow_tensorboard_cb,
)


class _DummyEnv:
    num_envs = 3


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
            "info": [],
        },
    )

    event_file = next(tmp_path.glob("events.*"))
    accumulator = EventAccumulator(str(event_file))
    accumulator.Reload()
    tags = set(accumulator.Tags()["scalars"])
    assert PPO_TENSORBOARD_TAGS <= tags
    assert SAC_TENSORBOARD_TAGS.isdisjoint(PPO_TENSORBOARD_TAGS)

    for tag in PPO_TENSORBOARD_TAGS:
        events = accumulator.Scalars(tag)
        assert events[-1].step == 6
        assert events[-1].value == events[-1].value

