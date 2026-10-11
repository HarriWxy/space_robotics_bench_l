"""CPU contracts for controlled pMF ablations and saved config reconstruction."""

from pathlib import Path

import pytest
import yaml

pytest.importorskip("isaaclab")

from train_isaacbase_fpo import _load_fpo_cfg, parse_args
from train_isaacbase_fsppo_joint import build_args


def test_pmf_ablation_changes_only_the_auxiliary_loss_coefficient() -> None:
    """A/B must keep the sampler, PPO, normalization and training budget equal."""
    common = ["--device", "cpu", "--num-envs", "2", "--total-env-steps", "128"]
    configs = []
    for variant in ("pg_only", "pg_pmf"):
        args = parse_args(build_args([*common, "--pmf-ablation", variant]))
        config, _ = _load_fpo_cfg(args)
        assert config.max_iterations == 2
        configs.append(config.to_dict())
    coefficient_a = configs[0]["algorithm"].pop("fsppo_joint_aux_loss_coef")
    coefficient_b = configs[1]["algorithm"].pop("fsppo_joint_aux_loss_coef")
    assert coefficient_a == 0.0
    assert coefficient_b > 0.0
    assert configs[0] == configs[1]


def test_saved_agent_config_reconstructs_policy_scales_and_tuple_settings(tmp_path: Path) -> None:
    """Playback must accept dump_yaml tuples and keep non-parameter policy settings."""
    args = parse_args(build_args(["--device", "cpu", "--num-envs", "2", "--max-iterations", "2"]))
    config, _ = _load_fpo_cfg(args)
    config.policy.actor_scale = 0.375
    config.policy.actor_mlp_output_scale = 0.625
    config.policy.pmf_time_eps = 0.02
    config.algorithm.adam_betas = (0.8, 0.98)
    saved_path = tmp_path / "agent.yaml"
    saved_path.write_text(yaml.dump(config.to_dict()), encoding="utf-8")
    replay_args = parse_args(build_args([
        "--eval", "--device", "cpu", "--fpo-config", str(saved_path),
        "--num-envs", "2", "--max-iterations", "2",
    ]))
    reconstructed, _ = _load_fpo_cfg(replay_args)
    assert reconstructed.to_dict() == config.to_dict()
