"""Check resolved experimental contracts without creating simulator scenes."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

pytest.importorskip("isaaclab")

from train_isaacbase_fpo import (
    _load_fpo_cfg,
    _prepare_env_cfg,
    _training_iterations,
    parse_args,
)
from train_isaacbase_fsppo_joint import build_args as build_legacy_args
from train_isaacbase_pmf_next import build_args


def test_profiles_resolve_common_sampling_and_equal_environment_budget():
    resolved = {}
    for variant in ("joint_pg", "pmd", "pmd_tr", "joint_pmf"):
        args = parse_args(build_args(["--variant", variant, "--device", "cpu"]))
        cfg, _ = _load_fpo_cfg(args)
        assert cfg.max_iterations * args.num_envs * cfg.num_steps_per_env == 15_003_648
        assert (
            cfg.policy.actor_hidden_dims
            == cfg.policy.critic_hidden_dims
            == [128, 128, 128]
        )
        assert cfg.policy.action_perturb_std == 0.1
        assert not cfg.policy.learn_action_perturb_std
        assert cfg.algorithm.action_perturb_std_final is None
        assert cfg.algorithm.ema_decay == 0.0
        assert cfg.algorithm.pmd_projection_epochs == cfg.algorithm.num_learning_epochs
        assert cfg.algorithm.num_mini_batches == 4
        resolved[variant] = cfg.to_dict()
    # B/C are the same pure projection experiment, differing only in trust region.
    b, c = resolved["pmd"], resolved["pmd_tr"]
    assert not b["algorithm"].pop("fsppo_joint_enable_budget")
    assert c["algorithm"].pop("fsppo_joint_enable_budget")
    b.pop("run_name")
    c.pop("run_name")
    assert b == c
    a, d = resolved["joint_pg"], resolved["joint_pmf"]
    assert a["algorithm"].pop("fsppo_joint_aux_loss_coef") == 0.0
    assert d["algorithm"].pop("fsppo_joint_aux_loss_coef") == 0.01
    a.pop("run_name")
    d.pop("run_name")
    assert a == d


def test_pmd_enables_native_final_observations_and_rejects_learned_sigma():
    args = parse_args(
        build_args(
            [
                "--variant",
                "pmd",
                "--device",
                "cpu",
                "--num-envs",
                "2",
                "--max-iterations",
                "2",
            ]
        )
    )
    cfg, _ = _load_fpo_cfg(args)
    env_cfg = _prepare_env_cfg(args, cfg)
    assert env_cfg.compute_final_obs
    assert env_cfg.scene.num_envs == 2
    args.resume = True
    runner = SimpleNamespace(tot_timesteps=48, current_learning_iteration=1)
    assert _training_iterations(args, cfg, runner) == 1
    args.max_iterations = None
    args.total_env_steps = 96
    assert _training_iterations(args, cfg, runner) == 1
    runner.tot_timesteps = 96
    assert _training_iterations(args, cfg, runner) == 0
    with pytest.raises(SystemExit):
        parse_args(build_args(["--variant", "pmd", "--learn-terminal-std"]))


def test_evaluation_preserves_distilled_and_aligned_saved_configs(tmp_path: Path):
    args = parse_args(
        build_args(["--variant", "joint_pg", "--device", "cpu", "--learn-terminal-std"])
    )
    cfg, _ = _load_fpo_cfg(args)
    assert cfg.policy.learn_action_perturb_std
    path = tmp_path / "agent.yaml"
    for policy, algorithm in (
        ("LearnedStdPMFActorCritic", "FSPPOJointAligned"),
        ("PMFActorCritic", "PMFFPO"),
    ):
        cfg.policy.class_name = policy
        cfg.algorithm.class_name = algorithm
        path.write_text(yaml.dump(cfg.to_dict()), encoding="utf-8")
        evaluation = parse_args(
            build_legacy_args(
                [
                    "--eval",
                    "--device",
                    "cpu",
                    "--fpo-config",
                    str(path),
                    "--max-iterations",
                    "2442",
                ]
            )
        )
        loaded, _ = _load_fpo_cfg(evaluation)
        assert evaluation.pmf_ablation is None
        assert loaded.policy.class_name == policy
        assert loaded.algorithm.class_name == algorithm
        assert loaded.algorithm.fsppo_joint_aux_loss_coef == 0.0


def test_saved_seed_and_relative_checkpoint_survive_native_launcher_chdir(
    tmp_path, monkeypatch
):
    """A saved profile must keep its seed and find paths relative to invocation."""
    from train_isaacbase_fpo import main

    args = parse_args(
        build_args(["--variant", "pmd", "--seed", "2", "--device", "cpu"])
    )
    cfg, _ = _load_fpo_cfg(args)
    run = tmp_path / "run"
    (run / "params").mkdir(parents=True)
    saved = run / "params" / "agent.yaml"
    saved.write_text(yaml.dump(cfg.to_dict()), encoding="utf-8")
    # dry-run only reads config/path metadata, never deserializes this marker.
    (run / "model.pt").touch()
    for explicit, expected in (([], 2), (["--seed", "9"], 9)):
        evaluation = parse_args(
            [
                "--eval",
                "--use-checkpoint-config",
                "--fpo-config",
                str(saved),
                "--device",
                "cpu",
                *explicit,
            ]
        )
        loaded, _ = _load_fpo_cfg(evaluation)
        assert loaded.seed == expected
    monkeypatch.chdir(tmp_path)
    assert (
        main(
            [
                "--eval",
                "--use-checkpoint-config",
                "--checkpoint",
                "run/model.pt",
                "--task",
                "Isaac-Velocity-Flat-H1",
                "--device",
                "cpu",
                "--dry-run",
            ]
        )
        == 0
    )
