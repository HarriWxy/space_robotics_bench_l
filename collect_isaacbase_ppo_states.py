"""Collect a reusable raw-state bank under a frozen native Gaussian PPO teacher."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from train_isaacbase_fpo import (
    _insert_fpo_root,
    _load_fpo_cfg,
    _positive_int,
    _prepare_env_cfg,
    parse_args,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Use the native H1 lifecycle; collect states without modifying the teacher."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Native launcher options such as --device, --task, --num-envs and --dry-run are accepted.",
    )
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-states", type=_positive_int, default=256)
    parser.add_argument("--samples-per-state", type=_positive_int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=256)
    parser.add_argument("--collection-steps", type=_positive_int, default=512)
    bank_args, remaining = parser.parse_known_args(
        sys.argv[1:] if argv is None else argv
    )
    if bank_args.warmup_steps < 0:
        parser.error("--warmup-steps must be non-negative")
    shared = parse_args(
        [
            "--fpo-config",
            str(Path(__file__).resolve().parent / "hyperparams" / "pmf_next_step.yaml"),
            "--task",
            "Isaac-Velocity-Flat-H1",
            "--num-envs",
            "8",
            "--seed",
            "12345",
            "--max-iterations",
            "1",
            *remaining,
        ]
    )
    if shared.eval or shared.resume or shared.pmf_next_step or shared.pmf_ablation:
        parser.error(
            "The collector runs a frozen PPO teacher; training/evaluation profile flags do not apply"
        )
    _insert_fpo_root(shared.fpo_root)
    from isaaclab_fpo.pmf_teacher import (
        collect_teacher_state_bank,
        load_native_ppo_teacher,
    )

    # CPU reconstruction also verifies the actual saved actor schema on dry run.
    teacher = load_native_ppo_teacher(
        bank_args.teacher_checkpoint, "cpu" if shared.dry_run else shared.device
    )
    agent_cfg, _ = _load_fpo_cfg(shared)
    env_cfg = _prepare_env_cfg(shared, agent_cfg)
    # Use the teacher's executed-action clipping contract, not the pMF config's.
    clip_actions = teacher.config.get("clip_actions")
    resolved = {
        "teacher_checkpoint": str(teacher.checkpoint_path),
        "task": shared.task,
        "num_envs": shared.num_envs,
        "num_states": bank_args.num_states,
        "warmup_steps": bank_args.warmup_steps,
        "collection_steps": bank_args.collection_steps,
        "raw_actor_observation_dim": teacher.num_actor_obs,
        "action_dim": teacher.num_actions,
        "wrapper_action_clip": clip_actions,
        "output": str(bank_args.output.resolve()),
    }
    if shared.dry_run:
        print(json.dumps(resolved, indent=2))
        return 0
    shared.enable_cameras = False
    shared.require_kit = bool(
        getattr(shared, "visualizer", None) and "kit" in shared.visualizer
    )
    import gymnasium as gym
    import torch
    from isaaclab.app import launch_simulation

    with launch_simulation(env_cfg, shared):
        from isaaclab_fpo import FpoRslRlVecEnvWrapper

        env = gym.make(shared.task, cfg=env_cfg)
        try:
            env = FpoRslRlVecEnvWrapper(env, clip_actions=clip_actions)
            bank = collect_teacher_state_bank(
                env,
                teacher,
                shared.task,
                num_states=bank_args.num_states,
                samples_per_state=bank_args.samples_per_state,
                warmup_steps=bank_args.warmup_steps,
                collection_steps=bank_args.collection_steps,
                seed=shared.seed,
            )
            bank_args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(bank, bank_args.output)
            resolved["environment_contract"] = bank["environment_contract"]
            bank_args.output.with_suffix(".json").write_text(
                json.dumps(resolved, indent=2), encoding="utf-8"
            )
        finally:
            env.close()
    print(f"Saved {bank_args.num_states} raw teacher states to {bank_args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
