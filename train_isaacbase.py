"""Launch or evaluate the native Isaac Lab Unitree H1 PPO baseline.

This file intentionally delegates training to the Isaac Lab checkout instead
of reimplementing its environment or RSL-RL runner.  The resulting run uses
the registered ``Isaac-Velocity-{Rough,Flat}-H1`` task and its official
RSL-RL PPO configuration.  Pass ``--eval`` to use Isaac Lab's native ``play``
entry point and open a visualizer with a saved checkpoint.
"""

from __future__ import annotations

import argparse
import math
import os
import shlex
import sys
from pathlib import Path
from typing import Sequence


DEFAULT_ISAACLAB_ROOT = Path("/root/isaaclab")
DEFAULT_TASK = "Isaac-Velocity-Flat-H1"
DEFAULT_NUM_ENVS = 256
DEFAULT_TOTAL_ENV_STEPS = 100_000_000
DEFAULT_SEED = 0
DEFAULT_RUN_NAME = "srb_h1_ppo_baseline"
DEFAULT_EVAL_NUM_ENVS = 8
DEFAULT_EVAL_VISUALIZER = "kit"

# Both base H1 RSL-RL agent configs use 24 rollout steps per environment.
ROLLOUT_STEPS_PER_ENV = {
    "Isaac-Velocity-Rough-H1": 24,
    "Isaac-Velocity-Flat-H1": 24,
}
TASKS = tuple(ROLLOUT_STEPS_PER_ENV)
PHYSICS_PRESETS = ("isaacsim_physx", "newton_kamino", "newton_mjwarp", "ovphysx")


def _positive_int(value: str) -> int:
    """Parse a strictly positive integer for a training-size argument."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse launcher options and leave trailing tokens for Isaac Lab Hydra."""
    parser = argparse.ArgumentParser(
        description="Run Isaac Lab's native RSL-RL PPO baseline on Unitree H1."
    )
    parser.add_argument(
        "--isaaclab-root",
        type=Path,
        default=Path(os.environ.get("ISAACLAB_ROOT", DEFAULT_ISAACLAB_ROOT)),
        help="Isaac Lab source checkout (default: /root/isaaclab or ISAACLAB_ROOT).",
    )
    parser.add_argument("--task", choices=TASKS, default=DEFAULT_TASK)
    parser.add_argument(
        "--num-envs",
        "--num_envs",
        dest="num_envs",
        type=_positive_int,
        default=DEFAULT_NUM_ENVS,
        help="Number of parallel environments.",
    )
    parser.add_argument(
        "--total-env-steps",
        type=_positive_int,
        default=DEFAULT_TOTAL_ENV_STEPS,
        help="Target environment transitions used when --max-iterations is omitted.",
    )
    parser.add_argument(
        "--max-iterations",
        "--max_iterations",
        dest="max_iterations",
        type=_positive_int,
        default=None,
        help="Explicit Isaac Lab policy iterations; overrides --total-env-steps.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--eval",
        "--play",
        dest="eval",
        action="store_true",
        help="Evaluate a checkpoint with Isaac Lab's play entry point.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path or selector (latest, best, pretrained); used for eval or resume.",
    )
    parser.add_argument(
        "--eval-num-envs",
        "--eval_num_envs",
        dest="eval_num_envs",
        type=_positive_int,
        default=DEFAULT_EVAL_NUM_ENVS,
        help="Number of environments used by visual evaluation.",
    )
    parser.add_argument(
        "--visualizer",
        "--viz",
        dest="visualizer",
        default=DEFAULT_EVAL_VISUALIZER,
        help="Visualizer passed to Isaac Lab play (default: kit).",
    )
    parser.add_argument("--video", action="store_true", help="Record a video during eval or training.")
    parser.add_argument(
        "--video-length",
        "--video_length",
        dest="video_length",
        type=_positive_int,
        default=None,
        help="Video length in environment steps when --video is enabled.",
    )
    parser.add_argument(
        "--physics",
        choices=PHYSICS_PRESETS,
        default="isaacsim_physx",
        help="Optional Isaac Lab physics preset, forwarded as physics=<value>.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help='Optional Isaac Lab simulation device, for example "cuda" or "cuda:1".',
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES value for the child process.",
    )
    parser.add_argument(
        "--experiment-name",
        "--experiment_name",
        dest="experiment_name",
        default=None,
    )
    parser.add_argument(
        "--run-name",
        "--run_name",
        dest="run_name",
        default=DEFAULT_RUN_NAME,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved command without launching Isaac Sim.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Additional Isaac Lab/Hydra overrides, e.g. env.scene.num_envs=2.",
    )
    return parser.parse_args(argv)


def build_command(args: argparse.Namespace) -> tuple[list[str], int | None]:
    """Build the Isaac Lab CLI arguments for the train or play workflow."""
    if args.eval:
        command = [
            "play",
            "--rl_library",
            "rsl_rl",
            "--task",
            args.task,
            "--checkpoint",
            args.checkpoint or "latest",
            "--num_envs",
            str(args.eval_num_envs),
            "--viz",
            args.visualizer,
            "--seed",
            str(args.seed),
        ]
        if args.device is not None:
            command.extend(["--device", args.device])
        if args.experiment_name is not None:
            command.extend(["--experiment_name", args.experiment_name])
        if args.video:
            command.append("--video")
            if args.video_length is not None:
                command.extend(["--video_length", str(args.video_length)])
        if args.physics is not None:
            command.append(f"physics={args.physics}")
        command.extend(args.overrides)
        return command, None

    rollout_steps = ROLLOUT_STEPS_PER_ENV[args.task]
    max_iterations = args.max_iterations
    if max_iterations is None:
        max_iterations = math.ceil(args.total_env_steps / (args.num_envs * rollout_steps))

    command = [
        "train",
        "--rl_library",
        "rsl_rl",
        "--task",
        args.task,
        "--num_envs",
        str(args.num_envs),
        "--max_iterations",
        str(max_iterations),
        "--seed",
        str(args.seed),
        # "--headless",
    ]
    if args.device is not None:
        command.extend(["--device", args.device])
    if args.experiment_name is not None:
        command.extend(["--experiment_name", args.experiment_name])
    if args.run_name:
        command.extend(["--run_name", args.run_name])
    if args.checkpoint is not None:
        command.extend(["--checkpoint", args.checkpoint])
    if args.video:
        command.append("--video")
        if args.video_length is not None:
            command.extend(["--video_length", str(args.video_length)])
    if args.physics is not None:
        command.append(f"physics={args.physics}")
    command.extend(args.overrides)
    return command, max_iterations


def run_isaaclab(
    argv: Sequence[str], *, cwd: Path, cuda_visible_devices: str | None = None
) -> int:
    """Run Isaac Lab in-process using the active Python environment.

    Isaac Lab's public console entry point reads ``sys.argv``.  This mirrors
    the local ``run_srb`` launchers while restoring process-global state after
    the workflow exits.
    """
    old_argv = sys.argv[:]
    old_cwd = Path.cwd()
    had_cuda_visible_devices = "CUDA_VISIBLE_DEVICES" in os.environ
    old_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        os.chdir(cwd)
        if cuda_visible_devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

        # Import only after the requested environment variables and cwd are in
        # place because Isaac Sim-related modules may read them at import time.
        from isaaclab.cli import cli

        sys.argv = [old_argv[0], *argv]
        cli()
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)
        if had_cuda_visible_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = old_cuda_visible_devices or ""
        else:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Validate the base checkout and launch Isaac Lab's native workflow."""
    args = parse_args(argv)
    isaaclab_root = args.isaaclab_root.expanduser().resolve()
    if not (isaaclab_root / "pyproject.toml").is_file():
        raise FileNotFoundError(f"Isaac Lab checkout not found: {isaaclab_root}")

    command, max_iterations = build_command(args)
    if args.eval:
        print(
            f"[isaacbase] mode=eval task={args.task} rl=rsl_rl(PPO) seed={args.seed} "
            f"num_envs={args.eval_num_envs} checkpoint={args.checkpoint or 'latest'} "
            f"visualizer={args.visualizer}"
        )
    else:
        rollout_steps = ROLLOUT_STEPS_PER_ENV[args.task]
        actual_env_steps = max_iterations * args.num_envs * rollout_steps
        print(
            f"[isaacbase] mode=train task={args.task} rl=rsl_rl(PPO) seed={args.seed} "
            f"num_envs={args.num_envs} rollout_steps={rollout_steps} "
            f"max_iterations={max_iterations} actual_env_steps={actual_env_steps}"
        )
    print(f"[isaacbase] cwd={isaaclab_root}")
    print(f"[isaacbase] command={shlex.join(['isaaclab', *command])}")

    if args.dry_run:
        return 0

    return run_isaaclab(
        command,
        cwd=isaaclab_root,
        cuda_visible_devices=args.cuda_visible_devices,
    )


if __name__ == "__main__":
    raise SystemExit(main())
