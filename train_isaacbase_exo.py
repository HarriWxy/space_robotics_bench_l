"""Run ExO-FPO on Isaac Lab's native Unitree H1 velocity task.

Isaac Lab provides the registered H1 environment and the native RSL-RL PPO
baseline, but it does not register ExO-FPO as an RL library.  This launcher
keeps the native environment and runs SRB's PyTorch ExO-FPO implementation in
the active Python process.  The native ``policy`` observation group is passed
through unchanged (and flattened by the ExO adapter), while native's
unbounded action space is replaced by the explicit flow-action clip range.

Examples (from the repository root, in ``conda activate srb``)::

    python train_isaacbase_exo.py --task Isaac-Velocity-Flat-H1 \
        --num-envs 256 --total-env-steps 100000000 --headless

    python train_isaacbase_exo.py --task Isaac-Velocity-Rough-H1 \
        --num-envs 256 --max-iterations 10 --headless

    python train_isaacbase_exo.py --eval --checkpoint latest \
        --task Isaac-Velocity-Flat-H1 --eval-num-envs 8 --viz kit

The launcher does not use ``subprocess`` or ``uv run``.  It uses the Isaac Lab
and ExO-PPO packages visible in the current Python environment.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_ISAACLAB_ROOT = Path("/root/isaaclab")
DEFAULT_EXO_ROOT = Path("/root/R2A/Algos/ExO-PPO")
DEFAULT_TASK = "Isaac-Velocity-Rough-H1"
DEFAULT_NUM_ENVS = 256
DEFAULT_EVAL_NUM_ENVS = 8
DEFAULT_SEED = 0
DEFAULT_EXPERIMENT_NAME = "h1_exofpo"
DEFAULT_RUN_NAME = "h1_exofpo"
DEFAULT_TOTAL_ENV_STEPS = 100_000_000
DEFAULT_EVAL_EPISODES = 5
DEFAULT_VIDEO_LENGTH = 200

TASKS = ("Isaac-Velocity-Rough-H1", "Isaac-Velocity-Flat-H1")
PHYSICS_PRESETS = ("isaacsim_physx", "newton_kamino", "newton_mjwarp", "ovphysx")
CHECKPOINT_SELECTORS = {"latest", "best"}


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse launcher options and preserve trailing Isaac Lab overrides."""

    parser = argparse.ArgumentParser(
        description="Run ExO-FPO on Isaac Lab's native Unitree H1 velocity task."
    )
    parser.add_argument(
        "--isaaclab-root",
        type=Path,
        default=Path(os.environ.get("ISAACLAB_ROOT", DEFAULT_ISAACLAB_ROOT)),
        help="Isaac Lab source checkout.",
    )
    parser.add_argument(
        "--exo-root",
        "--exoppo-root",
        dest="exo_root",
        type=Path,
        default=Path(os.environ.get("EXO_PPO_ROOT", DEFAULT_EXO_ROOT)),
        help="ExO-PPO checkout containing the flow package.",
    )
    parser.add_argument(
        "--exo-config",
        "--exofpo-config",
        dest="exo_config",
        type=Path,
        default=Path(__file__).resolve().parent / "hyperparams" / "exofpo.yaml",
        help="YAML file for the ExO-FPO runner and flow policy.",
    )
    parser.add_argument("--task", choices=TASKS, default=DEFAULT_TASK)
    parser.add_argument(
        "--num-envs",
        "--num_envs",
        dest="num_envs",
        type=_positive_int,
        default=DEFAULT_NUM_ENVS,
        help="Number of parallel environments for training.",
    )
    parser.add_argument(
        "--eval-num-envs",
        "--eval_num_envs",
        dest="eval_num_envs",
        type=_positive_int,
        default=DEFAULT_EVAL_NUM_ENVS,
        help="Number of parallel environments for evaluation.",
    )
    parser.add_argument(
        "--total-env-steps",
        "--total_env_steps",
        dest="total_env_steps",
        type=_positive_int,
        default=None,
        help=(
            f"Target training transitions; defaults to the max_iterations in the YAML "
            f"config ({DEFAULT_TOTAL_ENV_STEPS:,} is the usual 100M comparison)."
        ),
    )
    parser.add_argument(
        "--max-iterations",
        "--max_iterations",
        dest="max_iterations",
        type=_positive_int,
        default=None,
        help="Explicit ExO-FPO iterations; takes precedence over --total-env-steps.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--experiment-name", "--experiment_name", dest="experiment_name")
    parser.add_argument(
        "--run-name",
        "--run_name",
        dest="run_name",
        default=DEFAULT_RUN_NAME,
    )
    parser.add_argument("--log-root", "--log_root", dest="log_root", type=Path, default=None)
    parser.add_argument(
        "--clip-actions",
        "--clip_actions",
        dest="clip_actions",
        type=float,
        default=None,
        help="Flow action clip; defaults to clip_actions in the ExO-FPO YAML (1.0).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path, or 'latest'/'best' when evaluating or resuming.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume runner/optimizer state from --checkpoint or the latest run.",
    )
    parser.add_argument(
        "--eval",
        "--play",
        dest="eval",
        action="store_true",
        help="Evaluate a saved ExO-FPO checkpoint instead of training.",
    )
    parser.add_argument(
        "--untrained",
        action="store_true",
        help="Allow --eval without a checkpoint and evaluate a fresh policy.",
    )
    parser.add_argument(
        "--eval-episodes",
        "--eval_episodes",
        dest="eval_episodes",
        type=_positive_int,
        default=DEFAULT_EVAL_EPISODES,
        help="Approximate complete episodes per evaluation environment.",
    )
    parser.add_argument(
        "--eval-steps",
        "--eval_steps",
        dest="eval_steps",
        type=_nonnegative_int,
        default=None,
        help="Evaluation environment steps; 0 means use --eval-episodes * horizon.",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Enable Isaac Lab's native VideoRecorder during train/eval.",
    )
    parser.add_argument(
        "--video-length",
        "--video_length",
        dest="video_length",
        type=_positive_int,
        default=DEFAULT_VIDEO_LENGTH,
        help="Length of each native video clip in environment steps.",
    )
    parser.add_argument(
        "--video-interval",
        "--video_interval",
        dest="video_interval",
        type=_nonnegative_int,
        default=None,
        help="Interval between native video clips; the Isaac Lab default is used if omitted.",
    )
    parser.add_argument(
        "--physics",
        choices=PHYSICS_PRESETS,
        default="isaacsim_physx",
        help="Default Isaac Lab physics selector, forwarded as physics=<value>.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve config and print the run contract without launching Isaac Sim.",
    )

    # AppLauncher contributes --device, --viz/--visualizer, --kit_args, etc.
    # Importing it here is safe: constructing the launcher is deferred to the
    # Isaac Lab launch_simulation context below.
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Force headless Isaac Sim execution.",
    )
    # Use the same spelling accepted by Isaac Lab's CLI while still allowing
    # the active srb environment to choose cuda:0/cuda:1 explicitly.
    parser.set_defaults(device="cuda")

    args, remaining = parser.parse_known_args(argv)
    args.hydra_overrides = list(remaining)
    return args


def _insert_exo_root(exo_root: Path) -> None:
    """Make the Python 3.12-compatible ExO-PPO flow package importable."""

    flow_package = exo_root / "flow" / "__init__.py"
    if not flow_package.is_file():
        raise FileNotFoundError(
            f"ExO-PPO flow package not found under {exo_root}; expected {flow_package}"
        )
    resolved = exo_root.resolve()
    if str(resolved) not in sys.path:
        sys.path.insert(0, str(resolved))


def _load_exo_cfg(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    """Load the ExO-FPO YAML and apply launcher-owned values."""

    import yaml

    config_path = args.exo_config.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"ExO-FPO config not found: {config_path}")
    with config_path.open(encoding="utf-8") as config_file:
        loaded = yaml.safe_load(config_file) or {}
    if not isinstance(loaded, Mapping):
        raise TypeError(f"ExO-FPO config must contain a mapping: {config_path}")

    config: dict[str, Any] = dict(loaded)
    config["seed"] = int(args.seed)
    config["device"] = args.device

    rollout_steps = int(config.get("rollout_steps", 256))
    if rollout_steps <= 0:
        raise ValueError("ExO-FPO rollout_steps must be positive")
    if args.max_iterations is not None:
        max_iterations = int(args.max_iterations)
    elif args.total_env_steps is not None:
        max_iterations = math.ceil(
            int(args.total_env_steps) / (int(args.num_envs) * rollout_steps)
        )
    else:
        max_iterations = int(config.get("max_iterations", 1500))
    if max_iterations <= 0:
        raise ValueError("ExO-FPO max_iterations must be positive")
    config["max_iterations"] = max_iterations

    if args.clip_actions is not None:
        if args.clip_actions <= 0.0:
            raise ValueError("--clip-actions must be positive")
        config["clip_actions"] = float(args.clip_actions)
    elif config.get("clip_actions", 1.0) is not None:
        config["clip_actions"] = float(config["clip_actions"])

    config["eval_episodes"] = int(args.eval_episodes)
    if args.eval_steps is not None:
        config["eval_steps"] = int(args.eval_steps)

    # Native H1 exposes one mapping entry: observations["policy"].  The SRB
    # ExO wrapper treats actor_keys=None as exactly this policy group.  The
    # SRB-specific proprio/proprio_dyn/command contract must not be selected.
    obs_config = config.get("obs", {})
    if not isinstance(obs_config, Mapping):
        raise TypeError("ExO-FPO obs config must be a mapping")
    config["obs"] = {**obs_config, "actor_keys": None, "critic_keys": None}

    return config, config_path


def _resolve_log_root(args: argparse.Namespace, isaaclab_root: Path) -> Path:
    if args.log_root is not None:
        return args.log_root.expanduser().resolve()
    experiment_name = args.experiment_name or DEFAULT_EXPERIMENT_NAME
    return isaaclab_root / "logs" / "isaaclab_exofpo" / experiment_name


def _checkpoint_iteration(path: Path) -> int:
    match = re.fullmatch(r"model_(\d+)\.pt", path.name)
    return int(match.group(1)) if match else -1


def _latest_checkpoint(log_root: Path) -> Path | None:
    if not log_root.is_dir():
        return None
    candidates = [
        path
        for path in log_root.rglob("model_*.pt")
        if path.is_file() and _checkpoint_iteration(path) >= 0
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda path: (_checkpoint_iteration(path), path.stat().st_mtime),
    )


def _resolve_checkpoint(args: argparse.Namespace, log_root: Path) -> Path | None:
    requested = args.checkpoint
    if requested is not None and requested not in CHECKPOINT_SELECTORS:
        checkpoint = Path(requested).expanduser()
        if checkpoint.is_dir():
            candidates = [
                path
                for path in checkpoint.glob("model_*.pt")
                if path.is_file() and _checkpoint_iteration(path) >= 0
            ]
            if not candidates:
                raise FileNotFoundError(
                    f"No model_<iteration>.pt checkpoint found in {checkpoint}"
                )
            checkpoint = max(candidates, key=_checkpoint_iteration)
        return checkpoint.resolve()
    if requested in CHECKPOINT_SELECTORS or args.resume or args.eval:
        return _latest_checkpoint(log_root)
    return None


def _make_log_dir(
    args: argparse.Namespace,
    log_root: Path,
    checkpoint: Path | None,
) -> Path:
    # Evaluation writes its TensorBoard scalars alongside the checkpoint.  A
    # resumed run follows the same directory so its continuation is discoverable
    # without copying model files into a new timestamp directory.
    if checkpoint is not None and (args.eval or args.resume or args.checkpoint is not None):
        return checkpoint.parent
    suffix = f"_{args.run_name}" if args.run_name else ""
    return log_root / f"{datetime.now(UTC):%Y-%m-%d_%H-%M-%S}{suffix}"


def _has_physics_selector(args: argparse.Namespace) -> bool:
    return any(
        item.startswith(("physics=", "presets=")) for item in args.hydra_overrides
    )


def _requested_physics(args: argparse.Namespace) -> str:
    """Return the last explicit physics override or the CLI default."""

    for item in reversed(args.hydra_overrides):
        if item.startswith("physics="):
            return item.split("=", 1)[1]
    return args.physics


def _prepare_env_cfg(args: argparse.Namespace, config: Mapping[str, Any]):
    """Resolve the registered H1 task using Isaac Lab's native parser."""

    import isaaclab_tasks  # noqa: F401  # register built-in task IDs
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    overrides = list(args.hydra_overrides)
    if not _has_physics_selector(args):
        overrides.append(f"physics={args.physics}")
    num_envs = args.eval_num_envs if args.eval else args.num_envs
    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=num_envs,
        overrides=overrides,
    )
    env_cfg.seed = int(config["seed"])
    env_cfg.sim.device = args.device

    # Native H1 uses an infinite-horizon task with timeout bootstrapping.  The
    # ExO collector can use the pre-reset observation only when this flag is set
    # before ManagerBasedRLEnv is constructed.
    if hasattr(env_cfg, "compute_final_obs"):
        env_cfg.compute_final_obs = not bool(
            getattr(env_cfg, "is_finite_horizon", False)
        )
    return env_cfg


# def _set_native_action_space(env: Any, clip_actions: float | None) -> int:
#     """Give native H1 a finite Box space required by the flow action contract."""

#     import gymnasium as gym
#     import numpy as np

#     unwrapped = env.unwrapped
#     action_manager = getattr(unwrapped, "action_manager", None)
#     if action_manager is not None:
#         action_dim = int(action_manager.total_action_dim)
#     else:
#         action_space = getattr(unwrapped, "single_action_space", None)
#         if action_space is None or len(action_space.shape) != 1:
#             raise TypeError("native Isaac Lab environment must expose a flat action space")
#         action_dim = int(action_space.shape[0])
#     if action_dim <= 0:
#         raise ValueError("native Isaac Lab action dimension must be positive")

#     if clip_actions is None:
#         current_space = getattr(unwrapped, "single_action_space", None)
#         if not isinstance(current_space, gym.spaces.Box):
#             raise TypeError("native Isaac Lab action space must be a Box")
#         low = np.asarray(current_space.low, dtype=np.float32)
#         high = np.asarray(current_space.high, dtype=np.float32)
#         if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
#             raise ValueError(
#                 "native Isaac Lab publishes an unbounded action Box; set "
#                 "clip_actions in the ExO-FPO config or pass --clip-actions"
#             )
#         single_action_space = current_space
#     else:
#         clip = float(clip_actions)
#         if not math.isfinite(clip) or clip <= 0.0:
#             raise ValueError("clip_actions must be a finite positive scalar")
#         single_action_space = gym.spaces.Box(
#             low=-clip,
#             high=clip,
#             shape=(action_dim,),
#             dtype=np.float32,
#         )

#     unwrapped.single_action_space = single_action_space
#     unwrapped.action_space = gym.vector.utils.batch_space(
#         single_action_space,
#         int(unwrapped.num_envs),
#     )
#     return action_dim


def _configure_video(
    args: argparse.Namespace,
    env_cfg: Any,
    log_dir: Path,
    checkpoint: Path | None,
) -> None:
    """Use Isaac Lab 3's config-driven recorder instead of Gym RGB wrappers."""

    if not args.video:
        return
    from isaaclab_rl.entrypoints.common import apply_video_recording

    # apply_video_recording understands --video/--video_length and adds a
    # capture-capable visualizer before launch_simulation scans the config.
    apply_video_recording(
        env_cfg,
        str(log_dir),
        args,
        subdir="eval" if args.eval else "train",
        checkpoint_path=str(checkpoint) if checkpoint is not None else None,
    )
    args.enable_cameras = True


class _SimulationLifecycle:
    """Provide the ``is_running`` contract expected by the SRB collector.

    ``launch_simulation`` intentionally exposes the resolved physics config,
    not its private ``AppLauncher`` instance.  Kit-based runs can query the
    active Kit app; kitless Newton runs are bounded by the requested training
    iterations/evaluation steps and therefore remain active until the loop
    finishes.
    """

    @staticmethod
    def is_running() -> bool:
        try:
            from omni.kit.app import get_app

            app = get_app()
            return True if app is None else bool(app.is_running())
        except (ImportError, ModuleNotFoundError, RuntimeError):
            return True


def _run(args: argparse.Namespace) -> int:
    isaaclab_root = args.isaaclab_root.expanduser().resolve()
    if not (isaaclab_root / "pyproject.toml").is_file():
        raise FileNotFoundError(f"Isaac Lab checkout not found: {isaaclab_root}")

    _insert_exo_root(args.exo_root.expanduser())
    config, config_path = _load_exo_cfg(args)
    env_cfg = _prepare_env_cfg(args, config)
    log_root = _resolve_log_root(args, isaaclab_root)
    checkpoint = _resolve_checkpoint(args, log_root)
    if (
        (args.eval or args.resume)
        and checkpoint is None
        and not args.untrained
        and not args.dry_run
    ):
        raise FileNotFoundError(
            f"No ExO-FPO checkpoint found under {log_root}; "
            "pass --checkpoint /path/to/model_N.pt"
        )
    if checkpoint is not None and not checkpoint.is_file() and not args.dry_run:
        raise FileNotFoundError(f"ExO-FPO checkpoint not found: {checkpoint}")

    log_dir = _make_log_dir(args, log_root, checkpoint)
    args.require_kit = bool(
        args.video
        or (
            getattr(args, "visualizer", None)
            and "kit" in getattr(args, "visualizer", [])
        )
    )
    _configure_video(args, env_cfg, log_dir, checkpoint)

    selected_physics = _requested_physics(args)
    print(f"[isaacbase-exo] task={args.task} config={config_path}")
    print(
        f"[isaacbase-exo] mode={'eval' if args.eval else 'train'} algorithm=ExO-FPO "
        f"device={args.device} num_envs={args.eval_num_envs if args.eval else args.num_envs} "
        f"rollout_steps={config.get('rollout_steps', 256)} "
        f"max_iterations={config['max_iterations']}"
    )
    print(f"[isaacbase-exo] log_dir={log_dir}")
    print(
        "[isaacbase-exo] observation contract=Isaac Lab observations['policy'] "
        "(flat native H1 policy group); SRB actor_keys are disabled"
    )

    if args.dry_run:
        env_count = args.eval_num_envs if args.eval else args.num_envs
        if args.eval:
            print(
                f"[isaacbase-exo] physics={selected_physics} "
                f"eval_steps={config.get('eval_steps', 0)} "
                "(0 will be replaced by eval_episodes * native horizon after env creation)"
            )
        else:
            actual_env_steps = (
                int(config["max_iterations"])
                * int(env_count)
                * int(config.get("rollout_steps", 256))
            )
            print(
                f"[isaacbase-exo] physics={selected_physics} "
                f"actual_env_steps={actual_env_steps}"
            )
        print(f"[isaacbase-exo] action_clip={config.get('clip_actions', 1.0)}")
        print("[isaacbase-exo] subprocess=false (active Python environment)")
        return 0

    # launch_simulation selects the current Isaac Lab backend and only creates
    # Kit when the resolved config/visualizer/video path requires it.
    import gymnasium as gym
    from isaaclab.app import launch_simulation

    # If a presets=... override was supplied, let that resolved preset reach
    # launch_simulation instead of reapplying the default --physics value.
    if any(item.startswith("presets=") for item in args.hydra_overrides) and not any(
        item.startswith("physics=") for item in args.hydra_overrides
    ):
        args.physics = None

    with launch_simulation(env_cfg, args):
        env = gym.make(args.task, cfg=env_cfg)
        try:
            # action_dim = _set_native_action_space(env, config.get("clip_actions", 1.0))
            max_episode_length = int(env.unwrapped.max_episode_length)
            if args.eval and int(config.get("eval_steps", 0)) <= 0:
                config["eval_steps"] = int(args.eval_episodes) * max_episode_length
            policy_dim = int(env.unwrapped.observation_manager.group_obs_dim["policy"][0])
            print(
                f"[isaacbase-exo] native contract obs_dim={policy_dim} "
                f"action_dim={env.action_space.shape[0]} max_episode_length={max_episode_length} "
                f"compute_final_obs={getattr(env_cfg, 'compute_final_obs', None)}"
            )

            from srb.integrations.exoppo import exofpo_main

            exofpo_main.run(
                workflow="eval" if args.eval else "train",
                env=env,
                sim_app=_SimulationLifecycle(),
                env_id=args.task,
                env_cfg=env_cfg,
                agent_cfg=config,
                logdir=log_dir,
                model=checkpoint,
                continue_training=args.resume,
                untrained=args.untrained,
            )
        finally:
            env.close()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run native Isaac Lab H1 ExO-FPO in the active conda environment."""

    args = parse_args(argv)
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if not args.isaaclab_root.is_absolute():
        args.isaaclab_root = (project_root / args.isaaclab_root).resolve()
    if not args.exo_root.is_absolute():
        args.exo_root = (project_root / args.exo_root).resolve()
    if not args.exo_config.is_absolute():
        args.exo_config = (project_root / args.exo_config).resolve()
    if args.log_root is not None and not args.log_root.is_absolute():
        args.log_root = (project_root / args.log_root).resolve()
    if args.checkpoint is not None and args.checkpoint not in CHECKPOINT_SELECTORS:
        checkpoint = Path(args.checkpoint)
        if not checkpoint.is_absolute():
            args.checkpoint = str((project_root / checkpoint).resolve())

    old_cwd = Path.cwd()
    try:
        # Match Isaac Lab's official CLI path resolution while restoring the
        # user's working directory before returning to the caller.
        os.chdir(args.isaaclab_root)
        return _run(args)
    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    raise SystemExit(main())
