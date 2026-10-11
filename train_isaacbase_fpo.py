"""Train FPO on Isaac Lab's native Unitree H1 velocity task.

Isaac Lab ships the H1 task and PPO integrations, but it does not register FPO
as one of its built-in RL libraries.  This launcher composes the native
environment with the FPO implementation in ``fpo-control-saa`` without
starting a subprocess.  It is intentionally kept separate from
``train_isaacbase.py`` so the native RSL-RL PPO baseline remains unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_ISAACLAB_ROOT = Path("/root/isaaclab")
DEFAULT_FPO_ROOT = Path("/root/R2A/Algos/fpo-control-saa/isaaclab_experiments/isaaclab_fpo")
DEFAULT_TASK = "Isaac-Velocity-Rough-H1"
DEFAULT_NUM_ENVS = 256
DEFAULT_EVAL_NUM_ENVS = 8
DEFAULT_SEED = 0
DEFAULT_RUN_NAME = "h1_fpo"
DEFAULT_TOTAL_ENV_STEPS = 100_000_000
DEFAULT_VIDEO_LENGTH = 200

TASKS = ("Isaac-Velocity-Rough-H1", "Isaac-Velocity-Flat-H1")
PHYSICS_PRESETS = ("isaacsim_physx", "newton_kamino", "newton_mjwarp", "ovphysx")

_RUNNER_CONFIG_KEYS = (
    "seed",
    "device",
    "num_steps_per_env",
    "max_iterations",
    "empirical_normalization",
    "randomize_reset_episode_progress",
    "clip_actions",
    "save_interval",
    "experiment_name",
    "run_name",
    "logger",
    "neptune_project",
    "wandb_project",
    "eval_episodes",
    "flow_eval_modes",
    "flow_eval_fixed_seed",
    "enable_post_training_eval",
    "post_eval_checkpoint_interval",
    "resume",
    "load_run",
    "load_checkpoint",
)


def _configure_torch_matmul_precision() -> None:
    """Enable TF32-backed float32 matmuls before building the policy."""
    import torch

    torch.set_float32_matmul_precision("high")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse launcher/FPO options and preserve remaining Hydra overrides."""
    supplied = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="Train FPO on Isaac Lab's native Unitree H1 velocity task."
    )
    parser.add_argument(
        "--isaaclab-root",
        type=Path,
        default=Path(os.environ.get("ISAACLAB_ROOT", DEFAULT_ISAACLAB_ROOT)),
        help="Isaac Lab source checkout.",
    )
    parser.add_argument(
        "--fpo-root",
        type=Path,
        default=Path(os.environ.get("FPO_ROOT", DEFAULT_FPO_ROOT)),
        help="isaaclab_fpo checkout containing the FPO runner.",
    )
    parser.add_argument(
        "--fpo-config",
        type=Path,
        default=Path(__file__).resolve().parent / "hyperparams" / "fpo.yaml",
        help="YAML file for FPO runner/policy/algorithm settings.",
    )
    parser.add_argument("--task", choices=TASKS, default=DEFAULT_TASK)
    parser.add_argument("--num-envs", "--num_envs", dest="num_envs", type=_positive_int, default=DEFAULT_NUM_ENVS)
    parser.add_argument(
        "--eval-num-envs",
        "--eval_num_envs",
        dest="eval_num_envs",
        type=_positive_int,
        default=DEFAULT_EVAL_NUM_ENVS,
        help="Number of environments used by --eval.",
    )
    parser.add_argument(
        "--total-env-steps",
        "--total_env_steps",
        dest="total_env_steps",
        type=_positive_int,
        default=None,
        help=f"Target transitions; default keeps max_iterations from the YAML config ({DEFAULT_TOTAL_ENV_STEPS:,} is a common choice).",
    )
    parser.add_argument("--max-iterations", "--max_iterations", dest="max_iterations", type=_positive_int)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--experiment-name", "--experiment_name", dest="experiment_name",default=DEFAULT_TASK.split("-")[2])
    parser.add_argument("--run-name", "--run_name", dest="run_name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--log-root", "--log_root", dest="log_root", type=Path, default=None)
    parser.add_argument("--save-interval", "--save_interval", dest="save_interval", type=_positive_int)
    parser.add_argument("--checkpoint", type=Path, help="Checkpoint path, or 'latest' when resuming/evaluating.")
    parser.add_argument("--resume", action="store_true", help="Resume optimizer and runner state from a checkpoint.")
    parser.add_argument("--eval", action="store_true", help="Evaluate a saved FPO checkpoint instead of training.")
    parser.add_argument("--eval-episodes", "--eval_episodes", dest="eval_episodes", type=_positive_int)
    parser.add_argument("--flow-eval-fixed-seed", type=int, default=None)
    parser.add_argument(
        "--flow-eval-modes",
        "--flow_eval_modes",
        dest="flow_eval_modes",
        nargs="+",
        choices=("zero", "fixed_seed", "random"),
        default=None,
    )
    parser.add_argument(
        "--pmf-ablation", choices=("pg_only", "pg_pmf"),
        help="FSPPOJoint A/B ablation: disable or enable the complete pMF auxiliary loss.",
    )
    parser.add_argument(
        "--pmf-loss-coef", type=float, default=None,
        help="Positive pMF loss weight for --pmf-ablation pg_pmf (default: YAML weight or 0.01).",
    )
    parser.add_argument(
        "--pmf-next-step", choices=("joint_pg", "pmd", "pmd_tr", "joint_pmf"),
        help="Select an independent Next Step experiment using the common YAML settings.",
    )
    parser.add_argument(
        "--learn-terminal-std", action="store_true",
        help="Additional A/D ablation: learn per-action terminal Gaussian scales.",
    )
    parser.add_argument(
        "--pmf-validation", action="store_true",
        help="With --eval, compare the one-step pMF map with diagonal-field ODE integration.",
    )
    parser.add_argument(
        "--use-checkpoint-config", action="store_true",
        help="Load params/agent.yaml beside the checkpoint for faithful policy reconstruction.",
    )
    parser.add_argument("--validation-output", type=Path)
    parser.add_argument("--state-bank-in", type=Path, help="Reuse raw observations and paired initial noises from a saved bank.")
    parser.add_argument("--state-bank-out", type=Path, help="Also save the collected bank to this path for other checkpoints.")
    parser.add_argument("--validation-states", type=_positive_int, default=256)
    parser.add_argument("--validation-noises", type=_positive_int, default=8)
    parser.add_argument("--validation-seed", type=int, default=12345)
    parser.add_argument("--validation-batch-size", type=_positive_int, default=256)
    parser.add_argument("--ode-steps", type=_positive_int, nargs="+", default=[8, 32, 128])
    parser.add_argument("--video", action="store_true", help="Record the first evaluation/training episode.")
    parser.add_argument("--video-length", "--video_length", dest="video_length", type=_positive_int, default=DEFAULT_VIDEO_LENGTH)
    parser.add_argument("--physics", choices=PHYSICS_PRESETS, default="isaacsim_physx")
    parser.add_argument("--dry-run", action="store_true", help="Resolve and print the run without launching Isaac Sim.")

    # Add Isaac Lab's own launcher arguments lazily so importing this file does
    # not launch Isaac Sim.  The active conda environment supplies the package.
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    # Isaac Lab 3.0 moved headless selection to environment/config intent, but
    # keep the familiar flag for this standalone launcher.  It is added after
    # AppLauncher validation so the value can still be passed to AppLauncher.
    parser.add_argument("--headless", action="store_true", help="Force headless Isaac Sim execution.")
    parser.set_defaults(device="cuda")

    args, remaining = parser.parse_known_args(supplied)
    args.seed_explicit = any(item == "--seed" or item.startswith("--seed=") for item in supplied)
    args.hydra_overrides = list(remaining)
    if args.pmf_validation and not args.eval:
        parser.error("--pmf-validation requires --eval and a saved checkpoint")
    if args.use_checkpoint_config and not (args.eval or args.resume):
        parser.error("--use-checkpoint-config requires --eval or --resume")
    if args.pmf_loss_coef is not None:
        if args.pmf_ablation != "pg_pmf":
            parser.error("--pmf-loss-coef requires --pmf-ablation pg_pmf")
        if not math.isfinite(args.pmf_loss_coef) or args.pmf_loss_coef <= 0:
            parser.error("--pmf-loss-coef must be finite and positive")
    if args.pmf_next_step is not None and args.pmf_ablation is not None:
        parser.error("--pmf-next-step and --pmf-ablation select different experiment families")
    if args.learn_terminal_std and args.pmf_next_step not in ("joint_pg", "joint_pmf"):
        parser.error("--learn-terminal-std requires --pmf-next-step joint_pg or joint_pmf")
    return args


def _insert_fpo_root(fpo_root: Path) -> None:
    """Make the external FPO package importable without installing it globally."""
    package_init = fpo_root / "isaaclab_fpo" / "__init__.py"
    if not package_init.is_file():
        raise FileNotFoundError(f"isaaclab_fpo package not found under {fpo_root}")
    fpo_root = fpo_root.resolve()
    if str(fpo_root) not in sys.path:
        sys.path.insert(0, str(fpo_root))


def _load_fpo_cfg(args: argparse.Namespace):
    """Build the FPO config and apply only fields understood by the runner."""
    _insert_fpo_root(args.fpo_root.expanduser())

    from isaaclab_fpo import (
        FpoRslRlOnPolicyRunnerCfg,
        FpoRslRlPpoActorCriticCfg,
        FpoRslRlPpoAlgorithmCfg,
    )
    from isaaclab_fpo.patches import apply_isaaclab_patches

    # Isaac Lab 3 binds configclass.from_dict directly at decoration time.
    # Use the updater after patching so already-created config classes also
    # accept MISSING fields, int-to-float values and None defaults from YAML.
    apply_isaaclab_patches()
    from isaaclab.utils.dict import update_class_from_dict

    import yaml

    config_path = args.fpo_config.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"FPO config not found: {config_path}")
    # Isaac Lab's dump_yaml writes Python tuple tags for configuration tuples.
    # Accept just that tag without enabling arbitrary Python object loading.
    class ConfigLoader(yaml.SafeLoader):
        pass

    ConfigLoader.add_constructor(
        "tag:yaml.org,2002:python/tuple",
        lambda loader, node: tuple(loader.construct_sequence(node)),
    )
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.load(config_file, Loader=ConfigLoader) or {}
    if not isinstance(config, Mapping):
        raise TypeError(f"FPO config must contain a mapping: {config_path}")
    dynamics_cfg = config.get("dynamics_encoder")
    if isinstance(dynamics_cfg, Mapping) and dynamics_cfg.get("enabled", False):
        raise ValueError(
            "dynamics_encoder.enabled is not supported by this generic Isaac Lab FPO launcher; "
            "use the SRB FPO entry point or add a dedicated policy adapter"
        )

    # Start with the external H1 defaults.  The policy input dimension is
    # resolved from the actual Isaac Lab observation manager at runtime, so the
    # rough-terrain height scan does not need a separate network definition.
    agent_cfg = FpoRslRlOnPolicyRunnerCfg(
        policy=FpoRslRlPpoActorCriticCfg(
            init_noise_std=1.0,
            actor_hidden_dims=[256, 256, 256],
            critic_hidden_dims=[768, 768, 768],
            activation="elu",
        ),
        algorithm=FpoRslRlPpoAlgorithmCfg(),
    )

    runner_values = {key: config[key] for key in _RUNNER_CONFIG_KEYS if key in config}
    if runner_values:
        update_class_from_dict(agent_cfg, runner_values)
    if isinstance(config.get("policy"), Mapping):
        update_class_from_dict(agent_cfg.policy, dict(config["policy"]))
    if isinstance(config.get("algorithm"), Mapping):
        update_class_from_dict(agent_cfg.algorithm, dict(config["algorithm"]))

    if args.pmf_next_step is not None:
        variant = args.pmf_next_step
        joint = variant in ("joint_pg", "joint_pmf")
        agent_cfg.policy.class_name = "LearnedStdPMFActorCritic" if joint else "PMFActorCritic"
        agent_cfg.policy.learn_action_perturb_std = args.learn_terminal_std
        agent_cfg.algorithm.class_name = "FSPPOJointAligned" if joint else "PMFPMD"
        agent_cfg.algorithm.fsppo_joint_enable_budget = variant == "pmd_tr"
        agent_cfg.algorithm.fsppo_joint_aux_loss_coef = 0.01 if variant == "joint_pmf" else 0.0
        agent_cfg.algorithm.fsppo_joint_entropy_coef = 0.01 if joint else 0.0
        # Adaptive conditional Gaussian KL is supported by A/D. B/C improve
        # candidates with Q and fit pMF directly, with no PPO actor surrogate.
        agent_cfg.algorithm.schedule = "adaptive" if joint else "fixed"

    # CLI values are authoritative for this launcher.
    if not args.use_checkpoint_config or args.seed_explicit:
        agent_cfg.seed = args.seed
    agent_cfg.device = args.device
    if args.experiment_name is not None:
        agent_cfg.experiment_name = args.experiment_name
    if args.run_name is not None:
        agent_cfg.run_name = args.run_name
    if args.save_interval is not None:
        agent_cfg.save_interval = args.save_interval
    if args.eval_episodes is not None:
        agent_cfg.eval_episodes = args.eval_episodes
    if args.flow_eval_modes is not None:
        agent_cfg.flow_eval_modes = args.flow_eval_modes
    if args.flow_eval_fixed_seed is not None:
        agent_cfg.flow_eval_fixed_seed = args.flow_eval_fixed_seed
    elif agent_cfg.flow_eval_fixed_seed is None:
        agent_cfg.flow_eval_fixed_seed = 12345
    if args.pmf_ablation is not None:
        if (agent_cfg.policy.class_name, agent_cfg.algorithm.class_name) != (
            "PMFActorCritic", "FSPPOJoint"
        ):
            raise ValueError("--pmf-ablation requires PMFActorCritic with FSPPOJoint")
        if args.pmf_ablation == "pg_only":
            agent_cfg.algorithm.fsppo_joint_aux_loss_coef = 0.0
        else:
            coefficient = args.pmf_loss_coef
            if coefficient is None:
                coefficient = agent_cfg.algorithm.fsppo_joint_aux_loss_coef or 0.01
            if not math.isfinite(coefficient) or coefficient <= 0:
                raise ValueError("The PG+pMF auxiliary coefficient must be finite and positive")
            agent_cfg.algorithm.fsppo_joint_aux_loss_coef = coefficient
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations
    elif args.total_env_steps is not None:
        agent_cfg.max_iterations = math.ceil(
            args.total_env_steps / (args.num_envs * agent_cfg.num_steps_per_env)
        )
    agent_cfg.resume = args.resume

    if agent_cfg.num_steps_per_env <= 0 or agent_cfg.max_iterations <= 0:
        raise ValueError("FPO num_steps_per_env and max_iterations must be positive")
    rollout_size = args.num_envs * agent_cfg.num_steps_per_env
    if rollout_size % agent_cfg.algorithm.num_mini_batches != 0:
        raise ValueError(
            "num_envs * num_steps_per_env must be divisible by algorithm.num_mini_batches "
            f"({rollout_size} % {agent_cfg.algorithm.num_mini_batches} != 0)"
        )
    return agent_cfg, config_path


def _resolve_log_root(args: argparse.Namespace, agent_cfg: Any, isaaclab_root: Path) -> Path:
    if getattr(args, "log_root", None) is not None:
        return args.log_root.expanduser().resolve()
    return isaaclab_root / "logs" / "isaaclab_fpo" / agent_cfg.experiment_name


def _checkpoint_iteration(path: Path) -> int:
    match = re.fullmatch(r"model_(\d+)\.pt", path.name)
    return int(match.group(1)) if match else 0


def _latest_checkpoint(log_root: Path) -> Path | None:
    candidates = [path for path in log_root.glob("*/model_*.pt") if path.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: (_checkpoint_iteration(path), path.stat().st_mtime))


def _resolve_checkpoint(args: argparse.Namespace, log_root: Path) -> Path | None:
    requested = args.checkpoint
    if requested is not None and str(requested) not in {"latest", "best"}:
        checkpoint = requested.expanduser()
        if checkpoint.is_dir():
            files = [path for path in checkpoint.glob("model_*.pt") if path.is_file()]
            checkpoint = max(files, key=_checkpoint_iteration) if files else checkpoint / "model_latest.pt"
        return checkpoint.resolve()
    if requested in {Path("latest"), Path("best")} or args.resume or args.eval:
        checkpoint = _latest_checkpoint(log_root)
        if checkpoint is not None:
            return checkpoint.resolve()
    return None


def _make_log_dir(args: argparse.Namespace, agent_cfg: Any, isaaclab_root: Path, checkpoint: Path | None) -> Path:
    if args.eval and checkpoint is not None:
        return checkpoint.parent
    log_root = _resolve_log_root(args, agent_cfg, isaaclab_root)
    suffix = f"_{agent_cfg.run_name}" if agent_cfg.run_name else ""
    return log_root / f"{datetime.now():%Y-%m-%d_%H-%M-%S}{suffix}"


def _requested_physics(args: argparse.Namespace) -> str:
    """Return the last explicit ``physics=...`` Hydra override, if present."""
    for item in reversed(args.hydra_overrides):
        if item.startswith("physics="):
            return item.split("=", 1)[1]
    return args.physics


def _prepare_env_cfg(args: argparse.Namespace, agent_cfg: Any):
    """Resolve the registered H1 task with the same preset mechanism as Isaac Lab CLI."""
    import isaaclab_tasks  # noqa: F401  # register the built-in tasks
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    overrides = list(args.hydra_overrides)
    physics = _requested_physics(args)
    if not any(item.startswith(("physics=", "presets=")) for item in overrides):
        overrides.append(f"physics={physics}")
    num_envs = args.eval_num_envs if args.eval else args.num_envs
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=num_envs, overrides=overrides)
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args.device
    if agent_cfg.algorithm.class_name == "PMFPMD":
        if not hasattr(env_cfg, "compute_final_obs"):
            raise ValueError("PMFPMD requires native support for pre-reset final observations")
        env_cfg.compute_final_obs = True
    return env_cfg


def _write_run_files(log_dir: Path, env_cfg: Any, agent_cfg: Any, task: str) -> None:
    from isaaclab.utils.io import  dump_yaml

    from isaaclab_rl.entrypoints.common import write_run_manifest

    write_run_manifest(
        str(log_dir),
        library="isaaclab_fpo",
        task=task,
        metadata={
            "agent": agent_cfg.algorithm.class_name,
            "policy": agent_cfg.policy.class_name,
            "runner": "isaaclab_fpo",
            "seed": agent_cfg.seed,
            "num_envs": env_cfg.scene.num_envs,
            "num_steps_per_env": agent_cfg.num_steps_per_env,
            "max_iterations": agent_cfg.max_iterations,
            "pmf_aux_loss_coef": agent_cfg.algorithm.fsppo_joint_aux_loss_coef,
            "terminal_std_learned": agent_cfg.policy.learn_action_perturb_std,
            "terminal_std_initial": agent_cfg.policy.action_perturb_std,
            "pmd_trust_region": (
                agent_cfg.algorithm.fsppo_joint_enable_budget
                if agent_cfg.algorithm.class_name == "PMFPMD" else None
            ),
        },
    )
    dump_yaml(str(log_dir / "params" / "env.yaml"), env_cfg)
    dump_yaml(str(log_dir / "params" / "agent.yaml"), agent_cfg)


def _training_iterations(args: argparse.Namespace, agent_cfg: Any, runner: Any) -> int:
    """Treat the new experiments' resume budget as a final cumulative target."""
    if args.resume and agent_cfg.algorithm.class_name in ("PMFPMD", "FSPPOJointAligned"):
        if args.total_env_steps is not None and args.max_iterations is None:
            remaining = args.total_env_steps - runner.tot_timesteps
            return max(0, math.ceil(remaining / (args.num_envs * agent_cfg.num_steps_per_env)))
        return max(0, agent_cfg.max_iterations - runner.current_learning_iteration)
    return agent_cfg.max_iterations


def _run(args: argparse.Namespace) -> int:
    _configure_torch_matmul_precision()

    isaaclab_root = args.isaaclab_root.expanduser().resolve()
    if not (isaaclab_root / "pyproject.toml").is_file():
        raise FileNotFoundError(f"Isaac Lab checkout not found: {isaaclab_root}")

    _insert_fpo_root(args.fpo_root.expanduser())
    agent_cfg, config_path = _load_fpo_cfg(args)
    args.physics = _requested_physics(args)
    log_root = _resolve_log_root(args, agent_cfg, isaaclab_root)
    checkpoint = _resolve_checkpoint(args, log_root)
    if (args.eval or args.resume) and checkpoint is None and not args.dry_run:
        raise FileNotFoundError(
            f"No checkpoint found under {log_root}; pass --checkpoint /path/to/model_N.pt"
        )
    if checkpoint is not None and not checkpoint.is_file() and not args.dry_run:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if args.use_checkpoint_config:
        if checkpoint is None:
            raise FileNotFoundError("--use-checkpoint-config needs a resolvable checkpoint")
        args.fpo_config = checkpoint.parent / "params" / "agent.yaml"
        agent_cfg, config_path = _load_fpo_cfg(args)
    if args.pmf_validation and agent_cfg.policy.class_name not in (
        "PMFActorCritic", "LearnedStdPMFActorCritic"
    ):
        raise ValueError("--pmf-validation requires a supported pMF transport policy")
    env_cfg = _prepare_env_cfg(args, agent_cfg)

    log_dir = _make_log_dir(args, agent_cfg, isaaclab_root, checkpoint)
    args.enable_cameras = bool(getattr(args, "enable_cameras", False) or args.video)
    args.require_kit = bool(
        args.video
        or (
            getattr(args, "visualizer", None)
            and "kit" in args.visualizer
        )
    )

    print(f"[isaacbase-fpo] task={args.task} config={config_path}")
    print(
        f"[isaacbase-fpo] mode={'eval' if args.eval else 'train'} device={args.device} "
        f"num_envs={args.eval_num_envs if args.eval else args.num_envs} "
        f"num_steps_per_env={agent_cfg.num_steps_per_env} max_iterations={agent_cfg.max_iterations} "
        f"policy={agent_cfg.policy.class_name} algorithm={agent_cfg.algorithm.class_name}"
    )
    print(f"[isaacbase-fpo] log_dir={log_dir}")
    if args.pmf_ablation is not None:
        print(
            f"[isaacbase-fpo] pmf_ablation={args.pmf_ablation} "
            f"aux_loss_coef={agent_cfg.algorithm.fsppo_joint_aux_loss_coef}"
        )
    if args.pmf_validation:
        print(
            f"[isaacbase-fpo] pmf_validation=true ode_steps={args.ode_steps} "
            f"state_bank={args.state_bank_in} seed={args.validation_seed}"
        )
    print(
        "[isaacbase-fpo] observation contract=Isaac Lab policy group -> flattened tensor; "
        "FPO YAML obs/dynamics_encoder sections are not used by the generic wrapper"
    )

    if args.dry_run:
        if args.eval:
            print(
                f"[isaacbase-fpo] physics={args.physics} "
                f"eval_episodes_per_mode={agent_cfg.eval_episodes} modes={agent_cfg.flow_eval_modes}"
            )
        else:
            actual_env_steps = agent_cfg.max_iterations * args.num_envs * agent_cfg.num_steps_per_env
            print(f"[isaacbase-fpo] physics={args.physics} actual_env_steps={actual_env_steps}")
        print("[isaacbase-fpo] subprocess=false (active Python environment)")
        return 0

    from isaaclab.app import launch_simulation

    import gymnasium as gym

    with launch_simulation(env_cfg, args):
        from isaaclab_fpo import FpoRslRlVecEnvWrapper
        from isaaclab_fpo.runners import OnPolicyRunner
        env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)
        try:
            if args.video:
                video_dir = log_dir / "videos" / ("eval" if args.eval else "train")
                env = gym.wrappers.RecordVideo(
                    env,
                    video_folder=str(video_dir),
                    episode_trigger=lambda episode_id: episode_id == 0,
                    video_length=args.video_length,
                    disable_logger=True,
                )

            env = FpoRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
            runner_log_dir = None if args.eval else str(log_dir)
            runner = OnPolicyRunner(env, agent_cfg, log_dir=runner_log_dir, device=agent_cfg.device)
            runner.add_git_repo_to_log(__file__)

            if checkpoint is not None:
                print(f"[isaacbase-fpo] loading checkpoint={checkpoint}")
                if args.eval:
                    results = runner.evaluate_checkpoint(str(checkpoint), _checkpoint_iteration(checkpoint))
                    if results is None:
                        raise RuntimeError(f"FPO evaluation failed for {checkpoint}")
                    for mode, metrics in results.items():
                        print(
                            f"[isaacbase-fpo] eval/{mode}: "
                            f"reward={metrics['mean_reward']:.4f} +/- {metrics['std_reward']:.4f}, "
                            f"length={metrics['mean_length']:.2f} +/- {metrics['std_length']:.2f}"
                        )
                    if args.pmf_validation:
                        from isaaclab_fpo.pmf_validation import validate_pmf_checkpoint

                        output_dir = args.validation_output or (
                            checkpoint.parent / "pmf_validation" / checkpoint.stem
                            / f"{datetime.now():%Y-%m-%d_%H-%M-%S_%f}"
                        )
                        report = validate_pmf_checkpoint(
                            runner, checkpoint, output_dir,
                            state_bank_path=args.state_bank_in,
                            state_bank_output_path=args.state_bank_out,
                            num_states=args.validation_states,
                            samples_per_state=args.validation_noises,
                            seed=args.validation_seed,
                            ode_steps=args.ode_steps,
                            batch_size=args.validation_batch_size,
                            task_name=args.task,
                        )
                        report["closed_loop_evaluation"] = results
                        report["closed_loop_policy"] = "one_step_transport_without_conditional_gaussian"
                        report["agent_config_path"] = str(config_path)
                        report["hydra_overrides"] = args.hydra_overrides
                        report["physics"] = args.physics
                        with (output_dir / "report.json").open("w", encoding="utf-8") as report_file:
                            json.dump(report, report_file, indent=2, allow_nan=False)
                        print(f"[isaacbase-fpo] pMF validation report={output_dir / 'report.json'}")
                    return 0
                runner.load(str(checkpoint), load_optimizer=True)

            _write_run_files(log_dir, env_cfg, agent_cfg, args.task)
            learning_iterations = _training_iterations(args, agent_cfg, runner)
            if learning_iterations == 0:
                print("[isaacbase-fpo] checkpoint already meets the requested training budget")
                return 0
            runner.learn(
                num_learning_iterations=learning_iterations,
                init_at_random_ep_len=False,
            )
        finally:
            env.close()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run FPO in the active conda environment, without ``subprocess``."""
    args = parse_args(argv)
    project_root = Path(__file__).resolve().parent
    if not args.fpo_root.is_absolute():
        args.fpo_root = (project_root / args.fpo_root).resolve()
    if not args.fpo_config.is_absolute():
        args.fpo_config = (project_root / args.fpo_config).resolve()
    for name in ("validation_output", "state_bank_in", "state_bank_out"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    if args.checkpoint is not None and args.checkpoint not in (Path("latest"), Path("best")):
        args.checkpoint = args.checkpoint.expanduser().resolve()
    if args.log_root is not None:
        args.log_root = args.log_root.expanduser().resolve()
    isaaclab_root = args.isaaclab_root.expanduser().resolve()
    args.isaaclab_root = isaaclab_root
    old_cwd = Path.cwd()
    try:
        # Isaac Lab's task/Hydra paths and relative log conventions are rooted
        # at its checkout, just like the official ``isaaclab train`` command.
        os.chdir(isaaclab_root)
        return _run(args)
    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    raise SystemExit(main())
