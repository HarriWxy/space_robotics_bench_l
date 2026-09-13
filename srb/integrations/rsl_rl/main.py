"""Run Isaac Lab's RSL-RL PPO agent on an SRB environment."""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch

from srb.integrations.rsl_rl.logging import SrbRslRlLogger
from srb.integrations.rsl_rl.wrapper import SrbRslRlVecEnvWrapper
from srb.utils import logging
from srb.wrappers import maybe_wrap_action_smoothing

if TYPE_CHECKING:
    import gymnasium

    from srb._typing import AnyEnv, AnyEnvCfg

    from isaacsim.simulation_app import SimulationApp


FRAMEWORK_NAME = "rsl_rl"


_RUNNER_DEFAULTS: dict[str, Any] = {
    "class_name": "OnPolicyRunner",
    "seed": 42,
    "device": None,
    "num_steps_per_env": 24,
    "init_at_random_ep_len": True,
    "max_iterations": 500,
    "empirical_normalization": False,
    "obs_groups": {
        "actor": ["proprio", "proprio_dyn", "command"],
        "critic": ["proprio", "proprio_dyn", "command"],
    },
    "clip_actions": None,
    "check_for_nan": True,
    "save_interval": 50,
    "experiment_name": "srb_rsl_rl_ppo",
    "run_name": "",
    "logger": "tensorboard",
    "neptune_project": "isaaclab",
    "wandb_project": "isaaclab",
    "resume": False,
    "load_run": ".*",
    "load_checkpoint": "model_.*.pt",
    "eval_steps": 10_000,
    "eval_log_interval": 100,
}

_ACTOR_DEFAULTS: dict[str, Any] = {
    "class_name": "MLPModel",
    "hidden_dims": [128, 128, 128],
    "activation": "elu",
    "obs_normalization": False,
    "distribution_cfg": {
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
    },
}

_CRITIC_DEFAULTS: dict[str, Any] = {
    "class_name": "MLPModel",
    "hidden_dims": [128, 128, 128],
    "activation": "elu",
    "obs_normalization": False,
}

_ALGORITHM_DEFAULTS: dict[str, Any] = {
    "class_name": "PPO",
    "value_loss_coef": 1.0,
    "use_clipped_value_loss": True,
    "clip_param": 0.2,
    "entropy_coef": 0.01,
    "num_learning_epochs": 5,
    "num_mini_batches": 4,
    "learning_rate": 1.0e-3,
    "schedule": "adaptive",
    "gamma": 0.99,
    "lam": 0.95,
    "desired_kl": 0.01,
    "max_grad_norm": 1.0,
    "optimizer": "adam",
    "normalize_advantage_per_mini_batch": False,
    "share_cnn_encoders": False,
    "rnd_cfg": None,
    "symmetry_cfg": None,
}

_CHECKPOINT_PATTERN = re.compile(r"^model_(\d+)\.pt$")


def run(
    workflow: Literal["train", "eval"],
    algo: str,
    env: "AnyEnv | gymnasium.Env",
    sim_app: "SimulationApp",
    env_id: str,
    env_cfg: "AnyEnvCfg | None",
    agent_cfg: Mapping[str, Any] | object,
    logdir: Path,
    model: Path | None = None,
    continue_training: bool | None = None,
    **kwargs: Any,
) -> None:
    """Train or evaluate RSL-RL PPO inside an already-created SRB env."""
    if algo != "ppo":
        raise ValueError(
            f"Unsupported RSL-RL algorithm '{algo}'. SRB currently exposes only rsl_rl_ppo."
        )

    try:
        from rsl_rl.runners import OnPolicyRunner
    except ImportError as exc:
        raise ImportError(
            "rsl_rl_ppo requires rsl-rl-lib and Isaac Lab's isaaclab_rl package. "
            "Install the SRB rsl_rl extra in the srb environment."
        ) from exc

    raw_cfg = _as_dict(agent_cfg)
    smoothing_cfg = raw_cfg.pop("smoothing", {})
    env = maybe_wrap_action_smoothing(env, smoothing_cfg)  # type: ignore[arg-type]

    env_device = _resolve_env_device(env)
    cfg = _build_config(
        raw_cfg,
        env_cfg=env_cfg,
        env_id=env_id,
        env_device=env_device,
    )

    wrapped_env = SrbRslRlVecEnvWrapper(
        env,
        clip_actions=cfg["clip_actions"],
        obs_groups=cfg["obs_groups"],
    )
    observations = wrapped_env.get_observations()
    _validate_runtime_contract(wrapped_env, observations, cfg)
    _write_resolved_config(logdir, cfg, env_id=env_id, workflow=workflow)

    from_checkpoint = _resolve_checkpoint(
        workflow=workflow,
        logdir=logdir,
        model=model,
        continue_training=bool(continue_training),
    )
    if from_checkpoint:
        logging.info(f"Loading RSL-RL checkpoint from {from_checkpoint}")
        if not (from_checkpoint.parent / "rsl_rl_config.json").is_file():
            logging.warning(
                "Checkpoint has no SRB RSL-RL observation manifest; matching tensor "
                "shapes do not prove that observation ordering and semantics match."
            )

    runner_logdir = logdir if workflow == "train" else None
    runner = OnPolicyRunner(
        wrapped_env,
        train_cfg=copy.deepcopy(cfg),
        log_dir=runner_logdir.as_posix() if runner_logdir else None,
        device=cfg["device"],
    )
    runner.logger = SrbRslRlLogger(runner.logger)
    runner.add_git_repo_to_log(__file__)

    if from_checkpoint:
        runner.load(from_checkpoint.as_posix(), map_location=cfg["device"])

    if workflow == "train":
        runner.learn(
            num_learning_iterations=int(cfg["max_iterations"]),
            init_at_random_ep_len=bool(cfg["init_at_random_ep_len"]),
        )
    else:
        _evaluate(
            runner=runner,
            env=wrapped_env,
            sim_app=sim_app,
            logdir=logdir,
            cfg=cfg,
        )


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return copy.deepcopy(dict(value))
    if hasattr(value, "to_dict"):
        return copy.deepcopy(value.to_dict())
    if hasattr(value, "__dict__"):
        return copy.deepcopy(vars(value))
    raise TypeError(f"RSL-RL agent config must be a mapping, got {type(value).__name__}")


def _deep_merge(base: Mapping[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _build_config(
    agent_cfg: Mapping[str, Any],
    *,
    env_cfg: Any,
    env_id: str,
    env_device: torch.device,
) -> dict[str, Any]:
    cfg = _deep_merge(
        _RUNNER_DEFAULTS,
        {
            **agent_cfg,
            "actor": _deep_merge(_ACTOR_DEFAULTS, agent_cfg.get("actor", {})),
            "critic": _deep_merge(_CRITIC_DEFAULTS, agent_cfg.get("critic", {})),
            "algorithm": _deep_merge(
                _ALGORITHM_DEFAULTS, agent_cfg.get("algorithm", {})
            ),
        },
    )

    configured_device = cfg.get("device")
    if configured_device in (None, "auto"):
        cfg["device"] = str(env_device)
    else:
        try:
            requested_device = torch.device(configured_device)
        except (RuntimeError, TypeError) as exc:
            raise ValueError(f"Invalid RSL-RL device '{configured_device}'.") from exc
        cfg["device"] = str(requested_device)

    if "seed" not in agent_cfg and env_cfg is not None:
        cfg["seed"] = int(getattr(env_cfg, "seed", cfg["seed"]))
    cfg["experiment_name"] = cfg.get("experiment_name") or (
        f"srb_{env_id.rsplit('/', 1)[-1]}_rsl_rl_ppo"
    )

    for key in ("num_steps_per_env", "max_iterations", "save_interval"):
        if int(cfg[key]) <= 0:
            raise ValueError(f"RSL-RL config '{key}' must be positive, got {cfg[key]}")
    for key in ("eval_steps", "eval_log_interval"):
        if int(cfg[key]) <= 0:
            raise ValueError(f"RSL-RL config '{key}' must be positive, got {cfg[key]}")

    obs_groups = cfg.get("obs_groups")
    if (
        not isinstance(obs_groups, Mapping)
        or not obs_groups.get("actor")
        or not obs_groups.get("critic")
    ):
        raise ValueError(
            "RSL-RL config must define non-empty obs_groups for both actor and critic."
        )
    if not isinstance(cfg.get("actor"), Mapping) or not isinstance(cfg.get("critic"), Mapping):
        raise ValueError("RSL-RL config must define actor and critic mappings.")
    if not isinstance(cfg.get("algorithm"), Mapping):
        raise ValueError("RSL-RL config must define an algorithm mapping.")

    return cfg


def _resolve_env_device(env: Any) -> torch.device:
    unwrapped = getattr(env, "unwrapped", env)
    device = getattr(unwrapped, "device", getattr(env, "device", "cpu"))
    return torch.device(device)


def _validate_runtime_contract(
    env: SrbRslRlVecEnvWrapper,
    observations: Any,
    cfg: Mapping[str, Any],
) -> None:
    if observations.batch_size != torch.Size([env.num_envs]):
        raise ValueError(
            "RSL-RL expects a TensorDict with batch shape "
            f"[{env.num_envs}], got {observations.batch_size}."
        )
    if int(env.num_actions) <= 0:
        raise ValueError(f"SRB environment exposes an invalid action dimension: {env.num_actions}")


def _resolve_checkpoint(
    *,
    workflow: Literal["train", "eval"],
    logdir: Path,
    model: Path | None,
    continue_training: bool,
) -> Path | None:
    if model is not None:
        checkpoint = Path(model).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"RSL-RL checkpoint does not exist: {checkpoint}")
        return checkpoint

    if workflow == "eval" or continue_training:
        checkpoint = _last_checkpoint(logdir)
        if checkpoint is None:
            raise FileNotFoundError(
                f"No RSL-RL checkpoint matching model_<iteration>.pt was found in {logdir}"
            )
        return checkpoint
    return None


def _last_checkpoint(logdir: Path) -> Path | None:
    if not logdir.is_dir():
        return None
    checkpoints: list[tuple[int, Path]] = []
    for path in logdir.glob("model_*.pt"):
        match = _CHECKPOINT_PATTERN.match(path.name)
        if match:
            checkpoints.append((int(match.group(1)), path))
    return max(checkpoints, key=lambda item: item[0])[1] if checkpoints else None


def _write_resolved_config(
    logdir: Path,
    cfg: Mapping[str, Any],
    *,
    env_id: str,
    workflow: str,
) -> None:
    logdir.mkdir(parents=True, exist_ok=True)
    payload = {
        "framework": FRAMEWORK_NAME,
        "algorithm": "ppo",
        "workflow": workflow,
        "env_id": env_id,
        "config": _jsonable(cfg),
    }
    (logdir / "rsl_rl_config.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (Path, torch.device)):
        return str(value)
    return value


def _evaluate(
    *,
    runner: Any,
    env: SrbRslRlVecEnvWrapper,
    sim_app: Any,
    logdir: Path,
    cfg: Mapping[str, Any],
) -> None:
    writer = _make_summary_writer(logdir / "eval")
    policy = runner.get_inference_policy(device=cfg["device"])
    observations = env.get_observations().to(cfg["device"])
    episode_returns = torch.zeros(env.num_envs, device=env.device)
    episode_lengths = torch.zeros(env.num_envs, device=env.device)
    completed_episodes = 0

    for step in range(int(cfg["eval_steps"])):
        if hasattr(sim_app, "is_running") and not sim_app.is_running():
            break

        with torch.inference_mode():
            actions = policy(observations)
            observations, rewards, dones, extras = env.step(actions.to(env.device))
            observations = observations.to(cfg["device"])

        episode_returns += rewards
        episode_lengths += 1
        done = dones.bool()
        global_step = (step + 1) * env.num_envs

        if writer is not None:
            if (step + 1) % int(cfg["eval_log_interval"]) == 0:
                _write_eval_step_metrics(writer, extras, global_step)
            if done.any():
                writer.add_scalar(
                    "eval/episode_reward",
                    episode_returns[done].mean().item(),
                    global_step,
                )
                writer.add_scalar(
                    "eval/episode_length",
                    episode_lengths[done].mean().item(),
                    global_step,
                )
                _write_eval_episode_metrics(writer, extras, global_step)

        if done.any():
            completed_episodes += int(done.sum().item())
            episode_returns[done] = 0.0
            episode_lengths[done] = 0.0

    if writer is not None:
        writer.flush()
        writer.close()
    logging.info(
        "RSL-RL evaluation finished after %d environment steps and %d completed episodes.",
        (step + 1) * env.num_envs if "step" in locals() else 0,
        completed_episodes,
    )


def _make_summary_writer(logdir: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        logging.warning("TensorBoard is unavailable; evaluation scalars will not be written.")
        return None
    return SummaryWriter(log_dir=logdir.as_posix())


def _write_eval_step_metrics(writer: Any, extras: Mapping[str, Any], step: int) -> None:
    for key, value in extras.items():
        if not key.startswith("metrics/"):
            continue
        name = key.removeprefix("metrics/")
        if name.startswith("episode_"):
            continue
        scalar = _tensor_mean(value)
        if scalar is not None:
            writer.add_scalar(f"eval/metrics/{name}", scalar, step)


def _write_eval_episode_metrics(
    writer: Any,
    extras: Mapping[str, Any],
    step: int,
) -> None:
    names = {
        "episode_success": "eval/episode_success_rate",
        "episode_failed": "eval/episode_failure_rate",
        "episode_tracking_fraction": "eval/episode_tracking_fraction",
        "episode_duration_s": "eval/episode_duration_s",
    }
    completed = _tensor_sum(extras.get("metrics/episode_completed"))
    if completed is None or completed <= 0.0:
        return
    for source_name, target_name in names.items():
        value = _tensor_sum(extras.get(f"metrics/{source_name}"))
        if value is not None:
            writer.add_scalar(target_name, value / completed, step)


def _tensor_mean(value: Any) -> float | None:
    if value is None:
        return None
    try:
        tensor = torch.as_tensor(value, dtype=torch.float32)
    except (TypeError, ValueError):
        return None
    tensor = tensor[torch.isfinite(tensor)]
    return float(tensor.mean().item()) if tensor.numel() else None


def _tensor_sum(value: Any) -> float | None:
    if value is None:
        return None
    try:
        tensor = torch.as_tensor(value, dtype=torch.float32)
    except (TypeError, ValueError):
        return None
    tensor = tensor[torch.isfinite(tensor)]
    return float(tensor.sum().item()) if tensor.numel() else None
