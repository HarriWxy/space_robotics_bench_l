"""Run the FPO++ agent against an already-launched SRB environment."""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal

import gymnasium
import torch
from isaaclab_fpo.runners.on_policy_runner import OnPolicyRunner

from srb.integrations.fpo.wrapper import SrbFpoEnvWrapper
from srb.integrations.tensorboard import CanonicalScalarWriter
from srb.utils import logging
from srb.utils.cfg import stamp_dir
from srb.wrappers import maybe_wrap_action_smoothing

if TYPE_CHECKING:
    from isaacsim.simulation_app import SimulationApp

    from srb._typing import AnyEnv, AnyEnvCfg


FRAMEWORK_NAME = "fpo"


_RUNNER_DEFAULTS = {
    "seed": 42,
    "device": None,
    "num_steps_per_env": 24,
    "max_iterations": 1500,
    "empirical_normalization": True,
    "randomize_reset_episode_progress": 0.0,
    "clip_actions": 1.0,
    "save_interval": 50,
    "experiment_name": "srb_fpo",
    "run_name": "",
    "logger": "tensorboard",
    "neptune_project": "isaaclab",
    "wandb_project": "isaaclab",
    "eval_episodes": 10,
    "flow_eval_modes": ["zero", "random"],
    "flow_eval_fixed_seed": 12345,
    "enable_post_training_eval": False,
    "post_eval_checkpoint_interval": 1,
    "resume": False,
    "load_run": ".*",
    "load_checkpoint": "model_.*.pt",
    "dynamics_encoder": {
        "enabled": False,
        "state_keys": ["proprio", "proprio_dyn"],
        "latent_dim": 32,
        "hidden_dim": 128,
        "history_length": 16,
        "num_heads": 4,
        "num_layers": 2,
        "transformer_ff_dim": None,
        "physics_dim": 1,
        "dropout": 0.0,
        "transition_loss_weight": 1.0,
        "physics_loss_weight": 1.0,
        "update_interval": 64,
        "learning_rate": 3.0e-4,
        "weight_decay": 1.0e-4,
        "max_grad_norm": 1.0,
        "physics_target_key": "metrics/gravity_magnitude",
        "physics_target_scale": 9.80665,
    },
}

_POLICY_DEFAULTS = {
    "class_name": "ActorCritic",
    "init_noise_std": 1.0,
    "actor_hidden_dims": [256, 256, 256],
    "critic_hidden_dims": [256, 256, 256],
    "activation": "elu",
    "actor_scale": 1.0,
    "actor_mlp_output_scale": 1.0,
    "actor_final_layer_weight_scale": None,
    "timestep_embed_dim": 8,
    "training_sampling_steps": None,
    "cfm_loss_t_inverse_cdf_beta": 1.0,
    "sampling_steps": 8,
    "cfm_loss_reduction": "sqrt",
    "action_perturb_std": 0.02,
}

_ALGORITHM_DEFAULTS = {
    "class_name": "FPO",
    "num_learning_epochs": 16,
    "num_mini_batches": 4,
    "learning_rate": 1.0e-4,
    "weight_decay": 1.0e-4,
    "adam_betas": (0.9, 0.999),
    "schedule": "fixed",
    "gamma": 0.99,
    "lam": 0.95,
    "knn_entropy_coef": 0.0,
    "knn_entropy_k": 1,
    "desired_kl": 1.0e-4,
    "max_grad_norm": 1.0,
    "value_loss_coef": 1.0,
    "use_clipped_value_loss": False,
    "clip_param": 0.05,
    "trust_region_mode": "aspo",
    "normalize_advantage": True,
    "normalize_advantage_per_mini_batch": False,
    "advantage_clamp": (100.0, 100.0),
    "n_samples_per_action": 16,
    "cfm_diff_clamp_max": 10.0,
    "cfm_loss_clamp": 20.0,
    "cfm_loss_clamp_negative_advantages": True,
    "cfm_loss_clamp_negative_advantages_max": 20.0,
    "storage_action_noise_std": 0.0,
    "ema_decay": 0.95,
    "ema_warmup_steps": 500,
}


class _EnvironmentStepWriter(CanonicalScalarWriter):
    """Write FPO scalars with canonical tags and environment-step x-axis."""

    def __init__(self, writer: Any, runner: Any):
        super().__init__(
            writer,
            step_fn=lambda: int(runner.tot_timesteps),
        )


def _install_environment_step_logging(runner: Any) -> None:
    """Use canonical tags, environment-step x-coordinates and episode rates."""

    original_log = runner.log

    def log_with_environment_steps(*args: Any, **kwargs: Any):
        if args and isinstance(args[0], dict):
            _aggregate_episode_event_logs(args[0])
        if runner.writer is not None and not isinstance(
            runner.writer, _EnvironmentStepWriter
        ):
            runner.writer = _EnvironmentStepWriter(runner.writer, runner)
        return original_log(*args, **kwargs)

    runner.log = log_with_environment_steps


def _aggregate_episode_event_logs(locs: dict[str, Any]) -> None:
    """Replace sparse per-step event counters with rates for FPO's logger.

    The upstream runner averages every value in ``ep_infos``.  A sparse event
    counter would therefore be divided by rollout steps instead of completed
    episodes.  Collapse the event tensors before the runner sees them while
    retaining its normal logging path for all other metrics.
    """

    episode_infos = locs.get("ep_infos")
    if not isinstance(episode_infos, list) or not episode_infos:
        return

    event_names = {
        "rollout/metrics/episode_completed",
        "rollout/metrics/episode_success",
        "rollout/metrics/episode_failed",
        "rollout/metrics/episode_tracking_fraction",
        "rollout/metrics/episode_duration_s",
    }
    sums: dict[str, torch.Tensor] = {}
    for episode_info in episode_infos:
        if not isinstance(episode_info, Mapping):
            continue
        for name in event_names:
            if name not in episode_info:
                continue
            value = torch.as_tensor(episode_info[name], dtype=torch.float32)
            value = value[torch.isfinite(value)]
            if value.numel() > 0:
                sums[name] = (
                    sums.get(name, torch.zeros((), device=value.device)) + value.sum()
                )

    completed = float(sums.get("rollout/metrics/episode_completed", 0.0))
    # Remove the sparse counters from the upstream average and, when an
    # episode completed, add one scalar per rate to the first mapping, which is
    # the key set the runner visits.
    filtered_infos = []
    for episode_info in episode_infos:
        if isinstance(episode_info, Mapping):
            filtered_infos.append(
                {
                    key: value
                    for key, value in episode_info.items()
                    if key not in event_names
                }
            )
        else:
            filtered_infos.append(episode_info)

    first_info = filtered_infos[0]
    if not isinstance(first_info, dict):
        return
    if completed <= 0.0:
        locs["ep_infos"] = filtered_infos
        return
    first_info["rollout/episode_success_rate"] = (
        float(sums.get("rollout/metrics/episode_success", 0.0)) / completed
    )
    first_info["rollout/episode_failure_rate"] = (
        float(sums.get("rollout/metrics/episode_failed", 0.0)) / completed
    )
    first_info["rollout/episode_tracking_fraction"] = (
        float(sums.get("rollout/metrics/episode_tracking_fraction", 0.0)) / completed
    )
    first_info["rollout/episode_duration_s"] = (
        float(sums.get("rollout/metrics/episode_duration_s", 0.0)) / completed
    )
    first_info["rollout/metrics/episode_completed"] = completed
    locs["ep_infos"] = filtered_infos


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return copy.deepcopy(dict(value))
    if hasattr(value, "to_dict"):
        return copy.deepcopy(value.to_dict())
    if hasattr(value, "__dict__"):
        return copy.deepcopy(vars(value))
    raise TypeError(f"FPO agent config must be a mapping, got {type(value).__name__}")


def _deep_merge(
    defaults: Mapping[str, Any], overrides: Mapping[str, Any]
) -> dict[str, Any]:
    result = copy.deepcopy(dict(defaults))
    for key, value in overrides.items():
        if isinstance(result.get(key), Mapping) and isinstance(value, Mapping):
            result[key] = _deep_merge(result[key], value)  # type: ignore[arg-type]
        else:
            result[key] = copy.deepcopy(value)
    return result


def _to_namespace(value: Any) -> Any:
    if isinstance(value, Mapping):
        return SimpleNamespace(
            **{key: _to_namespace(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return [_to_namespace(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_namespace(item) for item in value)
    return value


def _build_config(
    agent_cfg: Any,
    *,
    env_device: torch.device,
    env_id: str,
) -> SimpleNamespace:
    raw_cfg = _as_dict(agent_cfg)
    merged = _deep_merge(
        {
            **_RUNNER_DEFAULTS,
            "policy": _POLICY_DEFAULTS,
            "algorithm": _ALGORITHM_DEFAULTS,
            "obs": {
                "actor_keys": ["proprio", "proprio_dyn", "command"],
                "critic_keys": None,
            },
        },
        raw_cfg,
    )

    configured_device = merged.get("device")
    if configured_device is not None and torch.device(configured_device) != env_device:
        logging.warning(
            f"FPO device {configured_device!s} does not match SRB environment "
            f"device {env_device}; using the environment device."
        )
    merged["device"] = str(env_device)

    if not merged.get("experiment_name"):
        merged["experiment_name"] = f"srb_{env_id.rsplit('/', 1)[-1]}_fpo"

    return _to_namespace(merged)


def _configure_dynamics_encoder(
    wrapped_env: SrbFpoEnvWrapper,
    *,
    cfg: SimpleNamespace,
    workflow: Literal["train", "eval"],
) -> Any | None:
    """Attach the optional dynamics encoder without changing baseline FPO."""

    dynamics_cfg = getattr(cfg, "dynamics_encoder", None)
    if dynamics_cfg is None or not bool(getattr(dynamics_cfg, "enabled", False)):
        return None

    from dyn.dyn_encoder import (
        DynamicsEncoder,
        DynamicsEncoderConfig,
        DynamicsEncoderTrainer,
    )

    state_keys = tuple(getattr(dynamics_cfg, "state_keys", ("proprio", "proprio_dyn")))
    state = wrapped_env.observation_tensor(state_keys, name="dynamics_state")
    encoder = DynamicsEncoder(
        DynamicsEncoderConfig(
            state_dim=int(state.shape[1]),
            action_dim=wrapped_env.num_actions,
            latent_dim=int(dynamics_cfg.latent_dim),
            hidden_dim=int(dynamics_cfg.hidden_dim),
            history_length=int(dynamics_cfg.history_length),
            num_heads=int(dynamics_cfg.num_heads),
            num_layers=int(dynamics_cfg.num_layers),
            transformer_ff_dim=(
                None
                if dynamics_cfg.transformer_ff_dim is None
                else int(dynamics_cfg.transformer_ff_dim)
            ),
            physics_dim=int(dynamics_cfg.physics_dim),
            dropout=float(dynamics_cfg.dropout),
            transition_loss_weight=float(dynamics_cfg.transition_loss_weight),
            physics_loss_weight=float(dynamics_cfg.physics_loss_weight),
        )
    ).to(wrapped_env.device)
    trainer = (
        DynamicsEncoderTrainer(
            encoder,
            learning_rate=float(dynamics_cfg.learning_rate),
            weight_decay=float(dynamics_cfg.weight_decay),
            max_grad_norm=float(dynamics_cfg.max_grad_norm),
        )
        if workflow == "train"
        else None
    )
    target_key = getattr(dynamics_cfg, "physics_target_key", None)
    target_scale = getattr(dynamics_cfg, "physics_target_scale", None)
    wrapped_env.attach_dynamics_encoder(
        encoder,
        state_keys=state_keys,
        trainer=trainer,
        update_interval=int(dynamics_cfg.update_interval),
        physics_target_key=target_key,
        physics_target_scale=(None if target_scale is None else float(target_scale)),
    )
    logging.info(
        "Dynamics encoder enabled: state_dim=%d, action_dim=%d, latent_dim=%d, "
        "history_length=%d, transformer_layers=%d, update_interval=%d",
        encoder.state_dim,
        encoder.action_dim,
        encoder.latent_dim,
        encoder.history_length,
        int(dynamics_cfg.num_layers),
        int(dynamics_cfg.update_interval),
    )
    return encoder


def _last_checkpoint(logdir: Path) -> Path | None:
    checkpoints = []
    for checkpoint in logdir.glob("model_*.pt"):
        match = re.fullmatch(r"model_(\d+)\.pt", checkpoint.name)
        if match and checkpoint.is_file():
            checkpoints.append((int(match.group(1)), checkpoint))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda item: item[0])[1]


def _resolve_checkpoint(
    *,
    workflow: Literal["train", "eval"],
    logdir: Path,
    model: Path | None,
    continue_training: bool | None,
) -> Path | None:
    if model:
        return Path(model)
    if workflow == "eval" or continue_training:
        checkpoint = _last_checkpoint(logdir)
        if checkpoint is None and workflow == "eval":
            raise FileNotFoundError(f"No FPO checkpoint (model_*.pt) found in {logdir}")
        return checkpoint
    return None


def _install_action_consistency(runner: OnPolicyRunner, clip_actions: float | None) -> None:
    """Keep FPO's stored action equal to the action sent to SRB.

    The upstream FPO wrapper clips only inside env.step. FPO computes its
    initial CFM loss before that call, so a clipped action could otherwise
    leave the transition and simulator out of sync. Recompute the CFM terms
    only when clipping changes an action.
    """

    if clip_actions is None:
        return

    clip = float(clip_actions)
    if clip <= 0.0:
        raise ValueError(f"clip_actions must be positive or None, got {clip}")

    original_act = runner.alg.act

    def act(obs: torch.Tensor, critic_obs: torch.Tensor):
        actions = original_act(obs, critic_obs)
        clipped = actions.clamp(-clip, clip)
        if torch.equal(actions, clipped):
            return actions

        transition = runner.alg.transition
        initial_cfm_loss, x1_pred, _ = runner.alg.policy.get_cfm_loss(
            obs,
            clipped,
            transition.cfm_loss_eps,
            transition.cfm_loss_t,
        )
        transition.actions = clipped
        transition.initial_cfm_loss = initial_cfm_loss.detach()
        transition.x1_pred = x1_pred.detach()
        return clipped

    runner.alg.act = act


def run(
    workflow: Literal["train", "eval"],
    env: AnyEnv | gymnasium.Env,
    sim_app: SimulationApp,
    env_id: str,
    env_cfg: AnyEnvCfg | None,
    agent_cfg: dict,
    logdir: Path,
    model: Path | None = None,
    continue_training: bool | None = None,
    untrained: bool = False,
    **kwargs,
):
    del env_cfg, kwargs

    try:
        from isaaclab_fpo.runners import OnPolicyRunner
    except ImportError as exc:
        raise ImportError(
            "The SRB FPO integration requires isaaclab_fpo in the conda srb "
            "environment. Activate it with 'conda activate srb', then run: "
            "python -m pip install --no-deps -e "
            "/root/R2A/Algos/fpo-control-saa/isaaclab_experiments/isaaclab_fpo"
        ) from exc

    raw_cfg = _as_dict(agent_cfg)
    obs_cfg = raw_cfg.get("obs", {}) or {}
    smoothing_cfg = raw_cfg.get("smoothing", {}) or {}

    if smoothing_cfg.get("enabled", False):
        logging.warning(
            "Action smoothing is enabled for FPO. The policy transition records "
            "pre-smoothing actions; disable smoothing for strict on-policy action consistency."
        )
        env = maybe_wrap_action_smoothing(env, smoothing_cfg)

    wrapped_env = SrbFpoEnvWrapper(
        env,
        actor_keys=obs_cfg.get("actor_keys", ["proprio", "proprio_dyn", "command"]),
        critic_keys=obs_cfg.get("critic_keys"),
        clip_actions=raw_cfg.get("clip_actions", 1.0),
    )
    cfg = _build_config(
        agent_cfg,
        env_device=wrapped_env.device,
        env_id=env_id,
    )
    wrapped_env.clip_actions = cfg.clip_actions
    dynamics_encoder = _configure_dynamics_encoder(
        wrapped_env,
        cfg=cfg,
        workflow=workflow,
    )

    from_checkpoint = _resolve_checkpoint(
        workflow=workflow,
        logdir=Path(logdir),
        model=model,
        continue_training=continue_training,
    )
    if from_checkpoint:
        logging.info(f"Loading FPO checkpoint from {from_checkpoint}")

    runner_logdir = Path(logdir)
    if workflow == "eval":
        runner_logdir = stamp_dir(runner_logdir.joinpath("eval"))

    runner = OnPolicyRunner(
        wrapped_env,
        train_cfg=cfg,
        log_dir=runner_logdir.as_posix(),
        device=cfg.device,
    )
    if dynamics_encoder is not None:
        # FPO creates its optimizer before this optional module is attached, so
        # the encoder is updated only by DynamicsEncoderTrainer. Registering it
        # here still makes the normal FPO checkpoint contain its weights.
        runner.alg.policy.add_module("dynamics_encoder", dynamics_encoder)
    runner.add_git_repo_to_log(__file__)
    _install_action_consistency(runner, cfg.clip_actions)
    _install_environment_step_logging(runner)

    if from_checkpoint:
        runner.load(
            from_checkpoint.as_posix(),
            load_optimizer=workflow == "train",
        )
    elif workflow == "eval" and not untrained:
        raise FileNotFoundError("An FPO checkpoint is required for evaluation")

    if workflow == "train":
        runner.learn(
            num_learning_iterations=cfg.max_iterations,
            init_at_random_ep_len=True,
        )
        return

    policy = runner.get_inference_policy(device=wrapped_env.device)
    observations, _ = wrapped_env.reset()
    with torch.inference_mode():
        while sim_app.is_running():
            actions = policy(observations)
            observations, _, _, _ = wrapped_env.step(actions)
