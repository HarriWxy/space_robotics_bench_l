import collections
import os
import sys
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import gymnasium
import numpy as np
import torch


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    else:
        value = np.asarray(value)

    if value.ndim > 0 and value.shape[0] == 1:
        value = value[0]
    return value


def _to_bool(value) -> bool:
    return bool(np.asarray(_to_numpy(value)).reshape(-1)[0])


def _prepare_request(obs: Mapping[str, Any], prompt: str) -> Dict[str, Any]:
    request = {key: _to_numpy(value) for key, value in obs.items()}
    request["prompt"] = prompt
    return request


def _resolve_visual_env_id(env_id: str) -> str:
    from srb.utils import logging

    if env_id.endswith("_visual"):
        return env_id

    visual_env_id = f"{env_id}_visual"
    try:
        gymnasium.spec(visual_env_id)
    except gymnasium.error.Error:
        logging.warning(
            "Visual VLA environment '%s' is not registered. Falling back to '%s'.",
            visual_env_id,
            env_id,
        )
        return env_id

    logging.info(
        "Switching VLA rollout environment from '%s' to visual variant '%s'.",
        env_id,
        visual_env_id,
    )
    return visual_env_id


def _resolve_hydra_config_path(config: str | None) -> Path | None:
    from srb.utils import logging

    if config is None or config.upper() in ("DEFAULT", "IGNORE", "NONE", "NULL"):
        return None

    config_path = Path(config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Hydra config path does not exist: {config_path}")
    if config_path.suffix.lower() not in (".yaml", ".yml"):
        logging.warning(
            f"The provided Hydra config path does not appear to be a YAML file: {config_path}"
        )
    logging.info(f"Using custom Hydra config from: {config_path}")
    return config_path


def run_vla_rollout(
    env_id: str,
    config: str,
    prompt: str,
    host: str,
    port: int,
    seed: int,
    device: str,
    enable_cameras: bool,
    max_steps: int,
    replan_steps: int,
    log_interval: int,
    headless: bool,
    hide_ui: bool,
    forwarded_args: Sequence[str] = (),
    **kwargs,
) -> None:
    if not find_spec("openpi_client"):
        raise ImportError(
            'The "openpi_client" package is required to run VLA rollouts.'
        )

    from srb.core.app import AppLauncher
    from srb.utils import logging
    from srb.utils.cache import update_offline_srb_cache
    from srb.utils.isaacsim import hide_isaacsim_ui
    from srb.utils.path import SRB_APPS_DIR

    resolved_headless = headless or not os.environ.get("DISPLAY")
    if resolved_headless != headless:
        logging.warning(
            "DISPLAY is not set. Forcing headless mode for the VLA rollout."
        )

    kwargs["enable_cameras"] = enable_cameras or env_id.endswith("_visual")
    kwargs["experience"] = SRB_APPS_DIR.joinpath(
        f"srb.{'headless.' if resolved_headless else ''}{'rendering.' if kwargs['enable_cameras'] else ''}{'xr.' if kwargs.get('xr') else ''}kit"
    )

    launcher = AppLauncher(headless=resolved_headless, **kwargs)
    try:
        update_offline_srb_cache()

        from srb.utils.hydra.sim import hydra_task_config

        from omni.physx import acquire_physx_interface

        env_id = _resolve_visual_env_id(env_id)
        acquire_physx_interface().overwrite_gpu_setting(1)
        if hide_ui and not resolved_headless:
            hide_isaacsim_ui()

        config_path = _resolve_hydra_config_path(config)
        if not any(arg.startswith("hydra.output_subdir=") for arg in forwarded_args):
            sys.argv.extend(["hydra.output_subdir=null"])

        @hydra_task_config(
            task_name=env_id,
            agent_cfg_entry_point=None,
            config_path=config_path.as_posix() if config_path else None,
        )
        def hydra_main(env_cfg: Dict[str, Any], agent_cfg: Dict[str, Any] | None = None):
            import gymnasium
            from openpi_client import websocket_client_policy as _websocket_client_policy

            env_cfg.seed = seed
            env_cfg.num_envs = 1
            env_cfg.scene.num_envs = 1
            env_cfg.sim.device = device

            env = gymnasium.make(id=env_id, cfg=env_cfg)
            try:
                client = _websocket_client_policy.WebsocketClientPolicy(host, port)
                logging.info("Server metadata: %s", client.get_server_metadata())

                obs, _ = env.reset()
                action_plan: collections.deque[np.ndarray] = collections.deque()
                episode_idx = 0

                for step in range(max_steps):
                    if not action_plan:
                        result = client.infer(_prepare_request(obs, prompt))
                        action_chunk = np.asarray(result["actions"], dtype=np.float32)
                        if action_chunk.ndim != 2:
                            raise ValueError(
                                f"Expected action chunk with 2 dims, got {action_chunk.shape}"
                            )
                        if len(action_chunk) < replan_steps:
                            raise ValueError(
                                f"Need at least {replan_steps} planned actions, got {len(action_chunk)}"
                            )
                        action_plan.extend(action_chunk[:replan_steps])

                    action = np.asarray(action_plan.popleft(), dtype=np.float32)
                    expected_action_dim = int(env.unwrapped.single_action_space.shape[0])
                    if action.shape[-1] != expected_action_dim:
                        raise ValueError(
                            f"Policy produced action dim {action.shape[-1]}, but SRB env expects {expected_action_dim}. "
                            "Update SRBDataConfig.action_dim to match the environment action space."
                        )
                    low = np.asarray(env.unwrapped.single_action_space.low, dtype=np.float32)
                    high = np.asarray(env.unwrapped.single_action_space.high, dtype=np.float32)
                    action = np.clip(action, low, high)
                    action_tensor = torch.as_tensor(
                        action[None, ...],
                        device=env.unwrapped.device,
                        dtype=torch.float32,
                    )

                    obs, reward, terminated, truncated, _ = env.step(action_tensor)

                    if step % max(1, log_interval) == 0:
                        logging.info(
                            "step=%d reward=%.4f terminated=%s truncated=%s",
                            step,
                            float(np.asarray(_to_numpy(reward)).reshape(-1)[0]),
                            _to_bool(terminated),
                            _to_bool(truncated),
                        )

                    if _to_bool(terminated) or _to_bool(truncated):
                        logging.info(
                            "Episode %d finished at step %d; resetting environment",
                            episode_idx,
                            step,
                        )
                        obs, _ = env.reset()
                        action_plan.clear()
                        episode_idx += 1
            finally:
                env.close()

        hydra_main()
    finally:
        launcher.app.close()