import collections
import gc
import os
import socket
import sys
import time
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import gymnasium
import numpy as np
import torch


def _check_server_reachable(host: str, port: int, timeout: float = 5.0) -> bool:
    """Quick TCP probe to check if the VLA server is reachable."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (ConnectionRefusedError, TimeoutError, OSError):
        return False


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().contiguous().cpu().numpy()
    else:
        value = np.asarray(value)  # if value is already a numpy array, this will be a no-opeartion

    # if value.ndim > 0 and value.shape[0] == 1:
    #     value = value[0]
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

        from omni.physx import get_physx_interface

        env_id = _resolve_visual_env_id(env_id)
        # overwrite_gpu_setting values: -1=Schema Based, 0=Force CPU, 1=Force GPU
        # launcher.device_id is the CUDA device index (0 for cuda:0), NOT the overwrite value!
        get_physx_interface().overwrite_gpu_setting(1 if "cuda" in device else 0)
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
            import traceback as _tb
            import gymnasium
            from openpi_client import websocket_client_policy as _websocket_client_policy

            try:
                env_cfg.seed = seed
                env_cfg.num_envs = 1
                env_cfg.scene.num_envs = 1
                env_cfg.sim.device = device

                env = gymnasium.make(id=env_id, cfg=env_cfg, render_mode="rgb_array")
                env.reset(seed=seed)

                # ── 同步重连推理循环（不使用 asyncio，避免与 Isaac Sim 事件循环冲突）─
                _cuda_fatal = False  # Flag: CUDA context corrupted beyond recovery
                _max_connect_attempts = 30  # 最大连接尝试次数

                while not _cuda_fatal:
                    try:
                        # 1. 建立/重连 WebSocket
                        logging.info("Connecting to VLA server %s:%d ...", host, port)

                        # 先做快速 TCP 探测，避免 WebsocketClientPolicy 内部无限等待
                        if not _check_server_reachable(host, port, timeout=5.0):
                            logging.warning(
                                "VLA server %s:%d is not reachable. "
                                "Please start the server first, then re-run this script.",
                                host,
                                port,
                            )
                            # break
                            time.sleep(5)  # 等待一段时间再尝试连接

                        client = _websocket_client_policy.WebsocketClientPolicy(host, port)
                        logging.info("Server metadata: %s", client.get_server_metadata())

                        # 2. 初始化环境状态
                        obs, _ = env.reset()
                        action_plan: collections.deque[np.ndarray] = collections.deque()
                        episode_idx = 0
                        step = 0
                        last_reward = np.array([0.0])

                        # 3. 推理循环
                        while step < max_steps:
                            if not action_plan:
                                obs["done"] = False
                                obs["reward"] = last_reward
                                result = client.infer(_prepare_request(obs, prompt))
                                last_reward = np.array([0.0])
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

                            # Validate action: NaN/Inf values can cause physics explosions
                            # that lead to Warp CUDA illegal memory access errors.
                            if not np.all(np.isfinite(action)):
                                logging.warning(
                                    "Non-finite action detected at step %d, clamping to zero.", step
                                )
                                action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)

                            low = np.asarray(env.unwrapped.single_action_space.low, dtype=np.float32)
                            high = np.asarray(env.unwrapped.single_action_space.high, dtype=np.float32)
                            action = np.clip(action, low, high)
                            action_tensor = torch.as_tensor(
                                action[None, ...],
                                device=env.unwrapped.device,
                                dtype=torch.float32,
                            )

                            obs, reward, terminated, truncated, _ = env.step(action_tensor)
                            last_reward = np.asarray(_to_numpy(reward)) #.reshape(-1)

                            # Periodic GPU memory cleanup to prevent memory fragmentation
                            # and Warp CUDA illegal memory access errors (typically at ~2250 steps).
                            # This is needed because Isaac Sim's camera rendering pipeline
                            # creates new tensors each step without explicit cleanup.
                            if step > 0 and step % 500 == 0:
                                torch.cuda.empty_cache()
                                gc.collect()

                            if step % max(1, log_interval) == 0:
                                logging.info(
                                    "step=%d reward=%.4f terminated=%s truncated=%s",
                                    step,
                                    float(last_reward[0]),
                                    _to_bool(terminated),
                                    _to_bool(truncated),
                                )

                            if _to_bool(terminated) or _to_bool(truncated):
                                logging.info(
                                    "Episode %d finished at step %d; resetting environment",
                                    episode_idx,
                                    step,
                                )

                                obs["done"] = True
                                obs["reward"] = last_reward
                                result = client.infer(_prepare_request(obs, prompt))

                                obs, _ = env.reset()
                                action_plan.clear()
                                last_reward = np.array([0.0])
                                episode_idx += 1

                            step += 1

                        if _cuda_fatal:
                            break

                        logging.info("Reached max_steps=%d, stopping.", max_steps)
                        obs["done"] = True
                        obs["reward"] = last_reward
                        client.infer(_prepare_request(obs, prompt))
                        # break  # 正常结束

                    except KeyboardInterrupt:
                        obs["done"] = True
                        obs["reward"] = last_reward
                        client.infer(_prepare_request(obs, prompt))
                        logging.info("Interrupted by user.")
                        break

                    except RuntimeError as e:
                        # Catch CUDA runtime errors specifically — the CUDA context may
                        # be corrupted (Warp errors cascade: kernel→stream→sync→free).
                        if "illegal memory access" in str(e) or "CUDA error" in str(e):
                            logging.error("CUDA error caught: %s", e)
                            logging.error(
                                "The Isaac Sim CUDA context is corrupted and cannot be recovered. "
                                "This is typically caused by physics simulation instability (NaN/Inf "
                                "propagation) or a Warp rendering pipeline bug."
                            )
                            _cuda_fatal = True
                            break
                        _max_connect_attempts -= 1
                        if _max_connect_attempts <= 0:
                            logging.error("Max reconnection attempts reached. Exiting.")
                            break
                        logging.warning("Connection lost or error: %s — reconnecting in 5s... (%d attempts left)", e, _max_connect_attempts)
                        time.sleep(5)

                    except Exception as e:
                        _max_connect_attempts -= 1
                        if _max_connect_attempts <= 0:
                            logging.error("Max reconnection attempts reached. Exiting.")
                            break
                        logging.warning("Connection lost or error: %s — reconnecting in 5s... (%d attempts left)", e, _max_connect_attempts)
                        time.sleep(5)

                if _cuda_fatal:
                    logging.error(
                        "Exiting due to unrecoverable CUDA error. Suggestions:\n"
                        "  1. Run with CUDA_LAUNCH_BLOCKING=1 to get a precise error stacktrace.\n"
                        "  2. Reduce camera resolution or disable cameras to isolate the issue.\n"
                        "  3. Check if the physics simulation is producing NaN/Inf (robot explosion).\n"
                        "  4. Ensure the NVIDIA driver and CUDA toolkit versions match Isaac Sim requirements."
                    )

                # finally:
                env.close()

            except Exception:
                _tb.print_exc()
                raise

        hydra_main()
    finally:
        launcher.app.close()