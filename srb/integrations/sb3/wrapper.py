from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import gymnasium
import numpy
import torch
from gymnasium.core import ObservationWrapper, ObsType, WrapperObsType
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnv,
    VecEnvObs,
    VecEnvStepReturn,
)

if TYPE_CHECKING:
    from srb._typing import AnyEnv


class SelectDictObsWrapper(ObservationWrapper):
    """Select top-level entries from an SRB dictionary observation.

    SRB environments expose a batched ``Dict`` observation with one entry per
    observation category.  SB3's ``MultiInputPolicy`` consumes every entry in
    that dictionary, so this wrapper must be applied before :class:`Sb3EnvWrapper`
    and update the underlying single-environment space as well.
    """

    def __init__(self, env: "AnyEnv", observation_keys: Sequence[str]):
        super().__init__(env)

        keys = tuple(observation_keys)
        if not keys:
            raise ValueError("observation_keys must contain at least one key")
        if len(keys) != len(set(keys)):
            raise ValueError("observation_keys must not contain duplicates")

        single_observation_space = getattr(
            env.unwrapped, "single_observation_space", None
        )
        if not isinstance(single_observation_space, gymnasium.spaces.Dict):
            raise TypeError(
                "SelectDictObsWrapper requires a Dict single_observation_space"
            )

        missing = [key for key in keys if key not in single_observation_space.spaces]
        if missing:
            available = ", ".join(single_observation_space.spaces.keys())
            raise KeyError(
                f"Unknown observation keys {missing}; available keys: [{available}]"
            )

        self.observation_keys = keys
        selected_space = gymnasium.spaces.Dict(
            {key: single_observation_space.spaces[key] for key in self.observation_keys}
        )

        # Sb3EnvWrapper resolves the space from ``env.unwrapped``.  Keep both
        # the single-environment and batched spaces consistent with the
        # filtered observations returned below.
        self.unwrapped.single_observation_space = selected_space  # type: ignore
        self.unwrapped.observation_space = gymnasium.vector.utils.batch_space(
            selected_space,
            self.unwrapped.num_envs,  # type: ignore
        )
        self._observation_space = selected_space

    def observation(self, observation: ObsType) -> WrapperObsType:
        """Return only the configured top-level observation categories."""

        if not isinstance(observation, Mapping):
            raise TypeError(
                "SelectDictObsWrapper expects a mapping observation, got "
                f"{type(observation).__name__}"
            )
        return {key: observation[key] for key in self.observation_keys}


class Sb3EnvWrapper(VecEnv):
    def __init__(self, env: "AnyEnv", ignore_extras: bool = True):
        # Initialize the wrapper
        self.env = env
        # Collect common information
        self.num_envs = self.unwrapped.num_envs
        self.sim_device = self.unwrapped.device
        self.render_mode = self.unwrapped.render_mode

        if ignore_extras:
            self._default_infos = [{"episode": None} for _ in range(self.num_envs)]
            self._process_extras = self._process_extras_ignore

        # Obtain gym spaces
        # Note: stable-baselines3 does not like when we have unbounded action space so
        #   we set it to some high value here. Maybe this is not general but something to think about.
        observation_space = self.unwrapped.single_observation_space  # type: ignore
        action_space = self.unwrapped.single_action_space  # type: ignore
        if isinstance(
            action_space, gymnasium.spaces.Box
        ) and not action_space.is_bounded("both"):
            action_space = gymnasium.spaces.Box(
                low=-1.0, high=1.0, shape=action_space.shape
            )

        # Initialize vec-env
        VecEnv.__init__(self, self.num_envs, observation_space, action_space)
        # Add buffer for logging episodic information
        self._ep_rew_buf = torch.zeros(self.num_envs, device=self.sim_device)
        self._ep_len_buf = torch.zeros(self.num_envs, device=self.sim_device)

        # Use the appropriate observation processing function
        if isinstance(observation_space, gymnasium.spaces.Dict):
            self._process_obs = self._process_obs_dict  # type: ignore

    def __str__(self):
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        return str(self)

    @classmethod
    def class_name(cls) -> str:
        return cls.__name__

    @property
    def unwrapped(self) -> "AnyEnv":
        return self.env.unwrapped  # type: ignore

    def get_episode_rewards(self) -> Sequence[float]:
        return self._ep_rew_buf.cpu().tolist()

    def get_episode_lengths(self) -> Sequence[int]:
        return self._ep_len_buf.cpu().tolist()

    def seed(self, seed: int | None = None) -> Sequence[int | None]:
        return [self.unwrapped.seed(seed)] * self.unwrapped.num_envs  # type: ignore

    def reset(self) -> VecEnvObs:
        obs_dict, _ = self.env.reset()
        # Reset episodic information buffers
        self._ep_rew_buf.zero_()
        self._ep_len_buf.zero_()
        # Convert data types to numpy depending on backend
        return self._process_obs(obs_dict)  # type: ignore

    def step_async(self, actions):
        # Convert input to numpy array
        if not isinstance(actions, torch.Tensor):
            actions = numpy.asarray(actions)
            actions = torch.from_numpy(actions).to(
                device=self.sim_device, dtype=torch.float32
            )
        else:
            actions = actions.to(device=self.sim_device, dtype=torch.float32)
        # Convert to tensor
        self._async_actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        # Record step information
        obs_dict, reward, terminated, truncated, extras = self.env.step(
            self._async_actions  # type: ignore
        )
        # Update episode un-discounted return and length
        self._ep_rew_buf += reward
        self._ep_len_buf += 1

        # Convert data types to numpy depending on backend
        obs = self._process_obs(obs_dict)  # type: ignore

        if isinstance(reward, torch.Tensor):
            reward = reward.detach().cpu().numpy()
        elif not isinstance(reward, numpy.ndarray):
            reward = numpy.array((reward,), dtype=numpy.float32)

        if isinstance(terminated, torch.Tensor):
            terminated = terminated.detach().cpu().numpy()
        elif not isinstance(terminated, numpy.ndarray):
            terminated = numpy.array((terminated,), dtype=bool)

        if isinstance(truncated, torch.Tensor):
            truncated = truncated.detach().cpu().numpy()
        elif not isinstance(truncated, numpy.ndarray):
            truncated = numpy.array((truncated,), dtype=bool)

        # Compute reset ids
        dones = terminated | truncated
        reset_ids = numpy.argwhere(dones > 0).flatten()
        # Convert extra information to list of dicts
        infos = self._process_extras(
            obs,  # type: ignore
            terminated,  # type: ignore
            truncated,  # type: ignore
            extras,
            reset_ids,  # type: ignore
        )

        # Reset info for terminated environments
        self._ep_rew_buf[reset_ids] = 0
        self._ep_len_buf[reset_ids] = 0

        return obs, reward, dones, infos  # type: ignore

    def close(self):
        self.env.close()

    def get_attr(self, attr_name: str, indices: Sequence[int] | slice | None = None):
        # Resolve indices
        if indices is None:
            indices = slice(None)
            num_indices = self.num_envs
        else:
            num_indices = len(indices)  # type: ignore
        # Obtain attribute value
        attr_val = getattr(self.env, attr_name)
        # Return the value
        if not isinstance(attr_val, torch.Tensor):
            return [attr_val] * num_indices
        else:
            return attr_val[indices].detach().cpu().numpy()

    def set_attr(
        self, attr_name: str, value: Any, indices: Sequence[int] | None = None
    ):
        raise NotImplementedError("Setting attributes is not supported.")

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: Sequence[int] | None = None,
        **method_kwargs,
    ):
        if method_name == "render":
            # Gymnasium does not support changing render mode at runtime
            return self.env.render()
        else:
            # This isn't properly implemented but it is not necessary.
            # Mostly done for completeness.
            env_method = getattr(self.env, method_name)
            return env_method(*method_args, indices=indices, **method_kwargs)

    def env_is_wrapped(self, wrapper_class, indices=None):
        raise NotImplementedError(
            "Checking if environment is wrapped is not supported."
        )

    def get_images(self):
        raise NotImplementedError("Getting images is not supported.")

    def _process_obs(self, obs: numpy.ndarray | torch.Tensor) -> numpy.ndarray:
        if self.num_envs == 1 and obs.ndim == 1:
            return (
                obs.detach().cpu().numpy() if isinstance(obs, torch.Tensor) else obs
            ).reshape((1, *obs.shape))

        return obs.detach().cpu().numpy() if isinstance(obs, torch.Tensor) else obs

    def _process_obs_dict(
        self, obs: dict[str, numpy.ndarray | torch.Tensor]
    ) -> Mapping[str, numpy.ndarray]:
        _first_obs = next(iter(obs.values()))
        if isinstance(_first_obs, torch.Tensor):
            obs = {k: v.detach().cpu().numpy() for k, v in obs.items()}  # type: ignore

        if self.num_envs == 1 and _first_obs.ndim == 1:
            obs = {k: v.reshape((1, *v.shape)) for k, v in obs.items()}

        return obs  # type: ignore

    def _process_extras(
        self,
        obs: numpy.ndarray,
        terminated: numpy.ndarray,
        truncated: numpy.ndarray,
        extras: dict[str, Any],
        reset_ids: numpy.ndarray,
    ) -> Sequence[Mapping[str, Any]]:
        # Create empty list of dictionaries to fill
        infos: list[dict[str, Any]] = [
            dict.fromkeys(extras.keys()) for _ in range(self.num_envs)
        ]
        # Fill-in information for each sub-environment
        # Note: This loop becomes slow when number of environments is large.
        for idx in range(self.num_envs):
            # Fill-in episode monitoring info
            if idx in reset_ids:
                infos[idx]["episode"] = {}
                infos[idx]["episode"]["r"] = float(self._ep_rew_buf[idx])
                infos[idx]["episode"]["l"] = float(self._ep_len_buf[idx])
            else:
                infos[idx]["episode"] = None
            # Fill-in bootstrap information
            infos[idx]["TimeLimit.truncated"] = truncated[idx] and not terminated[idx]
            # Fill-in information from extras
            for key, value in extras.items():
                # 1. remap extra episodes information safely
                # 2. for others just store their values
                if key == "log":
                    # Only log this data for episodes that are terminated
                    if infos[idx]["episode"] is not None:
                        for sub_key, sub_value in value.items():
                            infos[idx]["episode"][sub_key] = sub_value
                elif key == "reward_terms" and isinstance(value, dict):
                    infos[idx][key] = self._extract_reward_terms(value, idx)
                else:
                    infos[idx][key] = value[idx]
            # Add information about terminal observation separately
            if idx in reset_ids:
                # Extract terminal observations
                if isinstance(obs, dict):
                    terminal_obs = dict.fromkeys(obs.keys())
                    for key, value in obs.items():
                        terminal_obs[key] = value[idx]
                else:
                    terminal_obs = obs[idx]
                # Add info to dict
                infos[idx]["terminal_observation"] = terminal_obs
            else:
                infos[idx]["terminal_observation"] = None
        # Return list of dictionaries
        return infos

    def _process_extras_ignore(
        self,
        obs: numpy.ndarray,
        terminated: numpy.ndarray,
        truncated: numpy.ndarray,
        extras: dict[str, Any],
        reset_ids: numpy.ndarray,
    ) -> Sequence[Mapping[str, Any]]:
        infos = deepcopy(self._default_infos)

        for idx in reset_ids:
            # Fill-in episode monitoring info
            infos[idx]["episode"] = {
                "r": float(self._ep_rew_buf[idx]),
                "l": float(self._ep_len_buf[idx]),
            }

            # Fill-in bootstrap information
            infos[idx]["TimeLimit.truncated"] = truncated[idx] and not terminated[idx]

            # Add information about terminal observation separately
            if isinstance(obs, dict):
                terminal_obs = {key: value[idx] for key, value in obs.items()}
            else:
                terminal_obs = obs[idx]
            infos[idx]["terminal_observation"] = terminal_obs

        reward_terms = extras.get("reward_terms")
        if isinstance(reward_terms, dict):
            for idx in range(self.num_envs):
                infos[idx]["reward_terms"] = self._extract_reward_terms(
                    reward_terms,
                    idx,
                )

        metrics = {
            key: value
            for key, value in extras.items()
            if isinstance(key, str) and key.startswith("metrics/") and key != "metrics/"
        }
        if metrics:
            metric_values = self._extract_batch_values(metrics)
            for metric_idx, key in enumerate(metrics):
                for env_idx in range(self.num_envs):
                    infos[env_idx][key] = float(metric_values[metric_idx, env_idx])

        return infos

    def _extract_batch_values(self, values: Mapping[str, Any]) -> numpy.ndarray:
        """Move scalar or per-environment extras to host memory in one batch."""

        columns = []
        for name, value in values.items():
            tensor = torch.as_tensor(value, device=self.sim_device).detach().float()
            tensor = tensor.reshape(-1)
            if tensor.numel() == 1:
                tensor = tensor.expand(self.num_envs)
            elif tensor.numel() != self.num_envs:
                raise ValueError(
                    f"SB3 metric {name!r} must be scalar or have one value per "
                    f"environment, got {tensor.numel()} values for {self.num_envs} "
                    "environments"
                )
            columns.append(tensor)
        return torch.stack(columns).cpu().numpy()

    @staticmethod
    def _extract_reward_terms(
        reward_terms: dict[str, Any],
        idx: int,
    ) -> dict[str, float]:
        extracted_reward_terms: dict[str, float] = {}
        for reward_term, value in reward_terms.items():
            if isinstance(value, torch.Tensor):
                extracted_reward_terms[reward_term] = float(
                    value[idx].detach().cpu().item()
                )
            elif isinstance(value, numpy.ndarray):
                extracted_reward_terms[reward_term] = float(value[idx])
            else:
                extracted_reward_terms[reward_term] = float(value)

        return extracted_reward_terms
