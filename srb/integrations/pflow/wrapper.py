from collections import deque
from functools import cached_property
from typing import  Any, Dict, Tuple
# import warnings

import gymnasium

import torch

# from  import Wrapper
from policyflow_torch.env.base import Wrapper


class SrbEnvWrapper(Wrapper):
    def __init__(
        self,
        env,
        using_historical_obs: bool = False,
        critic_obs_len: int = 1,
        actor_obs_len: int = 1,
    ) -> None:
        self._env = env
        self._unwrapped = env.unwrapped
        self._device = torch.device(self._unwrapped.device)

        self._using_historical_obs = using_historical_obs
        if self._using_historical_obs:
            self.actor_obs_buffer = deque(maxlen=actor_obs_len)
            self.critic_obs_buffer = deque(maxlen=critic_obs_len)

    @cached_property
    def action_space(self) -> gymnasium.Space:
        return gymnasium.spaces.Box(
            low=-1.0, high=1.0, shape=super().action_space.shape
        )

    @cached_property # 缓存只计算一次, 修饰 func
    def observation_space(self) -> gymnasium.Space:
        if hasattr(self._unwrapped, "single_observation_space"):
            obs_space = self._unwrapped.single_observation_space
        else:
            obs_space = self._unwrapped.observation_space

        if self._obs_keys:
            return gymnasium.spaces.Dict(
                {key: obs_space[key] for key in self._obs_keys}
            )
        else:
            return obs_space

    @cached_property
    def state_space(self) -> gymnasium.Space | None:
        """State space"""
        if hasattr(self._unwrapped, "state_space"):
            return self._unwrapped.state_space

        if hasattr(self._unwrapped, "single_observation_space"):
            obs_space = self._unwrapped.single_observation_space
        else:
            obs_space = self._unwrapped.observation_space

        if self._state_keys is None:
            return None
        elif self._state_keys:
            return gymnasium.spaces.Dict(
                {key: obs_space[key] for key in self._state_keys}
            )
        else:
            return obs_space
        
    # def _sanitize_tensor(self, tensor: torch.Tensor, name: str = "tensor") -> torch.Tensor:
    #     """检测并修复张量中的 NaN/Inf 值，用 0 替换。返回是否包含非法值。"""
    #     nan_mask = torch.isnan(tensor)
    #     inf_mask = torch.isinf(tensor)
    #     bad_mask = nan_mask | inf_mask
    #     if bad_mask.any():
    #         n_bad = bad_mask.sum().item()
    #         n_total = tensor.numel()
    #         warnings.warn(
    #             f"[NaN Guard] {name}: {n_bad}/{n_total} values are NaN/Inf, replacing with 0"
    #         )
    #         tensor = tensor.clone()
    #         tensor[bad_mask] = 0.0
    #     return tensor

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Any]:
        obs_dict, reward, terminated, truncated, env_info = self._env.step(actions)
        dones = (terminated | truncated).to(dtype=torch.long)

        if self._using_historical_obs:
            env_ids = dones.nonzero(as_tuple=False).flatten()
            for i in range(self.actor_obs_buffer.maxlen):
                self.actor_obs_buffer[i][env_ids] *= 0.0
            for i in range(self.critic_obs_buffer.maxlen):
                self.critic_obs_buffer[i][env_ids] *= 0.0

            self.actor_obs_buffer.append(obs_dict["policy"])

            if "critic" in obs_dict:
                self.critic_obs_buffer.append(obs_dict["critic"])
            else:
                self.critic_obs_buffer.append(obs_dict["policy"])

            actor_obs = torch.cat(
                [self.actor_obs_buffer[i] for i in range(self.actor_obs_buffer.maxlen)],
                dim=-1,
            )
            critic_obs = torch.cat(
                [
                    self.critic_obs_buffer[i]
                    for i in range(self.critic_obs_buffer.maxlen)
                ],
                dim=-1,
            )
            observations_dict = {
                "actor_observations": actor_obs,
                "critic_observations": critic_obs,
            }
        else:
            obs_act =  torch.cat([obs_dict["state"],obs_dict["proprio"]],dim=-1)
            observations_dict = {
                "actor_observations": obs_act,
                "critic_observations": obs_act,
            }

        if hasattr(self._unwrapped.cfg, "is_finite_horizon"):
            if not self._unwrapped.cfg.is_finite_horizon:
                env_info["time_outs"] = truncated
        else:
            env_info["time_outs"] = truncated

        # observations_dict = self._sanitize_tensor(observations_dict, name="observations_dict")
        # for key in observations_dict:
        #     observations_dict[key] = self._sanitize_tensor(observations_dict[key], name=f"observations_dict/{key}")
        # reward = self._sanitize_tensor(reward, name="reward")
        return (
            observations_dict,
            reward,
            dones,
            env_info,
        )

    def reset(self) -> Tuple[Dict[str, torch.Tensor], Any]:
        obs_dict, env_info = self._env.reset()

        if self._using_historical_obs:
            for _ in range(self.actor_obs_buffer.maxlen):
                self.actor_obs_buffer.append(torch.zeros_like(obs_dict["policy"]))
            if "critic" in obs_dict:
                for _ in range(self.critic_obs_buffer.maxlen):
                    self.critic_obs_buffer.append(torch.zeros_like(obs_dict["critic"]))
            else:
                for _ in range(self.critic_obs_buffer.maxlen):
                    self.critic_obs_buffer.append(torch.zeros_like(obs_dict["policy"]))

            self.actor_obs_buffer.append(obs_dict["policy"])
            if "critic" in obs_dict:
                self.critic_obs_buffer.append(obs_dict["critic"])
            else:
                self.critic_obs_buffer.append(obs_dict["policy"])

            actor_obs = torch.cat(
                [self.actor_obs_buffer[i] for i in range(self.actor_obs_buffer.maxlen)],
                dim=-1,
            )
            critic_obs = torch.cat(
                [
                    self.critic_obs_buffer[i]
                    for i in range(self.critic_obs_buffer.maxlen)
                ],
                dim=-1,
            )
            observations_dict = {
                "actor_observations": actor_obs,
                "critic_observations": critic_obs,
            }
        else:
            obs_act =  torch.cat([obs_dict["state"],obs_dict["proprio"]],dim=-1)
            observations_dict = {
                "actor_observations": obs_act,
                "critic_observations": obs_act,
            }
        return observations_dict, env_info

    @property
    def render_mode(self) -> str | None:
        return self._env.render_mode

    def close(self):
        return self._env.close()
