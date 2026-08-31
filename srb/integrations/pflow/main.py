from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

import gymnasium
from isaacsim.simulation_app import SimulationApp

from srb.integrations.pflow.wrapper import SrbEnvWrapper
from srb.utils import logging
from srb.utils.cfg import last_file, stamp_dir
from srb.wrappers import maybe_wrap_action_smoothing

if TYPE_CHECKING:
    from srb._typing import AnyEnv, AnyEnvCfg

import os 
# os.environ['CUDA_VISIBLE_DEVICES']='1'

FRAMEWORK_NAME = "policyflow"


def run(
    workflow: Literal["train", "eval"],
    env: "AnyEnv | gymnasium.Env",
    sim_app: SimulationApp,
    env_id: str,
    env_cfg: "AnyEnvCfg | None",
    agent_cfg: dict,
    logdir: Path,
    model: Path,
    continue_training: bool | None = None,
    **kwargs,
):
    from policyflow_torch.agents import PolicyFlow
    from policyflow_torch.modules import (  # for intellisense
        ContinuousNormalizingFlow,
        ConditionLinearLayer,
        FlowMlp,
        Network,
    )
    from policyflow_torch.runners import IsaaclabRunner
    from policyflow_torch.storage import ReplayBuffer

    # Pop the entire smoothing config dictionary to be handled separately.
    smoothing_cfg = agent_cfg.pop("smoothing", {})

    # Determine checkpoint path
    if model:
        from_checkpoint = model
    elif workflow == "eval" or continue_training:
        from_checkpoint = last_file(logdir.joinpath("checkpoints"), modification_time=True)
    else:
        from_checkpoint = ""
    if from_checkpoint:
        logging.info(f"Loading model from {from_checkpoint}")

    # Special handling for eval workflow
    if workflow == "eval":
        logdir = stamp_dir(logdir.joinpath("eval"))

    # Pop PolicyFlow runner config
    max_iterations = agent_cfg.pop("max_iterations", 40000)
    rollouts = agent_cfg.pop("rollouts", 24)
    save_interval = agent_cfg.pop("save_interval", 100)

    # Pop model architecture config
    obs_embedding_dims = agent_cfg.pop("obs_embedding_dims", 64)
    flow_sample_steps = agent_cfg.pop("flow_sample_steps", 10)
    using_historical_obs = agent_cfg.pop("using_historical_obs", False)
    critic_obs_len = agent_cfg.pop("critic_obs_len", 1)
    actor_obs_len = agent_cfg.pop("actor_obs_len", 1)

    actor_hidden_dims = agent_cfg.pop("actor_hidden_dims", [512, 256, 128])
    actor_activations = agent_cfg.pop("actor_activations", ["mish", "mish", "mish", "linear"])  #
    critic_hidden_dims = agent_cfg.pop("critic_hidden_dims", [512, 256, 128])
    critic_activations = agent_cfg.pop("critic_activations", ["mish", "mish", "mish", "linear"])

    # Enable action smoothing if enabled
    env = maybe_wrap_action_smoothing(
        env,  # type: ignore
        smoothing_cfg,
    )

    # Wrap the environment for PolicyFlow
    wrapped_env = SrbEnvWrapper(
        env=env,  # type: ignore
        using_historical_obs=using_historical_obs,
        critic_obs_len=critic_obs_len,
        actor_obs_len=actor_obs_len,
    )

    # Get environment dimensions
    obs_dict, _ = wrapped_env.reset()
    critic_obs_size = obs_dict["critic_observations"].shape[1]
    actor_obs_size = obs_dict["actor_observations"].shape[1]

    if hasattr(env.unwrapped, "action_manager"):
        num_actions = env.unwrapped.action_manager.total_action_dim
    else:
        num_actions = gymnasium.spaces.flatdim(env.unwrapped.single_action_space)

    # Build actor network (Continuous Normalizing Flow)
    nn_flow = FlowMlp(
        x_dim=num_actions,
        emb_dim=obs_embedding_dims,
        hidden_dims=actor_hidden_dims,
        activations=actor_activations,
        timestep_emb_type="fourier",
    ).to(wrapped_env.device)

    nn_condition = ConditionLinearLayer(
        cond_dim=actor_obs_size,
        emb_dim=obs_embedding_dims,
    ).to(wrapped_env.device)

    actor = ContinuousNormalizingFlow(
        x_dims=num_actions,
        nn_flow=nn_flow,
        nn_condition=nn_condition,
        sample_steps=flow_sample_steps,
        interpolation_type="rectified_flow",
        device=wrapped_env.device,
    )

    # Build critic network
    critic = Network(
        input_size=critic_obs_size,
        output_size=1,
        hidden_dims=critic_hidden_dims,
        activations=critic_activations,
        init_fade=True,
        init_gain=1.0,
        input_normalization=False,
        recurrent=False,
    ).to(wrapped_env.device)

    models = {
        "critic": critic,
        "actor": actor,
    }

    # Build replay buffer
    replay_buffer = ReplayBuffer(
        memory_size=rollouts,
        num_envs=wrapped_env.num_envs,
        device=wrapped_env.device,
    )

    # Build agent
    agent = PolicyFlow(
        models=models,
        replay_buffer=replay_buffer,
        device=wrapped_env.device,
        cfg=agent_cfg,
    )
    agent.init_replay_buffer(
        critic_observation_size=critic_obs_size,
        actor_observation_size=actor_obs_size,
        action_size=num_actions,
    )

    # Build runner
    runner_cfg = {
        "max_iterations": max_iterations,
        "rollouts": rollouts,
        "save_interval": save_interval,
        "log_dir": str(logdir),
        "experiment_name": None,
    }
    runner = IsaaclabRunner(
        env=wrapped_env,
        agent=agent,
        cfg=runner_cfg,
    )

    # Run the workflow
    if from_checkpoint:
        runner.load(str(from_checkpoint))

    if workflow == "train":
        runner.train(return_epochs=100)
    elif workflow == "eval":
        runner.evaluate(steps=max_iterations, return_epochs=100, log=True)

    # Close the environment
    env.close()
