import sys

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from srb.__main__ import main

def _configure_torch_matmul_precision() -> None:
    """Enable TF32-backed float32 matmuls before building the policy."""
    import torch

    torch.set_float32_matmul_precision("high")

def run_srb(argv):
    old_argv = sys.argv[:]
    try:
        sys.argv = [old_argv[0], *argv]
        main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    _configure_torch_matmul_precision()
    run_srb([
        "agent",
        "train",
        # "eval",
        # "--untrained",
        # "rand",
        "--algo",
        "rsl_rl_ppo",  # policyflow  fpo 
        "--env",
        "locomotion_velocity_tracking_c",
        # "--cfg",
        # "IGNORE",
        # Isaac Lab H1 Flat uses Earth gravity and a global ground plane.
        "env.domain=moon", # keep moon environ
        # "env.curriculum.enabled=false",
        # "env.curriculum.fixed_stage=0",
        # "env.curriculum.command_mode=isaac_flat",
        # "env.curriculum.joint_position_ranges=[[1.0,1.0],[1.0,1.0],[1.0,1.0]]",
        # "env.episode_length_s=20.0",
        # "env.terminations.tracking_linear_error_threshold=0.5",
        # "env.terminations.tracking_angular_error_threshold=0.8",
        # "env.terminations.tracking_min_body_up_z=-1.0",
        # "env.terminations.success_min_duration_s=20.0",
        # "env.terminations.success_settle_time_s=0.0",
        # "env.terminations.success_tracking_fraction=0.9",
        "env.sim.physics.enable_ccd=false",
        # "env.sample=primitive",
        "env.robot=unitree_h1",
        # "env.stage=2",
        # "env.stage2_easy=True",
        # "env.episode_length_s=8.0",
        "env.malloc_scale=0.25",
        "env.num_envs=1024",
        # "env.num_envs=1",
        "env.sim.device=cuda",
        "--headless",
        # "--model=./logs/sample_collection/sbx_td3/20260527T092754/ckpt/srb-sample_collection.zip",
        # "--continue_training",
        # "env.curriculum.enabled=false",
        # "env.curriculum.fixed_stage=2",
        # "agent.ent_coef=0.003",
        # "agent.clip_range=0.15",
        # "agent.learning_rate=lin_0.00007",
    ])
