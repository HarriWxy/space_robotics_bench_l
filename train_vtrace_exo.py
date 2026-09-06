import os
import sys

from srb.__main__ import main

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def run_srb(argv):
    old_argv = sys.argv[:]
    try:
        sys.argv = [old_argv[0], *argv]
        main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    run_srb([
        "agent",
        "train",
        # "eval",
        # "rand",
        "--algo",
        "exoppo",  # policyflow
        "--env",
        "locomotion_velocity_tracking_c",
        # "--cfg",
        # "IGNORE",
        "env.domain=moon",
        "env.curriculum.enabled=false",
        "env.curriculum.fixed_stage=2",
        "env.curriculum.command_mode=forward",
        "env.curriculum.forward_command=[0.35,0.0,0.0]",
        "env.curriculum.joint_position_ranges=[[0.9,1.1],[0.9,1.1],[0.9,1.1]]",
        "env.episode_length_s=30.0",
        "env.terminations.tracking_linear_error_threshold=0.1",
        "env.terminations.tracking_angular_error_threshold=0.1",
        "env.terminations.tracking_min_body_up_z=0.9",
        "env.terminations.success_min_duration_s=30.0",
        "env.terminations.success_settle_time_s=1.0",
        "env.terminations.success_tracking_fraction=0.9",
        # "env.sample=primitive",
        "env.robot=unitree_h1",
        # "env.stage=2",
        # "env.stage2_easy=True",
        # "env.episode_length_s=8.0",
        # "env.num_envs=2",
        "env.num_envs=256",
        # 6104 * 64 * 256 ~= 100M environment transitions.
        "agent.max_iterations=6104",
        "env.malloc_scale=0.25",
        "env.sim.device=cuda",
        "--headless",
        # "env.curriculum.enabled=false",
        # "env.curriculum.fixed_stage=2",
        # "--model=./logs/sample_collection/sbx_td3/20260527T092754/ckpt/srb-sample_collection.zip",
        # "--continue_training",
        # "agent.ent_coef=0.003",
        # "agent.clip_range=0.15",
        # "agent.learning_rate=lin_0.00007",
    ])
