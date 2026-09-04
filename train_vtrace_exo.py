import os
import sys

from srb.__main__ import main

# os.environ.pop("OGN_REG_DEBUG", None)
# os.environ.pop("OGN_DEBUG", None)

# os.environ["EXP_PATH"] = "/root/isaac-sim/apps"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def run_srb(argv):
    old_argv = sys.argv[:]
    try:
        sys.argv = [old_argv[0], *argv]
        main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    # print(os.environ["EXP_PATH"])
    run_srb([
        "agent",
        "train",
        # "eval",
        # "rand",
        "--algo",
        "exoppo", # policyflow
        "--env",
        "locomotion_velocity_tracking_c",
        # "--cfg",
        # "IGNORE",
        "env.domain=moon",
        # "env.sample=primitive",
        "env.robot=unitree_h1",
        # "env.stage=2",
        # "env.stage2_easy=True",
        # "env.episode_length_s=8.0",
        # "env.num_envs=2",
        "env.num_envs=256",
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