from srb.__main__ import main

import sys

# sys.path.append("D:\\Program Files\\Blender Foundation\\Blender 4.5")  # Add parent directory to sys.path to import impl.py

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
        # "rand",
        "eval", 
        "--algo",
        "sbx_td3",
        "--env",
        "sample_collection",
        # "--cfg",
        # "IGNORE",
        "env.domain=moon",
        # "env.sample=moon_rock",
        "env.sample=primitive",
        # "env.robot=ur5+robotiq_hand_e",
        # "env.stage=2",
        "env.episode_length_s=8.0",
        # "env.scenery=ground_plane",
        # "env.debug_vis=true",
        "env.num_envs=1", # 
        "env.stage=2",
        "env.stage2_easy=True",
        "env.sim.device=cuda",
        "env.robot=ur5+robotiq_hand_e",
        "--model=./logs/sample_collection/sbx_td3/20260527T210054/ckpt/srb-sample_collection.zip",
        # "--headless",
        # "--continue_training",
    ])


# pip install omniverse-kit --extra-index-url https://pypi.nvidia.com