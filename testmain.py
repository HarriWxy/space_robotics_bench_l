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
        # "eval", #
        "train",
        "--algo",
        "sbx_ppo",
        "--env",
        "sample_collection",
        "env.domain=moon",
        "env.sample=moon_rock",
        # "env.scenery=ground_plane",
        # "env.debug_vis=true",
        "env.num_envs=128", # 
        "env.sim.device=cuda",
        # "--model=./logs/landing/sbx_ppo/20260414T103257/ckpt/srb-landing.zip",
        "env.robot=ur5+robotiq_hand_e",
        "--headless",
        "--continue_training",
    ])


# pip install omniverse-kit --extra-index-url https://pypi.nvidia.com