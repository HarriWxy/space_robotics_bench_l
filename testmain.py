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
        # "--cfg",
        # "IGNORE",
        "env.domain=moon",
        "env.sample=moon_rock",
        "env.robot=ur5+robotiq_hand_e",
        "env.stage=1",
        "env.episode_length_s=6.5",
        "env.num_envs=256",
        "env.sim.device=cuda",
        # "--headless",
        "--continue_training",
        # "agent.ent_coef=0.01",
    ])

