import sys
# from pathlib import Path

from srb.__main__ import main
# from srb.utils.cfg import last_logdir
# from srb.utils.path import SRB_LOGS_DIR

# sys.path.append("D:\\Program Files\\Blender Foundation\\Blender 4.5")  # Add parent directory to sys.path to import impl.py


# def resolve_model_path() -> Path:
#     logdir = last_logdir(
#         env_id="sample_collection",
#         workflow="sbx_td3",
#         root=SRB_LOGS_DIR,
#         modification_time=True,
#     )

#     candidates = (
#         logdir.joinpath("ckpt", "srb-sample_collection.zip"),
#         logdir.joinpath("srb-sample_collection.zip"),
#     )
#     for candidate in candidates:
#         if candidate.exists():
#             return candidate.resolve()

#     raise FileNotFoundError(
#         f"Could not find a checkpoint zip under {logdir}. Checked: "
#         + ", ".join(candidate.as_posix() for candidate in candidates)
#     )


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
        "rand",
        # "eval", 
        # "--algo",
        # "sb3_td3",
        "--env",
        "velocity_tracking",
        # "lunar_obstacle_crossing",
        # "--cfg",
        # "IGNORE",
        "env.domain=moon",
        # "env.sample=moon_rock",
        # "env.sample=primitive",
        "env.robot=unitree_g1",
        # "env.stage=2",
        # "env.episode_length_s=8.0",
        # "env.scenery=ground_plane",
        # "env.debug_vis=true",
        "env.num_envs=1", # 
        # "env.stage=2",
        # "env.stage2_easy=True",
        "env.sim.device=cuda:0",
        # "env.camera_data_types=[rgb]",
        # "env.camera_resolution=[640,640]",
        # "env.robot=ur5+robotiq_hand_e",
        # "--model=./logs/sample_collection/sbx_td3/20260527T210054/ckpt/srb-sample_collection.zip",
        # "--headless",
        # "--continue_training",
        # "--video",
        # "--livestream=2",
        # f"--model={resolve_model_path().as_posix()}",
    ])


# pip install omniverse-kit --extra-index-url https://pypi.nvidia.com