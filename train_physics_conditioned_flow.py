"""Launch the reproducible Physics-Conditioned Flow Policy paper condition.

Use ``--algo exoppo`` for the proposed method and ``--algo flowppo`` for the
matched one-step Flow-PPO baseline.  The gravity is deliberately global: PhysX
shares one scene across vectorized environments, so this launcher uses
30-second gravity blocks rather than claiming per-environment randomization.
"""

from __future__ import annotations

import argparse
import sys

MOON_GRAVITY = 1.62496
MARS_GRAVITY = 3.72076


def run_srb(argv: list[str]) -> None:
    # Delay Isaac Sim/SRB imports so ``--help`` works in a plain Python shell.
    from srb.__main__ import main as srb_main

    old_argv = sys.argv[:]
    try:
        sys.argv = [old_argv[0], *argv]
        srb_main()
    finally:
        sys.argv = old_argv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algo", choices=("exoppo", "flowppo"), default="exoppo")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=6104)
    parser.add_argument(
        "--no-gravity-context",
        action="store_true",
        help="Keep the gravity schedule but remove its scalar from the actor input.",
    )
    return parser.parse_args()


def build_srb_argv(args: argparse.Namespace) -> list[str]:
    return [
        "agent",
        "train",
        "--algo",
        args.algo,
        "--env",
        "locomotion_velocity_tracking_c",
        f"env.seed={args.seed}",
        f"agent.seed={args.seed}",
        # The scene assets start in the Moon domain; the explicit schedule
        # below spans the Moon--Mars training range.
        "env.domain=moon",
        "env.robot=unitree_h1",
        f"env.num_envs={args.num_envs}",
        "env.sim.device=cuda",
        "env.command_vis=false",
        "env.episode_length_s=30.0",
        "env.curriculum.enabled=false",
        "env.curriculum.fixed_stage=2",
        "env.curriculum.command_mode=omnidirectional",
        f"env.include_gravity_magnitude={'false' if args.no_gravity_context else 'true'}",
        "env.gravity_magnitude_reference=9.80665",
        f"env.physics_conditioning.gravity_magnitude_range=[{MOON_GRAVITY},{MARS_GRAVITY}]",
        "env.physics_conditioning.gravity_interval_s=[30.0,30.0]",
        f"agent.max_iterations={args.iterations}",
        "--headless",
    ]


if __name__ == "__main__":
    run_srb(build_srb_argv(parse_args()))
