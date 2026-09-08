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
GRAVITY_REFERENCE = 9.80665
GRAVITY_BIN_EDGES = tuple(
    (MOON_GRAVITY + (MARS_GRAVITY - MOON_GRAVITY) * index / 4.0)
    / GRAVITY_REFERENCE
    for index in range(1, 4)
)


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
        "--conditioning",
        choices=("none", "concat", "film"),
        default="concat",
        help="actor physics contract: no context, PC-Concat, or FiLM",
    )
    parser.add_argument(
        "--stratified-replay",
        action="store_true",
        help="balance ExO replay minibatches across four fixed gravity bins",
    )
    parser.add_argument(
        "--no-gravity-context",
        action="store_true",
        help="Keep the gravity schedule but remove its scalar from the actor input.",
    )
    return parser.parse_args()


def build_srb_argv(args: argparse.Namespace) -> list[str]:
    requested_conditioning = getattr(args, "conditioning", "concat")
    stratified_replay = bool(getattr(args, "stratified_replay", False))
    no_gravity_context = bool(getattr(args, "no_gravity_context", False))
    conditioning = "none" if no_gravity_context else requested_conditioning
    if stratified_replay and args.algo != "exoppo":
        raise ValueError("stratified replay is currently implemented for exoppo")
    if stratified_replay and conditioning != "film":
        raise ValueError("stratified replay requires --conditioning film")
    include_concat = conditioning == "concat"
    include_physics = conditioning == "film"
    interval_s = 64.0 / 25.0 if stratified_replay else 30.0
    schedule_mode = "stratified_cycle" if stratified_replay else "interval_uniform"
    bin_edges = ",".join(f"{edge:.9f}" for edge in GRAVITY_BIN_EDGES)
    argv = [
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
        f"env.include_gravity_magnitude={'true' if include_concat else 'false'}",
        f"env.include_physics_context={'true' if include_physics else 'false'}",
        "env.gravity_magnitude_reference=9.80665",
        f"env.physics_conditioning.gravity_magnitude_range=[{MOON_GRAVITY},{MARS_GRAVITY}]",
        f"env.physics_conditioning.gravity_interval_s=[{interval_s},{interval_s}]",
        f"env.physics_conditioning.schedule_mode={schedule_mode}",
        "env.physics_conditioning.num_buckets=4",
        f"agent.obs.physics_keys={'[physics]' if include_physics else '[]'}",
        f"agent.physics_film={'true' if include_physics else 'false'}",
        "agent.physics_embed_dim=32",
        f"agent.physics_stratified_replay={'true' if stratified_replay else 'false'}",
        "agent.physics_num_buckets=4",
        "agent.physics_context_index=0",
        f"agent.physics_bin_edges=[{bin_edges}]",
        f"agent.max_iterations={args.iterations}",
        "--headless",
    ]
    if stratified_replay:
        argv.extend(("agent.replay_N=4", "agent.warmup_rollouts=4"))
    return argv


if __name__ == "__main__":
    run_srb(build_srb_argv(parse_args()))
