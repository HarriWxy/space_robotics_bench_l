import argparse
import os
import shlex
import sys
from typing import Sequence


def run_srb(argv: Sequence[str]) -> None:
    from srb.__main__ import main

    old_argv = sys.argv[:]
    try:
        sys.argv = [old_argv[0], *argv]
        main()
    finally:
        sys.argv = old_argv


def _resolve_visual_env_id(env_id: str, use_visual_env: bool) -> str:
    if not use_visual_env or env_id.endswith("_visual"):
        return env_id
    return f"{env_id}_visual"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convenience launcher for SRB VLA rollouts with visual observations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-e", "--env", dest="env_id", default="sample_collection")
    parser.add_argument(
        "--visual",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer the visual task variant by appending '_visual' to the env id.",
    )
    parser.add_argument(
        "--prompt",
        default="collect the sample",
        help="Prompt sent to the VLA policy server.",
    )
    parser.add_argument("--host", default=os.environ.get("SRB_VLA_HOST", "0.0.0.0"))
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("SRB_VLA_PORT", "8000")),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default=os.environ.get("SRB_VLA_DEVICE", "cuda:0"),
    )
    parser.add_argument("--cfg", default="DEFAULT")
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--replan-steps", type=int, default=4)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=not bool(os.environ.get("DISPLAY")),
        help="Run Isaac Sim in headless mode.",
    )
    parser.add_argument(
        "--enable-cameras",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force camera rendering on the SRB CLI side.",
    )
    parser.add_argument(
        "--hide-ui",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Hide most Isaac Sim UI elements in GUI mode.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generated 'srb vla' argv without launching.",
    )
    return parser


def _build_srb_argv(args: argparse.Namespace, forwarded_args: Sequence[str]) -> list[str]:
    env_id = _resolve_visual_env_id(args.env_id, args.visual)
    argv = [
        "vla",
        "--env",
        env_id,
        "--prompt",
        args.prompt,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--cfg",
        args.cfg,
        "--max-steps",
        str(args.max_steps),
        "--replan-steps",
        str(args.replan_steps),
        "--log-interval",
        str(args.log_interval),
        "env.sample=primitive",
        "env.robot=ur5+robotiq_hand_e",
    ]
    if args.headless:
        argv.append("--headless")
    if args.enable_cameras:
        argv.append("--enable-cameras")
    if args.hide_ui:
        argv.append("--hide_ui")
    argv.extend(forwarded_args)
    return argv


def main() -> None:
    parser = _build_parser()
    args, forwarded_args = parser.parse_known_args()
    argv = _build_srb_argv(args, forwarded_args)

    if args.dry_run:
        print("Generated argv:")
        print(" ".join(shlex.quote(arg) for arg in argv))
        return

    run_srb(argv)


if __name__ == "__main__":
    main()

