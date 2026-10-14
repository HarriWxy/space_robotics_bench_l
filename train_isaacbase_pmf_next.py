"""Launch the independent A/B/C/D experiments from tasks.md Next Step."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from train_isaacbase_fpo import main as run_fpo
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def _has_option(argv: Sequence[str], *names: str) -> bool:
    return any(
        item == name or item.startswith(f"{name}=") for item in argv for name in names
    )


def build_args(argv: Sequence[str] | None = None) -> list[str]:
    """Select one profile while preserving explicit budget and launcher options."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Other options are forwarded to train_isaacbase_fpo.py, including --device, --seed and --dry-run.",
    )
    parser.add_argument(
        "--variant", choices=("joint_pg", "pmd", "pmd_tr", "joint_pmf"), default="pmd_tr"  # required=True
    )
    selected, supplied = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    if _has_option(supplied, "--pmf-next-step", "--pmf-ablation"):
        parser.error("Use --variant to select this experiment")
    if _has_option(supplied, "--eval", "--resume", "--use-checkpoint-config"):
        # The saved resolved profile is authoritative on resume/evaluation.
        profile = []
        if not _has_option(supplied, "--use-checkpoint-config"):
            supplied.append("--use-checkpoint-config")
    else:
        profile = ["--pmf-next-step", selected.variant]
    defaults = {
        "--fpo-config": str(
            Path(__file__).resolve().parent / "hyperparams" / "pmf_next_step.yaml"
        ),
        "--task": "Isaac-Velocity-Flat-H1",
        "--num-envs": "1024",
        "--experiment-name": "h1_pmf_next_step",
        "--run-name": selected.variant,
        "--log-root": str(
            Path(__file__).resolve().parent
            / "logs"
            / "pmf_next_step"
            / selected.variant
        ),
    }
    result = list(profile)
    for name, value in defaults.items():
        alias = "--" + name[2:].replace("-", "_")
        if not _has_option(supplied, name, alias):
            result.extend((name, value))
    if not _has_option(
        supplied,
        "--max-iterations",
        "--max_iterations",
        "--total-env-steps",
        "--total_env_steps",
    ):
        result.extend(("--total-env-steps", "15000000"))
    if not _has_option(
        supplied, "--visualizer", "--viz", "--xr", "--livestream", "--headless"
    ):
        result.append("--headless")
    return [*result, *supplied]


def main(argv: Sequence[str] | None = None) -> int:
    return run_fpo(build_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
