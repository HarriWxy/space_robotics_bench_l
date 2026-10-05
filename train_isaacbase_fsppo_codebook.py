"""Compare joint and exact marginal PPO on the same fixed-codebook policy.

Examples::

    python train_isaacbase_fsppo_codebook.py --codebook-ratio-mode joint --seed 0
    python train_isaacbase_fsppo_codebook.py --codebook-ratio-mode marginal --seed 0

The environment, network, codebook and sigma schedule are identical; only
the PPO ratio changes. Use --num-envs 2 --max-iterations 2 for a smoke run.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

from train_isaacbase_fpo import main as _run_fpo_entrypoint
from train_isaacbase_fsppo_joint import _has_option
from train_isaacbase_fsppo_joint import build_args as _joint_args


def build_args(argv: Sequence[str] | None = None) -> list[str]:
    """Use a separate experiment/config while sharing native H1 launch options."""
    supplied = list(sys.argv[1:] if argv is None else argv)
    ratio_mode = "marginal"
    for index, item in enumerate(supplied):
        if item.startswith("--codebook-ratio-mode="):
            ratio_mode = item.split("=", 1)[1]
        elif item == "--codebook-ratio-mode" and index + 1 < len(supplied):
            ratio_mode = supplied[index + 1]
    defaults = []
    values = {
        "--fpo-config": str(Path(__file__).resolve().parent / "hyperparams/fsppo_codebook.yaml"),
        "--experiment-name": "h1_fsppo_codebook",
        "--run-name": ratio_mode,
    }
    for name, value in values.items():
        if not _has_option(supplied, name):
            defaults.extend((name, value))
    return _joint_args([*defaults, *supplied])


def main(argv: Sequence[str] | None = None) -> int:
    """Run either likelihood objective with the identical discrete sampler."""
    return _run_fpo_entrypoint(build_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
