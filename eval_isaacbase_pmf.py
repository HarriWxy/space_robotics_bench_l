"""Validate native H1 pMF checkpoints through the existing training launcher.

Loads the checkpoint's saved agent configuration by default. Reusing a
``--state-bank-in`` pairs identical raw states and noises across checkpoints.
Pass ``--fpo-config`` explicitly for a checkpoint without ``params/agent.yaml``.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from train_isaacbase_fsppo_joint import _has_option, main as _run_joint_entrypoint


def main(argv: Sequence[str] | None = None) -> int:
    """Run complete-episode playback and a paired pMF consistency diagnostic."""
    supplied = list(sys.argv[1:] if argv is None else argv)
    defaults = ["--eval", "--pmf-validation"]
    if not _has_option(supplied, "--fpo-config"):
        defaults.append("--use-checkpoint-config")
    if not _has_option(supplied, "--eval-episodes") and not _has_option(supplied, "--eval_episodes"):
        defaults.extend(("--eval-episodes", "32"))
    if not _has_option(supplied, "--flow-eval-modes") and not _has_option(supplied, "--flow_eval_modes"):
        defaults.extend(("--flow-eval-modes", "random"))
    return _run_joint_entrypoint([*defaults, *supplied])


if __name__ == "__main__":
    raise SystemExit(main())
