"""Debug/train joint-latent FSPPO on Isaac Lab's native H1 velocity task.

This thin entry point delegates Isaac Lab lifecycle, checkpoint handling, and
Hydra overrides to ``train_isaacbase_fpo.py``.  Its separate configuration
selects ``PMFActorCritic`` with ``FSPPOJoint``: a one-NFE pMF transport map
whose training action has an explicitly modelled conditional Gaussian noise.

Defaults intentionally run a two-environment, two-iteration smoke job.  For a
full experiment, for example::

    python train_isaacbase_fsppo_joint.py \
        --task Isaac-Velocity-Rough-H1 --num-envs 256 \
        --total-env-steps 100000000 --run-name h1_fsppo_joint
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

from train_isaacbase_fpo import main as _run_fpo_entrypoint

_DEFAULT_CONFIG = Path(__file__).resolve().parent / "hyperparams" / "fsppo_joint.yaml"
_DEFAULT_VALUE_ARGS = (
    "--task",
    "Isaac-Velocity-Flat-H1",
    "--num-envs",
    "1024",
    "--max-iterations",
    "100000",
    "--experiment-name",
    "h1_fsppo_joint",
    "--run-name",
    "debug",
)
_DEFAULT_FLAGS = ("--headless",)


def _has_option(argv: Sequence[str], name: str) -> bool:
    """Return whether an option is present in ``--name value`` or ``--name=value`` form."""

    aliases = {
        "--num-envs": ("--num-envs", "--num_envs"),
        "--max-iterations": ("--max-iterations", "--max_iterations"),
        "--total-env-steps": ("--total-env-steps", "--total_env_steps"),
        "--experiment-name": ("--experiment-name", "--experiment_name"),
        "--run-name": ("--run-name", "--run_name"),
    }.get(name, (name,))
    return any(
        item == alias or item.startswith(f"{alias}=")
        for item in argv
        for alias in aliases
    )


def build_args(argv: Sequence[str] | None = None) -> list[str]:
    """Build FSPPOJoint arguments while allowing every smoke default to be overridden."""

    supplied = list(sys.argv[1:] if argv is None else argv)
    config_args = [] if _has_option(supplied, "--fpo-config") else [
        "--fpo-config",
        str(_DEFAULT_CONFIG),
    ]

    defaults: list[str] = []
    for name, value in zip(_DEFAULT_VALUE_ARGS[::2], _DEFAULT_VALUE_ARGS[1::2]):
        # The shared launcher derives iterations from --total-env-steps; do
        # not allow the smoke maximum to mask that requested total.
        if name == "--max-iterations":
            if _has_option(supplied, name) or _has_option(
                supplied, "--total-env-steps"
            ):
                continue
        elif _has_option(supplied, name):
            continue
        defaults.extend((name, value))

    visual_app_requested = any(
        _has_option(supplied, option)
        for option in ("--visualizer", "--viz", "--xr", "--livestream")
    )
    if not visual_app_requested:
        defaults.extend(
            flag for flag in _DEFAULT_FLAGS if not _has_option(supplied, flag)
        )
    return [*config_args, *defaults, *supplied]


def main(argv: Sequence[str] | None = None) -> int:
    """Run FSPPOJoint using the shared native Isaac Lab FPO launcher."""

    return _run_fpo_entrypoint(build_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
