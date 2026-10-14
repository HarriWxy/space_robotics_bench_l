"""Debug/train joint-latent FSPPO on Isaac Lab's native H1 velocity task.

This thin entry point delegates Isaac Lab lifecycle, checkpoint handling, and
Hydra overrides to ``train_isaacbase_fpo.py``.  Its separate configuration
selects ``PMFActorCritic`` with ``FSPPOJoint``: a one-NFE pMF transport map
whose training action has an explicitly modelled conditional Gaussian noise.

Defaults select the current long-running Flat-H1 experiment. For a bounded
smoke job, pass ``--num-envs 2 --max-iterations 2``. For a full experiment::

    python train_isaacbase_fsppo_joint.py \
        --task Isaac-Velocity-Rough-H1 --num-envs 256 \
        --total-env-steps 100000000 --run-name h1_fsppo_joint

The controlled ``--pmf-ablation pg_only`` / ``--pmf-ablation pg_pmf`` pair
changes only the pMF auxiliary-loss coefficient. Checkpoints can be diagnosed
with ``eval_isaacbase_pmf.py`` using a shared state/noise bank.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

from train_isaacbase_fpo import main as _run_fpo_entrypoint

import os

# Default to GPU 0 but respect an explicit selection so the pg_only/pg_pmf
# ablation pair can run concurrently on two devices without editing this file.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# Default ablation; override with FSPPPO_VARIANT=pg_pmf or --pmf-ablation.
variant = os.environ.get("FSPPPO_VARIANT", "pg_only")
_DEFAULT_CONFIG = Path(__file__).resolve().parent / "hyperparams" / "fsppo_joint.yaml"
_DEFAULT_VALUE_ARGS = (
    "--task",
    "Isaac-Velocity-Flat-H1",
    "--num-envs",
    "256",
    "--max-iterations",
    "100000",
    "--experiment-name",
    "h1_pmf_validation",
    "--pmf-ablation",
    variant,
    "--run-name",
    variant,
    "--log-root",
    "/root/space_robotics_bench_l/logs/pmf_ablation",
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


def _option_value(argv: Sequence[str], name: str) -> str | None:
    """Return the value of a ``--name value`` or ``--name=value`` option."""

    for index, item in enumerate(argv):
        if item == name and index + 1 < len(argv):
            return argv[index + 1]
        if item.startswith(f"{name}="):
            return item.split("=", 1)[1]
    return None


def build_args(argv: Sequence[str] | None = None) -> list[str]:
    """Build FSPPOJoint arguments while allowing each training default to be overridden."""

    supplied = list(sys.argv[1:] if argv is None else argv)
    config_args = [] if _has_option(supplied, "--fpo-config") else [
        "--fpo-config",
        str(_DEFAULT_CONFIG),
    ]

    defaults: list[str] = []
    for name, value in zip(_DEFAULT_VALUE_ARGS[::2], _DEFAULT_VALUE_ARGS[1::2]):
        # Keep the run name aligned with an explicitly selected ablation so
        # log directories carry the variant actually being trained.
        if name == "--run-name":
            value = _option_value(supplied, "--pmf-ablation") or value
        # Evaluation must reconstruct the saved algorithm and auxiliary
        # coefficient, including ordinary PMFFPO teacher-distilled artifacts.
        if name == "--pmf-ablation" and _has_option(supplied, "--eval"):
            continue
        # The shared launcher derives iterations from --total-env-steps; do
        # not allow the default maximum to mask that requested total.
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
