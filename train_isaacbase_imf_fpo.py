"""Train Isaac Lab's native H1 task with experimental iMF-FPO.

This entry point reuses the native Isaac Lab/FPO launcher while selecting the
Improved MeanFlow actor and its matching ``IMFFPO`` optimizer configuration.
Keeping the variant selection in a separate script makes it difficult to
accidentally resume an ordinary FPO checkpoint with the incompatible iMF actor
layout.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

import train_isaacbase_fpo as _fpo


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "hyperparams" / "imf_fpo.yaml"
DEFAULT_EXPERIMENT_NAME = "h1_imf_fpo"
DEFAULT_RUN_NAME = "h1_imf_fpo"


def _has_option(argv: Sequence[str], *options: str) -> bool:
    """Return whether an argparse option is present in ``argv``."""
    return any(
        argument == option or argument.startswith(f"{option}=")
        for argument in argv
        for option in options
    )


def _option_value(argv: Sequence[str], option: str) -> str | None:
    """Read a value supplied as ``--option value`` or ``--option=value``."""
    for index, argument in enumerate(argv):
        if argument.startswith(f"{option}="):
            return argument.split("=", 1)[1]
        if argument == option:
            if index + 1 >= len(argv):
                raise ValueError(f"{option} requires a value")
            return argv[index + 1]
    return None


def _resolve_config_path(argv: Sequence[str]) -> Path:
    """Resolve the config path using the same project-relative convention as the base launcher."""
    value = _option_value(argv, "--fpo-config")
    config_path = DEFAULT_CONFIG if value is None else Path(value).expanduser()
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    return config_path.resolve()


def _validate_imf_config(config_path: Path) -> None:
    """Fail early if a custom config would select the legacy FPO pair."""
    import yaml

    if not config_path.is_file():
        raise FileNotFoundError(f"iMF-FPO config not found: {config_path}")
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    if not isinstance(config, dict):
        raise TypeError(f"iMF-FPO config must contain a mapping: {config_path}")

    policy = config.get("policy", {})
    algorithm = config.get("algorithm", {})
    policy_name = policy.get("class_name") if isinstance(policy, dict) else None
    algorithm_name = (
        algorithm.get("class_name") if isinstance(algorithm, dict) else None
    )
    if policy_name != "IMFActorCritic" or algorithm_name != "IMFFPO":
        raise ValueError(
            "iMF-FPO requires policy.class_name=IMFActorCritic and "
            "algorithm.class_name=IMFFPO; "
            f"got policy={policy_name!r}, algorithm={algorithm_name!r} "
            f"in {config_path}"
        )


def _prepare_argv(argv: Sequence[str] | None) -> list[str]:
    """Add iMF-FPO defaults without overriding explicit user arguments."""
    prepared = list(sys.argv[1:] if argv is None else argv)
    if not _has_option(prepared, "--fpo-config"):
        prepared.extend(["--fpo-config", str(DEFAULT_CONFIG)])
    if not _has_option(prepared, "--experiment-name", "--experiment_name"):
        prepared.extend(["--experiment-name", DEFAULT_EXPERIMENT_NAME])
    if not _has_option(prepared, "--run-name", "--run_name"):
        prepared.extend(["--run-name", DEFAULT_RUN_NAME])
    return prepared


def main(argv: Sequence[str] | None = None) -> int:
    """Run the iMF-FPO launcher in the active Isaac Lab environment."""
    prepared = _prepare_argv(argv)
    _validate_imf_config(_resolve_config_path(prepared))
    return _fpo.main(prepared)


if __name__ == "__main__":
    raise SystemExit(main())
