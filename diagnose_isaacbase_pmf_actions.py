"""Compare full PPO/pMF action covariances on a shared raw-state bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from train_isaacbase_fpo import DEFAULT_FPO_ROOT, _insert_fpo_root, _positive_int


def main() -> int:
    """Run Torch-only fixed-state diagnostics without starting Isaac Sim."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--pmf-checkpoint", type=Path, required=True)
    parser.add_argument("--state-bank", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--pmf-weights", choices=("online", "evaluation"), default="online"
    )
    parser.add_argument("--samples-per-state", type=_positive_int, default=256)
    parser.add_argument("--batch-size", type=_positive_int, default=64)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=_positive_int, default=4)
    parser.add_argument("--fpo-root", type=Path, default=DEFAULT_FPO_ROOT)
    args = parser.parse_args()
    _insert_fpo_root(args.fpo_root)

    import torch
    from isaaclab_fpo.pmf_teacher import (
        covariance_report,
        load_native_ppo_teacher,
        load_pmf_checkpoint,
        load_raw_state_bank,
    )

    torch.set_num_threads(args.threads)
    bank = load_raw_state_bank(args.state_bank)
    teacher = load_native_ppo_teacher(args.teacher_checkpoint, args.device)
    pmf = load_pmf_checkpoint(
        args.pmf_checkpoint, args.device, weights=args.pmf_weights
    )
    report, samples = covariance_report(
        teacher,
        pmf,
        bank,
        samples_per_state=args.samples_per_state,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report["state_bank_path"] = str(args.state_bank.resolve())
    with (args.output_dir / "action_covariance_report.json").open("w") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    torch.save(samples, args.output_dir / "action_samples.pt")
    print(
        json.dumps(
            {
                "report": str(args.output_dir / "action_covariance_report.json"),
                "pmf_terminal_sigma": report["terminal_sigma"],
                "covariance_trace": {
                    name: value["mean_covariance_trace"]
                    for name, value in report["distributions"].items()
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
