"""Freeze a native Gaussian PPO teacher and fit pMF with plain MeanFlow loss."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from train_isaacbase_fpo import DEFAULT_FPO_ROOT, _insert_fpo_root, _positive_int


def main() -> int:
    """Run the diagnostic offline teacher experiment without Isaac Sim."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--state-bank", type=Path, required=True)
    parser.add_argument(
        "--student-config",
        type=Path,
        default=Path(__file__).parent / "hyperparams" / "pmf_next_step.yaml",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--updates", type=_positive_int, default=5000)
    parser.add_argument("--batch-size", type=_positive_int, default=256)
    parser.add_argument("--flow-samples", type=_positive_int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--ode-steps", nargs="*", type=_positive_int, default=[])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=_positive_int, default=4)
    parser.add_argument("--fpo-root", type=Path, default=DEFAULT_FPO_ROOT)
    args = parser.parse_args()
    _insert_fpo_root(args.fpo_root)

    import torch
    from isaaclab_fpo.pmf_teacher import (
        distill_teacher,
        load_native_ppo_teacher,
        load_raw_state_bank,
        read_agent_config,
    )

    torch.set_num_threads(args.threads)
    teacher = load_native_ppo_teacher(args.teacher_checkpoint, args.device)
    report = distill_teacher(
        teacher,
        load_raw_state_bank(args.state_bank),
        read_agent_config(args.student_config),
        args.output_dir,
        updates=args.updates,
        batch_size=args.batch_size,
        flow_samples=args.flow_samples,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        ode_steps=tuple(args.ode_steps),
    )
    print(
        json.dumps(
            {
                "checkpoint": report["checkpoint"],
                "last_update": report["history"][-1],
                "fixed_state_distribution_mean_error": report[
                    "fixed_state_distribution"
                ]["mean_squared_mean_error"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
