"""Plot recorded evidence preceding the Gaussian residual FSPPOJoint failure."""

import argparse
import os
import struct
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/srb_fsppo_nonfinite_mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.compat.proto.event_pb2 import Event


def read_scalars(run: Path) -> dict[str, np.ndarray]:
    """Read complete event records within a fixed file-size snapshot."""
    rows = defaultdict(list)
    for path in sorted(run.glob("*tfevents*")):
        size = path.stat().st_size
        with path.open("rb") as stream:
            while stream.tell() + 12 <= size:
                header = stream.read(12)
                length = struct.unpack("<Q", header[:8])[0]
                if stream.tell() + length + 4 > size:
                    break
                event = Event.FromString(stream.read(length))
                stream.read(4)
                for value in event.summary.value:
                    if value.HasField("simple_value"):
                        rows[value.tag].append((event.step, value.simple_value))
    return {tag: np.asarray(values, dtype=np.float64) for tag, values in rows.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start", type=int, default=1200)
    parser.add_argument("--stop", type=int, default=1374)
    args = parser.parse_args()
    data = read_scalars(args.run)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    panels = (
        (
            (
                ("Metrics/action_std", "Rollout raw action std"),
                ("Metrics/joint/action_noise_std", "Learned conditional sigma"),
            ),
            "Action output dispersion vs. sampling noise",
            "Action units (log)",
            "log",
        ),
        (
            (
                ("Loss/pmf_aux_loss", "Auxiliary loss"),
                ("Loss/value_loss", "Value loss"),
            ),
            "Auxiliary and critic losses",
            "Recorded loss (log)",
            "log",
        ),
        (
            (("Metrics/mean_grad_norm_before_clip", "Before clipping"),),
            "Gradient norm before clipping",
            "L2 norm (log)",
            "log",
        ),
        (
            (("Train/mean_reward", "Episode return"),),
            "Mean return of completed episodes",
            "Episode return (symlog)",
            "symlog",
        ),
    )
    for ax, (curves, title, ylabel, scale) in zip(axes.flat, panels):
        for tag, label in curves:
            values = data[tag]
            selected = (values[:, 0] >= args.start) & (values[:, 0] <= args.stop)
            selected &= np.isfinite(values[:, 1])
            if scale == "log":
                selected &= values[:, 1] > 0
            ax.plot(
                values[selected, 0], values[selected, 1], label=label, linewidth=1.3
            )
        if scale == "symlog":
            ax.set_yscale(scale, linthresh=10)
        else:
            ax.set_yscale(scale)
        ax.axvline(1295, color="0.45", linestyle="--", linewidth=0.8)
        ax.axvline(1304, color="0.45", linestyle=":", linewidth=0.8)
        ax.set(title=title, xlabel="Training iteration (zero-based)", ylabel=ylabel)
        ax.set_xlim(args.start, args.stop)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9)
    fig.suptitle(
        "H1 Gaussian residual FSPPOJoint: recorded evidence before failure\n"
        "Dashed: auxiliary loss rise (1295); dotted: large action outlier (1304)"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    plt.close(fig)
    print(args.out)


if __name__ == "__main__":
    main()
