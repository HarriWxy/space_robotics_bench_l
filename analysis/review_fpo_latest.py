"""Plot and summarize the latest FPO scalar log without launching Isaac Sim."""
import json
import argparse
import os
import struct
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/srb_fpo_review_mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.compat.proto.event_pb2 import Event

parser = argparse.ArgumentParser()
parser.add_argument("--algo", choices=("fpo", "exofpo", "exoppo", "sb3_ppo", "sb3_sac"), default="fpo")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
runs = root / "logs/locomotion_velocity_tracking_c" / args.algo
run = max(p for p in runs.iterdir() if list(p.rglob("*tfevents*")))
out = root / "logs/_analysis" / f"{args.algo}_{run.name}"
out.mkdir(parents=True, exist_ok=True)
series = defaultdict(list)
for path in sorted(run.rglob("*tfevents*")):
    limit = path.stat().st_size
    with path.open("rb") as stream:
        while stream.tell() < limit:
            header = stream.read(12)
            if len(header) != 12:
                break
            size = struct.unpack("<Q", header[:8])[0]
            if stream.tell() + size + 4 > limit:
                break
            event = Event.FromString(stream.read(size))
            stream.read(4)
            for value in event.summary.value:
                if value.HasField("simple_value"):
                    series[value.tag].append((event.step, value.simple_value))
series = {tag: np.asarray(rows) for tag, rows in series.items()}
np.savez_compressed(out / "scalars.npz", **series)
summary = {"run": str(run), "windows": {}}
for lo in range(0, 100, 2):
    stats = {}
    for tag, data in series.items():
        selected = data[(data[:, 0] > lo * 1e6) & (data[:, 0] <= (lo + 2) * 1e6), 1]
        if len(selected):
            stats[tag] = float(selected.mean())
    summary["windows"][f"{lo}-{lo + 2}M"] = stats
(out / "summary.json").write_text(json.dumps(summary, indent=2))
panels = [
    ("rollout/ep_rew_mean", "Episode return", 1),
    ("rollout/ep_len_mean", "Episode duration (s)", .04),
    ("rollout/metrics/undesired_contact", "Undesired contact (%)", 100),
    ("train/approx_kl", "CFM surrogate KL" if args.algo == "fpo" else "Approx KL (behavior/current)", 1),
    ("train/clip_fraction", "Clip fraction", 1),
    ("rollout/metrics/command_lin_error", "Planar velocity error (m/s)", 1),
]
fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
for ax, (tag, label, scale) in zip(axes.flat, panels):
    if tag not in series:
        ax.set_visible(False)
        continue
    data = series[tag]
    bins = np.maximum(0, np.ceil(data[:, 0] / 500000).astype(int) - 1)
    count = np.bincount(bins)
    valid = count > 0
    x = np.bincount(bins, weights=data[:, 0])[valid] / count[valid]
    y = np.bincount(bins, weights=data[:, 1])[valid] / count[valid]
    ax.plot(x / 1e6, y * scale, color="#984ea3")
    ax.set(xlabel="Environment transitions (M)", ylabel=label)
    ax.grid(alpha=.2)
fig.suptitle(f"{args.algo.upper()} {run.name}: 0.5M-transition bins")
fig.savefig(out / "curves.png", dpi=160)
plt.close(fig)
for key, stats in summary["windows"].items():
    if stats:
        print(key, {tag: round(stats[tag], 6) for tag, _, _ in panels if tag in stats})
for tag in ("train/approx_kl", "train/clip_fraction"):
    if tag not in series:
        continue
    data = series[tag]
    tail = data[data[:, 0] > 80e6, 1]
    if len(tail):
        print(tag, "after80M_zero_fraction", float((tail == 0).mean()))
print(out)
