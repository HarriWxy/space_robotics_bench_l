"""Read native Isaac Lab FSPPO events without starting the simulator.

Only fresh runs are supported: the runner logs zero-based iteration indices,
so transitions equal (iteration + 1) * num_envs * num_steps_per_env.
"""

import argparse
import hashlib
import json
import os
import struct
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/srb_fsppo_review_mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from tensorboard.compat.proto.event_pb2 import Event


def read_run(run, out):
    """Read complete records up to a fixed event-file size snapshot."""
    agent_path, env_path = run / "params/agent.yaml", run / "params/env.yaml"
    agent = yaml.load(agent_path.read_text(), Loader=yaml.BaseLoader)
    env = yaml.load(env_path.read_text(), Loader=yaml.BaseLoader)
    if str(agent.get("resume", "false")).lower() == "true":
        raise ValueError(f"Resumed run requires a verified step offset: {run}")
    scale = int(agent["num_steps_per_env"]) * int(env["scene"]["num_envs"])
    dt = float(env["sim"]["dt"]) * int(env["decimation"])
    rows, snapshots = defaultdict(list), []
    for path in sorted(run.glob("*tfevents*")):
        stat = path.stat()
        item = {"path": str(path), "bytes": stat.st_size, "mtime": stat.st_mtime}
        with path.open("rb") as stream:
            while stream.tell() + 12 <= stat.st_size:
                header = stream.read(12)
                size = struct.unpack("<Q", header[:8])[0]
                if stream.tell() + size + 4 > stat.st_size:
                    break
                event = Event.FromString(stream.read(size))
                stream.read(4)
                for value in event.summary.value:
                    if value.HasField("simple_value"):
                        rows[value.tag].append((event.step, value.simple_value, event.wall_time))
            item["read_bytes"] = stream.tell()
        snapshots.append(item)
    series = {tag: np.asarray(values, dtype=np.float64) for tag, values in rows.items()}
    target = out / run.parent.name / run.name
    target.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target / "scalars.npz", **series)
    summary = {
        "run": str(run), "events": snapshots, "transitions_per_iteration": scale,
        "control_dt": dt, "episode_seconds": float(env["episode_length_s"]),
        "agent": agent, "config_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (agent_path, env_path)
        }, "scalar_tag_count": len(series), "last": {}, "windows": {},
    }
    iteration_tags = [tag for tag in series if not tag.endswith("/time")]
    last_it = max((int(series[tag][-1, 0]) for tag in iteration_tags), default=-1)
    summary["last_iteration"] = last_it
    summary["transitions"] = (last_it + 1) * scale
    for tag in iteration_tags:
        data = series[tag]
        summary["last"][tag] = float(data[-1, 1])
    upper = summary["transitions"] / 1e6
    windows = [(lo, lo + 10, f"{lo}-{lo + 10}M") for lo in range(0, int(upper), 10)]
    windows += [(max(0, upper - 10), upper, "last10M"), (max(0, upper - 1), upper, "last1M")]
    for lo, hi, label in windows:
        result = {}
        for tag in iteration_tags:
            data = series[tag]
            x = (data[:, 0] + 1) * scale / 1e6
            values = data[(x > lo) & (x <= hi), 1]
            finite = values[np.isfinite(values)]
            if not len(finite):
                continue
            result[tag] = {
                "n": len(values), "nonfinite": int(len(values) - len(finite)),
                "mean": float(finite.mean()), "median": float(np.median(finite)),
                "min": float(finite.min()), "max": float(finite.max()),
                "zero_fraction": float((finite == 0).mean()),
            }
        if result:
            summary["windows"][label] = result
    (target / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    return series, summary


def binned(data, scale, width=1_000_000):
    x = (data[:, 0] + 1) * scale
    bins = np.maximum(0, np.ceil(x / width).astype(int) - 1)
    count = np.bincount(bins)
    valid = count > 0
    return (np.bincount(bins, weights=x)[valid] / count[valid] / 1e6,
            np.bincount(bins, weights=data[:, 1])[valid] / count[valid])


def plot_learning(runs, out):
    panels = [
        ("Train/mean_reward", "Episode return (last 100 completed)", 1),
        ("Train/mean_episode_length", "Episode duration (s)", None),
        ("Metrics/base_velocity/error_vel_xy", "Mean XY velocity error (m/s)", 1),
        ("Metrics/base_velocity/error_vel_yaw", "Mean yaw velocity error (rad/s)", 1),
        ("Episode_Termination/base_contact", "Latest-termination base-contact indicator", 1),
        ("Metrics/action_std", "Pooled rollout action std (not noise sigma)", 1),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    for ax, (tag, title, multiplier) in zip(axes.flat, panels):
        for data, meta in runs:
            if tag not in data:
                continue
            x, y = binned(data[tag], meta["transitions_per_iteration"])
            label = meta["agent"]["algorithm"]["class_name"]
            if label != "FSPPOJoint":
                label += " (different settings; reference only)"
            ax.plot(x, y * (meta["control_dt"] if multiplier is None else multiplier), label=label)
        ax.set(xlabel="Environment transitions (millions)", ylabel=title)
        ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Native Flat H1: recorded training behavior (1M-transition bin means)")
    fig.savefig(out / "learning.png", dpi=150)
    plt.close(fig)


def plot_optimizer(data, meta, out):
    panels = [
        (["joint/accepted_updates", "joint/attempted_batches"], "Optimizer steps per iteration"),
        (["joint/rejected_candidates", "joint/early_stop"], "Rejected candidates / early-stop indicator"),
        (["joint/probe_kl_mean", "joint/audit_kl_mean", "joint/kl_target"], "Post-update joint map KL"),
        (["clip_fraction", "joint/kl_coefficient_used"], "Clip fraction / KL penalty coefficient"),
        (["joint/last_step_learning_rate"], "LR of last accepted optimizer step"),
        (["explained_variance"], "Critic explained variance"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    for ax, (tags, title) in zip(axes.flat, panels):
        for tag in tags:
            key = f"Metrics/{tag}"
            if key in data:
                x, y = binned(data[key], meta["transitions_per_iteration"])
                ax.plot(x, y, label=tag.removeprefix("joint/"))
        ax.set(xlabel="Environment transitions (millions)", ylabel=title)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle(f"FSPPOJoint {Path(meta['run']).name}: 1M-transition bin means")
    fig.savefig(out / "optimization.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    runs = [read_run(run, args.out) for run in args.run]
    plot_learning(runs, args.out)
    for data, meta in runs:
        if data and meta["agent"]["algorithm"]["class_name"] == "FSPPOJoint":
            plot_optimizer(data, meta, args.out)
            break
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "runs": [meta["run"] for _, meta in runs]}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for _, meta in runs:
        print(meta["run"], "tags", meta["scalar_tag_count"], "iteration", meta["last_iteration"],
              "transitions", meta["transitions"])
        for window in ("0-10M", "30-40M", "60-70M", "100-110M", "150-160M", "last10M"):
            stats = meta["windows"].get(window, {})
            print(window, {k: round(v["mean"], 7) for k, v in stats.items() if k.startswith(("Train/", "Metrics/", "Episode_Termination/", "Perf/learning_time"))})
    print(args.out)


if __name__ == "__main__":
    main()
