"""Freeze and compare native H1 Rough codebook, continuous-joint, and PPO logs."""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/srb_fsppo_codebook_review_mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from review_fsppo_curves import binned, read_run


RUNS = {
    "codebook": (
        "/root/isaaclab/logs/isaaclab_fpo/h1_fsppo_codebook/2026-10-05_22-02-31_marginal",
        "Codebook marginal, LR 3e-4",
        "#1565c0",
    ),
    "joint_3e4": (
        "/root/isaaclab/logs/isaaclab_fpo/h1_fsppo_joint/2026-10-04_23-04-46_debug",
        "Continuous joint, LR 3e-4",
        "#ef6c00",
    ),
    "joint_1e4": (
        "/root/isaaclab/logs/isaaclab_fpo/h1_fsppo_joint/2026-10-04_20-16-52_debug",
        "Continuous joint, LR 1e-4",
        "#78909c",
    ),
    "ppo": (
        "/root/isaaclab/logs/rsl_rl/h1_rough/2026-09-16_17-22-34_srb_h1_ppo_baseline",
        "PPO, adaptive LR (reference)",
        "#2e7d32",
    ),
}


def window_stats(data, meta, tag, lo, hi):
    if tag not in data:
        return None
    array = data[tag]
    x = (array[:, 0] + 1) * meta["transitions_per_iteration"] / 1e6
    values = array[(x > lo) & (x <= hi), 1]
    finite = values[np.isfinite(values)]
    if not len(finite):
        return None
    return {
        "n": len(values), "nonfinite": int(len(values) - len(finite)),
        "mean": float(finite.mean()), "median": float(np.median(finite)),
        "p95": float(np.quantile(finite, 0.95)), "max": float(finite.max()),
    }


def milestones(data, meta):
    """Use a fixed transition window across different rollout batch sizes."""
    array = data["Train/mean_reward"]
    x = (array[:, 0] + 1) * meta["transitions_per_iteration"]
    left = np.searchsorted(x, x - 3_276_800, side="right")
    sums = np.concatenate(([0.0], np.cumsum(array[:, 1])))
    means = (sums[np.arange(len(x)) + 1] - sums[left]) / (np.arange(len(x)) + 1 - left)
    result = {}
    for threshold in (0, 5, 10, 15, 20):
        found = np.flatnonzero((x >= 3_276_800) & (means >= threshold))
        if len(found):
            i = int(found[0])
            result[str(threshold)] = {
                "transitions_M": float(x[i] / 1e6),
                "logged_elapsed_minutes": float((array[i, 2] - array[0, 2]) / 60),
            }
        else:
            result[str(threshold)] = None
    return result


def learning_plot(runs, out):
    upper = runs["codebook"][1]["transitions"] / 1e6
    panels = (
        ("Train/mean_reward", "Episode return", 1),
        ("Train/mean_episode_length", "Episode duration (s)", "dt"),
        ("Metrics/base_velocity/error_vel_xy", "XY velocity error (m/s)", 1),
        ("Metrics/base_velocity/error_vel_yaw", "Yaw velocity error (rad/s)", 1),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for ax, (tag, ylabel, multiplier) in zip(axes.flat, panels):
        for key, (data, meta) in runs.items():
            if tag not in data:
                continue
            array = data[tag]
            array = array[(array[:, 0] + 1) * meta["transitions_per_iteration"] <= upper * 1e6]
            x, y = binned(array, meta["transitions_per_iteration"])
            factor = meta["control_dt"] if multiplier == "dt" else multiplier
            ax.plot(x, y * factor, label=RUNS[key][1], color=RUNS[key][2])
        ax.set(xlabel="Environment transitions (millions)", ylabel=ylabel, xlim=(0, upper))
        ax.grid(alpha=0.2)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("H1 Rough: shared step range; 1M-transition bin means\n"
                 "PPO has different training settings; continuous joint is not a same-codebook ratio ablation")
    for suffix in ("png", "svg"):
        fig.savefig(out / f"learning.{suffix}", dpi=160)
    plt.close(fig)


def diagnostics_plot(runs, out):
    data, meta = runs["codebook"]
    scale = meta["transitions_per_iteration"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), layout="constrained")
    series = (
        (("posterior_entropy", 1 / np.log(8), "Posterior entropy / log(8)"),
         ("effective_components", 1 / 8, "Effective components / 8")),
        (("joint_clip_fraction", 1, "Joint clip fraction"),
         ("marginal_clip_fraction", 1, "Marginal clip fraction")),
        (("clip_jensen_gap", 1, "Clip Jensen gap"),),
        (("posterior_kl", 1, "Posterior KL"),
         ("marginal_kl_mc", 1, "Marginal raw KL estimate")),
    )
    for ax, tags in zip(axes.flat, series):
        for tag, factor, label in tags:
            x, y = binned(data[f"Metrics/codebook/{tag}"], scale)
            ax.plot(x, y * factor, label=label)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        ax.set(xlabel="Environment transitions (millions)")
    axes.flat[0].set_ylim(0, 1.05)
    axes.flat[3].set_yscale("symlog", linthresh=0.001)
    end = meta["transitions"] / 1e6
    ax = axes.flat[4]
    for tag, label in (("marginal_kl_mc", "Marginal raw KL, minibatch average"),
                       ("joint_kl_analytic_probe", "Joint analytic probe, after all updates")):
        array = data[f"Metrics/codebook/{tag}"]
        x = (array[:, 0] + 1) * scale / 1e6
        keep = x > end - 3
        ax.plot(x[keep], array[keep, 1], label=label)
    ax.set(xlabel="Environment transitions (millions)", ylabel="KL: last 3M, raw records", yscale="symlog")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    ax = axes.flat[5]
    for key in ("codebook", "joint_3e4", "joint_1e4"):
        d, m = runs[key]
        array = d["Metrics/action_std"]
        x = (array[:, 0] + 1) * m["transitions_per_iteration"] / 1e6
        keep = x <= max(end, runs["joint_3e4"][1]["transitions"] / 1e6)
        ax.plot(x[keep], array[keep, 1], label=RUNS[key][1], color=RUNS[key][2], alpha=0.8)
    ax.set(xlabel="Environment transitions (millions)", ylabel="Raw rollout action std (not sigma)", yscale="log")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.suptitle("Codebook marginal: overlap, clipping benefit, and late policy-change warning\n"
                 "First four panels: 1M bin means; probe KL and minibatch KL use different measurement points")
    for suffix in ("png", "svg"):
        fig.savefig(out / f"diagnostics.{suffix}", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("logs/_analysis/fsppo_codebook_review_20261005"))
    parser.add_argument("--refresh", action="store_true", help="Replace frozen scalar snapshots with current logs")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    runs = {}
    for key, (path, _, _) in RUNS.items():
        run = Path(path)
        target = args.out / run.parent.name / run.name
        if args.refresh or not (target / "summary.json").exists():
            data, meta = read_run(run, args.out)
        else:
            with np.load(target / "scalars.npz") as archive:
                data = {tag: archive[tag] for tag in archive.files}
            meta = json.loads((target / "summary.json").read_text())
        runs[key] = data, meta
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "runs": {}}
    tags = ["Train/mean_reward", "Train/mean_episode_length", "Metrics/base_velocity/error_vel_xy",
            "Metrics/base_velocity/error_vel_yaw", "Perf/total_fps", "Perf/learning_time",
            "Metrics/action_std", "Metrics/approx_kl", "Metrics/mean_grad_norm_before_clip",
            "Loss/value_loss", "Metrics/explained_variance", "Metrics/joint/action_noise_std",
            "Curriculum/terrain_levels"]
    tags += [f"Metrics/codebook/{tag}" for tag in (
        "posterior_entropy", "posterior_max_probability", "effective_components", "posterior_kl",
        "clip_jensen_gap", "joint_clip_fraction", "marginal_clip_fraction", "marginal_kl_mc",
        "joint_kl_k3", "marginal_kl_k3", "joint_ratio_mean", "joint_kl_analytic_probe")]
    for key, (data, meta) in runs.items():
        upper = meta["transitions"] / 1e6
        windows = {name: {tag: s for tag in tags if (s := window_stats(data, meta, tag, lo, hi)) is not None}
                   for name, lo, hi in (("30-40M", 30, 40), ("90-100M", 90, 100),
                                        ("110-120M", 110, 120), ("last3.2768M", max(0, upper - 3.2768), upper))}
        report["runs"][key] = {"path": meta["run"], "transitions_M": upper,
                                "event_snapshots": meta["events"], "windows": windows,
                                "milestones": milestones(data, meta), "last": meta["last"]}
    (args.out / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    learning_plot(runs, args.out)
    diagnostics_plot(runs, args.out)
    for key, entry in report["runs"].items():
        print(key, "Msteps", entry["transitions_M"], "milestones", entry["milestones"])
    print(args.out)


if __name__ == "__main__":
    main()
