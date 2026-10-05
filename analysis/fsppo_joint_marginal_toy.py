"""Numerically illustrate joint versus marginal PPO; this is not robot-run evidence."""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    out = Path(__file__).resolve().parents[1] / "logs/_analysis/fsppo_joint_theory"
    out.mkdir(parents=True, exist_ok=True)
    action = np.linspace(-6.0, 6.0, 40001)
    sigma, clip = 0.5, 0.2
    old_means = np.array([-1.0, 1.0])
    advantage = np.tanh(action)

    def log_component(means):
        return (
            -0.5 * ((action[None, :] - means[:, None]) / sigma) ** 2
            - np.log(sigma * np.sqrt(2.0 * np.pi))
        )

    def integrate(values):
        return np.trapz(values, action, axis=-1)

    def clipped(ratio):
        return np.minimum(ratio * advantage, np.clip(ratio, 1 - clip, 1 + clip) * advantage)

    old_log = log_component(old_means)
    old_component = np.exp(old_log)
    old_log_marginal = np.logaddexp.reduce(old_log, axis=0) - np.log(2.0)
    old_marginal = np.exp(old_log_marginal)
    alphas = np.linspace(0.0, 1.0, 101)
    joint_kl, marginal_kl, joint_clip, marginal_clip = [], [], [], []
    for alpha in alphas:
        new_log = log_component((1.0 - 2.0 * alpha) * old_means)
        new_log_marginal = np.logaddexp.reduce(new_log, axis=0) - np.log(2.0)
        joint_ratio = np.exp(new_log - old_log)
        marginal_ratio = np.exp(new_log_marginal - old_log_marginal)
        joint_kl.append(float(integrate(old_component * (old_log - new_log)).mean()))
        marginal_kl.append(float(integrate(old_marginal * (old_log_marginal - new_log_marginal))))
        joint_clip.append(float(integrate(old_component * clipped(joint_ratio)).mean()))
        marginal_clip.append(float(integrate(old_marginal * clipped(marginal_ratio))))

    # First-update score gradient, with respect to the two component means.
    score = (action[None, :] - old_means[:, None]) / sigma**2
    posterior = np.exp(old_log - old_log_marginal[None, :] - np.log(2.0))
    joint_grad_mean = integrate(0.5 * old_component * advantage * score)
    marginal_grad = advantage[None, :] * posterior * score
    marginal_grad_mean = integrate(old_marginal[None, :] * marginal_grad)
    joint_second_moment = np.diag(integrate(0.5 * old_component * (advantage * score) ** 2))
    marginal_second_moment = integrate(
        old_marginal[None, None, :] * marginal_grad[:, None, :] * marginal_grad[None, :, :]
    )
    covariance_gap = joint_second_moment - marginal_second_moment
    swap_log = log_component(-old_means)
    swap_ratio = np.exp(swap_log - old_log)
    stats = {
        "scope": "Exact finite two-component mixture toy, not a diagnosis of the H1 run.",
        "sigma": sigma,
        "clip_epsilon": clip,
        "old_means": old_means.tolist(),
        "swapped_means": (-old_means).tolist(),
        "old_density_integral": float(integrate(old_marginal)),
        "swap_joint_kl": joint_kl[-1],
        "swap_marginal_kl": marginal_kl[-1],
        "swap_joint_unclipped_surrogate": float(integrate(0.5 * old_component * swap_ratio * advantage).sum()),
        "swap_joint_clipped_surrogate": joint_clip[-1],
        "swap_marginal_clipped_surrogate": marginal_clip[-1],
        "max_joint_minus_marginal_clipped_surrogate": float(np.max(np.array(joint_clip) - marginal_clip)),
        "on_policy_joint_gradient_mean": joint_grad_mean.tolist(),
        "on_policy_marginal_gradient_mean": marginal_grad_mean.tolist(),
        "max_gradient_mean_difference": float(np.max(np.abs(joint_grad_mean - marginal_grad_mean))),
        "covariance_gap_eigenvalues": np.linalg.eigvalsh(covariance_gap).tolist(),
    }
    (out / "toy_stats.json").write_text(json.dumps(stats, indent=2) + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.1))
    swap_marginal = np.exp(np.logaddexp.reduce(swap_log, axis=0) - np.log(2.0))
    axes[0].plot(action, old_marginal, label="Old action density", lw=2.5)
    axes[0].plot(action, swap_marginal, "--", label="Labels swapped", lw=2)
    axes[0].set(xlim=(-3, 3), xlabel="Action", ylabel="Density", title="Same executed-action distribution")
    axes[1].plot(alphas, joint_kl, label="Joint KL")
    axes[1].plot(alphas, marginal_kl, label="Action marginal KL")
    axes[1].set(xlabel="Interpolation toward label swap", ylabel="KL(old || new)", title="Extra latent-association cost")
    axes[2].plot(alphas, joint_clip, label="Joint clipped surrogate")
    axes[2].plot(alphas, marginal_clip, label="Marginal clipped surrogate")
    axes[2].set(xlabel="Interpolation toward label swap", ylabel="Maximization objective", title="Joint clipping is more pessimistic")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("Two Gaussian components; A(a)=tanh(a), sigma=0.5, PPO clip=0.2")
    fig.tight_layout()
    fig.savefig(out / "joint_vs_marginal_toy.png", dpi=180)
    fig.savefig(out / "joint_vs_marginal_toy.svg")
    plt.close(fig)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
