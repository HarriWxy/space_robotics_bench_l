# FSPPOJoint implementation

This is an independently selectable successor to the experimental pMF-score
FSPPO. The existing `FPO`, `PMFFPO`, and `FSPPO` implementations and experiment
configs remain available for comparisons.

## Policy and objective

The training policy explicitly samples

```text
xi ~ N(0, I)
mean = F_theta(obs, xi) = actor_scale * x_mean(obs, xi, h=1)
a ~ N(mean, sigma^2 I), sigma = policy.action_perturb_std > 0
```

Each rollout transition stores the actual generating `xi`, raw Gaussian action,
behavior mean, and conditional action log-probability. The prior on `xi` is fixed
and cancels from the joint importance ratio:

```text
log_ratio = log N(a; F_new(obs, xi), sigma^2 I)
          - log N(a; F_old(obs, xi), sigma^2 I)
```

The ratio is exact for the joint latent/action distribution. It is generally
different from the marginal action ratio; joint PPO clipping can be more
conservative. No pMF residual, JVP, CFM-score clamp, or auxiliary head enters this
ratio. Raw actions are retained separately from the tensor passed to the
environment, so deterministic action clipping/scaling belongs to the environment
transition, not to the Gaussian density evaluation.

For fixed sigma and the same latent in both maps,

```text
D_map = E_xi sum_action (F_new - F_old)^2
K_joint = D_map / (2 sigma^2)
KL(marginal_old || marginal_new) <= K_joint
loss = clipped_PPO_loss + value_loss_coef * value_loss
     + kl_coefficient * estimated_K_joint
     + fsppo_joint_aux_loss_coef * pmf_regression_loss
```

The equality and marginal bound concern the prior expectation at a fixed state.
Logged map statistics are finite-sample estimates on rollout observations.
They are not measured marginal KL or population guarantees.

The design follows the latent/Markov policy-gradient construction in
[ReinFlow](https://arxiv.org/html/2505.22094v3). The map is parameterized using
[pMF](https://arxiv.org/html/2601.22158v3). These ingredients alone are not a
claim of a new likelihood theorem or monotonic policy improvement.

## Empirical budget and rollback

At the start of every update, freeze the online behavior actor and sample a fixed
probe of rollout observations and independent prior latents. For each minibatch:

1. Compute PPO, map-KL, value, and optional auxiliary gradients.
2. Snapshot the full policy and Adam state before the candidate step.
3. Apply the candidate, then measure the probe mean KL.
4. If the candidate is non-finite or exceeds the budget, restore both snapshots
   and retry the same gradient with a smaller learning rate.
5. If all retries fail, restore the last accepted state and end this update.

The configured base learning rate is restored after each accepted trial. Adam
momentum and step counters advance only for accepted candidates. The KL snapshot
remains the behavior actor throughout all epochs; it is not updated per minibatch.
This also checks the first candidate, where the map penalty initially has zero
gradient.

A fresh probe after the update reports how well the empirical budget generalizes
to new samples. That audit does not trigger another rollback and may exceed the
acceptance budget. The next penalty coefficient is

```text
lambda_next = clip(lambda + dual_lr * (audit_mean_KL / target_KL - 1), 0, lambda_max)
```

This implementation is single-device; distributed use is rejected until candidate
acceptance and rollback can be synchronized across ranks.

The new learner also sets float32 matmul precision to `highest` for its process.
The shared launcher otherwise enables TF32, which measurably changes the same
actor's output between small rollout batches and large update batches. With a
small fixed sigma these differences affect log-probability replay. Matmul
precision is therefore part of the saved sampling contract and cannot change
mid-run.

## Configuration

The native H1 entry point is `train_isaacbase_fsppo_joint.py`, with its own
`hyperparams/fsppo_joint.yaml` and experiment name `h1_fsppo_joint`.
Defaults select a two-environment, two-iteration smoke run.

| Field | Default | Meaning |
|---|---:|---|
| `policy.action_perturb_std` | 0.02 | Fixed Gaussian action noise in public action coordinates |
| `algorithm.fsppo_joint_kl_target` | 0.01 | Mean KL budget on the fixed acceptance probe |
| `algorithm.fsppo_joint_kl_coef` | 1.0 | Initial coefficient on normalized map KL |
| `algorithm.fsppo_joint_dual_lr` | 0.1 | Penalty coefficient adaptation; zero disables adaptation |
| `algorithm.fsppo_joint_map_samples` | 2 | Independent map samples per observation |
| `algorithm.fsppo_joint_probe_size` | 1024 | Maximum observations in each acceptance/audit probe |
| `algorithm.fsppo_joint_max_backtracks` | 5 | Retries after the initial candidate |
| `algorithm.fsppo_joint_backtrack_factor` | 0.5 | Step-size multiplier for each retry |
| `algorithm.fsppo_joint_aux_loss_coef` | 0.0 | Independent positive pMF regression weight |

These are initial experimental settings, not tuned locomotion hyperparameters.
`desired_kl` and legacy CFM-score parameters do not control the new trust region.
Map sampling is independent of `n_samples_per_action`; the latter is used only
when optional auxiliary regression is enabled. Action dimension is summed, not
averaged, for the Gaussian KL identity.

At the default zero auxiliary coefficient, no JVP is evaluated and only the
one-step transport is optimized: this does not establish a learned MeanFlow at
other intervals. Setting a positive auxiliary coefficient enables positive pMF
regression separately from the advantage-weighted PPO loss. Its exact global
anchor count is randomly assigned rather than tied to fixed environment IDs.

## Running and evaluating

```bash
conda activate srb
python train_isaacbase_fsppo_joint.py --dry-run
python train_isaacbase_fsppo_joint.py --num-envs 2 --max-iterations 2

# Independent full run; do not compare to old runs with different rollout sizes.
python train_isaacbase_fsppo_joint.py \
  --num-envs 1024 --max-iterations 10000 --seed 0 --run-name joint_seed0
```

The external repository's own train/play CLI also accepts
`--algorithm fsppo_joint`. The native wrapper selects the variant via its YAML;
edit that YAML or pass `--fpo-config` for algorithm overrides.

Resume only from FSPPOJoint checkpoints with the same sampling contract. The
checkpoint records adaptive penalty/counters and retains online weights paired
with the optimizer even when optional EMA is enabled.
This is training-state resume, not a bitwise continuation of an episode:
simulator state and random-number-generator state are not checkpointed.

Existing playback modes remain explicit deployment choices: `zero` uses zero
latent and `random` samples the prior; neither adds the training Gaussian action
perturbation. Consequently their returns need not equal returns of the stochastic
training policy. `sample_transport()` is the interface for sampling that exact
training distribution.

Monitor `Metrics/joint/probe_kl_mean`, `audit_kl_mean`, `audit_kl_p95`,
`rejected_candidates`, `accepted_updates`, and the separate auxiliary metrics,
together with episode duration, base-contact termination, tracking error, and
reward. A small probe KL alone does not imply that locomotion has improved.

## Validation

Focused tests live in the external package's `tests/test_fsppo_joint.py` and
`tests/test_joint_storage.py`. They cover Gaussian ratio/KL identities, generating
latent replay, auxiliary-head isolation, optional regression, lossless minibatch
coverage, timeout-aware GAE, full optimizer rollback, checkpoint compatibility,
and online-versus-EMA resume. The legacy `tests/test_imf_fpo.py` covers regression
of the older variants.

```bash
conda activate srb
cd /root/R2A/Algos/fpo-control-saa/isaaclab_experiments/isaaclab_fpo
python -m pytest tests/test_fsppo_joint.py tests/test_joint_storage.py tests/test_imf_fpo.py -q
```

Final result: **44 tests passed** on CPU. Focused Ruff `E9,F`, formatting of
the new algorithm/storage/test files, and external-repository `git diff --check`
also passed. The sandbox CPU test process emitted a CUDA-initialization warning;
GPU validation below was run separately with host-device access.

### GPU smoke evidence (2026-09-21)

The native `Isaac-Velocity-Flat-H1` task ran on `cuda:1` with seed 0,
two environments and 32 steps per environment. This is a functional check,
not a locomotion learning comparison.

| Update | Accepted minibatch steps | Rejected candidates | Probe mean KL | Fresh audit mean KL |
|---|---:|---:|---:|---:|
| First rollout | 5 | 30 | 0.009675 | 0.009531 |
| Second rollout | 2 | 12 | 0.009244 | 0.009305 |

Both updates stopped early at the empirical KL budget (`target=0.01`), and
all saved weights were finite. A rejected candidate is a rolled-back trial,
not an additional accepted optimizer step. The number of backtracks shows
that these conservative initial settings still need equal-step ablations.

Successful initial run:
`logs/fsppo_joint_smoke/2026-09-21_16-22-44_validation_fp32/model_2.pt`.
It saved the adaptive coefficient `0.98836496` and update counter `2`.
The follow-up run
`logs/fsppo_joint_smoke/2026-09-21_16-23-33_resume_validation/model_3.pt`
successfully resumed from that checkpoint and completed another rollout.

The final resume check under
`logs/fsppo_joint_smoke/2026-09-21_16-29-57_resume_counter_validation`
also verifies the new runner bookkeeping: it restores the per-environment
clock to `64` (not `128`), then reports `192` total transitions after one
additional 64-transition rollout. The joint branch saves these runner counters
separately from the algorithm's rollout-update counter; legacy branches retain
their existing behavior. Playback-only loads do not restore training counters.

The earlier TF32 run under
`logs/fsppo_joint_smoke/2026-09-21_16-16-04_validation` was deliberately
retained: it failed the behavior-replay check before the first optimizer
update. A separate CUDA check reproduced batch-shape-dependent mean/log-density
differences; forcing full FP32 fixed the actual H1 run without weakening that
check.
