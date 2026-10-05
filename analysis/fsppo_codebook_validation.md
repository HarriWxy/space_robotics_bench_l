# Fixed-codebook joint / marginal PPO 对照

入口：`train_isaacbase_fsppo_codebook.py`。配置：`hyperparams/fsppo_codebook.yaml`。

新增独立 `CodebookPMFActorCritic` / `FSPPOCodebook` variant。两种模式使用同一采样分布：均匀抽固定 latent code，再加条件 Gaussian。只切换 PPO ratio；budget、aux、entropy bonus、EMA 在本对照中关闭。原有连续 latent 和 Gaussian residual variants 保留。

## 启动对照

在现有 Isaac Lab 训练 Python 环境中，从本仓库执行：

```bash
python train_isaacbase_fsppo_codebook.py --codebook-ratio-mode joint --seed 0 --num-envs 1024 --total-env-steps 100000000
python train_isaacbase_fsppo_codebook.py --codebook-ratio-mode marginal --seed 0 --num-envs 1024 --total-env-steps 100000000
```

默认任务 `Isaac-Velocity-Rough-H1`、1024 envs × 32 horizon、K=8、独立 codebook seed=12345、LR=3e-4、sigma=0.5→0.05/800M transitions。两个命令共用所有这些参数。100M 是初次观察机制的实验预算，不是收敛保证；预算按完整 rollout 向上取整。

日志默认在 `/root/isaaclab/logs/isaaclab_fpo/h1_fsppo_codebook/` 下，以 `joint` / `marginal` 区分 run 名。`--run-name`、`--log-root`、`--device`、`--checkpoint` 等共享入口参数照常有效。**未自动启动以上长训练。**

小规模 Isaac 仿真 smoke 命令：

```bash
python train_isaacbase_fsppo_codebook.py --codebook-ratio-mode joint --num-envs 2 --max-iterations 2
python train_isaacbase_fsppo_codebook.py --codebook-ratio-mode marginal --num-envs 2 --max-iterations 2
```

## 概率、缓存和恢复契约

- Codebook 为 persistent buffer；独立 CPU generator 初始化，不消耗 actor/environment RNG。
- rollout 只前向选中的 component，存 raw action、实际 component index/code、behavior mean 和 conditional old logprob。
- 完整 rollout 收集完、首次 optimizer step 前，分块计算并缓存全部 old component logprob `[T,N,K]`。训练过程中该缓存保持冻结；每个 minibatch 重用。
- 每个 component 先对 action dimensions 求 Gaussian 联合 logprob，再对 components 做 `logsumexp - log(K)`；不是逐坐标混合。
- joint mode 使用生成 component 的新旧 conditional ratio；marginal mode 使用全部 components 的新旧 mixture ratio。uniform prior 在两者的 ratio 中抵消。
- 两种 mode 在 update 中均枚举 K，采样仅一次 transport；训练 likelihood 的计算成本为 O(K)。实际 sample efficiency 和 wall time 都需记录。
- checkpoint 保存实际 codebook、其身份契约、ratio mode、noise schedule 和当前 sigma。不同 K/codebook/ratio mode/schedule 的载入会拒绝；评估也恢复保存时的 sigma。
- `random` 评估从保存的 codebook 抽 component 并加 Gaussian，是实际 stochastic policy。`mean` 是全部 component 均值的确定性参考；`fixed_seed` 使用本地 generator 重复采样。legacy `zero` 对新 variant 是 `mean` 的别名。
- `params/codebook_sources/` 保存本次实际 variant 与相关父类源码及 SHA256，避免 untracked 新模块没有出现在 git diff 中。

改变 `policy.codebook_size` 或 `policy.codebook_seed` 时，两个对照运行都要使用相同新配置。K=1 退化为单 Gaussian 的概率目标，但不意味着整体训练栈与 native PPO 相同。原连续 latent joint 实验与本方法的对比同时改变 prior，不能当作仅 ratio 的消融。

## TensorBoard 指标

新诊断在 `Metrics/codebook/...`，通用 `approx_kl` / `clip_fraction` 对应当前选中的 ratio。

| 指标 | 用途 |
| --- | --- |
| `posterior_entropy`、`posterior_max_probability`、`effective_components` | 旧策略 latent 后验是否接近 one-hot；若接近，边缘化收益可能很小 |
| `posterior_kl` | old/new latent posterior 的额外变化成本；对 old actions 的总体期望等于 joint KL 减 marginal KL |
| `clip_jensen_gap` | marginal clipped objective 减 old-posterior-averaged joint objective；理论上非负，微小浮点负保留 |
| `joint_kl_k3`、`marginal_kl_k3` | 同一 sampled old action 下两种 ratio 的 k3 KL 估计 |
| `marginal_kl_mc` | old||new 的普通 log-ratio MC 估计；有限样本可以为负 |
| `joint_clip_fraction`、`marginal_clip_fraction` | 两种 clipping 比例；差值没有非负保证 |
| `joint_ratio_mean`、`marginal_ratio_mean`、`selected_ratio_mean` | 独立检查 density replay 与实际优化 ratio |
| `joint_kl_analytic_probe` | 最终策略相对冻结旧 actor 的 joint Gaussian KL，枚举全部 K、平均一小组 rollout 状态 |
| `size`、`marginal_ratio_enabled` | 记录实际 K 和目标分支 |

除 `joint_kl_analytic_probe` 外，loss 内新指标在 PPO minibatches/epochs 上平均；probe 是本轮更新结束后的值，不能直接与不同时间点的 minibatch KL 相减。

评估顺序：先看 posterior 是否有重叠，以及 posterior KL / Jensen gap 是否显著，再看相同 global transitions 下 return、XY/yaw tracking、episode length 和 terrain curriculum，同时记录吞吐与 learning time。单 seed 的好转只能作为初步证据。

## 已完成验证

- 新 variant 的 12 项 CPU contract tests：独立 `torch.distributions.MixtureSameFamily` likelihood 和 action/component/actor gradients、small-sigma far tails、component swap、K=1、RNG 隔离、实际两种 loss 分支、旧缓存冻结、checkpoint identity / noise、真实 mixture evaluation。
- 临时副本把两种模式强制改为 joint ratio，实际 marginal loss oracle 正确失败。
- 与既有 joint/storage 测试合跑：61 passed。
- 真实 `OnPolicyRunner.learn` 在轻量 CPU 环境分别执行两次迭代，写出 TensorBoard 和 `model_2.pt`，并恢复正确 transition count、codebook 与退火 sigma。
- 两种 native H1 入口 dry-run 通过，解析正确 task、config 和 ratio override。
- 在第二张 GPU（RTX A6000）上，joint / marginal 各完成 native H1 Rough 的 2 envs × 32 horizon × 2 iterations = 128 transitions。CUDA 编译、实际仿真 rollout、梯度更新和 checkpoint 写出均成功。
- 两个真实 checkpoint 均保存 K=8、19 维 codebook、正确 objective、退火 sigma 和 128 transitions；每组 16 个 codebook TensorBoard 指标均有限。
- 新文件 Ruff format/lint 与补丁 whitespace 检查通过。

CPU smoke 使用轻量环境；随后在沙箱外 GPU 上补跑了真实 Isaac 仿真 smoke。两次迭代仅验证实现链路，不能判断学习效果是否优于 joint 或 PPO。

真实 smoke 的日志、checkpoint、源码快照与汇总在 `logs/_analysis/fsppo_codebook_smoke/`，汇总文件为 `validation_summary.json`。

Focused tests（本仓库已配置好的 Python）：

```bash
PYTHONPATH=/root/SpaceRobot/fpo-control-saa/isaaclab_experiments/isaaclab_fpo .venv/bin/python -m pytest -p no:cacheprovider tests/test_fsppo_codebook.py -q
```

实现的外部增量补丁另存为 `analysis/fsppo_codebook.patch`。
