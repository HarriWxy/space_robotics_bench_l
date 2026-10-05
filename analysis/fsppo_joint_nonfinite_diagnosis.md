# FSPPOJoint 非有限梯度：已有日志证据

分析对象是 `/root/isaaclab/logs/isaaclab_fpo/h1_fsppo_joint/` 下
`2026-10-05_11-32-44_gaussian_residual_aw` 和
`2026-10-05_20-37-48_gaussian_residual_aw`，实际算法为 `FSPPOJoint`，策略为
`GaussianResidualPMFActorCritic`。

两次运行均最后成功记录在 **iteration 1374**（零起始编号，共 45,023,232 个环境
transitions）。79 个训练标量全程逐项完全一致，区别主要是运行耗时。

| iteration | 原始动作 std | 辅助 loss | value loss | 裁剪前梯度 norm |
|---:|---:|---:|---:|---:|
| 1290 | 1.23 | 0.000128 | 0.00847 | 1.34 |
| 1295 | 1.26 | 0.235 | 0.0114 | 7.13 |
| 1300 | 2.14 | 1.87 | 0.0199 | 68.9 |
| 1304 | 18.44 | 305.5 | 477 | 16,833 |
| 1308 | 1,207.7 | 38.16 | 2.87e10 | 9.35e9 |
| 1374 | 2,294.9 | 49.93 | 1.38e12 | 5.72e11 |

辅助项先出现明显增长，随后原始动作输出发散，动作变化惩罚大幅恶化，critic loss
和梯度增长数个数量级。学习到的条件噪声 **σ 仍有限**：最后均值约 0.998，范围
0.380–1.361；`Metrics/action_std` 是 rollout 动作的标准差，不能当作 σ。

保存的环境配置将未经裁剪的 `last_action` 输入策略；动作 manager 和 wrapper
也均未裁剪。这提供了动作递归反馈放大的可能路径。历史 PPO baseline 的
`clip_actions` 同样为 `null`，因此不能将问题归为遗漏 PPO 原有裁剪设置。

日志没有失败 batch 或接近失败时的 checkpoint，**尚不能确认具体哪个运算产生
非有限梯度**，也不能单凭这里的证据区分梯度元素非有限与 FP32 梯度范数计算溢出。
辅助项增长、动作反馈和 critic 放大是已记录的过程，具体触发原因仍需捕获验证。

曲线：[diagnosis.png](../logs/_analysis/fsppo_joint_nonfinite/diagnosis.png)。
可复现绘图：

```bash
env -u PYTHONPATH .venv/bin/python analysis/plot_fsppo_joint_nonfinite.py \
  --run /root/isaaclab/logs/isaaclab_fpo/h1_fsppo_joint/2026-10-05_20-37-48_gaussian_residual_aw \
  --out logs/_analysis/fsppo_joint_nonfinite/diagnosis.png
```

## 已安装的修复

外部 `isaaclab_fpo` 的 `fsppo_joint.py` 现在按 advantage 符号在 log-ratio 空间
计算 PPO clipped surrogate。这与原目标逐点等价：正 advantage 使用上截断，负
advantage 使用下截断，零 advantage 恒为零。避免正 advantage 已被截断时，先计算
`exp(100)` 再在 backward 中得到 `0 × Inf = NaN`。Codebook 两种 ratio 模式复用
同一实现，概率分布与 ratio 定义没有修改。

梯度在原地裁剪前先计算范数。若 FP32 范数非有限，先检查每个参数的梯度元素；
元素有限时用 FP64 重新计算范数并裁剪。真实 NaN/Inf 继续报错，错误包含迭代、
minibatch、参数名、输入与 log-ratio 范围，不替换坏梯度或提交该 optimizer step。
日志新增完整 update 的 `joint/log_ratio_min/max`。

另修复固定 sigma 的查询重入问题：重复查询曾经原地改写先前 Gaussian density
保存的 scale buffer，造成 backward 的 tensor-version 错误。现在每次 density
取得自己的 scale snapshot，旧 actor 与 sigma schedule 仍使用原 buffer。

当前 `hyperparams/fsppo_joint.yaml` 将 **执行动作**限制为 `[-10, 10]`，阻断
`last_action` 无界反馈。rollout 保存的仍是原始 Gaussian sample 与对应 logprob，
没有用裁剪后的动作替换 likelihood 输入。这个上限是新增的环境动作变换；它不
限制 sigma，也不保证 raw action 输出有界。将来比较 PPO 时应对齐执行动作上限。

可审查的增量补丁为
[`fsppo_joint_numerics.patch`](patches/fsppo_joint_numerics.patch)。安装前核对了原始
文件哈希，外部包仅更新三份生产文件和两份测试文件，备份留在
`/tmp/fsppo-joint-numerics-hc_42ba5/installed_backup/`。

## 验证范围

外部 package 全部 CPU 测试为 **118 passed、1 skipped**；额外 Codebook 测试为
**12 passed**。三项新增数值回归在原实现上都因预期缺陷失败，在修复后通过；
固定 sigma 的既有合同也扩展了 backward 验证。数值参考使用独立的 FP64 PPO
公式与真实优化器提交后的参数更新，不只检查配置或源码。

修改后的实际外部包与配置已完成真实 CUDA H1 **2 环境、2 轮 rollout/update**。
另外，在真实 H1 中强制 Gaussian 主干均值为 1000：64 个步骤中 raw action 最大
1002.31，执行动作及 `last_action` observation 最大均为 10。逐步确认 wrapper
没有改写 raw sample，rollout storage 保留 raw actions，两轮更新完成且所有参数
有限。摘要保存于
[`saturated_smoke_result.json`](../logs/_analysis/fsppo_joint_nonfinite/saturated_smoke_result.json)。

有迭代上限的旧实现故障捕获仍在运行。GPU 1 与原始 GPU 0 的早期轨迹已有细小
差异，因此不能假设会在原来的第 1375 轮复现；捕获结果将在完成后补充。
小规模 smoke 只验证采样、环境与 backward 连通，不能代表长训练稳定性或回报改善。
