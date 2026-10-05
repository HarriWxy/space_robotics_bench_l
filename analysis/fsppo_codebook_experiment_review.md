# H1 Rough：Codebook 实验分析，2026-10-05

有效实验：`/root/isaaclab/logs/isaaclab_fpo/h1_fsppo_codebook/2026-10-05_22-02-31_marginal`。
事件快照止于零起始 iteration 3845，即 3846 个 rollout、**126,025,728 环境 transitions**。
每轮 1024 environments × 32 steps = 32768 transitions；控制周期 0.02 s。
本报告按实际保存的配置和源码快照分析，不使用当前 YAML 推断历史运行。

另一次 `2026-10-05_20-49-42_marginal` 的 event 文件只有 88 bytes、file-version
记录，没有训练标量或 checkpoint，不能作为性能重复实验。两次配置、14 份源码快照
及其 manifest 完全相同，均为 seed 0、fresh start。

## 结论

Codebook 比学习率 1e-4 的旧 continuous-joint 更快；与学习率同为 3e-4 的旧
continuous-joint 相比，早期有领先，随后基本接近。还不能将提速归因于 marginal ratio。
相对 PPO，主要差距仍是 yaw tracking 的学习速度。边缘化的理论机制在诊断中可见，
但实际减少的 clipping 幅度较小。

这轮已停止，事件文件在核查期间没有增长，原训练 PID 83873 已不存在。末尾连续
出现大策略更新和动作输出放大；不能把此轮判断为长期稳定。目录没有 stdout/stderr
或退出 traceback，停止原因尚不明确，不能断言发生了 NaN 或具体异常。

学习曲线：[learning.png](../logs/_analysis/fsppo_codebook_review_20261005/learning.png)
；诊断：[diagnostics.png](../logs/_analysis/fsppo_codebook_review_20261005/diagnostics.png)。
曲线按 1M transitions 分箱；表格按完整 90–100M 窗口均值。

## 相同步数的结果

| 实验 | 初始/实际 LR | 回报 | Episode 时长，s | XY error，m/s | Yaw error，rad/s |
|---|---|---:|---:|---:|---:|
| Codebook marginal，10-05 22:02 | fixed 3e-4 | 11.71 | 19.37 | 0.209 | 1.680 |
| Continuous joint，10-04 23:04 | fixed 3e-4 | 11.29 | 19.32 | 0.200 | 1.694 |
| Continuous joint，10-04 20:16 | fixed 1e-4 | 8.48 | 19.07 | 0.218 | 1.751 |
| PPO，09-16 17:22 | adaptive，initial 1e-3 | 22.81 | 19.80 | 0.136 | 0.324 |

Codebook 与同 LR joint 的回报差约 3.7%，yaw 接近、XY 略差，没有一致的大幅提升。
Codebook 最新约 3.28M 步窗口回报 12.28，尚未达到 15。

以相同 **3,276,800 transitions 的滑动窗口**判断首次达到回报阈值：

| 实验 | Return ≥ 5 | Return ≥ 10 | Return ≥ 15 |
|---|---:|---:|---:|
| Codebook marginal，3e-4 | 18.42M | 81.85M | 未达到，止于 126.03M |
| Continuous joint，3e-4 | 29.36M | 80.48M | 未达到，随后发散 |
| Continuous joint，1e-4 | 44.37M | 117.80M | 214.27M |
| PPO | 31.21M | 52.89M | 61.98M |

达到回报 10 时，Codebook 比旧 1e-4 joint 少用约 30.5% transitions，但与 3e-4
joint 基本相同；比 PPO 多用约 55%。事件时间推算分别约 35.0、33.9、47.6、14.1
分钟，从各自第一条 reward 记录开始，排除启动过程。硬件负载及 PPO 并行配置
不完全相同，耗时仅作运行参考。

90–100M 时 Codebook 学习耗时约 0.213 s/iteration，joint 3e-4 为 0.170 s，
约多 25%；总吞吐分别 41.3k 和 43.1k transitions/s。枚举 K=8 有计算开销，
不能指望 marginal 本身带来运行速度收益。

## 理论机制：有效，但当前收益有限

最近 501 次 update（约 16.42M transitions）的诊断：

| 指标 | 均值 |
|---|---:|
| Posterior entropy | 2.0118，上限 log(8)=2.0794 |
| Effective components | 7.22 / 8 |
| Joint / marginal clip fraction | 24.37% / 23.32% |
| Clip Jensen gap | 0.000957 |
| Joint raw KL，采样估计 | 0.034154 |
| Marginal raw KL，采样估计 | 0.028864 |
| Posterior KL | 0.005323 |

Joint raw KL 由 `joint_kl_k3 - joint_ratio_mean + 1` 恢复。
Joint 与 marginal raw-KL 差为 0.005289，接近 posterior KL 0.005323，符合
KL 链式分解的预期；有限样本存在 MC 误差。Jensen gap 为正，marginal objective
确实减少了 joint clipping 的额外悲观性。但 clip fraction 仅减少约 1.06 个百分点，
不足以单凭这项诊断期待数倍训练加速。

Effective components≈7.2 的含义是**同一个动作能由多个 component 解释**，
不意味着学出了七种不同动作。估计条件互信息
`log(8) - E[posterior_entropy] ≈ 0.0677 nats` 很小，说明 component index
对动作的信息较少。可能 component means 接近，也可能 sigma 使分布高度重叠；
仅靠这些指标无法区分，需要对实际 rollout states 记录 component mean spread/σ、
两两距离和协方差谱。

## 末尾稳定性警报

最后五轮记录：

| Iteration，零起始 | 环境步数，M | Raw action std | Marginal raw KL，MC | Joint analytic probe KL |
|---:|---:|---:|---:|---:|
| 3841 | 125.894656 | 0.912 | 0.248 | 0.0313 |
| 3842 | 125.927424 | 1.193 | 0.758 | 0.1249 |
| 3843 | 125.960192 | 1.762 | 0.894 | 1.7418 |
| 3844 | 125.992960 | 1.033 | 0.099 | 1.5236 |
| 3845 | 126.025728 | 1.635 | 1.309 | 10.6898 |

此前 raw action std 通常约 0.67；条件 Gaussian sigma 末尾仍约 0.429，
因此这次动作方差增加不能归为 sigma schedule 增大。所有已记录标量仍有限；
最后 value loss=0.0381，critic 尚未出现旧 joint 后期的巨大数值。
Episode return 聚合已完成 episodes，响应会滞后，末尾回报仍约 12 不能排除策略失稳。

该实验 fixed LR=3e-4，budget=false，16/16 optimizer updates 始终接受，
没有 early stop，也没有随 KL 减小 LR 的机制。
旧 continuous-joint 3e-4 在 137.95M 首次出现 action std>2，随后放大到 1.57e5、
value loss 达 1e19、回报变成巨大负数。Codebook 末尾形态值得警惕，但日志不足以
断定两轮的触发原因相同。

也需要区分 **k3 importance estimator 的尖峰**和真正的大 policy change。
Codebook iteration 3723 的 marginal k3=5.086、ratio mean=6.065，而 raw KL≈0.0212，
主要是 rare ratio 放大估计。最后 iteration 3845 的 raw KL=1.309，同时最终
joint analytic probe=10.69，才构成更强的更新幅度警报。
Probe KL 测量整轮更新后的 1024 states，其他诊断是 update 期间 16 个 minibatch
的平均；不能直接相减解释 KL 分解。

## 比较合同与下一步

三轮 FSPPO 实验的 env.yaml 字节完全相同，网络均 `[512,256,128]`、ELU、1 NFE，
aux=0、sigma 0.5→0.05/800M、seed=0、clip_actions=null、budget/EMA 关闭。
Codebook 与旧 continuous-joint 同时改变了 latent prior/策略 family 和 ratio；
计算诊断的 RNG 消费路径也不同。只有一个有效 seed，未估计重复实验方差。
PPO 另有环境数、rollout 长度、更新 epochs、adaptive LR、learned sigma、entropy
和 critic clipping 等差异，因此这里作为整体性能参照。

两轮 Codebook 的保存源码仍是**今日数值修复之前**的实现，尚未采用稳定 log-ratio
surrogate 和梯度范数回退。当前安装的新实现及 workspace joint YAML 的执行动作
clip=10 不属于本轮设置，不能将本轮结果用于验证这些修复。

下一次最有诊断价值的实验是：用相同源码、相同 K=8/sigma/网络/LR 的
**Codebook joint vs Codebook marginal**，只切换 ratio mode。考虑此次末尾警报，
两侧可统一先用已安装数值修复和 fixed LR=1e-4，再比较；这是另一组控制实验，
不能与本轮混为同一设置。若要采用 KL adaptive LR，也应两侧同时启用。
补充 component spread 诊断能进一步判断固定 codebook 是否提供了实际的表达收益。
本次分析没有修改训练代码/配置，也没有启动训练。

## 复现

快照与全部窗口/阈值数据：
[`comparison.json`](../logs/_analysis/fsppo_codebook_review_20261005/comparison.json)。

```bash
.venv/bin/python analysis/review_fsppo_codebook.py
```

默认读取本报告的固定 scalar snapshots；`--refresh` 才重新读取 live event files。
每个事件文件只读取固定大小内的完整记录，迭代步数按 `(iteration+1)*32768` 转换。
