# Gaussian residual FSPPOJoint 与 reward-weighted pMF

当前入口：`train_isaacbase_fsppo_joint.py`；配置：`hyperparams/fsppo_joint.yaml`。

## 策略与概率比

新 policy 为 `GaussianResidualPMFActorCritic`：

$$
a=\mu_\theta(s)+R_\phi(s,z)+\sigma_\theta\odot\epsilon,
\qquad z,\epsilon\sim\mathcal N(0,I).
$$

Gaussian 主干只看 observation。pMF residual 的最后输出层初始化为零，初始 action distribution 因此精确为 Gaussian；前面的 residual 特征层正常初始化，PPO 更新可以启动该分支。主干与两个 pMF endpoint heads 共用明确的 public-action 坐标换算。

sigma 是每个 action 维度独立的参数，使用 sigmoid 约束在 `[action_perturb_std_min, action_perturb_std_max]` 内。参数与 bounds 放在 actor 内，old actor、EMA、checkpoint 都保存完整 conditional policy。默认初值 0.5，边界 0.05/2.0。

rollout 保存实际 latent、raw action、behavior mean、behavior std 与完整 Gaussian logprob。更新 ratio 使用当前 mean/std 与保存的 old logprob，包含 log determinant。learnable noise 禁止同时启用手工退火。

新策略的 budget 使用完整 `KL(old || new)`，包含 mean 与 variance 变化；旧 PMF 的历史 L1 budget 保持兼容并明确命名。当前 YAML 仍关闭 budget。新策略关闭 budget 时，KL 诊断复用实际 stored latent，省去额外 map samples 的 actor 前向。

conditional entropy 的默认系数为 0.01；它是 `H(a | s,z)`，不是 marginal action entropy。

## 正权重的辅助训练

默认设置：

```yaml
fsppo_joint_aux_loss_coef: 0.1
fsppo_joint_aux_weighting: advantage_exp
fsppo_joint_aux_temperature: 1.0
fsppo_joint_aux_weight_clip: 20.0
fsppo_joint_aux_target: means
n_samples_per_action: 4
```

每次 update 开始，在完整 rollout 上标准化冻结的 advantage，计算稳定的 `exp(A_normalized / temperature)`，mean-normalize 后 cap。所有 PPO epochs 使用同一份 detached weights。cap 后平均权重可能小于一，日志记录其均值和有效样本比例。

辅助目标使用保存的 behavior transport means，未包含最后的 conditional Gaussian。fresh flow noise 与这些 targets 做 pMF velocity/JVP 回归。这样不会在理想分布拟合时把末端 Gaussian 蒸馏进 map 后再注入一次。

这是 **reward-aware transport-mean distillation 的近似实现**。它不是 executed-action Gibbs policy 的精确投影：指数化 noisy advantage、全 rollout normalization、weight cap、mean target、有限 pMF 训练均引入近似。若需要精确的 state-wise reward target，需要另加同一状态多动作评估及归一化机制。这里没有单调回报或性能提升保证。

aux 系数为零时完全跳过 JVP。开启 aux 会增加训练计算量；默认四个 flow samples，PPO ratio 不依赖这个采样数。

## 启动与消融

训练入口及 task 保持 Rough H1，默认 run name 为 `gaussian_residual_aw`。新架构首次运行应从头训练；旧 PMF checkpoint 会被 class/contract 检查拒绝直接恢复到新模型。

小规模真实环境检查：

```bash
python train_isaacbase_fsppo_joint.py --num-envs 2 --max-iterations 2
```

仅验证方案 2：设置 `fsppo_joint_aux_loss_coef: 0.0`。验证 Gaussian-only 架构时再设置 `gaussian_residual_scale: 0.0`。每次实验只改变需要验证的开关。

固定噪声消融可设置 `learn_action_perturb_std: false`，同时将 `fsppo_joint_entropy_coef` 设为零；之后可使用旧 rollout-boundary sigma 退火配置。旧 `PMFActorCritic` / `FSPPOJoint` 组合及 Endpoint variant 继续支持。

主要新日志：`joint/action_noise_std_min/max`、`joint/conditional_entropy`、`joint/conditional_joint_kl_train`、`joint/aux_weight_mean/max`、`joint/aux_weight_ess_fraction`、`joint/aux_u_loss/v_loss`。实际 action std 与 conditional Gaussian std 分开记录。

## 验证

验证使用真实 PyTorch 与 Isaac Lab configclass。外部 package 的完整测试集为 **115 passed、1 skipped**；被跳过的原 GPU 测试受沙箱设备访问限制。CPU 合同测试覆盖 Gaussian 恢复、Normal likelihood/KL 独立 oracle、std 可训练性及边界、旧尺度回放、正权重跨 epoch 冻结、mean target、budget rollback，以及 checkpoint 恢复后的下一次更新复现。

另使用 active YAML 完成了完整的 CPU rollout/update，16 个 minibatch 更新接受，loss/metrics 有限、sigma 确实学习。退出沙箱设备限制后，在空闲 RTX A6000 上单独完成真实 CUDA 编译、两轮合成 rollout/update、零 latent 与随机 latent 推理；所有检查通过，没有替换 torch.compile。这不代表 H1 回报实验。

外部实现已更新，8 个安装文件的 SHA256 与验证版本一致，默认入口已确认加载实际外部 package 中的新 class。Ruff 的 Python 错误检查及两个工作目录的 whitespace 检查通过。

外部 package 的可审查补丁保存在 `analysis/patches/fsppo_joint_gaussian_residual.patch`，便于追踪 SRB 入口对应的外部实现改动。
