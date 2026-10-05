# FSPPOJoint：超出调参的算法改进方向

审查日期：2026-10-04。对象为当前 `fsppo_joint.py`、`actor_critic.py` 的实现；当前实验 `aux_loss_coef=0`，单步采样，固定 latent prior，一轮 PPO 内条件噪声 sigma 不变。

结论：joint policy gradient 合法，但 joint clipping 对 latent 对应关系施加额外保守性。优先验证动作边缘 likelihood 的目标；架构上提供可恢复 Gaussian PPO 的起点。若恢复 pMF 辅助训练，需要同时处理 reward 目标及末端 Gaussian 的分布一致性。以下严格恒等式不构成当前 H1 加速或收敛的保证。

## 1. 当前模型与成立条件

对固定状态，当前策略可写为

$$
z\sim p(z)=\mathcal N(0,I),\quad
a=F_\theta(s,z)+\sigma\epsilon,\quad\epsilon\sim\mathcal N(0,I),
$$
$$
q_\theta(z,a\mid s)=p(z)\mathcal N(a;F_\theta(s,z),\sigma^2I),\qquad
\pi_\theta(a\mid s)=\int q_\theta(z,a\mid s)\,dz.
$$

这里 `a` 是策略生成的 raw action。环境对 action 的后处理只要不额外依赖 z，给定状态与 action 后该次 latent 不影响 return。当前每步独立抽 latent，适用这一设定。

更新使用保存的实际 latent 计算 conditional ratio。因为 prior 固定，它就是 exact joint ratio；不是 action marginal ratio。这种 joint 优化有 [ReinFlow](https://arxiv.org/abs/2505.22094) 等工作的先例，不能称为概率实现错误。

实现对应：`fsppo_joint.py:474` 重算 stored-latent mean，随后计算 conditional logprob 与 PPO clipping；`actor_critic.py:833` 使用 unit-time embedding；`fsppo_joint.py:528` 在 aux=0 时跳过完整辅助回归。

## 2. Joint 目标中存在环境不需要的约束

对每个状态，严格 KL 链式分解为

$$
D_{KL}(q_o\Vert q_n)
=D_{KL}(\pi_o\Vert\pi_n)
+\mathbb E_{a\sim\pi_o}D_{KL}(q_o(z\mid a)\Vert q_n(z\mid a)).
$$

额外项衡量“哪个 latent 解释哪个动作”的变化。相同 sigma 时，joint KL 等于

$$
\mathbb E_z\frac{\|F_n(s,z)-F_o(s,z)\|^2}{2\sigma^2}.
$$

例如二维 Gaussian latent：旧映射为 z，新映射为正交旋转 Rz。动作边缘分布完全相同；若旋转角度为 phi，joint KL 为 `2(1-cos(phi))/sigma²`。90 度旋转、sigma=0.5 时，joint KL=8，marginal KL=0。

**当前日志的 `map_kl` 实际是 L1 位移，不是上述 KL。** budget 关闭时这项没有作为 penalty 生效，因此不能用它声称当前训练被 KL budget 卡住。下面的 clipping 推论仍适用于当前主目标。

令 r_j 为 joint ratio、r_m 为 marginal ratio，则

$$
r_m(s,a)=\mathbb E_{z\sim q_o(z\mid s,a)}r_j(s,z,a).
$$

标准 PPO 最大化目标 `g_A(r)=min(rA,clip(r)A)` 对 r 是凹函数：正 A 时是 `A min(r,1+epsilon)`，负 A 时是 `A max(r,1-epsilon)`。因此 Jensen 给出

$$
L^{clip}_{joint}\le L^{clip}_{marginal}.
$$

这是**同一策略 family、同一 advantage 下，population surrogate 的关系**，不是每条采样上的大小关系，也不是 marginal PPO 必然更快。PPO clipping 的定义来自 [PPO 原论文](https://arxiv.org/abs/1707.06347)；KL/Jensen 推导是对当前模型的分析。

在 on-policy 点，Fisher identity 还给出

$$
\mathbb E[A\nabla\log q_\theta(z,a\mid s)\mid s,a]
=A\nabla\log\pi_\theta(a\mid s).
$$

根据全协方差公式，边缘 score 消除了 latent 条件方差。这也适用于 unclipped importance surrogate；**不能直接推广为所有 PPO epoch 的 clipped gradient 都更低方差**，因为 clipping 后两个目标不同。

### 最小可验证版本：固定 codebook 的 mixture policy

固定并 checkpoint 保存 z_1,...,z_K，定义新的实际策略

$$
\pi_\theta^K(a\mid s)=\frac1K\sum_{k=1}^K\mathcal N(a;F_\theta(s,z_k),\sigma^2I).
$$

采样均匀 component 再加 Gaussian。保存 old **mixture** logprob；更新时通过 `logsumexp(component_logprob)-log(K)` 计算 current mixture logprob。这是新离散 latent family 的精确 likelihood；**不是连续 Gaussian latent 原策略的精确积分**。不能保留连续 latent rollout，却把少量新 latent 的 MC likelihood 当 exact policy density。

建议同一 codebook、网络、sigma、训练设置，分别比较 joint PPO 和 marginal PPO，先 K=4 或 8 做机制实验。K 的选择只是计算折中，不是理论最优值。训练 likelihood 计算量 O(K)，采样仍只需一个 component forward。old mixture logprob 可以在 rollout 后、actor 更新前批量计算。

记录 responsibility 集中度、joint/marginal 的 advantage 加权 clip gap、actor 梯度方差、样本数和 wall time。如果 posterior 几乎 one-hot，marginal 与 joint 差异可能很小。mixture entropy 无一般闭式，不能拿 component Gaussian entropy 代替。

更大的架构改动是可逆 flow，直接用 change-of-variables 获得 marginal density。其精确 likelihood 原理见 [Real NVP](https://arxiv.org/abs/1605.08803)。当前普通 pMF MLP 不保证可逆，不能删掉末端 sigma 后继续沿用 conditional Gaussian logprob。

## 3. 给策略一个明确的 Gaussian PPO 起点

可使用

$$
a=\mu_\theta(s)+R_\phi(s,z)+\sigma_\theta(s)\epsilon.
$$

residual 为零、sigma 与 PPO 同参数化时，动作分布及 ratio 恢复 Gaussian PPO。让 Gaussian 分支学习基本控制，latent residual 提供额外分布能力。这证明 family 包含 PPO，**不证明优化性能至少等于 PPO**。

residual 可将最后输出层初始化为零，前面的特征层正常初始化；不要同时把乘法 gate 和 residual 都设为零，否则相关参数没有梯度。learnable per-action sigma 比预设统一退火更灵活；[ReinFlow](https://arxiv.org/abs/2505.22094) 也使用 learnable noise，但它主要研究已有生成策略的 fine-tuning，不能套用性能数字到本次从零训练。

sigma 可学习后，logprob 必须包含新旧尺度及 log determinant；Gaussian KL 也必须使用完整公式。conditional entropy `H(a|s,z)` 不等于 action entropy：`H(a|s)=H(a|s,z)+I(a;z|s)`。

固定 sigma 时 score 为 `J_F^T epsilon/sigma`，joint 均值 Fisher 按 `1/sigma²` 缩放。这说明降低 sigma 同时改变探索与优化曲率，不能只视为减少动作噪声；也不意味着条件数必然恶化。

## 4. 重新引入 flow 训练前，先对齐目标与采样器

aux=0 后目前只训练一步 endpoint，未继续强制 MeanFlow identity。pMF 的一步性质依赖其场关系及训练目标，不能从使用同样 MLP 输出头直接继承图像生成结果；见 [pMF 原论文](https://arxiv.org/abs/2601.22158)。

现有 aux 使用 fresh eps 对 stored raw rollout actions 做 velocity/JVP regression，等权拟合，没有 reward 改进方向。它不是直接 endpoint MSE。

还存在理想化一致性问题。令 `nu_old=F_old#p`，则 `pi_old=nu_old*N_sigma`。若辅助训练理想地使 `nu_new=pi_old`，但执行仍加独立末端 Gaussian，则

$$
\pi_{new}=\pi_{old}*\mathcal N(0,\sigma^2I).
$$

有限二阶矩时 raw-action 协方差再增加 sigma² I。这是 noisy target 与 sampler 的一致性风险，**不是当前有限训练中已经观测到的方差递增定律**。aux 当前为零，它不能解释剩余 PPO 差距。

拟合保存的 behavior mean 分布可以避免理想情况下的重复卷积，但仅是旧策略 anchor，没有 policy improvement 保证。

若希望 flow 自身承担策略改进，可以从固定状态的 KL regularized improvement 推出

$$
\max_q\{\mathbb E_q A_o-\eta KL(q\Vert\pi_o)\},\qquad
q^*(a\mid s)=\frac{\pi_o(a\mid s)e^{A_o(s,a)/\eta}}{Z(s)}.
$$

其恒等式为 `E_q A - eta KL(q||pi_old) = eta log Z - eta KL(q||q*)`。理论基础见 [MPO](https://arxiv.org/abs/1806.06920)、[AWR](https://arxiv.org/abs/1910.00177)。reward-weighted regression 本身不是新的研究方向。

理想 weighted CFM 使用非负权重 `exp(A/eta)/Z(s)`，将 data target 改成 q*；不过 FM residual 一般不是 KL projection，当前带 stop-gradient JVP 的 pMF 更不能直接宣称有限更新单调改进。

实际要处理：同一状态的权重归一化（全局归一化会改变状态权重）、单个 noisy GAE 的指数偏差、固定 sigma family 的方差下界，以及 target 后再次加 Gaussian 的 mismatch。不能直接把 signed A 乘平方残差，负权重会鼓励残差增大。

更完整的路线是 same-state 多动作评估 → reward-weighted flow teacher → 单步 pMF 蒸馏，并明确实际 action distribution。当前 H1 仍慢时，建议先完成第 2 节 joint/marginal 机制验证。

## 5. 保留当前 family 的另一条严格方向：零期望 control variate

冻结 conditional-action-independent 的 B(s,z) 与 beta，定义最大化目标

$$
L_{cv}=L_{clip}-\beta\mathbb E_o[(r_j-1)B(s,z)].
$$

因为 `E_old[r_j|s,z]=1`，附加项对任意 theta 的 population 期望为零，不改变原 clipped surrogate。合适的 B 与 beta 可以降低梯度方差；不保证任何 learned baseline 都会改善。

这一项必须使用 **unclipped exact ratio**；B/beta 要 stop-gradient 并预先冻结，最好从过去数据或 cross-fitting 获取。不能把 `A-B(s,z)` 直接放进 PPO min 后宣称 clipping 目标仍然相同。保留 V(s) 做 GAE bootstrap。高 ratio 尾部及不准 baseline 可能增加方差。

learned Q 也可以作为 pathwise control variate，但必须保留 score residual 与校正项；单独使用 approximate Q 的 action gradient 不等于真实 policy gradient。相关原理见 [Q-Prop](https://arxiv.org/abs/1611.02247) 与 [LAX](https://arxiv.org/abs/1711.00123)。

## 6. 已完成的数值验证

运行 `MPLCONFIGDIR=/tmp/fsppo-theory-mpl python analysis/fsppo_joint_marginal_toy.py`。

toy 为两个等权 Gaussian component，mean=-1/+1、sigma=0.5、A(a)=tanh(a)、clip=0.2。交换两个 latent 标签后动作分布完全不变：

| 检查 | 数值 |
| --- | ---: |
| joint KL | 8.000000 |
| action marginal KL | 0 |
| joint unclipped surrogate | 约 0 |
| joint clipped surrogate | -0.619360 |
| marginal clipped surrogate | 约 0 |
| on-policy joint/marginal 梯度均值最大差 | 2.78e-17 |
| joint covariance 减 marginal covariance 的特征值 | 0.009377、0.036787 |

扫描 101 个插值策略时，joint clipped surrogate 均不大于 marginal surrogate。它验证了数学机制，没有使用机器人轨迹，不能量化该机制对当前 H1 的贡献。

输出图及统计在 `logs/_analysis/fsppo_joint_theory/`；PNG 与 SVG 均可独立导出。
