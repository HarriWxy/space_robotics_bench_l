# tips for training with FPO

## sample actions

严格说，不是。`n_samples_per_action` 是“每个 action 的 Monte Carlo 样本数”，不是 flow 的时间步数。

以当前 `pmffpo.yaml` 为例：

| 配置 | 当前值 | 含义 |
|---|---:|---|
| `sampling_steps` | 1 | action 生成时从噪声到 action 的 ODE/NFE 步数 |
| `n_samples_per_action` | 2 | 每个 action 额外采样 2 组 `eps/r/t`，分别计算 PMF score |
| `training_sampling_steps` | 1 | 设计上想表示训练步数，但当前代码没有实际读取 |

PMF 中的张量大致是：

```text
eps: [B, S, action_dim]
r,t: [B, S, 1]
```

其中 `S = n_samples_per_action`。所以当前配置表示：

```text
每个环境 action：
  生成阶段：1 次 mean-flow jump
  训练 score：用 2 组独立的噪声和时间样本估计
```

这两个样本不是串行的 `x_t -> x_{t-1}`，不会改变 action 的生成轨迹；它只增加 loss 估计的计算量，并降低 Monte Carlo 估计噪声。相关维度见 [actor_critic.py:769](/root/R2A/Algos/fpo-control-saa/isaaclab_experiments/isaaclab_fpo/isaaclab_fpo/modules/actor_critic.py:769)。

因此：

- 想改变 action 生成的 flow 步数：改 `sampling_steps`；
- 想提高训练 score 的采样精度：改 `n_samples_per_action`；
- `training_sampling_steps` 在当前 checkout 中仍只是预留配置，没有实际效果。

## infer



## pMF tips

### infer

不是漏了 `t`。这是 pMF 的 **h-only 条件化 + x-prediction**：

$$
r=t+dt,\quad dt<0,\quad h=t-r=-dt>0
$$

所以这里嵌入 `h=-dt`，是在告诉 actor：“这次要从当前 $t$ 跨多长区间到 $r$”。训练时也是传 `h=t-r`，不能把反向积分的负 `dt` 直接喂进去。

$$
\hat x=f_\theta(\text{obs},x_t,\phi(h)),\qquad
u=\frac{x_t-\hat x}{\max(t,\epsilon)},\qquad
x_r=x_t-(t-r)u
$$

其中：

- `cos/sin(2^k h)` 是固定 Fourier time embedding，让 MLP 更容易区分不同区间长度；`expand` 是因为一个 batch 在同一 solver step 共享同一个 $h$。
- actor 输出的前半段 `x_mean` 不是速度，而是“去噪 action-like”预测；`t_value` 没进 MLP，但必须留在分母中完成 $x\rightarrow u$ 转换。
- 更新 `x_t + mean_velocity * dt[i]` 等价于 $x_t-hu$，符号是对的。
- 默认单步时 $(t,r,h)=(1,0,1)$，所以
  $$
  x_0=x_1-\frac{x_1-\hat x}{1}=\hat x,
  $$
  即 pMF 的第一头直接给出最终 action。第二头只用于训练中的瞬时速度/JVP 辅助项。见 [本地实现](</root/R2A/Algos/fpo-control-saa/isaaclab_experiments/isaaclab_fpo/isaaclab_fpo/modules/actor_critic.py:687>) 和 [单步测试](</root/R2A/Algos/fpo-control-saa/isaaclab_experiments/isaaclab_fpo/tests/test_imf_fpo.py:186>)。

一个值得注意的边界：论文的一般记号是 $\hat x_\theta(z_t,r,t)$，但官方 released pMF 架构明确选择“不显式 condition on $t$，只 condition on $h=t-r$”；$t$ 只参与末端的 $x\to u/v$ 变换。[官方实现](https://github.com/Lyy-iiis/pMF/blob/torch/models/pmfDiT.py#L332-L370)；[论文公式](https://arxiv.org/html/2601.22158v3)。

所以这不是 bug，但它是一个结构性约束：若你把 `sampling_steps` 改成均匀多步，所有 step 的 $h=1/N$ 都相同，网络没有显式绝对时刻标记，只能从 `x_t` 和分母 $t$ 间接区分阶段。对 RL action flow，最干净的消融是比较 `[h]`、`[t,h]`、`[r,t]` 三种条件化。

