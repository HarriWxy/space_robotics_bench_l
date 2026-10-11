# pMF 一步生成与 Flow 一致性验证

本实现对应 [`tasks.md`](/root/R2A/Algos/fpo-control-saa/isaaclab_experiments/tasks.md) 的
One-step / Multi-step Consistency Test，沿用
`train_isaacbase_fsppo_joint.py → train_isaacbase_fpo.py → PMFActorCritic + FSPPOJoint`
训练路径。验证不改变训练的采样器、损失实现或默认训练规模。

## 受控 A/B 训练

`--pmf-ablation pg_only` 将 `fsppo_joint_aux_loss_coef` 设为零；
`--pmf-ablation pg_pmf` 使用 YAML 的正系数，若 YAML 为零则使用 0.01，
也可通过 `--pmf-loss-coef` 明确指定。两者的其余配置保持一致。
这里的 PG 是当前 FSPPOJoint 的 joint-Gaussian PPO surrogate，不是直接 `-Q(s,a)`。
价值损失、可选 KL 预算、EMA 和训练动作的条件 Gaussian 噪声仍按同一配置运行。

先验证实际训练入口：

```bash
conda activate srb
cd /root/space_robotics_bench_l
python train_isaacbase_fsppo_joint.py --dry-run --device cpu \
  --num-envs 2 --max-iterations 2 --pmf-ablation pg_pmf
```

GPU 环境中，先移除 `--dry-run --device cpu` 运行上述两迭代 smoke，再运行等步数实验：

```bash
for seed in 0 1 2; do
  for variant in pg_only pg_pmf; do
    python train_isaacbase_fsppo_joint.py \
      --task Isaac-Velocity-Flat-H1 --num-envs 1024 \
      --total-env-steps 100000000 --seed "$seed" \
      --pmf-ablation "$variant" --save-interval 500 \
      --experiment-name h1_pmf_validation --run-name "${variant}_seed${seed}" \
      --log-root /root/space_robotics_bench_l/logs/pmf_ablation
  done
done
```

总步数按完整 rollout 向上取整：1024×32=32768 transitions/iteration，
100M 对应 3052 次更新、100007936 transitions。A/B 使用相同任务、种子和环境覆盖项；
比较相同更新次数的检查点。保存的 `params/agent.yaml` 和 manifest 记录实际配置。
当前 YAML 中 `fsppo_joint_adv_v_loss_coef` 没有被 FSPPOJoint 消费，不能用它作为此消融开关。

`tasks.md` 的 C（交替 PMD+pMF）需要独立的策略改进和投影算法；当前代码没有该算法。
已有 advantage-exp 加权辅助回归也不等同于 C。本实现提供可用的 A/B 验证。

## 检查点评估与共享状态/噪声

将示例中的检查点路径替换为同种子、相同步数的 A/B 产物。第一份检查点采集 raw policy
observations 和显式 Gaussian 初始噪声；后续检查点复用相同 bank。

```bash
python eval_isaacbase_pmf.py \
  --task Isaac-Velocity-Flat-H1 --checkpoint /path/to/A/model_3052.pt \
  --eval-num-envs 8 --eval-episodes 32 --flow-eval-modes random \
  --validation-states 256 --validation-noises 8 --validation-seed 12345 \
  --ode-steps 8 32 128 \
  --state-bank-out /root/space_robotics_bench_l/logs/pmf_ablation/shared_bank.pt \
  --validation-output /root/space_robotics_bench_l/logs/pmf_ablation/eval_A

python eval_isaacbase_pmf.py \
  --task Isaac-Velocity-Flat-H1 --checkpoint /path/to/B/model_3052.pt \
  --eval-num-envs 8 --eval-episodes 32 --flow-eval-modes random \
  --ode-steps 8 32 128 \
  --state-bank-in /root/space_robotics_bench_l/logs/pmf_ablation/shared_bank.pt \
  --validation-output /root/space_robotics_bench_l/logs/pmf_ablation/eval_B
```

评估默认使用保存的 agent 配置；缺少 `params/agent.yaml` 的产物需明确传入
`--fpo-config /path/to/matching.yaml`。`--task`、physics preset 和环境 Hydra 覆盖项仍须匹配
训练。agent 配置的自动读取不等于自动恢复整个训练环境。
新增验证路径可先加 `--dry-run --device cpu` 检查配置，dry-run 不采集样本或验证 checkpoint 权重。

不同归一化器分别将 bank 中的同一份 raw observations 归一化，避免复用 A 的 normalized
observations 来评估 B。共享 bank 保证逐样本状态和噪声配对；状态分布仍来自采集它的策略。
必要时另用 B 的 bank 做反向诊断，以区分状态分布效应。

## 报告指标

每次验证输出 `report.json` 与 `state_bank.pt`，包括检查点和 bank 的来源、配置、
有限值比例、噪声引起的条件动作方差，以及两种分别命名的 ODE 参考场：

- `diagonal_mean`：`v=(z-x_mean(s,z,h=0))/t`，对应 `tasks.md` 的 `v=u(z,t,t)`。
- `auxiliary`：`v=(z-x_instantaneous(s,z,h=0))/t`，对应当前 pMF loss 的 JVP tangent。

`diagnostics.fields.diagonal_mean.resolutions[*].one_step_error.mean_squared_l2`
是任务定义的 `E_flow = mean(||a_one-a_ODE||²)`；`mse` 是再除以动作维度的数值。
所有比较使用相同 normalized 状态和初始噪声，一步调用真实 `transport_actions`。
ODE 在未缩放的 action 坐标中积分，最终应用一次 `actor_scale`；如 wrapper 配置 action clipping，
同时报告最终 clipping 前后的误差，积分过程不裁剪。

ODE 使用递减的几何时间网格与 Heun 法。默认在当前 `pmf_time_eps` 截止，末段用
`z_0=z_eps-eps*v(z_eps,eps)` 外推。因此 8/32/128 个积分区间的 NFE 是 17/65/257，
包括末段外推。`reference_error` 比较各分辨率和最细网格，判断数值离散误差是否足够小；
固定截止时间下的网格收敛不消除截止/末段误差，不能据此宣称精确 ODE 解。
现有 `sampling_steps>1` 使用有限间隔 MeanFlow 跳步，不能替代这里的瞬时场积分。

`closed_loop_evaluation` 单独记录真实完成 episode 的 return、length、数量和 timeout rate。
每种 noise mode 恰好统计 `--eval-episodes` 个 episode；没有真正 `done` 时有界失败，
不会用局部长度计数伪造 episode。未提供 timeout 信息时 timeout rate 为 null。
这一项评估一步 transport policy；它不注入训练用的末端 Gaussian action perturbation。
`random` 每步重采样 latent；`fixed_seed` 每步复用同一批 latent；`zero` 不是动作分布的均值。
timeout rate 表示到达时限，不是任务成功率。

条件动作方差描述噪声敏感性和动作多样性，不证明多模态覆盖。
降低 `E_flow` 也不等于提高回报或完整分布匹配；同时检查完整 episode 指标、有限值比例、
clipping 饱和比例、两种参考场及网格收敛结果。

## CPU 验证

```bash
python -m pytest tests/test_pmf_validation_launcher.py -q
python -m pytest \
  /root/R2A/Algos/fpo-control-saa/isaaclab_experiments/isaaclab_fpo/tests/test_pmf_consistency.py \
  /root/R2A/Algos/fpo-control-saa/isaaclab_experiments/isaaclab_fpo/tests/test_pmf_validation.py \
  /root/R2A/Algos/fpo-control-saa/isaaclab_experiments/isaaclab_fpo/tests/test_checkpoint_evaluation.py -q
```

解析场测试检查积分符号、对角场与辅助场区分、缩放、配对及收敛；工作流测试检查
checkpoint normalizer、共享 bank、报告和状态恢复。CPU 证据不替代 Isaac Sim GPU rollout。
