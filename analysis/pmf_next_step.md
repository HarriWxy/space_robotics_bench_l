# pMF Next Step 实验实现（2026-10-11）

对应外部 `isaaclab_experiments/tasks.md` 的 `## Next Step`。新增入口为
`train_isaacbase_pmf_next.py`，旧 `train_isaacbase_fsppo_joint.py` 与旧长跑 YAML 保留。
本版提供可执行研究原型，尚无真实 H1 GPU 训练收益结论。

## 四组训练

| 组 | `--variant` | Actor 更新 | 策略类 | 调度 / 约束 |
|---|---|---|---|---|
| A | `joint_pg` | Joint Gaussian PPO，pMF 辅助系数 0 | LearnedStdPMFActorCritic | conditional Gaussian KL 自适应 LR，无旧 map budget |
| B | `pmd` | Q 驱动候选重加权 → 纯 weighted pMF projection | PMFActorCritic | 固定 LR，无 trust region |
| C | `pmd_tr` | 同 B | PMFActorCritic | 累计平方 map Gaussian KL 探针约束、回溯与完整 Adam 回滚 |
| D | `joint_pmf` | 同 A，加原行为目标 pMF 辅助系数 0.01 | LearnedStdPMFActorCritic | 同 A |

共同配置在 `hyperparams/pmf_next_step.yaml`：Flat-H1/PhysX，256 环境 × 24 步，
actor/critic hidden widths `[128,128,128]`，ELU，无 observation normalization，
γ=0.99、λ=0.95、clip=0.2、clipped value loss，初始 LR=0.001，无 EMA，
不随机推进 episode 起点。一次推理 NFE=1，末端 Gaussian 默认固定 σ=0.1、不退火。
15M 预算向上取整为 **2442 次更新、15,003,648 个环境 transitions**。
A/D 每轮 5 epochs × 4 minibatches；B/C 投影同样最多 20 次 actor 更新。
C 可因预算拒绝提前结束；B/C 前 10 次更新只拟合 Q/V，这些环境步也计入总预算。
投影 microbatch=256，仅控制显存，累积完整 macrobatch 梯度后才执行 Adam。

同样的 hidden widths 不等于相同参数量：flow 额外输入 latent/time，输出双头；
B/C 另有 action Q 网络。不能用更新次数替代 wall-clock 计算成本，也不能把本配置
与旧 `[512,256,128]`、大噪声退火长跑的差距解释为单变量 PMD 收益。

```bash
conda activate srb
cd /root/space_robotics_bench_l

# 先做每组 2 环境、2 次更新的 GPU 启动检查；按需选择实际可用 GPU。
python train_isaacbase_pmf_next.py --variant pmd --num-envs 2 --max-iterations 2 --device cuda:0

# PMD至少12次更新才会越过默认10次Q/V warmup，检查实际投影。
python train_isaacbase_pmf_next.py --variant pmd --num-envs 2 --max-iterations 12 --device cuda:0

# 四组与三个独立 seed 顺序运行；日志分别进入 logs/pmf_next_step/<variant>/。
for seed in 0 1 2; do
  for variant in joint_pg pmd pmd_tr joint_pmf; do
    python train_isaacbase_pmf_next.py --variant "$variant" --seed "$seed" \
      --total-env-steps 15000000 --run-name "${variant}_s${seed}" --device cuda:0
  done
done
```

A/D 可额外加 `--learn-terminal-std`，使用带界限的逐动作尺度与 0.01 conditional
entropy 系数，σ 随 actor 快照/checkpoint 一起保存。默认固定 σ 时 entropy 项为常数。
学习 σ 会改变探索分布，应使用不同 run name；B/C 不接受此选项。
评估时 `random` 使用 latent transport，无额外 terminal Gaussian；探索协方差诊断
则明确包含该 Gaussian。完整控制评估与训练采样分布分别报告。

重新跑 Gaussian PPO 参考可以沿用官方 runner，并对齐本轮预算、seed、起点设置：

```bash
python train_isaacbase.py --task Isaac-Velocity-Flat-H1 --num-envs 256 \
  --total-env-steps 15000000 --seed 0 --run-name ppo_next_s0 --device cuda:0 \
  agent.init_at_random_ep_len=false \
  'agent.actor.hidden_dims=[128,128,128]' 'agent.critic.hidden_dims=[128,128,128]' \
  agent.actor.obs_normalization=false agent.critic.obs_normalization=false \
  agent.algorithm.learning_rate=0.001 agent.algorithm.schedule=adaptive \
  agent.algorithm.entropy_coef=0.01 agent.algorithm.desired_kl=0.01
```

原生 PPO 的 Gaussian 初始 std=1.0 且可学习，flow 的总方差来自 latent 与 σ=0.1
两部分；这仍是重要差别，需要下面的同状态协方差诊断，而不能声称探索已完全一致。

## PMD 的实际数据流与限制

`PMFPMD` 复用 Joint rollout 的显式 latent/action 存储，**替换整个 actor update**。
现有 V(s) 只做 GAE bootstrap；新增 Q(s,a) 用 rollout 中的 raw public action 和
冻结的 λ returns 拟合。Q/V/actor 是三个独立 optimizer。
PMD 打开 native `compute_final_obs=True`，wrapper 保存重置前 critic 观测，runner
使用冻结 normalizer 变换 terminal 观测，超时 reward 才加 γV(final state)。
A/D 与当前原生 RSL PPO 保留现有以 pre-action V 近似超时 bootstrap 的惯例；
这是 B/C 与 PPO 对照间另一个明确实现差异。

更新次序：拟合 V/Q → 冻结旧 actor 和已拟合 Q → 每状态采样 M=8 个旧均值 →
每个均值用 4 个独立末端 Gaussian 样本计算平滑 Q → 状态内 softmax。
低 ESS 状态增温至 `ESS/M >= 0.5`，整轮投影候选/权重不变。
每次投影独立采样 flow 起点及 (r,t)，绝不复用生成候选的 latent。
回归目标是候选**均值**，末端 Gaussian 在 rollout 中重新加入。
没有 PPO ratio、advantage-weighted actor surrogate 或另外叠加的 Joint pMF loss。

Q holdout 是当前 rollout 的随机 10% transitions，可能存在轨迹相关性；
报告 `pmd/q_validation_mse`、`pmd/q_validation_explained_variance`，但这些不能证明
离线候选动作 Q 可靠。`pmd/weighted_q_gain` 是估计 Q 的提升，需用真实完整 episode
确认收益。温度只自动增加，不自动减小。
C 的 KL 为同 latent 的平方 public-map 差 / (2σ²)，约束固定探针平均值，
每轮另抽 audit 探针报告均值/p95；不保证每状态 KL 或总体 marginal mixture KL。
仅支持单 learner device；尚不提供多 GPU Q/V 同步实现。

TensorBoard 同时看 reward、XY/yaw 误差、episode duration、termination，以及：

- Q holdout MSE/explained variance、`pmd/candidate_q_std`、`pmd/weighted_q_gain`；
- `pmd/ess_fraction_min/mean`、`pmd/weight_max`、温度；
- `pmd/u_loss`、`pmd/v_loss`、`pmd/jvp_norm`；
- C 的 probe/audit conditional joint KL、accepted/attempted projection steps；
- `pmd/actor_optimizer_trials`、`pmd/projection_flow_examples`、Q/V optimizer steps 与 update time。

resume 必须读已解析配置，Q/V 与三个 optimizer、温度和 counters 一并恢复：

```bash
python train_isaacbase_pmf_next.py --variant pmd --resume \
  --checkpoint /path/to/pmd/run/model_250.pt --total-env-steps 15000000 --device cuda:0
```

`--total-env-steps` 表示最终总预算。`--variant` 在 resume 只选择默认日志目录，
真正的 policy/algorithm 从 checkpoint 旁的 `params/agent.yaml` 重建。
默认保留 checkpoint 的 seed，只有显式 `--seed` 才覆盖。
使用原来的 num-envs、task、physics；不在 resume 再加 `--learn-terminal-std`。

## 共享 PPO 状态、协方差与教师蒸馏

先选择经过完整 episode 验证的 PPO checkpoint。下面现有路径只是已验证可加载的
示例，不代表该 checkpoint 最优。collector 使用 frozen teacher 的完整 Gaussian
采样动作，保存 RAW 观测、噪声、关节/观测/动作语义及采集步号；采集不是性能评估。

```bash
PPO_CKPT=/root/isaaclab/logs/rsl_rl/h1_rough_ablation/2026-10-11_16-24-07_srb_h1_ppo_baseline_abla/model_500.pt
FLOW_CKPT=/path/to/flow/run/model_250.pt
BANK=logs/pmf_next_step/diagnostics/ppo_states.pt

python collect_isaacbase_ppo_states.py --teacher-checkpoint "$PPO_CKPT" \
  --output "$BANK" --num-states 256 --num-envs 8 --warmup-steps 256 \
  --collection-steps 512 --seed 12345 --device cuda:0 --headless

# 相同固定状态各采样256次。默认online权重；--pmf-weights evaluation选择EMA评估权重。
python diagnose_isaacbase_pmf_actions.py --teacher-checkpoint "$PPO_CKPT" \
  --pmf-checkpoint "$FLOW_CKPT" --state-bank "$BANK" \
  --output-dir logs/pmf_next_step/diagnostics/covariance --samples-per-state 256 --device cpu

# Torch-only：PPO随机目标动作与新flow起点独立采样，纯普通pMF loss，无RL loss。
python distill_isaacbase_pmf.py --teacher-checkpoint "$PPO_CKPT" --state-bank "$BANK" \
  --output-dir logs/pmf_next_step/teacher_student --updates 5000 \
  --batch-size 256 --flow-samples 4 --ode-steps 8 32 128 --device cuda:0
```

协方差报告分别给出 PPO full Gaussian、pMF latent-only、pMF full，
以及 `Cov_total_theory = Cov_latent + diag(terminal_std_per_action²)`。
使用每状态条件协方差，不混入不同状态均值的变化；Monte Carlo full covariance
与理论值允许有限采样误差。std 还原到 checkpoint 已完成步数处的下一 rollout 边界。
`independent_same_state_action_delta_rms` 是同状态独立样本间的 RMS，
不能替代跨时间的动作抖动指标。

蒸馏保存 `model_distilled.pt`、resolved `params/agent.yaml`、teacher env/config/provenance
和 loss/分布误差。完整 teacher Gaussian 由 student latent transport 拟合，student
terminal noise=0，critic 不训练，仅用于模型格式兼容。离线固定状态覆盖有限，
成功蒸馏不保证闭环控制；这项诊断也不是从零训练的样本效率比较。

```bash
# 同一共享RAW bank做E_flow/ODE验证，同时统计真实完成的32个episode。
python train_isaacbase_fpo.py --eval --pmf-validation --use-checkpoint-config \
  --task Isaac-Velocity-Flat-H1 --checkpoint "$FLOW_CKPT" \
  --state-bank-in "$BANK" --eval-episodes 32 --flow-eval-modes random \
  --ode-steps 8 32 128 --device cuda:0 --headless

python train_isaacbase_fpo.py --eval --pmf-validation --use-checkpoint-config \
  --task Isaac-Velocity-Flat-H1 \
  --checkpoint logs/pmf_next_step/teacher_student/model_distilled.pt \
  --state-bank-in "$BANK" --eval-episodes 32 --flow-eval-modes random \
  --ode-steps 8 32 128 --device cuda:0 --headless
```

判别顺序：教师动作分布可否拟合 → E_flow 与闭环表现 → A 的噪声/Joint KL →
B/C 的真实任务收益和 Q 可靠性 → 多 seed 同环境步与计算时间比较。
训练 tag `Metrics/success_rate` 仍是速度达标代理，不能当作完整20s episode成功率。

## 本轮验证

四组 native Flat-H1 配置解析与预算 dry-run，真实 PPO checkpoint/配置 CPU 重建，
新增算法的真实 CPU rollout、JVP/backward、动作 Q 排序、ESS、冻结目标/新噪声、
trust rollback、超时 bootstrap 与 checkpoint resume 均有行为测试。
目前针对新增路径及相关旧实现的103项 CPU 测试通过，新增代码 Ruff 与编译检查通过。
教师/协方差 CLI 使用合成 RAW 状态只验证调用链，不能作为 H1 探索分布结果。
当前工具环境 CUDA 探测返回 Error 304、device_count=0；尚未执行真实 Isaac Sim GPU
smoke 或15M学习实验。上述 GPU 命令需在用户可用训练环境运行。
