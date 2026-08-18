# Space Robotics Bench

这个目录把 ACoT-VLA / OpenPI 的标准 VLA 推理与训练流程接到了 Space Robotics Bench。

ACoT-VLA 项目地址在: `/root/algos/R2A/Algos/ACoT-VLA_saa`

## 已接入的内容

- `src/openpi/policies/srb_policy.py`
  - 把 SRB 的 flat observation dict 转成模型输入。
  - 兼容 `state` / `proprio`，也兼容视觉任务里的 `image_cam_base` / `image_cam_wrist`。
  - 如果当前 SRB 任务没有相机观测，会自动补零图像，所以非视觉任务也能跑通同一套接口。
- `src/openpi/training/config.py`
  - 增加了 `SRBDataConfig`。
  - 提供了两个模板 config：`pi05_srb` 和 `pi0_fast_srb`。
- `examples/srb/serve.py`
  - 简化版 websocket policy server，避免直接处理 `scripts/serve_policy.py` 的 union CLI。
- `examples/srb/main.py`
  - 直接在 SRB 环境里 rollout，并通过 websocket 请求 policy action chunk。

## 推理

前提：

- 你运行这个脚本的 Python 环境必须已经能正常 import `srb`、`isaaclab`、`isaacsim`。
- 同一个环境里还要能 import 当前仓库的 `openpi` 和 `openpi_client`。
- 推荐直接在 SRB 的 Isaac Sim 环境里执行 `uv pip install -e /media/omnisky/sda/algos/R2A/Algos/ACoT-VLA_saa`。

启动 policy server：

```bash
uv run python examples/srb/serve.py \
  --config-name pi05_srb \
  --checkpoint-dir checkpoints/pi05_srb/<EXP_NAME>/<STEP>
```

启动 SRB rollout：

```bash
uv run python -m srb vla \
  --headless \
  --env srb/sample_collection_visual \
  --prompt "collect the sample" \
  --host 127.0.0.1 \
  --port 8899 \
  --device cuda:0
```

如果 policy server 不在本机，把 `127.0.0.1` 换成服务端的实际 IP。`0.0.0.0` 只适合服务端绑定监听，不适合客户端连接。

也可以继续执行 `python vla/main.py ...`。该脚本现在会直接转发到同一个 `python -m srb vla` 入口，避免再维护一套单独的启动链路。

如果你想在仓库内用一个更适合调试的本地 wrapper，可以直接运行：

```bash
python vla/vlamain.py --dry-run
python vla/vlamain.py --env sample_collection env.domain=moon env.sample=moon_rock
```

这个 wrapper 默认会：

- 优先把 `--env sample_collection` 这类环境切到 `sample_collection_visual`
- 默认打开 `--enable-cameras`
- 保留额外的 Hydra 覆盖参数透传给 `srb vla`
- 支持 `--no-visual`、`--no-enable-cameras` 和 `--dry-run` 做快速切换与检查

如果你传入的是非 visual 环境，例如 `--env srb/sample_collection`，VLA rollout 现在会优先自动切到对应的 `srb/sample_collection_visual`。只有在 visual 变体未注册时，才会回退到原环境。

说明：

- 对 manipulation visual 任务，SRB 会返回 `image_cam_base` 和 `image_cam_wrist`，adapter 会把它们映射到模型的 base / wrist image slots。
- 对非 visual 任务，adapter 会自动补零图像；这能跑通接口，但效果会明显依赖你训练时是否也是无视觉输入。
- 当前脚本把 SRB 环境固定成 `num_envs=1`，因为 OpenPI 的 policy server 接口是单条 observation 推理。
- 在没有 `DISPLAY` 的容器环境里，`srb vla` 会自动退回到 headless 启动，避免 GUI Kit 启动直接崩溃。

## 训练

### 1. 数据格式

训练仍然走 OpenPI 原本的 LeRobot 数据管线。你的 SRB 数据集建议保存这些 key：

- `proprio`: 机器人本体低维状态，推荐优先保留。
- `state`: 任务相关状态。
- `state_dyn`: 可选，动态状态。
- `proprio_dyn`: 可选，动态本体状态。
- `image_cam_base`: 可选，SRB base camera RGB。
- `image_cam_wrist`: 可选，SRB wrist camera RGB。
- `actions`: 环境动作，形状应为 `(action_dim,)`。
- `task`: 任务文本。配合 `prompt_from_task=True` 自动生成 prompt。

当前模板 config 默认：

- 输出动作维度 `action_dim=7`
- 状态拼接顺序 `("proprio", "state")`
- 图像 key `("image_cam_base", "image_cam_wrist")`

如果你的 SRB 任务动作维度不是 7，或者你想把 `state_dyn` / `proprio_dyn` 也拼进去，需要在 `SRBDataConfig` 里改这些字段。

### 2. 归一化统计

```bash
uv run python scripts/compute_norm_stats.py --config-name pi05_srb
```

如果你把数据集 repo id 改成自己的，需要同时覆盖对应字段，例如：

```bash
uv run python scripts/compute_norm_stats.py \
  --config-name pi05_srb \
  --data.repo-id your_hf_username/srb_manipulation
```

### 3. 启动训练

```bash
uv run python scripts/train.py \
  pi05_srb \
  --exp-name srb_pi05 \
  --data.repo-id your_hf_username/srb_manipulation \
  --overwrite
```

也可以使用 FAST 模型：

```bash
uv run python scripts/train.py \
  pi0_fast_srb \
  --exp-name srb_fast \
  --data.repo-id your_hf_username/srb_manipulation \
  --overwrite
```

## 重要限制

- 这次接入的是标准 VLA 训练 / 推理路径，不是 ACoT-VLA 的 `coarse_actions` 三路训练路径。
- 当前 `SRBInputs` 会把选中的状态向量裁到模型 `action_dim`，再做 zero padding。
- 对 `pi05_srb`，模型内部 state 维度是 32。如果你拼接后的 SRB 状态超过 32 维，超出的部分会被截断。
- 如果你不想接受截断，可以在 `SRBDataConfig(..., strict_state_dim=True)` 下显式报错，然后自己重新选 state keys。

## 推荐做法

- manipulation 任务优先用 visual 版本环境，例如 `*_visual`；即使传非 visual 名称，`srb vla` 现在也会优先切到对应 visual 变体，这样训练和推理都能得到稳定的 base / wrist 视角。
- 状态优先级建议是 `proprio` > `state` > `proprio_dyn` > `state_dyn`。
- 如果你后面要把 SRB 接到 ACoT-VLA 的 coarse action 训练，需要继续扩展 data loader，让 batch 从 `(obs, actions)` 变成 `(obs, actions, coarse_actions)`。




### 1

先沿着当前入口 vlamain.py 往下定位 VLA rollout 的实际观测构造点，再结合仓库里已有的视觉观测说明，确认传给算法 server 的 `observ` 是在哪一层定义的、字段各自含义是什么。


不是在 vlamain.py 里定义的。这个文件现在只负责把参数转发到 SRB 的 VLA 子命令。当前这条视觉 VLA 执行流里，发给算法 server 的 observ 定义链路是这样的：

- 发包位置在 openpi.py 和 openpi.py。这里会把环境返回的 observation 字典转成 numpy，再额外塞一个 prompt 字符串，然后调用 server 的 infer。
- 视觉任务把“普通任务观测”和“相机观测”合并，定义在 task_visual.py。
- 低维任务观测本身定义在 task.py、task.py、task.py、task.py。
- 图像观测由通用视觉扩展生成，定义在 impl.py。相机本体是 manipulation 环境里定义的 base 和 wrist 两个相机，见 visual_ext.py。
- 低维观测在真正发出去前，会按组扁平化成顶层的 state、state_dyn、proprio、proprio_dyn 四个向量，规则在 impl.py。这里是“按子字段名字排序后拼接”。

对当前默认的 sample_collection_visual 来说，server 实际收到的顶层键基本就是：state、state_dyn、proprio、proprio_dyn、image_cam_base、image_cam_wrist，再加一个 prompt。含义分别是：

- “state” 是任务语义摘要向量。它包含接触力统计、末端执行器与目标物/目标点的相对几何关系、夹爪张开度，以及任务阶段标志。比如：
  contact_forces_mean_robot / contact_forces_mean_end_effector / contact_forces_mean_end_effector_collision 是接触力统计；
  distance_xy_end_effector_to_obj、height_above_obj、height_above_terrain 是抓取前对位和安全高度；
  sample_lift_height、distance_obj_to_target、sample_lin_speed、sample_ang_speed 是目标物当前搬运状态；
  tf_pos_end_effector_to_obj、tf_pos_obj_to_target 是相对位置；
  tf_rot6d_end_effector_to_obj、tf_rot6d_obj_to_target 是 6D 旋转表示；
  gripper_aperture 是夹爪开合程度；
  pregrasp_ready、stable_grasp、transport_ready、success、end_effector_collision_undesired 是任务进度和安全标志。几个关键布尔量的判定在 task.py、task.py、task.py、task.py。
- “state_dyn” 是更原始的动态量。当前任务里主要是目标物线速度、角速度，以及机器人和末端执行器的未汇总接触力张量。
- “proprio” 是机器人本体低维状态里的静态位姿部分。当前任务里是末端执行器前向运动学位置 fk_pos_end_effector 和 6D 朝向 fk_rot6d_end_effector。
- “proprio_dyn” 是关节级动态本体状态。包括机器人和夹爪的归一化关节位置、关节速度、机器人关节加速度、机器人施加扭矩。
- “image_cam_base” 和 “image_cam_wrist” 是视觉观测。它们由 cfg.py 和 cfg.py 的默认配置控制，默认分辨率是 64×64，默认数据类型是 rgb 和 depth；随后在 impl.py 按通道拼接。所以按当前代码，发出去的通常不是纯 RGB，而是“RGB + 归一化 depth”的 4 通道 uint8 图像。base 相机更偏全局/底座视角，wrist 相机是末端执行器近景视角。

有一个你需要特别注意的点：文档在 README.md 和 README.md 里把 image_cam_base、image_cam_wrist 简写成给模型的 base / wrist 图像输入，但按当前仓库的实际运行代码，SRB 出站这两个键默认更接近 RGBD。至于算法 server 端到底是直接吃 4 通道，还是只取前三个 RGB 通道，那部分代码不在当前工作区里，当前仓库只能确定“发出去的是什么”，不能确认“server 端怎么解”。

如果你要继续往下钻，我可以接着做两件事：
1. 把 sample_collection_visual 的 state、state_dyn、proprio、proprio_dyn 按实际拼接顺序和每段维度展开成一张完整表。
2. 如果你把算法 server 那边的仓库也打开，我可以继续顺着 server 端的解包和输入适配逻辑追到模型前。

### 2

先把路线分成两条，不要一开始混在一起做。

当前这套接入里，SRB 侧已经把标准 OpenPI 风格的 VLA rollout 接通了，训练说明和输入约束在 README.md，在线 rollout 的实际链路在 openpi.py。但 SRB 自带的 collect 流在仿真 agent 侧还没实现，当前就是 NotImplemented，见 __main__.py。所以最稳的做法不是直接冲“完整 ACoT”，而是先做一条能闭环的标准 VLA 训练流，再把 coarse action 这条链补上。尤其要注意，你现在外部 ACoT-VLA 仓库里虽然已经有 SRB 配置和一些 coarse action 痕迹，但数据加载、norm stats 和训练分支之间还有接口不一致，直接照 README 盲跑风险很高。

**建议训练流**
1. 先做单任务、单机器人、视觉版基线。优先从 sample_collection_visual 开始，不要一上来混 debris capture、assembly、mobile manipulation。空间机器人 VLA 最难的是长时序和分布漂移，不是任务数量。
2. 自己补一层数据采集器。最简单的方式是复用 openpi.py 和 openpi.py 里那种 gymnasium.make + step 循环，在每一步把 prompt、obs dict、action、done、task id 全部记下来。数据源建议混三类：人遥操作示教、脚本化 teacher、已有 RL expert rollout。
3. 数据格式先严格对齐 LeRobot。最少保留 proprio、state、image_cam_base、image_cam_wrist、actions、task；如果后面要做真正 ACoT，再额外加入 coarse_actions。这里不要混用两种动作语义，整个数据集必须统一成同一种 action 表达。
4. 先跑标准 VLA 微调，不要先用快速版。原因很简单：当前 SRB adapter 会把拼接后的状态向量裁到模型允许的维度，pi05_srb 至少还能容 32 维状态，而 pi0_fast_srb 会更早遇到状态瓶颈。开发阶段应该强制开启状态维度检查，而不是接受静默截断。
5. 把验证前置，而不是把训练前置。先做小数据集冒烟，确认三件事：训练能读到 norm stats，batch 里的 state 和 image 没被错误裁切，server rollout 能在 SRB 里稳定闭环。
6. 基线稳定后，再补 coarse action。推荐离线生成，不要先改在线推理：先从成功轨迹里根据阶段规则或几何阈值生成 coarse_actions，再把训练 batch 扩成 observation、actions、coarse_actions 三元组，最后再切到 ACoT 分支。

**针对空间机器人的模型设计**
1. 先改状态输入，不要再靠截断。空间任务里最有价值的不是原始大向量，而是相对位姿、接触、速度和任务阶段。建议把 state 设计成两级：一级是高价值摘要特征，二级是动态补充特征；再用一个小 projector 或 state token 化把它压到模型输入维，而不是直接裁掉尾部。
2. 两视角一定保留，深度信息最好显式建模。SRB 现在是 base 和 wrist 两路视角，这对抓取、对接、插装都很关键。若继续沿用当前 RGB-only 适配，你会丢掉对低纹理、强阴影、反光表面的判别力。更合理的是把 depth 作为单独分支或 4 通道 stem，而不是默认忽略。
3. coarse action 不要做成“抽象标签”，要做成可执行的子目标。对采样抓取类任务，粗粒度链条可以是 approach、align、pregrasp、close、lift、transport、place。对轨道捕获类任务，可以是 relative approach、velocity nulling、capture、stabilize。这样 coarse reasoning 才真的对空间机器人有增益。
4. 训练策略用课程式，不要全任务混训起步。推荐顺序是：单任务单域，单任务多资产，单任务多域，多个相邻 manipulation 任务，最后再加入 mobile manipulation 和 orbital 任务。跨任务泛化建立在动作结构共享上，不是建立在 prompt 堆料上。
5. 语言设计不要只写一句自然语言。空间机器人任务很受场景条件影响，建议把 domain、object、goal、safety constraint 编到 prompt 模板里，比如把月球/火星、目标类型、碰撞约束、目标姿态偏好一起写进去。
6. 评价指标不要只看 success rate。你至少要同时看碰撞率、接触稳定性、阶段完成率、动作平滑度、prompt 改写鲁棒性和跨域泛化。SRB 任务本身就有不少阶段性信号，适合拿来做更细的诊断。

**当前最该先修的点**
- SRB 自带 collect 不能直接用，数据采集需要你自己补。
- 外部 ACoT 仓库当前的标准 SRB 配置本质上还是标准 VLA 微调，不是完整 ACoT 训练。
- 外部仓库里的 norm stats 脚本、data loader 和 coarse action 训练分支目前不完全一致，建议先修通这条窄链路，再上大规模训练。
- 现有 SRB adapter 的状态截断会成为性能上限，尤其对空间场景的接触和相对位姿推理影响很大。

如果你要继续，我建议直接做下面两件事之一：
1. 我先帮你补一套 SRB 到 LeRobot 的数据采集与导出脚本，把标准 pi05_srb 基线跑通。
2. 我直接帮你改外部 ACoT-VLA 仓库，把 coarse_actions、norm stats 和 data loader 这条训练链打通。

我继续把两块补全：先把 sample_collection_visual 的四组低维观测按实际拼接顺序和维度展开，再顺着外部 ACoT-VLA 仓库里的 srb_policy.py 追到 server 端的解包与模型输入适配逻辑。

已完成: *定位 ACoT-VLA 可见配置面* (2/4)

先更正上一条里最关键的一处：当前 SRB 视觉观测的真实出站键不是 image_cam_base 和 image_cam_wrist，而是 image_base 和 image_wrist。命名逻辑在 impl.py 和 str.py，其中 sanitize_cam_name 会把 cam_base 变成 base、cam_wrist 变成 wrist。视觉任务本身只是把普通任务观测和视觉扩展观测合并，见 task_visual.py。低维组内的最终拼接顺序由 impl.py 决定，规则就是对每组子键按字母序排序后再拼接。

**1. 观测展开**
我做了两次实际 reset 测量。

1. sample_collection_visual 默认机器人是 Franka，默认来源在 env.py。实测 shape 是：
state 41，state_dyn 39，proprio 9，proprio_dyn 32，image_base 和 image_wrist 都是 64×64×4，action space 是 7。
2. 通过 vlamain.py 那条本地 wrapper 默认会强制 env.robot=ur5+robotiq_hand_e。实测 shape 是：
state 41，state_dyn 36，proprio 9，proprio_dyn 28，image_base 和 image_wrist 都是 64×64×4，action space 是 8。

四组低维观测的定义位置在 task.py。

state 的实际拼接顺序和维度，两种机器人相同，总计 41 维：

| 顺序 | 字段 | 维度 |
|---|---|---:|
| 1 | contact_forces_mean_end_effector | 3 |
| 2 | contact_forces_mean_end_effector_collision | 3 |
| 3 | contact_forces_mean_robot | 3 |
| 4 | distance_obj_to_target | 1 |
| 5 | distance_xy_end_effector_to_obj | 1 |
| 6 | end_effector_collision_force_max | 1 |
| 7 | end_effector_collision_undesired | 1 |
| 8 | gripper_aperture | 1 |
| 9 | height_above_obj | 1 |
| 10 | height_above_terrain | 1 |
| 11 | pregrasp_ready | 1 |
| 12 | sample_ang_speed | 1 |
| 13 | sample_lift_height | 1 |
| 14 | sample_lin_speed | 1 |
| 15 | stable_grasp | 1 |
| 16 | success | 1 |
| 17 | tf_pos_end_effector_to_obj | 3 |
| 18 | tf_pos_obj_to_target | 3 |
| 19 | tf_rot6d_end_effector_to_obj | 6 |
| 20 | tf_rot6d_obj_to_target | 6 |
| 21 | transport_ready | 1 |

proprio 的实际拼接顺序和维度，两种机器人相同，总计 9 维：

| 顺序 | 字段 | 维度 |
|---|---|---:|
| 1 | fk_pos_end_effector | 3 |
| 2 | fk_rot6d_end_effector | 6 |

state_dyn 的实际拼接顺序和维度：

| 顺序 | 字段 | Franka 默认 | UR5 + Robotiq |
|---|---|---:|---:|
| 1 | contact_forces_end_effector | 9 | 9 |
| 2 | contact_forces_robot | 24 | 21 |
| 3 | sample_ang_vel | 3 | 3 |
| 4 | sample_lin_vel | 3 | 3 |
| 合计 |  | 39 | 36 |

这里 contact_forces_robot 的差异来自接触传感器监控的刚体数不同。机器人接触传感器挂载逻辑在 env.py。UR5 和 UR5RobotiqHandE 的定义在 universal_robots.py 和 universal_robots.py。Robotiq 两个滑块关节定义在 robotiq.py。

proprio_dyn 的实际拼接顺序和维度：

| 顺序 | 字段 | Franka 默认 | UR5 + Robotiq |
|---|---|---:|---:|
| 1 | joint_acc_robot | 7 | 6 |
| 2 | joint_applied_torque_robot | 7 | 6 |
| 3 | joint_pos_end_effector_normalized | 2 | 2 |
| 4 | joint_pos_robot_normalized | 7 | 6 |
| 5 | joint_vel_end_effector | 2 | 2 |
| 6 | joint_vel_robot | 7 | 6 |
| 合计 |  | 32 | 28 |

图像部分的 runtime 事实也需要一起纠正：
1. 实际键名是 image_base 和 image_wrist，不是 README 里写的 image_cam_base 和 image_cam_wrist。
2. 实际张量是 64×64×4，因为 impl.py 会把 rgb 和 depth 合并到一个 image_* 张量里。
3. 所以当前 SRB 出站图像是 RGBD 风格的 4 通道 uint8，而不是纯 RGB。

**2. Server 端解包与输入适配**
现在这条链路可以明确写成：

1. SRB rollout 端在 openpi.py 直接把环境 observation dict 原样转成 numpy，再塞一个 prompt，发给 websocket server。这里没有任何图像键重命名。
2. server 端的 serve 入口会根据 config_name 取训练配置，然后创建 trained policy。外部 ACoT-VLA 仓库里这一步会把训练时的数据 transforms 原样复用于推理。
3. 外部仓库的 SRBDataConfig 默认 observation_keys 是 proprio、state，默认 image_keys 还是 image_cam_base、image_cam_wrist；pi05_srb 和 pi0_fast_srb 也都沿用了这个设置。
4. create_trained_policy 的输入变换顺序是：
InjectDefaultPrompt → SRBInputs → Normalize → 模型自己的 transforms。
5. Policy.infer 会先跑完整个输入 transform 链，再 batch 化成 Observation，然后调用 model.sample_actions，最后再跑输出 transform 链。

外部仓库里 SRBInputs 这一层的真实行为有四个关键点：

1. 它按 observation_keys 的声明顺序拼接，不是按组名排序。默认就是先 proprio，再 state。
2. 它对图像键做的是精确匹配。如果没找到，就直接补零图像。
3. 它会把超过 3 通道的图像截到前 3 通道，所以即使你把键名改对，当前深度通道也会被丢掉。
4. 它会在进入模型前把 state 截断到模型 action_dim，再做 zero pad。

这四点叠在一起，得到三个直接后果：

1. 当前 SRB runtime 发的是 image_base 和 image_wrist，但外部 SRBDataConfig 和 SRBInputs 默认找的是 image_cam_base 和 image_cam_wrist。
结果不是“图像错位”，而是更糟的“图像直接没被取到，自动补零”。
2. 即使把图像键改对，SRB 真实出站是 4 通道 RGBD，SRBInputs 仍会把第 4 个通道裁掉，只留下 RGB。
3. 默认 state 拼接是 proprio 9 维 + state 41 维 = 50 维。
对 pi05_srb，进入 SRBInputs 时会截到 32 维，所以只保留完整 proprio 9 维和 state 的前 23 维。
这意味着它会保留到 tf_pos_end_effector_to_obj 的第 1 个分量为止，后面的相对位置剩余分量、两段 6D 旋转、transport_ready 全都会被裁掉。
对 pi0_fast_srb 更极端，因为模型 action_dim 是 7，默认配置下它只会保留拼接向量的前 7 维，也就是 proprio 的前 7 维，连完整 proprio 都留不全，更不用说任务 state。
4. rollout 还会在 openpi.py 检查 policy 输出动作维度是否等于环境 action space。
而 vlamain.py 默认强制的 UR5 + Robotiq 配置，实测 action space 是 8，不是 README 里写的 7。
所以当前外部 pi05_srb 和 pi0_fast_srb 这两个 7 维配置，和 wrapper 默认 UR5 rollout 也是不闭环的。

压缩成一句话就是：
当前这条 SRB ↔ ACoT-VLA 接入里，真正会影响结果的不是“模型有没有读到 depth”，而是更前面的两个兼容性问题：
图像键名已经漂移，动作维度也已经漂移。

如果你要继续推进，我建议下一步只做两件具体事：

1. 我直接帮你把 README.md 和外部 ACoT-VLA 仓库里的 SRBDataConfig 默认值统一到当前真实 runtime：
image_base、image_wrist，UR5 8 维动作。
2. 我继续把 pi05_srb 当前 50→32 的截断向量精确展开成“哪些维度被保留，哪些维度被丢弃”的一张表，然后顺手给你一个更合理的 observation_keys 和 state projector 设计。
