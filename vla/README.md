# Space Robotics Bench

这个目录把 ACoT-VLA / OpenPI 的标准 VLA 推理与训练流程接到了 Space Robotics Bench。

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
  --host 0.0.0.0 \
  --port 8000 \
  --device cuda:0
```

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

Searched for regex `observ|observation|obs_dict|vla` (`**/vla/**`), 28 results

Searched for regex `observ|observation|obs_dict|vla` (`**/srb/**`), 100 results

Read memory [](file:///root/.vscode-server/data/User/workspaceStorage/eb517b899284a574d0ab0f283eb93510/GitHub.copilot-chat/memory-tool/memories/repo/visual-observations.md)

Read memory [](file:///root/.vscode-server/data/User/workspaceStorage/eb517b899284a574d0ab0f283eb93510/GitHub.copilot-chat/memory-tool/memories/repo/vla-entrypoint.md)

已经定位到控制链路：入口只是把环境切到 `*_visual` 并调用 `srb vla`，真正把环境 observation 发给算法 server 的逻辑在 openpi.py，下一步我直接看这个文件和视觉扩展的 `_get_observations()`。

Searched for regex `class VisualExt|def _get_observations|image_cam_|cam_wrist|cam_base` (`**/srb/**`), 33 results

Read [](file:///root/space_robotics_bench_l/srb/integrations/openpi.py)

Read [](file:///root/space_robotics_bench_l/vla/README.md)

现在需要把“定义点”和“语义”拆开看：一层是 VLA 客户端把整个 obs dict 原样发出去，另一层是具体 visual 任务如何拼出这个 dict。我继续读默认的 `sample_collection_visual` 任务和通用 `VisualExt`。

Searched for regex `def _get_observations\(|return \{|state|proprio|proprio_dyn|state_dyn` (`**/srb/tasks/manipulation/sample_collection/**`), 80 results

Read [](file:///root/space_robotics_bench_l/srb/core/env/common/extension/visual/impl.py#1-1), lines 1 to 220

Read [](file:///root/space_robotics_bench_l/srb/tasks/manipulation/sample_collection/task_visual.py)

Read [](file:///root/space_robotics_bench_l/srb/core/env/manipulation/visual_ext.py)

视觉任务的 obs dict 不是单独重写一套，而是把常规任务观测和视觉扩展观测直接合并；现在我读默认 sample_collection 的低维观测定义，确认每个 key 的具体内容。

Searched for regex `image_cam_base|image_cam_wrist|prompt|infer\(|_prepare_request`, 44 results

Read [](file:///root/space_robotics_bench_l/srb/tasks/manipulation/sample_collection/task.py#700-700), lines 700 to 818

Read [](file:///root/space_robotics_bench_l/srb/tasks/manipulation/sample_collection/task.py#360-360), lines 360 to 520

还有一个关键点要确认：任务代码里先按 `state/proprio/...` 分组返回，但传给 server 时看起来是扁平 dict，所以我补一跳看环境基类在哪把这些组展开。

Searched for regex `flatten|unflatten|observation_group|single_observation_space|observations` (`**/srb/core/env/**`), 17 results

Searched for regex `StepReturn\(|_get_observations\(|observation_space` (`**/srb/tasks/manipulation/sample_collection/task.py`), 1 result

Read [](file:///root/space_robotics_bench_l/srb/core/env/common/base/direct/impl.py#430-430), lines 430 to 560

分组观测确实会在环境基类里做一次扁平化；我补上这段实现，这样能准确解释 server 看到的是“按组拼接后的向量”，还是“保留原字段名的独立张量”。

Read [](file:///root/space_robotics_bench_l/srb/core/env/common/base/direct/impl.py#556-556), lines 556 to 574

现在已经确认 VLA server 看到的是顶层六类 key：四个低维组加相机图像。还差两个细节：图像里到底包含哪些通道、默认是 RGB 还是 RGBD，以及相机配置本身在哪定义。

Searched for regex `data_types|width|height|resolution|CameraCfg\(|spawn=PinholeCameraCfg` (`**/srb/core/**`), 120 results

Searched for regex `class StepReturn|extract_step_return` (`**/srb/**`), 43 results

Read [](file:///root/space_robotics_bench_l/srb/core/env/common/extension/visual/cfg.py)

还差最后一块：`extract_step_return` 的数据结构定义。确认后就能把“任务内部定义”和“发给 server 的最终形态”一一对应起来。

Read [](file:///root/space_robotics_bench_l/srb/tasks/manipulation/sample_collection/task.py#260-260), lines 260 to 340

Read [](file:///root/space_robotics_bench_l/srb/core/env/common/base/direct/impl.py#140-140), lines 140 to 180

Read [](file:///root/space_robotics_bench_l/srb/_typing/step_return.py)

如果要把 `state` 里每个标志量讲清楚，还要补一段 reward/termination 逻辑附近的定义，尤其是 `stable_grasp`、`transport_ready`、`pregrasp_ready`、`success` 这些任务语义字段。

Searched for regex `stable_grasp|transport_ready|pregrasp_ready|success\s*=|distance_xy_to_obj|height_above_obj` (`**/srb/tasks/manipulation/sample_collection/task.py`), 55 results

Read [](file:///root/space_robotics_bench_l/srb/tasks/manipulation/sample_collection/task.py#520-520), lines 520 to 720

图像这边还有一个容易忽略的细节：默认不是纯 RGB，而是把 `rgb` 和 `depth` 合并到同一个张量里。我再补完整个通道拼接实现，避免解释错。

Read [](file:///root/space_robotics_bench_l/srb/core/env/common/extension/visual/impl.py#200-200), lines 200 to 252

README 里把相机键简称成 RGB，但代码路径看起来默认会补上 depth。我再确认一下 `CameraCfg` 的默认 `data_types`，避免把文档语义和实际运行时混在一起。

Searched for regex `class CameraCfg|data_types:|data_types =` (`**/srb/core/sensor/**`), no results

Searched for files matching `**/srb/core/sensor/**/*.py`, 2 matches

`CameraCfg` 是 re-export 出来的，我直接看传感器模块入口确认它来自哪里；如果是外部类，就只基于本仓库对它的覆盖逻辑来说明。

Searched for text `CameraCfg` (`**/srb/**`), 50 results

Read [](file:///root/space_robotics_bench_l/srb/core/sensor/__init__.py)

SRB 这边已经定位完了。为了区分“发出的数据结构”和“算法 server 内部如何消费它”，我最后确认一下当前工作区里有没有 server 侧的 `srb_policy.py`；没有的话我会明确告诉你哪部分只能根据 README 契约来解释。

Searched for files matching `**/srb_policy.py`, no matches

Searched for files matching `**/openpi/**`, no matches

我补一条仓库记忆，记录这个容易踩坑的点：当前 visual VLA 出站图像默认是合并后的 RGBD，而不是纯 RGB 文档表述。

Inserted into memory file [](file:///root/.vscode-server/data/User/workspaceStorage/eb517b899284a574d0ab0f283eb93510/GitHub.copilot-chat/memory-tool/memories/repo/visual-observations.md)

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
