# 训练历史记录

srb agent eval --algo sbx_ppo --env sample_collection ++agent.load_from=/root/ws/logs/sample_collection/sbx_ppo/20260427T225949/ckpt/ ++env.domain=moon ++env.robot=ur5+robotiq_hand_e ++env.sample=moon_rock ++env.num_envs=1 ++env.observations.rotation_type=quaternion ++env.sim.gravity=[0.0,0.0,-1.62] ++env.sim.physx.solver_type=1 ++env.sim.physx.collision_system=1 ++env.sim.physx.num_position_iterations=32 ++env.sim.physx.num_velocity_iterations=8 ++env.sim.physx.enable_ccd=True ++env.sim.physx.ccd_threshold=0.001 ++env.sim.physx.solve_articulation_contact_last=True ++env.sim.physx.enable_gpu_dynamics=True ++env.scenery.moon_surface.spawn.collision_props.collision_enabled=True ++env.scenery.moon_surface.spawn.collision_props.collision_approximation=triangleMesh ++env.scenery.moon_surface.spawn.collision_props.static_friction=4.0 ++env.object.spawn.collision_props.collision_enabled=True ++env.object.spawn.collision_props.collision_approximation=convexHull ++env.object.spawn.rigid_props.enable_ccd=True ++env.object.spawn.rigid_props.mass=1.5 ++env.robot.spawn.collision_props.collision_enabled=True ++env.robot.spawn.keep_all_collisions=True ++env.robot.spawn.enable_articulation_contacts=True ++env.robot.actuators.hand.stiffness=10000000.0 ++env.robot.actuators.hand.damping=100000.0 --video --livestream 2



Hydra 是一个开源的 Python 框架，旨在简化复杂应用程序的配置管理，特别是在深度学习和机器学习项目中。



## 奖励函数设置

[task.py](srb/tasks/manipulation/sample_collection/task.py)

### 这里奖励函数的结构

`_compute_step_return()` 里把奖励/惩罚分成几类，最后返回一个 `StepReturn`，其中：

- `state` / `state_dyn` / `proprio` / `proprio_dyn`：是观测数据
- 第二个字典：是各种奖励/惩罚项
- `termination` / `truncation`：是否结束 / 截断

---

### 主要惩罚项

1. `penalty_action_rate`
   - 目标：让动作变化平滑
   - 公式：`-0.5 * mean((act_current - act_previous)^2)`

2. `penalty_joint_torque`
   - 目标：惩罚大电机力矩
   - 公式：
     - 先把 `joint_applied_torque_robot` 限制到 `[-50, 50]`
     - 再做 `-0.000025 * sum(torque^2)`
     - 最终最小值 clamp 到 `-4.0`

3. `penalty_joint_acceleration`
   - 目标：惩罚大关节加速度
   - 公式：
     - 先把 `joint_acc_robot` 限制到 `[-20, 20]`
     - 再做 `-0.0005 * sum(acc^2)`
     - 最小值 clamp 到 `-4.0`

4. `penalty_undesired_robot_contacts`
   - 目标：避免机器人发生过大碰撞
   - 公式：如果机器人接触力最大值超过 `10`，则惩罚 `-1.0`

5. `penalty_time`
   - 目标：鼓励尽快完成
   - 公式：每步固定 `-0.005`

---

### 主要奖励项

1. `reward_top_down_orientation`
   - 目标：让末端执行器朝下
   - 计算末端 z 轴方向与世界向下向量 `(0,0,-1)` 的点积
   - 公式：`1 - tanh((1 - alignment)/0.15)`
   - alignment 越接近 1，奖励越高

2. `reward_distance_end_effector_to_obj`
   - 目标：让末端执行器接近目标物体
   - 公式：`2.5 * (1 - tanh(dist / 0.2))`
   - 距离越小，奖励越高

3. `reward_grasp`
   - 目标：检测是否已经抓到物体
   - 如果 end effector 的接触力最大值 > `2.0`，奖励 `4.0`
   - 否则奖励 `0`

4. `reward_lift`
   - 目标：鼓励抬起物体
   - 基于物体当前高度与初始高度之差
   - 公式实际上对提升 `0.15` 左右给予较大奖励，并对“过度”提升做软衰减
   - 这里的关键点是：当提升接近 `0.15` 时，奖励最大

5. `reward_success`
   - 目标：稀疏成功奖励
   - 如果物体上升高度超过 `0.15`，就奖励 `20.0`

6. `reward_distance_obj_to_target`
   - 目标：鼓励物体接近目标位置
   - 公式：`32.0 * (1 - tanh(dist_obj_target / 0.2))`
   - 物体离目标越近，奖励越高

---

### 新增的额外惩罚项

这段代码添加了两个额外约束：

1. `far_close_penalty`
   - 如果夹爪处于“闭合”状态，但末端执行器距离物体太远（> 0.15），则惩罚 `-1.0`

2. `fake_grasp_penalty`
   - 如果夹爪闭合但接触力很小（< 2.0 N），则惩罚 `-0.5`
   - 目的是避免“空抓”行为

---

### 你现在的奖励设计意思是什么

- 惩罚：动作突变、关节力矩、加速度、异常接触、时间
- 奖励：末端朝下、接近物体、抓取接触、抬起高度、成功完成、靠近目标
- 额外约束：远距离闭合和弱接触抓取都会被惩罚

---

### 纠正点

`reward_lift` 的写法比较特殊，它不是简单线性增长，而是对离 `0.15` 的高度差做了一个“软带宽”处理，因此：
- 小幅抬起接近 `0.15` 会得到最大值
- 继续过度抬高并不会无限增加奖励，反而会衰减

如果你想要“越抬越多”而不是“到某个高度为最优”，这部分可以再调整。

## warning

*WARN: Box high's precision lowered by casting to float32, current high.dtype=float64* 对应 [spaces.py](/root/isaaclab/source/isaaclab/isaaclab/envs/utils/spaces.py) 中 deserialize_space 函数：
`return gym.spaces.Box(low=np.array(obj["low"]).astype(np.float32), high=np.array(obj["high"]).astype(np.float32), shape=obj["shape"])`

## docker 增挂载

[blog](https://www.cnblogs.com/azureology/p/17296787.html)

pip install omniverse-kit --extra-index-url https://pypi.nvidia.com

from openpi_client import websocket_client_policy as w; c=w.WebsocketClientPolicy('0.0.0.0', 8899); r=c.infer({'prompt':'collect the sample'}); import numpy as np; a=np.asarray(r['actions']); print(type(r), a.shape, a.dtype)

I'll先对比 `num_envs=1` 和 `num_envs=2` 的相机/环境克隆配置，确认是窗口没有启动、相机落在空位置，还是多环境场景本身没有被正确复制；然后用同一脚本做一次窄范围复现。

Searched for regex `ViewerCfg|origin_type|env_index|set_camera_view|viewer` (`**/srb/**`), 33 results

Read , lines 135 to 190

Read , lines 1 to 180

Read 

No matches found. Your search pattern might be excluded completely by either the search.exclude settings or .*ignore files.
If you believe that it should have results, you can check into the .*ignore files and the exclude setting (here are some excluded patterns for reference:[**/node_modules,**/bower_components,**/*.code-search]).
Then if you want to include those files you can call the tool again by setting "includeIgnoredFiles" to true.

Ran terminal command:  grep -R "cfg.viewer\|viewer_cfg\|origin_type" -n /root/isaaclab/source/isaaclab/isaaclab/envs /root/isaaclab/source/isaaclab/isaaclab | head -n 160

Ran terminal command:  grep -R "cfg.viewer\|viewer_cfg\|origin_type" -n /root/isaaclab/source/isaaclab/isaaclab/envs /root/isaaclab/source/isaaclab/isaaclab | head -n 160

Ran terminal command:  find /root/isaaclab/source/isaaclab/isaaclab -type f -name '*.py' -print0 | xargs -0 grep -nE 'cfg\.viewer|viewer_cfg|origin_type' | head -n 180

Read 

相机配置本身是 `origin_type="env"`，默认应看 `env_0`，所以单纯增加环境数量不应让窗口消失。当前更可疑的是多环境时 USD 克隆/SimForge 多资产把根 prim 结构改变，导致 viewport 相机虽然启动，却指向一个空的环境根；我继续核对 `DirectRLEnv` 的初始化顺序和实际相机坐标。

Read 

Read 

No matches found. Your search pattern might be excluded completely by either the search.exclude settings or .*ignore files.
If you believe that it should have results, you can check into the .*ignore files and the exclude setting (here are some excluded patterns for reference:[**/node_modules,**/bower_components,**/*.code-search]).
Then if you want to include those files you can call the tool again by setting "includeIgnoredFiles" to true.

Ran terminal command:  grep -n "env_origins\|default_env_pose\|env_spacing" /root/isaaclab/source/isaaclab/isaaclab/scene/interactive_scene.py | head -n 80

Read 

Read 

我会启动一次当前的 `num_envs=2` 进程并检查 X11 窗口与 Kit 相机状态；这能直接区分“窗口不存在”和“窗口存在但视角/场景为空”。检查完成后会关闭这次诊断进程。

Ran terminal command:  conda activate srb && python train_stage2.py

Ran terminal command:  printf '%s\n' '--- processes ---'; ps -ef | grep -E 'train_stage2|isaac|kit' | grep -v grep | head -n 40; printf '%s\n' '--- windows ---'; command -v wmctrl >/dev/null && wmctrl -l || true; command -v xwininfo >/dev/null && xwininfo -root -tree | head -n 80 || true

Ran terminal command:  grep -nE -i 'Simulation App (Starting|Startup Complete|Shutting Down)|Completed setting up|Error|Traceback|IndexError|NameError|app ready|Scene manager|Environment spacing|Number of environments' /tmp/train_stage2_final.log 2>/dev/null | tail -n 80 || true

Ran terminal command:  for tool in xwininfo wmctrl xdotool xprop xdpyinfo import gnome-screenshot; do printf '%-18s' "$tool"; command -v "$tool" || true; done; echo "DISPLAY=$DISPLAY"; xdpyinfo 2>/dev/null | head -n 20 || true; xprop -root _NET_CLIENT_LIST 2>/dev/null || true

Read , lines 340 to 460

No matches found. Your search pattern might be excluded completely by either the search.exclude settings or .*ignore files.
If you believe that it should have results, you can check into the .*ignore files and the exclude setting (here are some excluded patterns for reference:[**/node_modules,**/bower_components,**/*.code-search]).
Then if you want to include those files you can call the tool again by setting "includeIgnoredFiles" to true.

Ran terminal command:  grep -nE "def step|render_enabled|sim.step|app.update" /root/isaaclab/source/isaaclab/isaaclab/envs/direct_rl_env.py | head -n 100

Read 

Read 

Ran terminal command:  python - <<'PY'
try:
    from PIL import ImageGrab
    print('PIL ImageGrab available')
except Exception as exc:
    print(type(exc).__name__, exc)
try:
    import mss
    print('mss available')
except Exception as exc:
    print(type(exc).__name__, exc)
PY

Ran terminal command:  python - <<'PY'
from PIL import ImageGrab
image = ImageGrab.grab()
print(image.size)
image.save('/tmp/srb-num-envs-2.png')
PY

当前多环境进程确实还活着，日志也已完成场景初始化，所以不是“启动失败”。我准备加一个临时诊断输出，打印 `has_gui`、`is_rendering`、环境原点和相机原点；这一步能直接证明多环境时到底是渲染被关掉，还是相机落到了错误位置。

已终止的终端

Generating patch (23 lines) in 


```

```


Ran terminal command:  conda activate srb && rm -f /tmp/train_stage2_gui_debug.log && timeout 25s python train_stage2.py > /tmp/train_stage2_gui_debug.log 2>&1; ret=$?; echo "RET=$ret"; grep -nE '\[GUI DEBUG\]|Scene manager|Number of environments|Environment spacing|Simulation App (Startup Complete|Shutting Down)|Traceback|Error executing job' /tmp/train_stage2_gui_debug.log | tail -n 80

诊断结果很关键：`num_envs=2` 时不是场景没生成，而是 `sim.has_gui=False`，同时 `sim.is_rendering=True`；相机原点是有效的 `[2, 0, 0]`。这说明当前多环境运行进入了“渲染但无 GUI”的分支，我再用同样参数只改 `num_envs=1` 对照验证。

Ran terminal command:  cd /root/space_robotics_bench_l && conda activate srb && rm -f /tmp/sample_collection_env1_gui_debug.log && timeout 25s python - <<'PY' > /tmp/sample_collection_env1_gui_debug.log 2>&1
import sys
from srb.__main__ import main
sys.argv = [
    "sample_collection_env1",
    "agent",
    "rand",
    "--env",
    "sample_collection",
    "env.domain=moon",
    "env.sample=primitive",
    "env.robot=ur5",
    "env.stage=2",
    "env.episode_length_s=8.0",
    "env.num_envs=1",
    "env.sim.device=cuda",
]
main()
PY
ret=$?; echo "RET=$ret"; grep -nE '\[GUI DEBUG\]|Scene manager|Number of environments|Environment spacing|Simulation App (Startup Complete|Shutting Down)|Traceback|Error executing job' /tmp/sample_collection_env1_gui_debug.log | tail -n 80

No matches found. Your search pattern might be excluded completely by either the search.exclude settings or .*ignore files.
If you believe that it should have results, you can check into the .*ignore files and the exclude setting (here are some excluded patterns for reference:[**/node_modules,**/bower_components,**/*.code-search]).
Then if you want to include those files you can call the tool again by setting "includeIgnoredFiles" to true.

Ran terminal command:  grep -R "has_gui\|has_active_visualizers\|resolve_visualizer_types" -n /root/isaaclab/source/isaaclab/isaaclab/sim | head -n 180

Read 

Read 

Ran terminal command:  grep -R "isaaclab/has_gui\|has_gui" -n /root/isaaclab/source /root/miniconda3/envs/srb/lib/python3.12/site-packages/isaaclab* 2>/dev/null | head -n 160

Ran terminal command:  grep -R "isaaclab/has_gui" -n /root/isaaclab /root/miniconda3/envs/srb/lib/python3.12/site-packages/isaacsim /root/space_robotics_bench_l/apps 2>/dev/null | head -n 120

已检查终端输出

已终止的终端

No matches found. Your search pattern might be excluded completely by either the search.exclude settings or .*ignore files.
If you believe that it should have results, you can check into the .*ignore files and the exclude setting (here are some excluded patterns for reference:[**/node_modules,**/bower_components,**/*.code-search]).
Then if you want to include those files you can call the tool again by setting "includeIgnoredFiles" to true.

Ran terminal command:  grep -n "has_gui\|visualizer/types\|render/offscreen" /root/isaaclab/source/isaaclab/isaaclab/sim/simulation_context.py | head -n 100; sed -n '960,1040p' /root/isaaclab/source/isaaclab/isaaclab/sim/simulation_context.py

Read 

Read 

Ran terminal command:  find /root/isaaclab/source /root/miniconda3/envs/srb/lib/python3.12/site-packages -path '*isaaclab_visualizers*' -type f -name '*.py' 2>/dev/null | head -n 100

Read 

Read 

Ran terminal command:  grep -R "has_gui" -n /root/isaaclab/source/isaaclab /root/isaaclab/source/isaaclab_visualizers /root/miniconda3/envs/srb/lib/python3.12/site-packages/isaaclab* 2>/dev/null | head -n 200

Read 

Ran terminal command:  grep -nE "visualizer/(types|explicit|disable_all|max_visible)|has_gui|render/offscreen" /root/isaaclab/source/isaaclab/isaaclab/app/app_launcher.py | head -n 160

Read 

No matches found. Your search pattern might be excluded completely by either the search.exclude settings or .*ignore files.
If you believe that it should have results, you can check into the .*ignore files and the exclude setting (here are some excluded patterns for reference:[**/node_modules,**/bower_components,**/*.code-search]).
Then if you want to include those files you can call the tool again by setting "includeIgnoredFiles" to true.

Ran terminal command:  grep -n "sync_visualizer" /root/isaaclab/source/isaaclab/isaaclab/app/app_launcher.py; grep -n "visualizer_explicit\|visualizer_disable_all" /root/isaaclab/source/isaaclab/isaaclab/app/app_launcher.py | head -n 80

Read 

No matches found. Your search pattern might be excluded completely by either the search.exclude settings or .*ignore files.
If you believe that it should have results, you can check into the .*ignore files and the exclude setting (here are some excluded patterns for reference:[**/node_modules,**/bower_components,**/*.code-search]).
Then if you want to include those files you can call the tool again by setting "includeIgnoredFiles" to true.

Ran terminal command:  grep -nE "compute_visualized|tiled_cam|set_viewport_camera|visible_env|env_ids|camera" /root/isaaclab/source/isaaclab_visualizers/isaaclab_visualizers/kit/kit_visualizer.py | head -n 160

Ran terminal command:  grep -nE "compute_visualized|tiled_cam|set_viewport_camera|visible_env|env_ids|camera" /root/isaaclab/source/isaaclab_visualizers/isaaclab_visualizers/kit/kit_visualizer.py | head -n 160

Ran terminal command:  grep -nE "def _compute_visualized_env_ids|def _set_viewport_camera|tiled_cam|resolve_visible_env|def _setup_viewport" /root/isaaclab/source/isaaclab_visualizers/isaaclab_visualizers/kit/kit_visualizer.py

Read 

Read 

No matches found. Your search pattern might be excluded completely by either the search.exclude settings or .*ignore files.
If you believe that it should have results, you can check into the .*ignore files and the exclude setting (here are some excluded patterns for reference:[**/node_modules,**/bower_components,**/*.code-search]).
Then if you want to include those files you can call the tool again by setting "includeIgnoredFiles" to true.

No matches found. Your search pattern might be excluded completely by either the search.exclude settings or .*ignore files.
If you believe that it should have results, you can check into the .*ignore files and the exclude setting (here are some excluded patterns for reference:[**/node_modules,**/bower_components,**/*.code-search]).
Then if you want to include those files you can call the tool again by setting "includeIgnoredFiles" to true.

Ran terminal command:  grep -R "default_visualizer_cfg\|visualizer_cfgs" -n /root/isaaclab/source/isaaclab/source /root/isaaclab/source/isaaclab/isaaclab 2>/dev/null | head -n 120

Read 

Read 

Read 

Ran terminal command:  grep -nE -i 'KitVisualizer|visualizer|viewport|has_gui|headless|camera|env_0/scenery|env_1' /tmp/train_stage2_gui_debug.log | tail -n 140

单环境对照结果反而显示 `has_gui=False`、`is_rendering=True`，与多环境完全一致，因此“env>1 没画面”不是 `prim_paths` 或相机原点单独造成的。下一步我会检查实际激活的 visualizer 数量、Kit 的 `SimulationApp.config["headless"]` 和 viewport 设置，找出为什么表面上在渲染但窗口没有正确承载画面。

Generating patch (22 lines) in 


```

```


Ran terminal command:  conda activate srb && rm -f /tmp/train_stage2_gui_debug.log && timeout 25s python train_stage2.py > /tmp/train_stage2_gui_debug.log 2>&1; ret=$?; echo "RET=$ret"; grep -nE '\[GUI DEBUG\]|Simulation App (Startup Complete|Shutting Down)|Traceback|Error executing job' /tmp/train_stage2_gui_debug.log | tail -n 40

Generating patch (30 lines) in 


```

```


Ran terminal command:  conda activate srb && rm -f /tmp/train_stage2_gui_debug.log && timeout 25s python train_stage2.py > /tmp/train_stage2_gui_debug.log 2>&1; ret=$?; echo "RET=$ret"; grep -nE '\[GUI DEBUG\]|Simulation App (Startup Complete|Shutting Down)|Traceback|Error executing job' /tmp/train_stage2_gui_debug.log | tail -n 40

Ran terminal command:  grep -nE "viewport_api\.|camera_path|camera_path" /root/isaaclab/source/isaaclab_visualizers/isaaclab_visualizers/kit/kit_visualizer.py | head -n 100; sed -n '520,590p' /root/isaaclab/source/isaaclab_visualizers/isaaclab_visualizers/kit/kit_visualizer.py

现在可以排除“程序被 headless 强制运行”：`KitVisualizer` 已初始化，`runtime_headless=False`，viewport window/API 都存在。剩下的最可疑点是 viewport 的活动相机最终被设置到了错误的 world 坐标，我会读取相机实际 transform 和两个环境根 prim 的位置做最后一次对照。

Generating patch (28 lines) in 


```

```


Ran terminal command:  conda activate srb && rm -f /tmp/train_stage2_camera_debug.log && timeout 25s python train_stage2.py > /tmp/train_stage2_camera_debug.log 2>&1; ret=$?; echo "RET=$ret"; grep -nE '\[GUI DEBUG\]|Simulation App (Startup Complete|Shutting Down)|Traceback|Error executing job' /tmp/train_stage2_camera_debug.log | tail -n 40

Generating patch (20 lines) in 


```

```


Ran terminal command:  conda activate srb && rm -f /tmp/train_stage2_camera_debug.log && timeout 25s python train_stage2.py > /tmp/train_stage2_camera_debug.log 2>&1; ret=$?; echo "RET=$ret"; grep -nE '\[GUI DEBUG\]|Simulation App (Startup Complete|Shutting Down)|Traceback|Error executing job' /tmp/train_stage2_camera_debug.log | tail -n 40

多环境时场景与相机坐标都正常：两个机器人分别在 `[2, 0, 0.25]` 和 `[-2, 0, 0.25]`，活动相机在 `[3.85, 0, 1.85]`，正对 `env_0`。因此问题已经集中到 Kit 窗口本身的可见性/启动配置，我再读取窗口 `visible` 状态和实际 `SimulationApp` 配置，随后给出针对性修复。

Generating patch (19 lines) in 


```

```


Ran terminal command:  conda activate srb && rm -f /tmp/train_stage2_gui_debug.log && timeout 25s python train_stage2.py > /tmp/train_stage2_gui_debug.log 2>&1; ret=$?; echo "RET=$ret"; grep -nE '\[GUI DEBUG\]|Simulation App (Startup Complete|Shutting Down)|Traceback|Error executing job' /tmp/train_stage2_gui_debug.log | tail -n 40

No matches found. Your search pattern might be excluded completely by either the search.exclude settings or .*ignore files.
If you believe that it should have results, you can check into the .*ignore files and the exclude setting (here are some excluded patterns for reference:[**/node_modules,**/bower_components,**/*.code-search]).
Then if you want to include those files you can call the tool again by setting "includeIgnoredFiles" to true.

Ran terminal command:  grep -nE "compute_visualized|_env_ids|env_indices|max_visible" /root/isaaclab/source/isaaclab_visualizers/isaaclab_visualizers/kit/kit_visualizer.py | head -n 120

Read 

No matches found. Your search pattern might be excluded completely by either the search.exclude settings or .*ignore files.
If you believe that it should have results, you can check into the .*ignore files and the exclude setting (here are some excluded patterns for reference:[**/node_modules,**/bower_components,**/*.code-search]).
Then if you want to include those files you can call the tool again by setting "includeIgnoredFiles" to true.

Ran terminal command:  grep -R "compute_visualized_env_ids" -n /root/isaaclab/source/isaaclab_visualizers /root/isaaclab/source/isaaclab | head -n 40

Read 

Ran terminal command:  grep -R "capture_viewport\|capture.*viewport\|viewport.*capture" -n /root/isaaclab/source /root/miniconda3/envs/srb/lib/python3.12/site-packages 2>/dev/null | head -n 100

Ran terminal command:  grep -R "capture_viewport\|capture.*viewport\|viewport.*capture" -n /root/isaaclab/source/isaaclab_visualizers /root/isaaclab/source/isaaclab/isaaclab /root/miniconda3/envs/srb/lib/python3.12/site-packages/isaacsim 2>/dev/null | head -n 100

Generating patch (24 lines) in 


```

```


Ran terminal command:  conda activate srb && rm -f /tmp/train_stage2_scene_debug.log && timeout 25s python train_stage2.py > /tmp/train_stage2_scene_debug.log 2>&1; ret=$?; echo "RET=$ret"; grep -nE '\[GUI DEBUG\]|Simulation App (Startup Complete|Shutting Down)|Traceback|Error executing job' /tmp/train_stage2_scene_debug.log | tail -n 40

多环境的 `env_0` 下确实有完整可渲染结构：`scenery0`、`scenery1`、`robot`、`pedestal`、`sample`，且相机正对机器人。因此不是“多环境没有复制场景”。我现在做最后一个对照：比较 `num_envs=1` 时的子 prim 名称，确认差异是否来自 SimForge 的多资产根节点。

Ran terminal command:  cd /root/space_robotics_bench_l && conda activate srb && rm -f /tmp/sample_collection_env1_scene_debug.log && timeout 25s python - <<'PY' > /tmp/sample_collection_env1_scene_debug.log 2>&1
import sys
from srb.__main__ import main
sys.argv = [
    "sample_collection_env1",
    "agent",
    "rand",
    "--env",
    "sample_collection",
    "env.domain=moon",
    "env.sample=primitive",
    "env.robot=ur5",
    "env.stage=2",
    "env.episode_length_s=8.0",
    "env.num_envs=1",
    "env.sim.device=cuda",
]
main()
PY
ret=$?; echo "RET=$ret"; grep -nE '\[GUI DEBUG\]|Simulation App (Startup Complete|Shutting Down)|Traceback|Error executing job' /tmp/sample_collection_env1_scene_debug.log | tail -n 40

Generating patch (27 lines) in 


```

```


Ran terminal command:  conda activate srb && rm -f /tmp/train_stage2_bounds_debug.log && timeout 25s python train_stage2.py > /tmp/train_stage2_bounds_debug.log 2>&1; ret=$?; echo "RET=$ret"; grep -nE '\[GUI DEBUG\]|Simulation App (Startup Complete|Shutting Down)|Traceback|Error executing job' /tmp/train_stage2_bounds_debug.log | tail -n 40

资产边界检查显示当前阶段 USD 的根 Xform 没有可计算的渲染包围盒，说明问题可能发生在 SimForge 生成模型的 USD 内容/加载状态，而不是相机位置。我要再检查 `scenery0/1` 是否有实际网格后代和可见性；这一步会决定是修复多环境资产数量，还是只修复 viewport。

Generating patch (15 lines) in 


```

```


Read , lines 130 to 210

Generating patch (16 lines) in 


```

```


Ran terminal command:  conda activate srb && rm -f /tmp/train_stage2_visibility_debug.log && timeout 25s python train_stage2.py > /tmp/train_stage2_visibility_debug.log 2>&1; ret=$?; echo "RET=$ret"; grep -nE '\[GUI DEBUG\]|Simulation App (Startup Complete|Shutting Down)|Traceback|Error executing job' /tmp/train_stage2_visibility_debug.log | tail -n 40

目前唯一会随 `num_envs` 改变且直接影响 viewport 内容的配置是程序化地形变体数：默认代码把 `n_procgen_variants` 设成 `scene.num_envs`，所以 `num_envs=2` 会在源环境里生成 `scenery0` 和 `scenery1` 两套地形。我要用 `env.n_procgen_variants=1` 做一个最小对照，验证是否正是多套地形导致画面异常。

Ran terminal command:  cd /root/space_robotics_bench_l && conda activate srb && rm -f /tmp/train_stage2_one_variant_debug.log && timeout 25s python - <<'PY' > /tmp/train_stage2_one_variant_debug.log 2>&1
import sys
from srb.__main__ import main
sys.argv = [
    "sample_collection_two_env_one_variant",
    "agent",
    "rand",
    "--env",
    "sample_collection",
    "env.domain=moon",
    "env.sample=primitive",
    "env.robot=ur5",
    "env.stage=2",
    "env.episode_length_s=8.0",
    "env.num_envs=2",
    "env.n_procgen_variants=1",
    "env.sim.device=cuda",
]
main()
PY
ret=$?; echo "RET=$ret"; grep -nE '\[GUI DEBUG\]|Scene manager|Number of environments|scenery|Simulation App (Startup Complete|Shutting Down)|Traceback|Error executing job' /tmp/train_stage2_one_variant_debug.log | tail -n 80

诊断已证明多环境场景和相机都有效，唯一明显差异是多环境会生成多套程序化地形；但当前临时输出已过多，我先清理所有诊断代码，再做一次较长时间运行，确认是否存在延迟出现的 GPU/Kit 错误，而不是把测试超时误当成用户的实际退出。

Read , lines 115 to 245

Generating patch (113 lines) in 


```

```


Ran terminal command:  conda activate srb && rm -f /tmp/train_stage2_long.log && timeout 70s python train_stage2.py > /tmp/train_stage2_long.log 2>&1; ret=$?; echo "RET=$ret"; grep -nE -i 'Simulation App (Startup Complete|Shutting Down)|Completed setting up|Traceback|Error executing job|reentrant|segmentation|abort|fatal|CUDA error|out of memory|NoneType' /tmp/train_stage2_long.log | tail -n 120

Ran terminal command:  find /root/miniconda3/envs/srb/lib/python3.12/site-packages/isaacsim /root/isaaclab -type f \( -name '*.py' -o -name '*.pyi' \) 2>/dev/null | xargs grep -lE 'capture_next_frame|capture_viewport|CaptureViewport|viewport.*capture|Capture' | head -n 100

Ran terminal command:  find /root/miniconda3/envs/srb/lib/python3.12/site-packages/isaacsim /root/isaaclab/source/isaaclab -type f \( -name '*.py' -o -name '*.pyi' \) 2>/dev/null | xargs grep -lE 'capture_next_frame|capture_viewport|CaptureViewport|viewport.*capture' | head -n 100

Read 

长时间运行结果没有任何延迟崩溃、GPU OOM 或 Kit 错误；`num_envs=2` 时进程持续运行，窗口与 viewport 也都处于 visible 状态。这里更接近“多环境共用一个 viewport，且默认相机只跟踪一个环境”，不是仿真没有启动；我会清理最后的临时改动并保留可复现的启动建议。

No matches found. Your search pattern might be excluded completely by either the search.exclude settings or .*ignore files.
If you believe that it should have results, you can check into the .*ignore files and the exclude setting (here are some excluded patterns for reference:[**/node_modules,**/bower_components,**/*.code-search]).
Then if you want to include those files you can call the tool again by setting "includeIgnoredFiles" to true.

Ran terminal command:  conda activate srb && python -m py_compile srb/tasks/manipulation/sample_collection/task.py train_stage2.py && git diff --check -- srb/tasks/manipulation/sample_collection/task.py train_stage2.py && grep -nE 'GUI DEBUG|env0_bounds|camera_state' srb/tasks/manipulation/sample_collection/task.py || true

Checked , no problems found

已确认：`num_envs>1` 时仿真并没有停止，Kit viewport 也处于可见状态；变化来自多环境网格布局和程序化地形变体数量，默认相机只跟踪 `env_0`，不会自动把所有环境平铺到画面中。临时诊断已全部移除，任务文件语法检查通过。



这不是 `env.num_envs>1` 后仿真初始化失败，而是 **多环境共用一个 viewport，默认相机只跟踪一个环境**。

当前配置中 `env.py:83` 使用了：

```python
viewer = ViewerCfg(
    eye=(1.85, 0.0, 1.85),
    lookat=(0.125, 0.0, 0.25),
    origin_type="env",
)
```

`origin_type="env"` 表示相机坐标相对于某个环境原点，默认 `env_index=0`。当 `num_envs=2` 时，环境被放置在类似：

```text
env_0: [ 2, 0, 0]
env_1: [-2, 0, 0]
```

但不会自动打开两个窗口，也不会自动缩放到同时显示所有环境。

我实际检查了 `num_envs=2`：

```text
Simulation App Startup Complete
Completed setting up the environment...
KitVisualizer: active
runtime_headless: False
viewport window: visible
robot roots: [[2.0, 0.0, 0.25], [-2.0, 0.0, 0.25]]
```

运行 70 秒期间也没有 `Traceback`、GPU 错误或 Kit 错误。测试返回 `124` 只是命令主动超时。

另外，`env_cfg.py:894` 中程序化资产默认会把：

```python
num_assets = scene.num_envs
```

因此 `num_envs=2` 时会在源环境中生成多个地形变体，例如：

```text
/World/envs/env_0/scenery0
/World/envs/env_0/scenery1
```

这也会让多环境画面比单环境更复杂。

**建议先把 `train_stage2.py` 的参数改成下面这样测试：**

```python
"env.num_envs=2",
"env.n_procgen_variants=1",
"env.viewer.origin_type=world",
"env.viewer.eye=[8.0,-8.0,8.0]",
"env.viewer.lookat=[0.0,0.0,0.0]",
```

含义是：

- 仍然运行两个并行环境
- 只生成一套程序化地形，减少重复地形对 viewport 的影响
- 相机改为观察世界坐标原点
- 拉远相机，使两个环境都进入视野

如果只想查看某一个环境，则使用：

```python
"env.viewer.origin_type=env",
"env.viewer.env_index=0",
```

查看第二个环境时改成：

```python
"env.viewer.env_index=1",
```

多环境训练通常建议使用 `num_envs>1` 配合 headless；需要人工观察场景时用 `num_envs=1` 或显式调整 world camera。

---

I'll先把当前 Robotiq 的 USD 加载链路拆开，确认 prototype 是来自外部 USD reference 还是 stage 内的 instance，然后给你两种可落地方案：离线修改源 USD，以及运行时在应用碰撞属性前解除实例化。

No matches found. Your search pattern might be excluded completely by either the search.exclude settings or .*ignore files.
If you believe that it should have results, you can check into the .*ignore files and the exclude setting (here are some excluded patterns for reference:[**/node_modules,**/bower_components,**/*.code-search]).
Then if you want to include those files you can call the tool again by setting "includeIgnoredFiles" to true.

Searching for regex `def create_prim|create_prim(` (`/root/isaaclab/source/isaaclab/isaaclab/sim`)

Read 

Read 

Ran terminal command:  grep -R "def create_prim" -n /root/isaaclab/source/isaaclab /root/miniconda3/envs/srb/lib/python3.12/site-packages/isaacsim 2>/dev/null | head -n 20; grep -R "SetInstanceable(False)\|make_uninstanceable" -n /root/isaaclab/source/isaaclab /root/miniconda3/envs/srb/lib/python3.12/site-packages/isaacsim 2>/dev/null | head -n 80

Ran terminal command:  grep -R "SetInstanceable(False)\|make_uninstanceable" -n /root/isaaclab/source/isaaclab /root/miniconda3/envs/srb/lib/python3.12/site-packages/isaacsim 2>/dev/null | head -n 80; grep -R "def create_prim" -n /root/isaaclab/source/isaaclab /root/miniconda3/envs/srb/lib/python3.12/site-packages/isaacsim 2>/dev/null | head -n 20

当前链路已经清楚：`robotiq_hand_e.usdz` 被 `create_prim()` 引用到场景后，内部 `Group*ROBOTIQ_HAND_E_DEFEATURE` 才作为实例出现；因此不能对场景中的 instance proxy 直接写属性。接下来我核对该 USD 的真实路径和可编辑方式，再给出最稳妥的离线 prototype 修改脚本，以及需要保留原始 `clone` 逻辑时的运行时写法。

Ran terminal command:  cd /root/space_robotics_bench_l && find assets/srb_assets -iname '*robotiq*' -o -iname 'robotiq_hand_e.usdz' | head -n 30; printf '--- resolved asset ---\n'; python - <<'PY'
from srb.utils.path import SRB_ASSETS_DIR_SRB_ROBOT
path = SRB_ASSETS_DIR_SRB_ROBOT.joinpath('gripper', 'robotiq_hand_e.usdz')
print(path)
print(path.exists())
PY

Ran terminal command:  grep -R "def spawn_from_usd" -n /root/simforge/simforge/integrations/isaaclab/spawner/from_files/impl.py /root/isaaclab/source/isaaclab/isaaclab/sim/spawners/from_files/from_files.py; grep -n "@clone" /root/simforge/simforge/integrations/isaaclab/spawner/from_files/impl.py

Read 

Ran terminal command:  cd /root/space_robotics_bench_l && conda activate srb && python - <<'PY'
from pxr import Usd
from srb.utils.path import SRB_ASSETS_DIR_SRB_ROBOT
path = SRB_ASSETS_DIR_SRB_ROBOT.joinpath('gripper', 'robotiq_hand_e.usdz').as_posix()
stage = Usd.Stage.Open(path)
print('stage:', bool(stage), 'root layer:', stage.GetRootLayer().identifier if stage else None)
if stage:
    instances = []
    meshes = []
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if prim.IsInstance():
            instances.append(prim.GetPath().pathString)
        if prim.GetTypeName() == 'Mesh':
            meshes.append((prim.GetPath().pathString, prim.IsInstanceProxy(), prim.HasAPI('UsdPhysicsCollisionAPI') if False else ''))
    print('instances:', instances[:30], 'count=', len(instances))
    print('meshes:', len(meshes), meshes[:12])
PY

Ran terminal command:  cd /root/space_robotics_bench_l && conda activate srb && python - <<'PY'
from pxr import Usd, UsdPhysics
from srb.utils.path import SRB_ASSETS_DIR_SRB_ROBOT
path = SRB_ASSETS_DIR_SRB_ROBOT.joinpath('gripper', 'robotiq_hand_e.usdz').as_posix()
stage = Usd.Stage.Open(path)
for prim in Usd.PrimRange(stage.GetPseudoRoot()):
    if not prim.IsInstance():
        continue
    prototype = prim.GetPrototype()
    print('instance=', prim.GetPath(), 'prototype=', prototype.GetPath())
    for child in Usd.PrimRange(prototype):
        if child.GetTypeName() == 'Mesh':
            print('  mesh=', child.GetPath(), 'collision=', child.HasAPI(UsdPhysics.CollisionAPI), 'mesh_collision=', child.HasAPI(UsdPhysics.MeshCollisionAPI))
PY

源 USD 已进一步确认：四个 Robotiq prototype 中的 Mesh 本身没有 `CollisionAPI` / `MeshCollisionAPI`，而场景里的对应节点只是它们的 instance proxy。因此这里不只是“参数写不进去”，还可能导致夹爪碰撞体没有按预期建立。正确做法是先解除源 USD 内部实例，再给实际 Mesh 写碰撞 schema，导出一个派生 USD，最后让 `RobotiqHandE` 引用这个派生文件。

Ran terminal command:  cd /root/space_robotics_bench_l && conda activate srb && rm -f /tmp/robotiq_hand_e_fixed.usda && python - <<'PY'
from pathlib import Path
from pxr import Usd, UsdGeom, UsdPhysics
from srb.utils.path import SRB_ASSETS_DIR_SRB_ROBOT
src = SRB_ASSETS_DIR_SRB_ROBOT.joinpath('gripper', 'robotiq_hand_e.usdz').as_posix()
out = '/tmp/robotiq_hand_e_fixed.usda'
stage = Usd.Stage.Open(src)
instance_roots = [prim for prim in Usd.PrimRange(stage.GetPseudoRoot()) if prim.IsInstance()]
for prim in instance_roots:
    prim.SetInstanceable(False)
mesh_count = 0
collision_count = 0
for prim in Usd.PrimRange(stage.GetPseudoRoot()):
    if not prim.IsA(UsdGeom.Mesh):
        continue
    mesh_count += 1
    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        UsdPhysics.CollisionAPI.Apply(prim)
    mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
    mesh_collision_api.GetApproximationAttr().Set('convexDecomposition')
    collision_count += 1
stage.Export(out)
print('source=', src)
print('instances_uninstanced=', len(instance_roots))
print('meshes_modified=', mesh_count, 'collision_meshes=', collision_count)
print('output=', out, 'bytes=', Path(out).stat().st_size)
check = Usd.Stage.Open(out)
remaining_instances = sum(1 for prim in Usd.PrimRange(check.GetPseudoRoot()) if prim.IsInstance())
check_meshes = []
for prim in Usd.PrimRange(check.GetPseudoRoot()):
    if prim.IsA(UsdGeom.Mesh):
        check_meshes.append((prim.GetPath().pathString, prim.HasAPI(UsdPhysics.CollisionAPI), prim.HasAPI(UsdPhysics.MeshCollisionAPI), UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()))
print('remaining_instances=', remaining_instances)
print('checked_meshes=', check_meshes)
PY

离线转换已经在当前环境实际验证过：四个 Robotiq instance root 可解除实例化，四个 Mesh 都能写入 `CollisionAPI`、`MeshCollisionAPI` 和 `convexDecomposition`，导出的派生 USD 不再含 instance。下面给出推荐做法，以及不想生成派生文件时的运行时版本。

推荐把 Robotiq 的源 `.usdz` 转成一个派生 USD：先解除源文件内部的四个 instance root，再给四个 Mesh 写入 `CollisionAPI`、`MeshCollisionAPI` 和 `convexDecomposition`，最后让 `robotiq.py:17-38` 引用派生文件。这个流程我已在当前环境实际跑通。

## 关键概念

`/World/envs/env_0/end_effector/...` 下面看到的是 **instance proxy**，不是可以直接写入的普通 Mesh。

不能这样做：

```python
UsdPhysics.MeshCollisionAPI.Apply(instance_proxy)
```

正确流程是：

1. 找到 instance root
2. 对 instance root 执行：

```python
instance_root.SetInstanceable(False)
```

3. 再遍历实际 Mesh
4. 给 Mesh 写入碰撞属性
5. 导出为新的 USD
6. 仿真时引用这个新的 USD

不要直接修改 `/__Prototype_N`。这些是 USD 运行时生成的 prototype，通常不是持久化编辑入口。

---

## 推荐方案：离线生成派生 USD

当前 Robotiq 文件是：

```text
assets/srb_assets/robot/gripper/robotiq_hand_e.usdz
```

可以用下面的脚本生成一个已经解除实例化、带碰撞属性的派生文件：

```python
from pathlib import Path

from pxr import Usd, UsdGeom, UsdPhysics

source_path = Path(
    "/root/space_robotics_bench_l/assets/srb_assets/robot/gripper/robotiq_hand_e.usdz"
)
output_path = Path.home() / ".cache/srb/robotiq_hand_e_collision.usda"
output_path.parent.mkdir(parents=True, exist_ok=True)

stage = Usd.Stage.Open(source_path.as_posix())
if stage is None:
    raise RuntimeError(f"Could not open USD asset: {source_path}")

# 必须先收集 instance root，再解除实例化
instance_roots = [
    prim
    for prim in Usd.PrimRange(stage.GetPseudoRoot())
    if prim.IsInstance()
]

for instance_root in instance_roots:
    instance_root.SetInstanceable(False)

mesh_count = 0

for prim in Usd.PrimRange(stage.GetPseudoRoot()):
    if not prim.IsA(UsdGeom.Mesh):
        continue

    mesh_count += 1

    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        UsdPhysics.CollisionAPI.Apply(prim)

    mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
    mesh_collision_api.GetApproximationAttr().Set("convexDecomposition")

stage.Export(output_path.as_posix())

print(f"Uninstanced roots: {len(instance_roots)}")
print(f"Modified meshes: {mesh_count}")
print(f"Output: {output_path}")
```

我在当前 Robotiq USD 上验证过：

```text
instance roots: 4
modified meshes: 4
remaining instances after export: 0
```

运行方式：

```bash
conda activate srb
python prepare_robotiq_usd.py
```

---

## 修改 Robotiq 配置

然后将 `robotiq.py:17-38` 中的 `usd_path` 改成派生文件：

```python
from pathlib import Path

ROBOTIQ_HAND_E_USD = (
    Path.home() / ".cache" / "srb" / "robotiq_hand_e_collision.usda"
)
```

然后：

```python
spawn=UsdFileCfg(
    usd_path=ROBOTIQ_HAND_E_USD.as_posix(),
    activate_contact_sensors=True,
    collision_props=CollisionPropertiesCfg(
        contact_offset=0.005,
        rest_offset=0.0,
    ),
    rigid_props=RigidBodyPropertiesCfg(
        disable_gravity=True,
        max_depenetration_velocity=5.0,
    ),
    articulation_props=ArticulationRootPropertiesCfg(
        enabled_self_collisions=False,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=0,
    ),
)
```

派生 USD 中已经写入了：

```python
MeshCollisionAPI
physics:approximation = "convexDecomposition"
```

所以 `mesh_collision_props` 可以继续保持移除状态。`collision_props` 则可以保留，因为派生 USD 已经不是 instance proxy，可以正常写入 `contact_offset` 和 `rest_offset`。

---

## 如果必须运行时处理

也可以在自定义 spawner 中，在 `create_prim()` 之后、环境 clone 之前做：

```python
from pxr import Usd, UsdGeom, UsdPhysics


def prepare_instanced_asset(stage, root_path: str) -> None:
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        raise RuntimeError(f"Invalid asset root: {root_path}")

    instance_roots = [
        prim
        for prim in Usd.PrimRange(root_prim)
        if prim.IsInstance()
    ]

    for instance_root in instance_roots:
        instance_root.SetInstanceable(False)

    for prim in Usd.PrimRange(root_prim):
        if not prim.IsA(UsdGeom.Mesh):
            continue

        UsdPhysics.CollisionAPI.Apply(prim)

        mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
        mesh_collision_api.GetApproximationAttr().Set(
            "convexDecomposition"
        )
```

调用顺序必须是：

```python
create_prim(...)
prepare_instanced_asset(stage, prim_path)
_apply_missing_apis(...)
apply_collision_properties(...)
clone_environment(...)
```

这个运行时方案比离线转换复杂，因为要确保它发生在 `InteractiveScene.clone_environments()` 之前，否则后续环境 clone 可能复制尚未处理完的 prototype。

## 建议

对 Robotiq 这种固定资产，优先使用：

```text
源 usdz -> 一次性生成派生 usda -> 配置引用派生文件
```

这样启动时不会反复解除实例化，也不会再出现 `modify_mesh_collision_properties` 警告。

