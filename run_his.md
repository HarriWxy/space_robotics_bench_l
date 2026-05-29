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


Optimized tool selection先定位动作空间定义和抓取任务用到的 action term，再确认每个输出维度是怎么映射到控制器的。

Searched for regex `ActionCfg|action_space|gripper|BinaryJointPositionAction|JointPositionAction|DifferentialInverseKinematicsAction` (`**/srb/**`), 200 results

Searched for regex `grasp|pick|lift|reach` (`**/srb/tasks/**`), 200 results

Searched for text `train_stage2|stage2` (`**/*.py`), 47 results

我已经定位到抓取任务的关键路径：stage2 的 sample_collection 任务里，动作向量前 6 维控制末端位姿，后面的维度在任务代码里被当作夹爪动作处理。接下来读训练入口和任务实现，把这个映射说清楚，并找最适合加限制的位置。

Read [](file:///root/space_robotics_bench_l/train_stage2.py)

Read [](file:///root/space_robotics_bench_l/srb/tasks/manipulation/sample_collection/task.py#260-260), lines 260 to 360

Read [](file:///root/space_robotics_bench_l/srb/tasks/manipulation/sample_collection/task.py#760-760), lines 760 to 810

下一步我在看这个任务到底挂了哪套机器人动作配置，因为维度语义最终由机器人 action group 决定，不是训练脚本本身决定。

Searched for regex `ur5\+robotiq_hand_e|robotiq_hand_e|UR5` (`**/srb/**`), 16 results

Searched for regex `class .*SampleCollection|SampleCollectionEnvCfg|sample_collection` (`**/srb/tasks/manipulation/sample_collection/**`), no results

Searched for regex `DifferentialInverseKinematicsActionCfg|TaskSpaceAction|OperationalSpace|BinaryJointPositionActionCfg|JointPositionToLimitsActionCfg` (`**/srb/**`), 200 results

Read [](file:///root/space_robotics_bench_l/srb/tasks/manipulation/sample_collection/task.py#1-1), lines 1 to 140

Read [](file:///root/space_robotics_bench_l/srb/assets/robot/manipulation/universal_robots.py#440-440), lines 440 to 540

Read [](file:///root/space_robotics_bench_l/srb/assets/object/tool/robotiq.py)

我已经确认你这条训练命令用的是 UR5 + Robotiq Hand-E。现在还差最后一块：系统是怎样把 UR5 的 6 维末端位姿控制和夹爪维度拼成一个总动作向量。

Searched for regex `class SerialManipulator|end_effector.*actions|ActionGroup\(|JointPositionBoundedActionGroup|InverseKinematicsActionGroup` (`**/srb/**`), 141 results

Searched for regex `single_action_space|detailed_action_space|action_terms|action_manager` (`**/srb/core/**`), 22 results

Read [](file:///root/space_robotics_bench_l/srb/interfaces/interface/ros.py#400-400), lines 400 to 490

Read [](file:///root/space_robotics_bench_l/srb/core/env/common/base/env_cfg.py#540-540), lines 540 to 810

Read [](file:///root/space_robotics_bench_l/srb/core/action/term/manipulation/differential_ik.py)

现在已经能确认拼接顺序了：机械臂 action term 先加入，主动末端执行器 action term 后加入，所以这个任务里一定是 “前 6 维 = UR5 末端位姿增量；后面维度 = Robotiq 夹爪”。我补一眼夹爪 action term 的具体维数和缩放方式。

Searched for regex `class JointPositionToLimitsAction|JointPositionToLimitsActionCfg` (`**/srb/**`), 8 results

Read [](file:///root/space_robotics_bench_l/srb/core/action/group/common/joint.py)

Read [](file:///root/space_robotics_bench_l/srb/core/action/group/manipulation/task_space.py)

Searched for regex `class JointPositionToLimitsAction\(|class EMAJointPositionToLimitsAction\(|rescale_to_limits` (`**/srb/core/action/**`), no results

Searched for files matching `**/srb/core/action/**/*limits*.py`, no matches

Searched for files matching `**/srb/core/action/**/*.py`, 26 matches

Read [](file:///root/space_robotics_bench_l/srb/core/action/__init__.py)

Read [](file:///root/space_robotics_bench_l/srb/core/action/action_group.py)

Read [](file:///root/space_robotics_bench_l/srb/core/action/term/common/__init__.py)

我再补精确锚点，主要是为了给你一个能直接改的落点，而不是泛泛建议。

Searched for regex `processed_actions = actions|processed_actions\[:, 6:\]|far_close_penalty|fake_grasp_penalty|gripper_closed` (`**/srb/tasks/manipulation/sample_collection/task.py`), 23 results

Searched for regex `self.actions = ActionGroup\(|map_cmd_to_action_fns.append\(|manipulator.end_effector.actions|setattr\(\s*self.actions,\s*f\"\{end_effector_name\}/` (`**/srb/core/env/common/base/env_cfg.py`), 8 results

Read [](file:///root/space_robotics_bench_l/srb/assets/robot/manipulation/universal_robots.py#315-315), lines 315 to 390

Searched for regex `class UR5\(|class UR5RobotiqHandE\(|actions: ActionGroup = InverseKinematicsActionGroup\(|scale=0.1` (`**/srb/assets/robot/manipulation/universal_robots.py`), 50 results

Searched for regex `class RobotiqHandE\(|actions: ActionGroup = JointPositionBoundedActionGroup\(|joint_names=\["Slider_\[1-2\]"\]|rescale_to_limits=True` (`**/srb/assets/object/tool/robotiq.py`), 4 results

Searched for regex `map_cmd_to_action\(self, twist: torch.Tensor, event: bool\)|return twist|return torch.Tensor\(\(-1.0 if event else 1.0,\)\)` (`**/srb/core/action/group/manipulation/task_space.py|srb/core/action/group/common/joint.py`), no results

Searched for regex `def map_cmd_to_action|return twist|return twist\[:3\]` (`**/srb/core/action/group/manipulation/task_space.py`), 5 results

Searched for regex `class JointPositionBoundedActionGroup|JointPositionToLimitsActionCfg|def map_cmd_to_action|return torch.Tensor` (`**/srb/core/action/group/common/joint.py`), 13 results

Searched for regex `--env|env.robot=ur5\+robotiq_hand_e|env.stage=2|env.stage2_easy=True` (`**/train_stage2.py`), 4 results

我已经能给出维度语义了。最后再补一条：当前动作空间是不是统一归一化到 [-1, 1]，以及哪种“限制某些维度”的方式最贴近你现在这套实现。

Searched for regex `single_action_space = gymnasium.spaces.Box|shape=\(self.action_manager.total_action_dim,\)` (`**/srb/core/env/common/base/direct/impl.py`), 2 results

Searched for regex `_map_action_term\(|msg.linear.x|msg.angular.x|BinaryJointPositionAction|Float32MultiArray` (`**/srb/interfaces/interface/ros.py`), 17 results

Read [](file:///root/space_robotics_bench_l/srb/core/env/common/base/direct/impl.py#350-350), lines 350 to 375

这里还有一个更关键的点：Robotiq 夹爪现在很可能是两维独立输出，但从抓取任务角度看，这两维通常是冗余的。这个地方如果改成单标量或强制同步，往往比只调奖励更直接。

Searched for regex `top_down_alignment|rot6d|height_above_obj|distance_xy_to_obj` (`**/srb/tasks/manipulation/sample_collection/task.py`), 38 results

Searched for regex `Slider_\[1-2\]|joint_pos_end_effector|gripper_aperture` (`**/srb/tasks/manipulation/sample_collection/task.py|srb/assets/object/tool/robotiq.py`), no results

Read [](file:///root/space_robotics_bench_l/srb/core/action/term/common/scalar_joint_position.py)

关键点确认了：仓库里已经有单标量夹爪 action term，可以把 Robotiq 从两维改成一维；对于抓取，这通常比保留两个独立 slider 输出更稳。

Searched for regex `gripper_aperture|joint_pos_end_effector` (`**/srb/tasks/manipulation/sample_collection/task.py`), 19 results

Searched for regex `class ScalarJointPositionAction|action_dim|open_command_expr|close_command_expr` (`**/srb/core/action/term/common/scalar_joint_position.py`), 5 results

Searched for regex `top_down_alignment = torch.sum|PREGRASP_ALIGNMENT_THRESHOLD|reward_top_down_orientation` (`**/srb/tasks/manipulation/sample_collection/task.py`), 6 results

Searched for regex `lateral_gate_threshold|upper_gate_threshold|min_close_scale|close_scale|close_action =|open_action =` (`**/srb/tasks/manipulation/sample_collection/task.py`), 13 results

Searched for regex `joint_pos_end_effector_normalized =|gripper_aperture =|torch.mean\(joint_pos_end_effector_normalized` (`**/srb/tasks/manipulation/sample_collection/task.py`), 4 results

Searched for text `command_type="pose"|use_relative_mode=True|ik_method="dls"` (`**/srb/assets/robot/manipulation/universal_robots.py`), 30 results

**维度含义**

对你当前这条训练配置 train_stage2.py，动作向量按代码应当理解为 8 维：

1. 第 0 维：末端执行器 TCP 在 x 方向的相对平移
2. 第 1 维：末端执行器 TCP 在 y 方向的相对平移
3. 第 2 维：末端执行器 TCP 在 z 方向的相对平移
4. 第 3 维：绕 x 轴的相对旋转增量
5. 第 4 维：绕 y 轴的相对旋转增量
6. 第 5 维：绕 z 轴的相对旋转增量
7. 第 6 维：Robotiq 第一个滑块关节的开合命令
8. 第 7 维：Robotiq 第二个滑块关节的开合命令

依据是这几处：

- UR5 机械臂动作是相对位姿的 Differential IK，配置在 universal_robots.py
- 这个 IK action group 对 pose 命令直接返回 6 维 twist，也就是平移 3 维加旋转 3 维，见 task_space.py
- Robotiq Hand-E 的动作是对两个 Slider 关节做 bounded position command，见 robotiq.py
- 环境组装动作时，先把机械臂 action term 加进去，再把末端执行器 action term 追加进去，所以顺序就是前 6 维机械臂、后 2 维夹爪，见 env_cfg.py 和 env_cfg.py
- 整个动作空间统一是 [-1, 1] 的归一化 Box，见 impl.py

**能不能限制某些维度**

可以，而且你这套任务里其实已经在这么做了。抓取任务在进入物理步前，会对第 6 维之后的夹爪闭合动作做状态相关缩放：离物体太远、太高时，闭合会被压小；另外还有远距离闭合惩罚和虚抓惩罚，见 task.py 和 task.py。

如果目标是更有效地学抓取，我建议优先这样做：

1. 把第 6-7 维夹爪动作强制同步成一个值。现在 Robotiq 实际给了两维，但任务里连抓爪开度观测都在对两指取平均，见 task.py。对抓取来说，这两维大多是冗余自由度，会增加探索难度。
2. 明显压小第 3-4 维，甚至直接冻结。这个任务奖励明确鼓励 top-down alignment，见 task.py。如果你的抓取主要是俯抓，roll 和 pitch 往往是干扰项。常见做法是只保留 xyz 和少量 yaw。
3. 把夹爪闭合从现在的软门控，改成更硬的门控。也就是只有当末端已经进入预抓取窗口时，才允许负向夹爪动作；否则只允许保持或张开。

如果要从结构上进一步减维，这个仓库里其实已经有单标量夹爪 action term，可直接把 Robotiq 从两维改成一维，见 scalar_joint_position.py。不过对你现在这套实验，最省事的做法不是改资产定义，而是在 task.py 里先做动作预处理：同步夹爪两维，再缩小第 3-5 维。

如果你要，我可以直接帮你做其中一种修改：

1. 只保留 xyz + yaw + 单一夹爪动作
2. 保留 6 维机械臂，但把夹爪两维强制同步
3. 在现有基础上把夹爪闭合改成硬门控，专门强化抓取阶段