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
