---
title: Gymnasium CPU 训练游乐场
emoji: 🎮
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 6.17.3
app_file: app.py
pinned: false
license: apache-2.0
---

# WalkingLab × Hands-On Modern RL · Gymnasium CPU Playground

**WalkingLab** 与开源课程 **Hands-On Modern RL（《动手学现代强化学习》）** 的 Gymnasium 在线训练合集。创空间在构建阶段显式预装 Gymnasium 的 Atari/ALE、Box2D、Classic Control、JAX CPU、MuJoCo、Toy Text 和 Robotics 运行时，以及 EGL、OSMesa、Mesa DRI 与 FFmpeg。应用进程启动时会分别创建并执行一次代表环境，预热原生库、ROM、模型资源和 JAX 编译缓存；正式训练仍使用全新的环境实例，避免不同实验共享状态。

- Project: <https://github.com/walkinglabs/hands-on-modern-rl>
- WalkingLab: <https://modelscope.cn/organization/walkinglab>
- Companion Notebook: <https://modelscope.cn/notebook/share/github/walkinglabs/hands-on-modern-rl/blob/main/code/online-experiments/hands-on-modern-rl-experiment-gymnasium.ipynb>

首批实验：

- Multi-Armed Bandit：ε-greedy 探索与利用
- Blackjack：首次访问蒙特卡洛方法
- GridWorld、FrozenLake、Taxi：Q-Learning
- CliffWalking：SARSA 与安全路径
- CartPole：DQN 与 PPO 对照
- MountainCar：表格 Q-Learning
- Acrobot、Pendulum：PPO
- MountainCarContinuous：SAC 连续动作控制

完整目录中的 Auto 项会检查动作空间：离散动作使用 DQN，连续动作使用 SAC，其他兼容动作空间使用 PPO。ALE 0.12 自带 ROM，无需在页面中下载；MuJoCo 与 Robotics 使用当前维护的 Python 绑定。目录也保留 Gymnasium 注册表中的历史版本，但建议优先选择当前版本（例如 MuJoCo v5、Fetch v4），避免已废弃的 `mujoco-py` 运行时。

页面会先为 12 个精选任务显示随代码部署的真实训练示例：表格任务使用策略图或热图，可渲染控制任务使用策略回放 GIF。训练过程中保留当前预览，训练完成后再用本次新生成的 GIF、策略图或结果 PNG 替换它。环境无法输出 RGB 帧时自动生成包含最终指标和学习轨迹的结果 PNG；训练或历史环境初始化失败时也会生成诊断结果图和 JSON 摘要。

课程项目：<https://github.com/walkinglabs/hands-on-modern-rl>

课程网站：<https://walkinglabs.github.io/hands-on-modern-rl/>
