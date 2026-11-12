# 24.3 VLA 模型

前两节把多模态强化学习扩展到音频理解与语音交互，但智能体仍在处理数字信号。VLA 进一步把视觉、语言和动作连成闭环：模型需要理解环境，并在物理世界中操控机器人、驾驶汽车或完成工业任务。真实动作有成本，环境不能随时 `reset()`，策略还必须服从动力学与安全约束。

这就是**具身智能（Embodied Intelligence）** 要解决的问题：让 AI 拥有"身体"，在物理世界中感知、决策和行动。

## 什么是具身智能？

具身智能是指**智能体通过物理身体与环境交互，在感知-决策-行动的闭环中完成任务的 AI 系统**。它不是纯粹在数字空间中处理数据的"离身智能"，而是强调智能必须通过身体与真实世界的交互来产生和体现。

这个概念来自认知科学的一个基本洞察：人类的智能不是在真空里"思考"出来的，而是在与物理世界的持续交互中塑造的。婴儿通过抓握、爬行、碰撞来理解物理规律，这种"身体经验"是认知发展的基础。具身智能把这个思想搬到了 AI 领域——让 AI 也通过"身体经验"来学习。

![Stanford Arm](./images/stanford_arm.jpg)

<div style="text-align: center; font-size: 0.9em; color: var(--vp-c-text-2); margin-top: -10px; margin-bottom: 20px;">
  <em>图 1：The Stanford arm，世界上第一台全电动并由计算机控制的机械臂，开启了机器人在物理世界中执行复杂操作的先河。来源：<a href="https://commons.wikimedia.org/wiki/File:The_Stanford_Arm.jpg" target="_blank" rel="noopener noreferrer">Wikimedia Commons</a></em>
</div>

### 核心特征

具身智能系统有三个核心特征：

1. **具身性**：智能体拥有物理或仿真的身体，能对环境施加力、移动物体、改变世界状态
2. **感知-行动闭环**：智能体的感知影响行动，行动改变环境，环境变化又反馈到感知——形成闭环
3. **物理约束**：动作受物理定律约束（重力、摩擦、碰撞），试错有成本，失败有后果

## 具身智能与普通 RL 的区别

你可能已经发现，具身智能用的也是 RL 的框架——状态、动作、奖励、策略。那它和前面学的 CartPole、PPO 训练 LLM 有什么本质区别？

| 维度       | 普通 RL（CartPole、LLM） | 具身智能 RL                        |
| ---------- | ------------------------ | ---------------------------------- |
| 动作空间   | 离散为主（左/右、token） | 连续高维（关节角度、力矩、速度）   |
| 状态空间   | 低维向量或 token 序列    | 多模态（视觉+力觉+本体感觉+语言）  |
| 环境动力学 | 快速模拟，可瞬时重置     | 物理仿真昂贵，真实环境几乎无法重置 |
| 试错成本   | 几乎为零                 | 高（时间、设备损耗、安全风险）     |
| 泛化要求   | 固定环境即可             | 必须适应多变物理条件               |
| 样本效率   | 可以大量试错             | 必须高效利用每一次交互             |

核心差异可以归结为一点：**在普通 RL 中，你不需要关心"动作如何改变物理世界"；在具身智能中，这是所有问题的起点。**

## 具身智能主流任务类型

具身智能的任务五花八门，但按智能体与环境的交互方式，可以归为几大类。在这些任务中，强化学习扮演着不同的角色，解决不同的痛点：

### 1. 导航（Navigation）

**核心目标**：智能体在未知或部分已知的环境中移动到目标位置（例如：扫地机器人回充、自动驾驶汽车从 A 点开到 B 点）。

- **任务细分**：
  - **PointGoal Navigation**：目标由相对坐标（“向前走 5 米，左转”）给出，主要考察避障和路径规划。
  - **ObjectGoal Navigation**：目标是语义概念（“找到厨房里的冰箱”），要求智能体具备常识推理和场景理解能力。
- **技术原理**：传统方法依赖于 SLAM（同步定位与建图）建图后再进行 A\* 等路径规划。而在具身 RL 中，流行**端到端（End-to-End）学习**：直接输入 RGB-D 摄像头和 LiDAR 数据，通过 CNN/Transformer 提取特征，输出离散（前进、左转、右转）或连续的底盘速度指令。
- **难点**：视觉特征与几何空间的映射（Mapping）、长时间记忆（探索过的死胡同不能再去）。

<OnlineTraining studios="ai2thor" compact />

### 2. 抓取（Grasping）

**核心目标**：机械臂从桌面上抓取指定形状、材质或堆叠状态的物体。

- **任务细节**：抓取是机器人操作（Manipulation）中最基础的子任务。对于平行夹爪（Parallel Jaw Gripper），智能体需要预测一个 **6-DoF（六自由度）的抓取位姿**（x, y, z 坐标加上 Roll, Pitch, Yaw 旋转角），并决定何时闭合夹爪。
- **技术原理**：
  - 在 RL 中，抓取通常被建模为一个多步马尔可夫决策过程。状态是桌面物体的点云（Point Cloud）或深度图。
  - **Reward Shaping（奖励塑形）** 在抓取训练中极其关键。如果只给“抓取成功”才给 +1 奖励，智能体会面临极度稀疏的奖励信号。因此，通常需要设置中间奖励：靠近物体（+0.1） -> 触碰到物体（+0.3） -> 成功抓起并抬高（+1.0）。

### 3. 灵巧操作（Dexterous Manipulation）

**核心目标**：使用多指灵巧手（如 Shadow Hand）进行旋转物体、双手协调、工具使用（如拧螺丝、切菜）。

- **技术原理**：比抓取更复杂的操作。灵巧手的自由度极高（比如 Shadow Hand 拥有 24 个关节自由度），动作空间维度动辄达到 20 维以上。在这个高维连续空间中进行探索非常困难。
- **经典案例**：OpenAI 训练机械手解魔方（Solving Rubik's Cube）。他们使用了 PPO 算法，并在仿真中引入了极端的**非对称域随机化（ADR）**，训练了一个能在现实世界中单手转动魔方的策略网络。
- **难点**：极度复杂的**接触动力学（Contact Dynamics）**——手指与物体表面的滑动、摩擦、微小形变，很难在仿真引擎中完美还原。

![PR2 Robot](./images/pr2_robot.jpg)

<div style="text-align: center; font-size: 0.9em; color: var(--vp-c-text-2); margin-top: -10px; margin-bottom: 20px;">
  <em>图 2：配备高级抓取手的 PR2 机器人，常被用于研究灵巧操作和人机交互。来源：<a href="https://commons.wikimedia.org/wiki/File:PR2_robot_with_advanced_grasping_hands.JPG" target="_blank" rel="noopener noreferrer">Wikimedia Commons</a></em>
</div>

### 4. 四足 / 人形行走（Locomotion）

**核心目标**：控制机器人的各个关节，使其在平地、楼梯、草地等复杂地形上保持动态平衡并完成移动（如波士顿动力的 Atlas、宇树科技的 Go2 和 H1）。

- **技术原理**：
  - 传统控制理论通常使用 ZMP（零矩点）和 MPC（模型预测控制）来计算落脚点。
  - 现代具身 RL 则通过神经网络直接输出**关节目标角度（Target Joint Angles）**。底层依然有一个高频（如 1000Hz）运行的 PD 控制器（比例微分控制器）将角度差转化为实际电机扭矩：$\tau = K_p (q_{target} - q_{current}) - K_d \dot{q}_{current}$。
  - RL 策略网络通常以 50Hz 运行，接收 IMU（姿态、角速度）和关节编码器数据，输出目标角度 $q_{target}$ 给到底层 PD 控制器。
- **难点**：欠驱动系统（Underactuated System）的平衡问题、双足机器人的高重心导致容易摔倒。

![BigDog Robot](./images/big_dog.jpg)

<div style="text-align: center; font-size: 0.9em; color: var(--vp-c-text-2); margin-top: -10px; margin-bottom: 20px;">
  <em>图 3：波士顿动力的 BigDog 四足机器人，能够在复杂地形中保持动态平衡和行走。来源：<a href="https://commons.wikimedia.org/wiki/File:Big_dog_military_robots.jpg" target="_blank" rel="noopener noreferrer">Wikimedia Commons</a></em>
</div>

```mermaid
flowchart TB
    subgraph "具身智能任务类型"
        N["导航 Navigation"]
        G["抓取 Grasping"]
        D["灵巧操作 Dexterous Manipulation"]
        L["运动控制 Locomotion"]
    end

    N --> N1["室内导航 / 自动驾驶"]
    G --> G1["物体拾取 / 放置"]
    D --> D1["旋转 / 工具使用 / 组装"]
    L --> L1["四足行走 / 人形平衡"]

    style N fill:#e3f2fd,stroke:#1976d2,color:#000
    style G fill:#e8f5e9,stroke:#388e3c,color:#000
    style D fill:#fff3e0,stroke:#f57c00,color:#000
    style L fill:#f3e5f5,stroke:#7b1fa2,color:#000
```

## 从数学与代码看 与 具身智能与传统 RL 的区别

我们在前面的章节（如 CartPole 或雅达利游戏）中学过的强化学习，和在具身智能（如机械狗、机械臂）上跑的 RL 到底有什么区别？我们可以从数学建模和代码实现两个维度来对比：

### 1. 从 MDP 到 POMDP（部分可观测性）

- **传统 RL（完全可观测）**：通常假设环境是马尔可夫决策过程（MDP）。比如倒立摆的状态 $S = [x, \dot{x}, \theta, \dot{\theta}]$ 只有 4 维，它包含了系统当前**所有的**物理信息，只要有这个状态，下一步的走向就完全确定了。
- **具身智能（部分可观测）**：物理世界是**部分可观测马尔可夫决策过程（POMDP）**。机器人永远无法获取“世界绝对真实的物理状态”（例如准确的质心速度、地面的摩擦系数），只能通过传感器得到带噪声的**观测（Observation）** $O_t$。
  - **数学推导**：策略不再是简单的 $\pi(a_t | s_t)$，而是依赖于历史观测序列来形成一个**置信状态（Belief State）**：$b_t(s) = \mathbb{P}(s_t | O_1, A_1, ..., O_t)$。
  - **代码实战**：在传统 RL 中，你可能只需要一个 MLP 就能搞定。但在具身 RL 中，必须使用**多帧观测堆叠（Frame Stacking）** 或者 RNN/LSTM 来恢复隐含信息（例如从连续两帧的位置估算速度）。

    ```python
    # 传统 RL 的观测空间 (CartPole)
    observation_space = gym.spaces.Box(low=-inf, high=inf, shape=(4,))

    # 具身智能 (如 Unitree Go2 四足机器狗) 的观测空间通常是一个高维字典，并包含历史帧
    observation_space = gym.spaces.Dict({
        'base_lin_vel': Box(shape=(3,)),         # 估算的质心线速度
        'base_ang_vel': Box(shape=(3,)),         # IMU 测量的角速度
        'projected_gravity': Box(shape=(3,)),    # 投影重力（反映机身倾斜）
        'dof_pos': Box(shape=(12,)),             # 12个关节的当前角度
        'dof_vel': Box(shape=(12,)),             # 12个关节的角速度
        'history': Box(shape=(5, 33))            # 过去 5 帧的观测历史堆叠
    })
    ```

### 2. 动作空间 与 从低维离散到高维连续

- **传统 RL（离散输出）**：雅达利游戏的动作是离散的（上下左右），$a_t \in \{0, 1, ..., N\}$，策略网络最后一层通常是 Softmax 输出各类动作的概率。
- **具身智能（连续控制）**：真实机器人的关节电机需要连续的电压、扭矩或目标角度输入。例如，一个拥有 19 个自由度的人形机器人（如 Unitree H1），其动作 $a_t \in \mathbb{R}^{19}$ 是一个 19 维连续向量。
  - **数学建模**：连续控制无法用 Softmax 穷举。我们通常假设策略服从**多维高斯分布（Multivariate Gaussian Distribution）**：
    $$
    \pi_\theta(a|s) = \frac{1}{\sqrt{(2\pi)^k |\Sigma|}} \exp\left(-\frac{1}{2}(a-\mu_\theta(s))^T \Sigma^{-1} (a-\mu_\theta(s))\right)
    $$
    神经网络输出的是一个 $k$ 维的均值向量 $\mu_\theta$（对应每个关节的目标动作）和一个 $k$ 维的标准差向量（决定探索的随机性）。
  - **代码实战**：网络输出的连续动作通常在 `[-1, 1]` 之间，必须通过**动作缩放（Action Scaling）** 映射到物理世界真实的关节角度范围：

    ```python
    # 策略网络输出动作 (假设是 12 自由度机器狗)
    action = policy_net(observation)  # 范围在 [-1.0, 1.0] 之间的 12 维向量

    # 转换为真实世界的目标关节角度 (rad)
    # default_dof_pos 是机器狗站立时的初始姿态，action_scale 通常设为 0.25 左右
    target_dof_pos = default_dof_pos + action * action_scale

    # 将 target_dof_pos 发送给底层的 PD 控制器产生电机扭矩
    ```

### 3. 环境动态与 Sim-to-Real 鸿沟

- **传统 RL（环境一致性）**：训练环境和测试环境的动态转移概率 $\mathcal{P}(s_{t+1}|s_t, a_t)$ 完全一致。你在电脑上训练的雅达利模型，测试时跑的还是那套雅达利代码。
- **具身智能（Sim-to-Real Gap）**：在真机上试错成本太高（动辄摔坏几万块的硬件），必须在仿真中训练。但仿真引擎的物理参数 $\mu_{sim}$（摩擦力、质量、延迟）不可能与真实世界 $\mu_{real}$ 完全一样。
  - **数学建模**：为了防止策略在仿真中过拟合，引入**域随机化（Domain Randomization）**。我们将物理参数 $\mu$ 视为一个分布 $p(\mu)$，目标从“最大化单一环境回报”变为“最大化参数分布下的平均回报”：
    $$
    J(\theta) = \mathbb{E}_{\mu \sim p(\mu)} \left[ \mathbb{E}_{\tau \sim \mathcal{P}_\mu}[R(\tau)] \right]
    $$
  - **代码实战**：在强化学习代码的 `reset()` 或 `step()` 环节，必须对物理世界的各种“脏数据”进行疯狂的随机化注入：

    ```python
    def apply_domain_randomization(self):
        # 1. 随机改变机器人的质量分布 (例如增加或减少 20%)
        self.model.body_mass[:] *= np.random.uniform(0.8, 1.2)

        # 2. 随机改变地面的摩擦力 (模拟瓷砖、地毯、水泥地等不同材质)
        self.model.geom_friction[:] *= np.random.uniform(0.5, 1.5)

        # 3. 模拟真实电机的延迟 (通信延迟)
        # 真实机器人的指令传达到电机有 10~20ms 的延迟，如果不模拟这个，实机必摔
        delayed_action = self.action_history[-self.delay_steps]

        # 4. 给观测数据注入高斯噪声 (模拟传感器误差)
        noisy_observation = true_observation + np.random.normal(0, noise_std)
        return noisy_observation
    ```

### 4. 奖励函数设计 与 从稀疏目标到极度密集的奖励工程 (Reward Shaping)

- **传统 RL（稀疏奖励）**：在雅达利游戏或围棋中，奖励通常极其简单。赢了得 +1，输了得 -1。RL 算法需要在几千步的延迟后自己搞清楚是哪一步做对了。
- **具身智能（密集工程）**：如果只给机器人设定“走到终点 +1，摔倒 -1”，在连续高维的物理世界里，机器人可能永远学不会走路（它可能会发现原地打滚也能蹭向终点）。
  - **核心 Insight**：具身 RL 的大部分精力花在设计一个包含十几个子项的**复合奖励函数（Composite Reward Function）**。
  - **数学建模**：奖励被拆解为“任务奖励（Task Reward）”和“正则化惩罚（Regularization Penalty）”的加权和：
    $$
    R_t = \sum_{i} w_i r_{task, i} - \sum_{j} \lambda_j c_{penalty, j}
    $$
  - **代码实战**：你看真正的四足机器人 RL 代码（比如 `unitree_rl_gym` 中），每一帧的奖励计算可能长达几十行，像是在精雕细琢一门艺术：

    ```python
    def compute_reward(self):
        # 1. 任务奖励：鼓励机器人朝着目标速度前进（追踪指令）
        lin_vel_error = np.sum(np.square(self.commands[:2] - self.base_lin_vel[:2]))
        reward_tracking = np.exp(-lin_vel_error / 0.25) * self.dt

        # 2. 姿态惩罚：惩罚机身倾斜，让它保持平稳
        penalty_orientation = np.sum(np.square(self.projected_gravity[:2])) * -0.5

        # 3. 能量惩罚：惩罚输出过大的电机扭矩，防止电机过热烧毁
        penalty_torques = np.sum(np.square(self.torques)) * -0.00001

        # 4. 动作平滑惩罚：惩罚相邻两帧动作变化过大，防止机器人抽搐（实机非常容易出现）
        penalty_action_rate = np.sum(np.square(self.last_actions - self.actions)) * -0.01

        return reward_tracking + penalty_orientation + penalty_torques + penalty_action_rate
    ```

### 5. 网络架构 Insight 与 非对称 Actor-Critic (Asymmetric AC)

- **传统 RL**：在 PPO 等 Actor-Critic 算法中，Actor 网络（策略 $\pi$）和 Critic 网络（价值 $V$）通常接收**完全一样**的观测数据（State/Observation）。
- **具身智能（Privileged Learning）**：在仿真环境中，我们其实拥有“上帝视角”。仿真引擎知道当前的绝对精确质量、摩擦系数、风速等隐藏信息。但真实的机器人到了物理世界，只能看到带噪声的相机和 IMU 数据。
  - **核心 Insight**：为什么不让 Critic 拥有“上帝视角”，而只让 Actor 保持“机器人视角”呢？这就是**非对称 Actor-Critic（Asymmetric Actor-Critic / 老师教学生范式）**。
  - **数学与工程优势**：由于 Critic 在训练时能看到没有噪声的特权状态（Privileged State $S_{priv}$），它对价值 $V(S_{priv})$ 的估计极其精准。这极大降低了策略梯度的方差，使 Actor $\pi(a|O_{noisy})$ 能够又快又稳地收敛。部署时，我们只需要带走 Actor 网络，不需要 Critic。
  - **代码实战**：网络输入发生了分离：

    ```python
    # 具身智能训练时的 Asymmetric Actor-Critic

    # 1. 真实世界的观测 (带噪声的 IMU、相机深度图等)
    obs = env.get_noisy_observation()

    # 2. 仿真器独有的特权信息 (绝对速度、摩擦系数、推力干扰)
    privileged_obs = env.get_privileged_state()

    # 3. 价值网络 (Critic) 吃特权信息，给出的 V 值极准
    value = critic_net(privileged_obs)

    # 4. 策略网络 (Actor) 只能吃带噪声的观测，学会在不确定性中做决策
    action = actor_net(obs)

    # 部署到真机时，扔掉 Critic，只保留 Actor
    ```

## 具身智能强化学习主流范式

具身智能的 RL 训练不是从头发明新算法，而是在我们前面学过的算法中选择最合适的，并针对物理世界的特点做适配。

### PPO——具身最常用

[PPO](https://arxiv.org/abs/1707.06347) 是具身智能中**使用最广泛的 RL 算法**。原因很简单：

- **稳定性**：PPO 的裁剪机制（clipping）天然适合连续动作空间，不会像 Vanilla PG 那样容易策略崩溃
- **可扩展性**：PPO 可以高效地在大量并行仿真器上训练，这在具身智能中至关重要
- **数据利用率**：相比 SAC 等离线策略算法，PPO 的数据利用效率略低，但训练吞吐量更高

在 [NVIDIA Isaac Gym](https://developer.nvidia.com/isaac-gym) 上，PPO 被用来同时训练数千个机器人——正是它在大规模并行采样上的优势让它成为具身 RL 的默认选择。OpenAI 的机械手解魔方（[Solving Rubik's Cube with a Robot Hand](https://openai.com/index/solving-rubiks-cube/)）、四足机器人 [ANYmal](https://www.anybotics.com/) 的地形行走，都是 PPO 的代表性应用。

### SAC 的定位

[SAC](https://arxiv.org/abs/1801.01290)（Soft Actor-Critic）在具身智能中扮演"精细调优"的角色。它的最大熵框架（Maximum Entropy）鼓励策略探索——这对灵巧操作等需要多样性行为的任务特别有价值。

SAC 的优势在于**样本效率**——它通过经验回放反复利用历史数据，每一步交互的数据都能被多次学习。在真实机器人上训练时（不是仿真），样本效率至关重要，因为真实交互的成本太高。

实践中，具身 RL 的典型范式是：**先用 PPO 在大规模仿真中预训练，再用 SAC 在真实机器人上微调**——利用 SAC 的样本效率来最小化真实世界的交互次数。

## 具身智能常用仿真环境

![Simulation Environment](./images/simulation.gif)

<div style="text-align: center; font-size: 0.9em; color: var(--vp-c-text-2); margin-top: -10px; margin-bottom: 20px;">
  <em>图 4：仿真环境中的机器人控制测试。高质量的物理仿真引擎是训练具身智能体不可或缺的基础设施。来源：<a href="https://commons.wikimedia.org/wiki/File:5R_robot.gif" target="_blank" rel="noopener noreferrer">Wikimedia Commons</a></em>
</div>

仿真环境是具身 RL 的基础设施——没有高质量的仿真器，具身智能就只能在真实机器人上试错，成本不可接受。以下是目前工业界和研究中最常用的几个仿真器：

### MuJoCo

![MuJoCo Simulation](./images/mujoco_sim.png)

<div style="text-align: center; font-size: 0.9em; color: var(--vp-c-text-2); margin-top: -10px; margin-bottom: 20px;">
  <em>图 5：机器人关节与动力学仿真示例。像 MuJoCo 这样的物理引擎能够提供精确的接触建模和运动学计算。来源：<a href="https://commons.wikimedia.org/wiki/File:SpaceRoboticsChallenge_Task2.png" target="_blank" rel="noopener noreferrer">Wikimedia Commons</a></em>
</div>

[MuJoCo](https://mujoco.org/)（Multi-Joint dynamics with Contact）是机器人仿真领域的"老牌标准"。它的核心优势是**接触动力学建模精确、仿真速度快**。MuJoCo 能精确模拟刚体之间的碰撞和接触力——这对抓取、行走等任务至关重要。DeepMind 的 [dm_control](https://github.com/google-deepmind/dm_control) suite 和 OpenAI Gym 的连续控制环境都基于 MuJoCo。2021 年 MuJoCo 被 DeepMind 收购并开源，目前免费可用。

### Isaac Gym / Isaac Sim

![Isaac Sim Simulation](./images/isaac_sim.png)

<div style="text-align: center; font-size: 0.9em; color: var(--vp-c-text-2); margin-top: -10px; margin-bottom: 20px;">
  <em>图 6：多机器人并行仿真框架示意。Isaac Gym/Sim 系列通过 GPU 并行能力，实现了大规模多机器人的同时训练。来源：<a href="https://commons.wikimedia.org/wiki/File:Asynchronous_Multi-Body_Framework,_AMBF.png" target="_blank" rel="noopener noreferrer">Wikimedia Commons</a></em>
</div>

NVIDIA 的 Isaac 系列是**GPU 并行机器人仿真**的代表。Isaac Gym（已 deprecated，继任者为 Isaac Lab/Isaac Sim）能在单块 GPU 上并行仿真数千个机器人环境，训练速度比 MuJoCo 快一到两个数量级。

::: info Isaac Lab 替代 Isaac Gym
NVIDIA 的 Isaac Gym 已被标记为 deprecated，其继任者 **Isaac Lab** 是目前 GPU 并行机器人仿真的标准工具。Isaac Lab 基于 Isaac Sim 构建，支持万级机器人同时训练，同时兼容 Gymnasium 接口。安装方式：`pip install isaacsim[all]`，详见 [Isaac Lab GitHub](https://github.com/isaac-sim/IsaacLab)。
:::

Isaac 系列的核心价值是**把仿真从 CPU 瓶颈解放出来**。PPO + Isaac Gym 的组合，让原本需要几天的训练缩短到几十分钟。

### ManiSkill / Sapien

![ManiSkill Simulation](./images/maniskill_sim.jpg)

<div style="text-align: center; font-size: 0.9em; color: var(--vp-c-text-2); margin-top: -10px; margin-bottom: 20px;">
  <em>图 7：使用机器人手臂进行抓取和操作（Manipulation）任务的仿真场景。这是 ManiSkill 和 Sapien 等仿真器最擅长的领域。来源：<a href="https://commons.wikimedia.org/wiki/File:Robotic_simulation_using_Robcad_software.jpg" target="_blank" rel="noopener noreferrer">Wikimedia Commons</a></em>
</div>

[ManiSkill](https://maniskill.readthedocs.io/)（基于 [Sapien](https://sapien.ucsd.edu/) 仿真器）是专注**机器人操作（Manipulation）** 的仿真基准。它提供了一系列标准化的操作任务——从简单的推物体到复杂的多步骤组装——以及统一的评测接口。如果你做的是抓取、放置、组装类任务，ManiSkill 是目前最成熟的 benchmark。

<OnlineTraining studios="maniskill" compact />

| 仿真器                | 核心优势                     | 最适合的场景                  |
| --------------------- | ---------------------------- | ----------------------------- |
| MuJoCo                | 接触建模精确，物理可信度高   | 精细控制、学术研究基准        |
| Isaac Lab / Isaac Sim | GPU 并行，万级环境同时训练   | 大规模策略训练、四足/人形行走 |
| ManiSkill / Sapien    | 操作任务标准化，评测体系完善 | 抓取、放置、组装类任务        |

## Sim-to-Real 与 仿真到现实迁移核心思想

![Sim-to-Real](./images/sim_to_real.png)

<div style="text-align: center; font-size: 0.9em; color: var(--vp-c-text-2); margin-top: -10px; margin-bottom: 20px;">
  <em>图 8：仿真与现实环境的对比。Sim-to-Real 的核心挑战在于跨越物理世界与数字世界之间的差异（Sim-to-Real Gap）。来源：<a href="https://commons.wikimedia.org/wiki/File:Xenobot_sim_to_real.png" target="_blank" rel="noopener noreferrer">Wikimedia Commons</a></em>
</div>

在仿真器中训练好策略后，下一步是迁移到真实机器人。这个过程叫做 **Sim-to-Real（从仿真到现实）**，是具身智能最关键也最困难的环节。

为什么不能直接把仿真策略部署到真实机器人？因为**仿真和现实之间存在差距（Sim-to-Real Gap）**：

- **物理差距**：模拟器中的摩擦力、重力、碰撞都是理想化的，真实世界的物理要复杂得多
- **感知差距**：模拟器提供精确的状态信息（关节角度、物体位置），真实机器人只能通过摄像头和力传感器来估计
- **动力学差距**：模拟器中的电机响应是即时的，真实电机有延迟和非线性

### 域随机化（Domain Randomization）

解决 Sim-to-Real Gap 最常用的技术。核心思想极其简单：**在训练时大量随机化仿真参数**——摩擦力在 0.3-0.7 之间随机、物体颜色随机、光照条件随机、传感器噪声随机。这样训练出来的策略不再依赖于特定的物理参数，而是学会了在各种条件下都能工作。

```python
def domain_randomized_env(base_params):
    """域随机化：在训练时随机化物理参数"""
    randomized = {}
    randomized["friction"] = np.random.uniform(0.3, 0.7)
    randomized["gravity"] = base_params["gravity"] * np.random.uniform(0.9, 1.1)
    randomized["joint_damping"] = np.random.uniform(0.01, 0.1)
    randomized["object_mass"] = base_params["mass"] * np.random.uniform(0.8, 1.2)
    return randomized
```

直觉上，这就像一个篮球运动员在训练时故意用不同气压的球、不同摩擦力的地板来练习——这样到了正式比赛，无论条件如何都能适应。

### 系统辨识（System Identification）

另一种思路是**让仿真更接近真实**：先用真实机器人的数据来估计物理参数（摩擦系数、关节阻尼等），然后在这些"校准过"的参数上训练策略。这比域随机化更精确，但也更耗时。

实践中两者常常结合使用：域随机化作为基础策略覆盖不确定性，系统辨识缩小需要随机的范围。

## 具身智能前沿

### 扩散策略（Diffusion Policy）

我们在第 8 章学过 PPO 的确定性策略输出——给定状态，网络输出一个动作向量。但很多操作任务的动作分布是多模态的：同一个目标，可能有多种有效的抓取方式。

扩散策略（[Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)）把动作生成建模为一个**去噪扩散过程**——从随机噪声开始，逐步去噪生成最终动作。这和图像扩散模型的思路完全一致，只是生成对象从"图像"变成了"动作序列"。扩散策略天然支持多模态动作分布，在灵巧操作任务上表现优异。

### 视觉-语言-动作模型（VLA）

![Vision-Language-Action](./images/vla_robot.jpg)

<div style="text-align: center; font-size: 0.9em; color: var(--vp-c-text-2); margin-top: -10px; margin-bottom: 20px;">
  <em>图 9：具备视觉感知和抓取能力的机器人系统。VLA 模型通过结合视觉输入和自然语言指令，直接生成机器人的物理动作，是实现通用具身智能的关键技术。来源：<a href="https://commons.wikimedia.org/wiki/File:Care-O-Bot_grasping_an_object_on_the_table_(5117071459).jpg" target="_blank" rel="noopener noreferrer">Wikimedia Commons</a></em>
</div>

VLA（Vision-Language-Action）是具身智能最新、最令人兴奋的方向。它把视觉理解、语言指令理解和动作生成统一到一个模型中。

典型流程：机器人收到指令"把桌子上的红色杯子拿起来"→ VLA 模型接收摄像头画面 + 语言指令 → 直接输出机器人的关节控制指令。

Google 的 [RT-2](https://robotics-transformer2.github.io/)（Robotic Transformer 2）是 VLA 的代表性工作：它用一个预训练的视觉语言模型，把机器人控制任务转化为"视觉问答"问题——输入图像和语言指令，输出动作 token。这和我们第 23 章 VLM RL 的思路一脉相承——只是输出从"文本回答"变成了"物理动作"。

```mermaid
flowchart LR
    V["视觉输入\n（摄像头画面）"] --> M["VLA 模型"]
    L["语言指令\n（'抓起红杯子'）"] --> M
    M --> A["动作输出\n（关节角度）"]

    style V fill:#e3f2fd,stroke:#1976d2,color:#000
    style L fill:#e8f5e9,stroke:#388e3c,color:#000
    style M fill:#fff3e0,stroke:#f57c00,color:#000
    style A fill:#f3e5f5,stroke:#7b1fa2,color:#000
```

VLA 的核心意义在于：它把具身智能从"针对特定任务训练特定策略"推进到了"通用策略理解语言指令并执行"——这是走向通用具身智能的关键一步。

### 从 RT-2 到 Gemini Robotics 1.5 与 VLA 的 2025 进化

RT-2（2023）证明了 VLA 范式的可行性，但还有几个根本限制：

- **动作空间简单**：RT-2 主要做抓取、放置等单步动作
- **缺少长 horizon 规划**：无法处理多步任务（如"做一道菜"，需要切菜、调味、烹饪）
- **没有显式推理**：模型直接给动作，没有"思考"过程

2025 年 3 月，Google DeepMind 发布了 [Gemini Robotics 1.0](https://deepmind.google/models/gemini-robotics/)——基于 Gemini 2.0 的 VLA 模型。2025 年 6 月又发布了 [Gemini Robotics 1.5](https://deepmind.google/blog/gemini-robotics-15-brings-ai-agents-into-the-physical-world/)，带来几个关键突破：

**突破 1：Embodied Thinking**

Gemini Robotics 1.5 引入了 **Embodied Thinking**——模型在执行动作前显式"思考"：

```text
指令："把蓝色积木放到红色积木上面"

模型思考：
1. 我需要先识别蓝色和红色积木的位置
2. 计算抓取蓝色积木的最佳角度
3. 移动到目标位置
4. 释放积木，确保稳定

模型执行：
[一系列精确的关节控制指令]
```

这种推理过程让机器人可以处理**多步、有依赖**的任务——这是 RT-2 做不到的。

**突破 2：多模态理解**

Gemini Robotics 1.5 不仅看图像，还整合：

- 视觉（摄像头）
- 语言（指令）
- 触觉（力反馈传感器）
- 本体感觉（关节位置、速度）

这种多模态融合让机器人理解更全面的环境状态。

**突破 3：泛化能力**

Gemini Robotics 1.5 在多个**未见过的环境**上展示泛化：

- 新的物体（训练时没见过的工具）
- 新的场景（不同的厨房布局）
- 新的指令组合（"先做 X 再做 Y"的组合指令）

泛化能力来自 Gemini base model 的强语义理解——把"通用语言能力"迁移到机器人控制。

**突破 4：Sim-to-Real**

Gemini Robotics 1.5 大量在仿真环境训练，部署到真实机器人：

- 仿真环境：大规模并行训练（10 万 + 并行环境）
- 真实部署：domain randomization + 适应训练

**Gemini Robotics 1.5 的工业意义**：

1. **VLA 进入实用阶段**：不再是实验室 demo，可以处理真实工业场景
2. **Embodied Thinking**：与文本 reasoning model 的"思考"对应——机器人也开始"思考"
3. **中国厂商追赶**：字节 RoboBrain、智元远征、银河通用等都在做 VLA

### VLA 的训练方法

VLA 训练与文本 LLM RL 有相似之处，但增加几个组件：

```text
┌──────────────────────────────────────────────────────────┐
│ Phase 1: 多模态预训练                                    │
│   - 视觉 + 语言 + 动作数据                              │
│   - 学习"指令 → 动作"的基础映射                         │
├──────────────────────────────────────────────────────────┤
│ Phase 2: 仿真环境 RL                                     │
│   - 在仿真环境做大量 RL                                 │
│   - 奖励：任务成功 + 动作平滑 + 效率                    │
├──────────────────────────────────────────────────────────┤
│ Phase 3: 真实环境适应                                    │
│   - 在真实机器人做少量 RL/SFT                           │
│   - 修正 sim-to-real gap                                │
├──────────────────────────────────────────────────────────┤
│ Phase 4: Embodied Thinking 训练                          │
│   - 让模型在动作前生成推理                              │
│   - 用 reasoning reward 强化思考行为                    │
└──────────────────────────────────────────────────────────┘
```

这种多阶段训练与 [DeepSeek-R1](../../chapter18_grpo/deepseek-dapo) 的多阶段范式高度相似——只是把"数学推理"换成了"机器人动作"。

## 用 PPO 训练一个仿真机器人奔跑

为了让你直观体验具身智能的训练过程，我们这里提供一个**计算资源消耗极小**的实战代码。你只需要一台普通的笔记本电脑（无需 GPU，仅用 CPU），在几分钟内就能训练出一个能够向前奔跑的虚拟机器人（HalfCheetah）。

我们将使用前文提到的 `MuJoCo` 物理仿真引擎（目前已开源并集成在 Gymnasium 中），以及 `Stable-Baselines3` 库中的 `PPO` 算法。

### 1. 环境与机器人来源

首先，确保你的环境中安装了以下依赖：

```bash
pip install gymnasium[mujoco] stable-baselines3
```

![HalfCheetah Simulation](./images/halfcheetah.gif)

<div style="text-align: center; font-size: 0.9em; color: var(--vp-c-text-2); margin-top: -10px; margin-bottom: 20px;">
  <em>图 10：Gymnasium 中的 HalfCheetah 机器人模型。你将使用 PPO 算法训练它从随机乱动到学会平稳向前奔跑。</em>
</div>

> **这个半猎豹（HalfCheetah）机器人从哪来的？**
> 它其实是 `gymnasium[mujoco]` 库中自带的经典连续控制基准测试模型（Benchmark Model）。你不需要自己画 3D 模型、设计关节或写物理计算代码，MuJoCo 环境已经内置好了这个由多个刚体（如躯干、大腿、小腿、脚）和旋转关节连接而成的生物力学模型。

### 2. 训练代码

创建一个名为 `train_cheetah.py` 的文件并运行以下代码。这段代码展示了经典的**仿真训练 → 渲染测试**的 Sim-to-Real 雏形（尽管这里只到了 Sim 阶段）：

```python
import gymnasium as gym
from stable_baselines3 import PPO

# ==========================================
# 在不可见（无渲染）的仿真中极速训练
# ==========================================
# 1. 创建 MuJoCo 仿真环境（半猎豹机器人，状态连续，动作连续）
# - 状态(Observation) 与 各关节的角度、角速度等（17 维）
# - 动作(Action) 与 施加在 6 个马达上的扭矩（6 维连续值）
# - 奖励(Reward) 与 向前的速度越快奖励越高，消耗能量或摔倒会扣分
env = gym.make("HalfCheetah-v4")

# 2. 初始化 PPO 智能体
# MlpPolicy 表示使用多层感知机作为策略网络
# verbose=1 会在终端打印训练进度（解释的方差、损失等）
model = PPO("MlpPolicy", env, verbose=1)

print("🚀 开始在 MuJoCo 仿真中训练 PPO...")
# 3. 开始训练
# HalfCheetah 相对比较好训练。在普通笔记本 CPU 上：
# - 跑 20万步（约 2~3 分钟） 与 机器人能学会手脚并用地向前蹭或跌跌撞撞地走。
# - 跑 100万步（约 10 分钟） 与 机器人能学会非常平稳、协调的奔跑姿势。
model.learn(total_timesteps=200_000)

# 4. 保存训练好的"大脑"
model.save("ppo_halfcheetah")
env.close()
print("✅ 训练完成，模型已保存！")

# ==========================================
# 加载"大脑"并进行可视化测试
# ==========================================
# 重新创建一个带可视化界面的环境 (render_mode="human")
eval_env = gym.make("HalfCheetah-v4", render_mode="human")
model = PPO.load("ppo_halfcheetah")

obs, info = eval_env.reset()

print("👀 开始渲染机器人的表现...")
for _ in range(1000):
    # 智能体根据当前状态输出动作（deterministic=True 表示不进行探索，直接输出最优动作）
    action, _states = model.predict(obs, deterministic=True)

    # 仿真环境物理引擎计算动作影响，返回下一帧画面、状态和奖励
    obs, reward, terminated, truncated, info = eval_env.step(action)

    if terminated or truncated:
        obs, info = eval_env.reset()

eval_env.close()
```

### 3. 你会看到什么？

1. **终端输出**：你会看到 PPO 在打印 `ep_rew_mean`（平均回合奖励）。你会发现这个数值从负数或者很小的正数，随着时间步（timesteps）的增加稳步上升。
2. **可视化窗口**：训练完成后，会弹出一个 MuJoCo 渲染窗口。你会看到一个两足的"半猎豹"机器人在物理世界中拼命向前奔跑，这就是 RL 纯靠**试错**和**奖励反馈**学习到的物理控制策略。

这就是具身智能最底层的逻辑：**物理仿真器提供交互反馈，RL 算法（如 PPO）优化控制策略网络。**

## 与前面章节的联系

| 前面章节的概念                    | 在具身智能中的对应                           |
| --------------------------------- | -------------------------------------------- |
| PPO 的稳定性与并行采样（第 8 章） | 具身 RL 最常用的训练算法，支持大规模并行仿真 |
| Actor-Critic 架构（第 7 章）      | SAC 的基础，用于真实机器人上的高效微调       |
| VLM RL 的多模态融合（第 23 章）    | VLA 模型：视觉 + 语言 → 动作                 |
| 策略梯度定理（(第 6 章）          | 连续动作空间策略优化的数学基础               |
| RLHF 的奖励建模（第 13 章）        | 人类偏好反馈用于对齐机器人行为               |

<details>
<summary>思考题：为什么 PPO 比 SAC 更常用于具身智能的大规模仿真训练？</summary>

核心原因是**训练吞吐量**。在大规模并行仿真（如 Isaac Lab）中，你可以在 GPU 上同时跑数千个环境。PPO 的在线策略特性意味着每一批数据用完就可以扔掉——不需要维护经验回放缓冲区，内存效率极高。

SAC 的离线策略框架虽然样本效率更高，但它需要一个大的经验回放缓冲区（replay buffer），在数千个并行环境同时产生的数据面前，缓冲区的内存开销和采样开销会成为瓶颈。

简单说：**PPO 牺牲了单样本利用率，换来了更高的整体训练吞吐量**——在大规模仿真中，这个权衡是划算的。

</details>

::: tip 继续阅读：Model-Based RL
具身智能真正走进物理世界时，最大瓶颈往往不是算法公式，而是真实交互的成本。下一节将 MBRL 展开为独立专题：[Model-Based RL：从 Model-Free 到 Model-Based](./model-based-rl/)。
:::

## 延伸阅读与参考资料 与 宇树科技（Unitree）开源生态

如果想在物理世界中真正部署一个能跑能跳的机器人，不仅需要理论，更需要靠谱的硬件和开源框架。国内机器人公司**宇树科技（Unitree Robotics）** 提供了目前工业界非常完善的强化学习开源生态。你可以参考以下资料，将我们在仿真中训练大脑的思路，应用到真实的四足或人形机器人上：

![Unitree RL GYM](./images/unitree_rl.jpeg)

<div style="text-align: center; font-size: 0.9em; color: var(--vp-c-text-2); margin-top: -10px; margin-bottom: 20px;">
  <em>图 11：宇树科技基于强化学习训练的 Go2 四足机器人和 H1 人形机器人。来源：<a href="https://github.com/unitreerobotics/unitree_rl_gym" target="_blank" rel="noopener noreferrer">unitree_rl_gym GitHub</a></em>
</div>

### 1. 核心训练框架 与 `unitree_rl_gym`

- **GitHub 仓库**：[unitreerobotics/unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym)
- **功能细节**：这是一个基于 Isaac Gym 和 MuJoCo 的强化学习代码库，专门为 Unitree Go2（四足）、G1 和 H1（人形）等机器人设计。
- **它解决了什么痛点？**
  - **端到端流程**：它定义了一个极其清晰的 pipeline：`Train -> Play -> Sim2Sim -> Sim2Real`。
  - **奖励函数设计**：你可以在代码中看到极度复杂的 `Reward Shaping`（例如：惩罚关节速度过快、惩罚脚体滑动、奖励跟踪目标速度等）。
  - **域随机化（Domain Randomization）**：代码里内置了对摩擦力、电机刚度、甚至延迟的随机化配置，确保模型能直接在真机上跑起来。

### 2. 传统与现代控制的融合 与 `unitree_guide`

- **GitHub 仓库**：[unitreerobotics/unitree_guide](https://github.com/unitreerobotics/unitree_guide)
- **功能细节**：如果你不仅想了解深度强化学习（RL），还想了解传统的机器人控制学，这个仓库是一个宝库。
- **为什么需要它？**
  - 真实机器人的控制通常不是“纯 RL”。在工业界，最成熟的方案往往是 **RL 与 MPC（模型预测控制）的结合**。
  - 这个仓库包含了如何通过传统的 WBC（全身控制）和 MPC 来解算机器人的动力学方程，你可以非常直观地对比传统控制方程与 RL 神经网络“端到端”黑盒控制的区别。

### 3. 具身智能开发者社区指南

- **文档链接**：[宇树科技支持文档 (Developer Support)](https://support.unitree.com/home/zh/developer)
- **学习建议**：在这里你可以找到从零开始配环境的保姆级教程（例如怎么装 PyTorch、Isaac Gym，怎么配置 `rsl_rl`）。对于想要亲手把强化学习算法部署到真实电机上的开发者，这是必读的实战手册。

VLA 讨论的是多模态模型如何感知并改变物理世界。下一节 [24.4 视觉生成与 DanceGRPO](../../chapter29_visual_generation/visual-generation-dancegrpo) 转向另一类动作空间：策略通过完整的去噪轨迹生成图像，再用视觉质量、文本一致性和人类偏好构造奖励。
