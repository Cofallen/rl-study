---
title: PPO 游戏项目实践导论
description: 从入门到进阶的 PPO 游戏项目选点指南，按技术主题自然展开，不贴难度标签，每个项目就事论事地讨论 PPO 怎么用、结果如何、能学到什么。
outline:
  level: [2, 3]
---

# 8.5 动手：PPO 游戏项目

学完 PPO 的数学原理和代码实现后，下一步是把它放到真实的游戏环境里跑通。本节的目的不是罗列所有能用 PPO 做的游戏，而是帮你建立一条从"跑通"到"理解边界"的实践路径。

## 学习目标

完成本节学习后，你应该能够：

1. **判断一个游戏是否适合作为 PPO 入门项目**：低维状态、小动作空间、即时反馈是三个关键指标。
2. **区分"能跑"和"能教"**：PPO 能跑通不代表它是最佳选择；有些项目更适合讨论探索、泛化或奖励设计。
3. **按自己的需求选项目**：从几小时能跑通的 Flappy Bird，到需要持续调试的 Pokemon Red，每个项目都有明确的目标和退出条件。

## 怎么评估一个项目

选项目时不看"难度标签"，而是看三个**技术维度**的叠加：

| 维度         | 含义             | 低要求       | 高要求            |
| ------------ | ---------------- | ------------ | ----------------- |
| **状态空间** | 策略看到什么     | 低维向量     | 原始像素 + 帧堆叠 |
| **动作空间** | 策略能做什么     | 离散、少量   | 连续、多维        |
| **反馈延迟** | 动作到奖励的距离 | 一步内见分晓 | 几十步甚至数百步  |

Flappy Bird 三个维度都低，跑起来最快；Pokemon Red 三个维度都高，失败模式也最多。下面逐个讨论每个项目在这三个维度上的表现，以及 PPO 具体是怎么用的、结果如何、你能从中学到什么。

---

### Flappy Bird 与 最简闭环

![Flappy Bird PPO 运行示例，来源：wangjia184/rl 项目](https://user-images.githubusercontent.com/44725090/67148880-e7dba280-f2a4-11e9-8dbf-d154842ee0cf.gif)

Flappy Bird 是验证"**PPO 真的能学**"的最快路径。状态几乎只有四个数——小鸟高度、速度、下一根管道距离、缺口位置，动作只有两个：拍翅膀或者不拍。失败反馈在**瞬间发生**，策略不需要记忆太多历史。

**Dhyanesh18** 的实现基于 **Stable-Baselines3**，用 `CnnPolicy` 处理像素输入，做了 **4 帧堆叠**，还写了一个**熵系数随时间退火**的 callback。不过对入门来说，先用**低维向量**跑通更有意义：一个 2-3 层的 MLP 就足够，学习率 **$3 \times 10^{-4}$**，gamma **0.99**，PPO clip **0.2**，训练到前几百个 episode 就能观察到策略从频繁撞管变成稳定穿越。像素版训练量更大，大约 **10M timesteps** 后策略能长时间稳定飞行。

这个项目的价值在于过程本身——你能在极短的时间里走完"**环境封装 → 策略网络 → 训练循环 → 可视化**"的完整链路，确认你的 PPO 代码没有 bug。

| 入口类型 | 链接                                                                                                                                                                        |
| -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 玩家入口 | [Flappy Bird 网页试玩](https://flappybird.gg/)                                                                                                                              |
| 训练环境 | [flappy-bird-gymnasium](https://pypi.org/project/flappy-bird-gymnasium/)                                                                                                    |
| PPO 参考 | [Yuanpeng-Li/Flappy-Bird-AI](https://github.com/Yuanpeng-Li/Flappy-Bird-AI)（PPO 实现）、[Dhyanesh18/flappbird-rl](https://github.com/Dhyanesh18/flappbird-rl)（PPO + A2C） |

---

### Snake 与 奖励设计的第一课

![Snake PPO2 训练结果示例，来源：gym-snake-rl 文章](https://d1k6fapei95iy6.cloudfront.net/imgs_taemin/ppo2-fullobs.gif)

Snake 比 Flappy Bird 多了一个维度：**身体长度随时间变化**。策略不能只看食物在哪，还得记住自己身体的形状，否则吃食物的途中就把自己缠死了。

这里最值得一做的是**奖励设计实验**。如果你只给"**吃到食物 +10，死亡 -10**"，策略大概率学会**原地转圈**——既不撞墙也不碰食物，靠无限苟活来维持正回报。需要引入**时间惩罚**或者**距离奖励**，才能逼它主动探索。用 **Stable-Baselines PPO2** 训练时，5×5 小棋盘上的最高分能到 **23**（接近理论最优），10×10 标准棋盘训练 **100M steps** 后平均约 **6-7 分**。对比 **DQN** 会发现，PPO 的峰值分数更高但收敛更慢，需要 **900k steps** 对 DQN 的 **140k**，这说明**策略梯度方法在小规模离散任务上的样本效率确实不如值函数方法**。

这个项目会让你第一次直面"**局部最优**"——策略发现转圈比觅食更安全，于是永远停留在舒适区。同样的 PPO 算法，奖励函数不同，策略行为完全不同。

| 入口类型 | 链接                                                                                                                     |
| -------- | ------------------------------------------------------------------------------------------------------------------------ |
| 玩家入口 | [Snake 网页试玩](https://snaketap.com/)                                                                                  |
| 训练环境 | [gym-snake-rl](https://jfpettit.svbtle.com/introducing-gym-snake-rl)、[Gym-Snake](https://github.com/grantsrb/Gym-Snake) |
| PPO 参考 | [Introducing gym-snake-rl](https://jfpettit.svbtle.com/introducing-gym-snake-rl)（PPO2 训练示例）                        |

---

### 即时奖励的陷阱

![2048 PPO 训练结果示例，来源：tejpshah/2048-DeepRL](https://github.com/tejpshah/2048-DeepRL/raw/main/gifs/PPO.gif)

2048 没有复杂画面，但引入了一个关键概念：**即时奖励会误导策略**。如果你只按合并得分给奖励，策略会疯狂合并小的 tile，很快把棋盘堵死。真正的挑战在于学会"**保留空间**"和"**把大数推到角落**"，而这些行为在几十步之后才会兑现成回报。

**arturf1** 的 **Unity ML-Agents PPO** 实现对比了两种奖励方案。**复杂奖励**（合并分 + 新最高 tile 奖励 + 到达 2048 的 bonus）训练 **215k 局**后胜率只有 **4%**；而**简单奖励**（仅首次达到新 tile 时给奖励）训练 **450k 局**后胜率达到 **37%**。这完全反直觉——**更简单的奖励函数反而让策略学会了更好的全局策略**。**tejpshah** 的 PyTorch PPO 用 one-hot 编码的 4×4 棋盘做输入，偶尔能达到 **512 tile**，多数停在 **256**；同一套代码里的 **DDQN** 反而频繁达到 **1024** 甚至 **2048**。

这个项目最有价值的不是跑通游戏，而是让你体会：**同一个 PPO 算法，奖励函数的设计决定了策略能走多远**。而且在这个任务上，**值函数方法确实比策略梯度更优**——因为 2048 的状态转移相对确定，值估计的方差小。

| 入口类型 | 链接                                                                                                                                                       |
| -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 玩家入口 | [2048 原版网页试玩](https://classic.play2048.co/)                                                                                                          |
| 训练环境 | [gymnasium-2048](https://pypi.org/project/gymnasium-2048/)、[2048 源码](https://github.com/gabrielecirulli/2048)                                           |
| PPO 参考 | [arturf1/2048](https://github.com/arturf1/2048)（Unity ML-Agents PPO）、[tejpshah/2048-DeepRL](https://github.com/tejpshah/2048-DeepRL)（PPO + DDQN 对比） |

---

### Tetris 与 长期规划的试炼

![Tetris 游戏截图，来源：Wikimedia Commons](https://upload.wikimedia.org/wikipedia/commons/5/5c/Tetris_freemade.png)

Tetris 的失败原因经常来自十几步前留下的洞。一个方块放错位置，可能在几十步之后才导致游戏结束。动作虽然是离散的，但长期规划的要求很高。

最近一篇用 bitboard 优化的 PPO 论文（arXiv:2603.26765）提出了 **afterstate-evaluating actor 网络**，在 10×10 小棋盘上训练 **61,440 steps**（约 3 分钟）平均得分 **3829**，最佳测试得分 **4124**。但把这个策略直接放到标准 10×20 棋盘上时，分数极高但方差极大——说明**小规模训练的策略根本没学会 Tetris 的核心规则，只是记住了小棋盘的局部模式**。实际上，**PPO 在 Tetris 上通常被精心设计的启发式算法和进化算法超越**，这让它成为讨论"**RL 对比专家知识**"的好案例。

起步时建议先用棋盘矩阵 + 当前方块做输入，不要直接上 RGB 像素。奖励可以从消行、高度惩罚、洞数惩罚、死亡惩罚四项开始组合实验。

| 入口类型 | 链接                                                                                                                      |
| -------- | ------------------------------------------------------------------------------------------------------------------------- |
| 玩家入口 | [Tetris 官方网页试玩](https://play.tetris.com/)                                                                           |
| 训练环境 | [ALE Tetris](https://ale.farama.org/environments/tetris/)、[tetris-gymnasium](https://pypi.org/project/tetris-gymnasium/) |
| PPO 参考 | [chirbard/ppo-Tetris-v5](https://huggingface.co/chirbard/ppo-Tetris-v5)（PPO 模型）                                       |

---

### Breakout 与 像素输入的标准流程

![Breakout PPO 示例，来源：Stable-Baselines3 文档](https://stable-baselines3.readthedocs.io/en/master/_images/breakout.gif)

Breakout 是第一次"**从像素画面玩游戏**"的理想选择。球的位置、速度、挡板位置不再作为显式数字给出，策略要从连续帧里推断出来。**Stable-Baselines3 的 PPO** 在这个环境上是强 baseline：**10M steps** 后平均得分 **398 ± 33**，**5M steps** 约 **300 分**（接近人类水平）。不同随机种子下分数在 **360-430** 之间波动。

这里的关键工程点是**预处理流水线**。单帧画面无法判断球的运动方向，所以**帧堆叠**（通常 **4 帧**）是必须的。**灰度化**和**图像裁剪**能大幅减少计算量——有实验表明，100k steps 时灰度输入约 **150 分**，而原始 RGB 只有 **-48 分**。平均回报数字不够，必须看回放视频确认策略是真的学会了追踪球，还是只靠运气。**ICLR Blog Track 2022** 对比了多个 PPO 实现，Stable-Baselines3 的 **398 分**与 CleanRL、Tianshou 相当，是 Atari 环境上**最可靠的实现之一**。

| 入口类型 | 链接                                                                                                                                                                         |
| -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 玩家入口 | [Atari Breakout 网页试玩](https://brickbreaker.app/atari-breakout/)                                                                                                          |
| 训练环境 | [ALE Breakout](https://ale.farama.org/environments/breakout/)                                                                                                                |
| PPO 参考 | [sb3/ppo-BreakoutNoFrameskip-v4](https://huggingface.co/sb3/ppo-BreakoutNoFrameskip-v4)（Stable-Baselines3 PPO）、[CleanRL PPO](https://docs.cleanrl.dev/rl-algorithms/ppo/) |

---

### Procgen 与 泛化 vs 记忆

![Procgen 多环境截图，来源：OpenAI procgen GitHub](https://raw.githubusercontent.com/openai/procgen/master/screenshots/procgen.gif)

Procgen 的关键不是"能不能跑"，而是**泛化**。训练关卡和测试关卡由不同的随机种子生成，PPO 是记住了地图，还是学到了游戏规则？OpenAI 的 benchmark 使用 **IMPALA-CNN** 架构，学习率 **$5 \times 10^{-4}$**，gamma **0.999**，做了**奖励归一化**。

从 `coinrun` 开始最合适——这是一个平台跳跃游戏，规则直观，PPO 在 easy 模式上训练分数**接近满分 10**，在未见过的测试关卡上平均 **8.31 ± 0.12**。这个分数已经很高，所以很多后续改进方法（如 **IDAAC**、**DAAC**）在 CoinRun 上只能取得微小提升。真正值得关注的是训练集和测试集的分数差距：差距大说明策略在**过拟合**，记住了训练地图的地形而不是学会了"**向右跑、跳、躲避**"的通用规则。

| 入口类型 | 链接                                                                                                                                                                         |
| -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 玩家入口 | [Procgen PyPI](https://pypi.org/project/procgen/)，运行 `python -m procgen.interactive --env-name coinrun`                                                                   |
| 训练环境 | [OpenAI Procgen Benchmark](https://openai.com/index/procgen-benchmark/)、[procgen PyPI](https://pypi.org/project/procgen/)                                                   |
| PPO 参考 | [OpenAI Procgen Benchmark](https://openai.com/index/procgen-benchmark/)（官方 PPO 结果）、[CleanRL ppo_procgen.py](https://docs.cleanrl.dev/rl-algorithms/ppo/#ppoprocgenpy) |

---

### CarRacing 与 像素 + 连续控制

![CarRacing PPO 运行截图，来源：Solving CarRacing with PPO](https://notanymike.github.io/img/posts/SolvingCarRacing/4.gif)

CarRacing 把像素输入和连续控制叠在一起。策略要同时处理 **96×96** 的画面，并输出三个**连续值**：转向、油门、刹车。预处理是关键——**灰度化**是最有效的改进，100k steps 时灰度版约 **150 分**而 RGB 版只有 **-48**。**帧堆叠 4 帧**也必要，否则策略无法判断弯道的曲率变化。

训练曲线通常分几个阶段：**400k-500k steps** 达到 **450-620 分**，能完成大部分赛道；**2M steps** 达到 **740-920+ 分**，接近最优驾驶；如果加上草地检测和加速奖励塑造，最高可达 **917 分**。对比 **A2C** 会很有启发——A2C 在这个环境上**基本失败**，分数约 **-90**，这直接说明了 **PPO 的 clipped surrogate objective 对连续控制稳定性的贡献**。

这个项目适合接在 BipedalWalker 后面做，因为它也是连续控制，但界面更像游戏。尝试加入 **LSTM** 层后你会发现，急弯和岔路处的表现有明显提升，因为策略需要记住几秒前的道路信息。

| 入口类型 | 链接                                                                                                                                                                                 |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 玩家入口 | [Gymnasium CarRacing 文档](https://gymnasium.farama.org/environments/box2d/car_racing/)                                                                                              |
| 训练环境 | [Gymnasium CarRacing-v3](https://gymnasium.farama.org/environments/box2d/car_racing/)                                                                                                |
| PPO 参考 | [Solving CarRacing with PPO](https://notanymike.github.io/Solving-CarRacing/)（PPO 完整方案）、[Rinnnt/ppo-CarRacing-v3](https://huggingface.co/Rinnnt/ppo-CarRacing-v3)（PPO 模型） |

---

### Super Mario Bros 与 奖励设计决定策略上限

![Super Mario Bros PPO 演示，来源：vietnh1009/Super-mario-bros-PPO-pytorch](https://github.com/vietnh1009/Super-mario-bros-PPO-pytorch/raw/master/demo/video-1-1.gif)

Mario 是经典项目，但**奖励很容易写歪**。只奖励向右走，策略可能**无脑冲进岩浆**；只奖励通关，探索又太稀疏。**vietnh1009** 的 PyTorch 实现用 **CNN Actor-Critic**，做了帧跳过、灰度化、resize 到 **84×84**、**4 帧堆叠**。学习率从 **$10^{-3}$** 到 **$10^{-5}$** 都有尝试，困难关卡 World 1-3 需要更小的学习率（如 **$7 \times 10^{-5}$**）才能稳定收敛。

最终结果是 **31/32 关卡通关**——比他早期用 **A3C** 实现的 **19/32** 有显著提升。**PPO 的 clipped surrogate objective** 在这里的作用是明确的：**A3C 无论如何调参都只能到 19 关，而 PPO 在同样的环境上多通了 12 关**。

动作空间不要一开始就给完整 NES 手柄。先用 **`RIGHT_ONLY`**（**2 个动作**）或者 **`SIMPLE_MOVEMENT`**（**5 个动作**），跑通 World 1-1 后再考虑扩展。对比不同奖励函数会很有教育意义：**纯通关奖励**可能让策略学会速通但频繁死亡；**纯向右奖励**可能让策略无脑冲刺；**混合奖励**才能真正学会跳跃时机和敌人躲避。

| 入口类型 | 链接                                                                                                                                                                                   |
| -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 玩家入口 | [gym-super-mario-bros CLI](https://github.com/Kautenja/gym-super-mario-bros)：运行 `gym_super_mario_bros -e SuperMarioBros-v0 -m human`                                                |
| 训练环境 | [Kautenja/gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros)、[PyPI](https://pypi.org/project/gym-super-mario-bros/)                                              |
| PPO 参考 | [vietnh1009/Super-mario-bros-PPO-pytorch](https://github.com/vietnh1009/Super-mario-bros-PPO-pytorch)（PPO 实现）、[super-mario-agent](https://github.com/nemanja-m/super-mario-agent) |

---

### Sonic / Gym Retro 与 迁移学习的价值

![Sonic / Retro Contest 截图，来源：OpenAI Retro Contest Results](https://images.ctfassets.net/kftzwdyauwt9/1gGJCzTHFTNJn6m4GkROAB/6cb51a280c362510874d3ee15f5da6e8/retro-contest-results.jpg?w=3840&q=90&fm=webp)

**Retro Contest 2018** 的核心设计是**泛化**：在训练关卡上训练，然后在完全不同的测试关卡上评估。人类平均水平约 **7438 分**，理论满分约 **10000**。OpenAI 提供的 **Joint PPO baseline** 只有 **3128 分**——**还不到人类的一半**。

但**冠军方案（Dharmaraja）**用了一个关键技巧：**迁移学习**。先在训练关卡上**预训练**网络权重，然后用 PPO **fine-tune** 测试关卡，分数跃升到 **4692**。亚军用 **Rainbow DQN** 从零训练，分数类似但训练效率低很多。这个结果说明了两件事：一是 **PPO 的 fine-tune 能力很强**；二是在跨关卡泛化上，**即使是冠军方案也仅达到人类 60% 的水平**。

观察 PPO 在陌生关卡上的失败模式很有意思：它通常**死在没见过的地形陷阱上**，而不是策略本身有根本性缺陷。这指向了 **PPO 策略网络在特征泛化上的局限**——它记住了训练关卡中的具体视觉模式，而不是**抽象的游戏规则**。

| 入口类型 | 链接                                                                                                                                                                        |
| -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 玩家入口 | [Gym Retro interactive script](https://retro.readthedocs.io/en/latest/getting_started.html)（需自备合法 ROM）                                                               |
| 训练环境 | [Gym Retro 文档](https://retro.readthedocs.io/en/latest/)、[OpenAI Gym Retro](https://openai.com/research/gym-retro)                                                        |
| PPO 参考 | [OpenAI Retro Contest](https://openai.com/index/retro-contest/)（PPO baseline）、[Retro Contest Results](https://openai.com/index/retro-contest-results/)（Joint PPO 方案） |

---

### Unity SoccerTwos 与 多智能体的自博弈

![Unity SoccerTwos PPO 示例，来源：Hugging Face Deep RL Course](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit10/soccertwos.gif)

SoccerTwos 是 2v2 足球，引入了一个全新维度：**多智能体**。PPO 在这里需要和 **self-play** 配合使用——智能体与历史版本的自己对战，通过 **ELO 评分**追踪进步。**Unity ML-Agents 内置 PPO** 是最直接的实现路径，batch_size **2048**、buffer_size **20480**、学习率 **$3 \times 10^{-4}$**、熵系数 **0.005**。

<OnlineTraining studios="unity" compact />

训练过程能看到明显的**策略演化**。初期所有智能体追着球跑，场面混乱；ELO 从初始 **1200** 慢慢爬到约 **1600** 后，开始出现简单的**前锋-后卫角色分化**。**SAC** 在这个环境上表现差很多，基本停留在 **1200-1250**，说明 **PPO 的稳定性对多智能体 competitive 场景很重要**。

**课程学习**的效果很显著。设计良好的课程（逐步调整初始位置、球速）能让 PPO 在 **250k iterations** 内平均奖励超过 **1.8**，节省约 **40%** 训练时间；但设计不好的课程反而降低性能（**0.35 vs 0.45**），这提醒我们**课程设计本身需要调优**。

**奖励设计是多智能体的核心难点**。只给进球奖励，训练极慢；过度奖励靠近球，所有智能体挤成一团。需要找到**平衡点**，让策略既有进攻动机又能学会位置分配。

| 入口类型 | 链接                                                                                                                                                                            |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 玩家入口 | [Hugging Face Space: ML-Agents SoccerTwos](https://huggingface.co/spaces/unity/ML-Agents-SoccerTwos)                                                                            |
| 训练环境 | [Unity ML-Agents 文档](https://unity-technologies.github.io/ml-agents/Training-ML-Agents/)、[HF Deep RL Course](https://huggingface.co/learn/deep-rl-course/unit7/introduction) |
| PPO 参考 | [blu666/ppo-SoccerTwos](https://huggingface.co/blu666/ppo-SoccerTwos)（PPO 模型）、[Sekiraw/SoccerTwos](https://huggingface.co/Sekiraw/SoccerTwos)（PPO 模型）                  |

---

### ViZDoom 与 部分可观测性

![ViZDoom PPO 运行示例，来源：GuillBla/RL-Doom](https://github.com/GuillBla/RL-Doom/raw/master/demos/demo_basic.gif)

第一人称视角引入了**部分可观测性**——当前画面不代表完整状态，敌人可能在身后，补给可能在墙后。**GuillBla** 的实现用 **Stable-Baselines3** 的 `CnnPolicy`，输入先做**灰度化**、resize 到 **100×160**、裁剪底部 UI 区域到 **85×160**。

场景难度分三级。**Basic** 最简单：3 个动作（左移、右移、射击），训练 **100k steps** 后策略学会直接向敌人移动并在最佳时机射击。**Defend** 场景稍难：3 个动作（左转、右转、射击），训练 **200k steps** 后策略能有效瞄准接近的敌人，但还会浪费一些弹药。**Deadly Corridor** 最难：7 个动作（移动 + 转向 + 射击），需要**奖励塑造**和**课程学习**，200k steps 后策略表现出一定防御行为但结果较随机，未稳定到达终点。

**Deadly Corridor 的失败很有启发性**——它说明 PPO 在**复杂 3D 导航 + 多动作空间**的组合上，**仅靠调参是不够的**。对比有/无 **LSTM** 的表现会发现，**记忆网络对部分可观测性有明显帮助**，但**无法完全弥补信息缺失**。

| 入口类型 | 链接                                                                                                                                                                     |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 玩家入口 | [ViZDoom 默认场景](https://vizdoom.farama.org/environments/default/)、[Freedoom](https://github.com/freedoom/freedoom)                                                   |
| 训练环境 | [ViZDoom 文档](https://vizdoom.farama.org/)、[ViZDoom default scenarios](https://vizdoom.farama.org/environments/default/)                                               |
| PPO 参考 | [GuillBla/RL-Doom](https://github.com/GuillBla/RL-Doom)（Stable-Baselines3 PPO）、[callumhay/vizdoom_ppo_rnd](https://github.com/callumhay/vizdoom_ppo_rnd)（PPO + RND） |

---

### Pokemon Red 与 极端延迟与工程补偿

![Pokemon Red 强化学习探索截图，来源：PWhiddy/PokemonRedExperiments](https://github.com/PWhiddy/PokemonRedExperiments/raw/master/assets/grid.png?raw=true)

Pokemon Red 的难点是**地图切换、剧情 flag、菜单、对话、战斗和长时程探索**——**奖励延迟远超前面所有项目**。**PWhiddy** 的 baseline 使用 **PPO + PyBoy** 模拟器，V2 版本用**坐标探索奖励**替代了原始的 KNN 帧相似度奖励，效率大幅提升，策略成功到达 **Cerulean City**（游戏中期城镇）。

但 PPO 在这个任务上有两个典型的**失败模式**。第一是**动作循环**——策略发现反复走同一条路能带来正的探索奖励，于是**无限重复**。第二是**菜单 spam**——无意义地打开关闭菜单，因为某些状态下按按钮本身不会导致负面后果。**PokeRL**（arXiv:2604.10812）在 PPO 基础上增加了 **Loop-Aware Environment Wrapper** 和 **Anti-Spam Mechanism**，把任务目标切到更早期的"**走出房间、到达常磐市、赢下第一场战斗**"，才稳定下来。

这个项目的**关键教训**是：当**奖励极度稀疏**时，**PPO 本身的能力边界就暴露出来了**。它需要**工程层面的 wrapper** 来补偿算法的局限，而**不是靠调参解决一切**。

| 入口类型 | 链接                                                                                                                                                         |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 玩家入口 | [PokemonRedExperiments 项目入口](https://github.com/PWhiddy/PokemonRedExperiments)（需自备合法游戏文件）                                                     |
| 训练环境 | [PWhiddy/PokemonRedExperiments](https://github.com/PWhiddy/PokemonRedExperiments)、[davidpaulius/PokemonRedRL](https://github.com/davidpaulius/PokemonRedRL) |
| PPO 参考 | [PokeRL 项目页](https://drubinstein.github.io/pokerl/)（PPO 研究）、[PokeRL 论文](https://arxiv.org/abs/2604.10812)（PPO baseline）                          |

---

### Crafter 与 PPO 的探索瓶颈

![Crafter 环境截图，来源：danijar/crafter GitHub](https://raw.githubusercontent.com/danijar/crafter/main/media/video.gif)

Crafter 有生存、采集、合成、睡觉、战斗和 **22 个分层成就**，但工程量比 Minecraft 小很多。评估标准不是单一分数，而是 **22 个成就的几何平均成功率**——这 favor **广域解锁**而非单一重复。人类专家约 **50.5%**，标准 PPO（**Stable-Baselines3 CNN**）只有 **4.6%**。换用更深的 **Impala ResNet** 后能到 **15.6%**，与 DreamerV3 相当。

成就解锁的分布很有规律。**喝水、觅食**等基础生存技能成功率较高；**石镐、煤**等中等成就偶尔解锁；**铁镐、钻石**等长时程成就基本 **unreachable**。**EnvGen** 的实验显示，PPO 在 **1.96M steps** 后解锁"制作铁镐"约在 **135k steps** 处（进入最后的 1M window），而"制作铁剑"要到 **925k steps**。如果用 LLM 生成的环境**预训练后再 fine-tune**，这两个成就分别在 **40k** 和 **192k steps** 就解锁了——说明**标准 PPO 的瓶颈不是网络容量，而是探索效率**。

这个项目适合用来讨论一个核心问题：**PPO 能在它"看到"的数据上学得很快，但它很难自己去"发现"那些需要 10+ 步精确序列才能到达的长时程目标**。

| 入口类型 | 链接                                                                                                               |
| -------- | ------------------------------------------------------------------------------------------------------------------ |
| 玩家入口 | [danijar/crafter](https://github.com/danijar/crafter)：运行 `python3 -m crafter.run_gui`                           |
| 训练环境 | [Crafter GitHub](https://github.com/danijar/crafter)、[Crafter project page](https://danijar.com/project/crafter/) |
| PPO 参考 | [Benchmarking and Improving RL Generalization with Crafter](https://arxiv.org/abs/2208.03374)（PPO baseline 研究） |

---

### MiniHack / NetHack 与 长时程信用分配的上限

![MiniHack / NetHack 子任务示例，来源：hr0nix/omega](https://github.com/hr0nix/omega/raw/main/images/river.gif)

MiniHack 把 NetHack 拆成过河、推石头、开门、找钥匙、打怪等子任务。它的观测不是像素，而是**符号化的 ASCII/Unicode 网格**——这改变了策略网络的输入形式，通常用 **CNN 或 Transformer** 编码符号。

在完整的 **NetHack Learning Environment（NLE）** 上，**Sample Factory** 的高吞吐量 PPO 用 **480 个并行环境**在单卡 RTX 2080Ti 上 **24 小时**内达到约 **700 分**的 NetHackScore，优于原始的 TorchBeast baseline（约 **400 分**）。但在 MiniHack 子任务上，PPO 很快**饱和**——ZombieHorde 约 **5 杀**，TreasureDash 约 **20-23 分**（最优约 **28**）。更值得注意的是，添加 **RND**、**NovelD**、**E3B** 等内在动机方法**通常不能显著超越**纯外在奖励的 PPO baseline，这说明问题**不是"探索不够"**，而是 **PPO 的长时程信用分配能力本身有上限**。

这个任务上后续研究（**SOL**、**MOTIF**）的方向很清晰：**不是改进 PPO 本身**，而是在 PPO 之上叠加**层次化选项**或更好的探索机制。PPO 在这里的角色是"**可靠的 lower bound**"——任何新方法都必须先超过它。

| 入口类型 | 链接                                                                                                                                                                 |
| -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 玩家入口 | [MiniHack trying out](https://minihack.readthedocs.io/en/latest/getting-started/trying_out.html)：运行 `python -m minihack.scripts.play_gui --env MiniHack-River-v0` |
| 训练环境 | [MiniHack 文档](https://minihack.readthedocs.io/)、[MiniHack GitHub](https://github.com/facebookresearch/minihack)                                                   |
| PPO 参考 | [hr0nix/omega: PPO and MuZero agents](https://github.com/hr0nix/omega)（PPO + MuZero）                                                                               |

---

## 快速索引表

| 游戏               | 核心讨论点               | PPO 典型结果                             |
| ------------------ | ------------------------ | ---------------------------------------- |
| Flappy Bird        | 验证 PPO 能学            | 稳定飞行，低维版本极快收敛               |
| Snake              | 奖励设计决定策略行为     | 5×5 最高 23 分，10×10 平均 6-7 分        |
| 2048               | 即时奖励 vs 长期布局     | 简单奖励 37% 胜率，但 DQN 更优           |
| Tetris             | 延迟反馈，RL vs 专家知识 | 10×10 约 3800 分，标准棋盘仍困难         |
| Breakout           | 像素预处理的标准流程     | 10M steps 约 398 分，接近人类            |
| Procgen            | 泛化 vs 记忆             | CoinRun 测试集 8.31±0.12，训练接近满分   |
| CarRacing          | 连续控制的稳定性         | 2M steps 740-920+ 分，A2C 约 -90         |
| Super Mario Bros   | 奖励设计决定策略上限     | 31/32 关卡，显著优于 A3C                 |
| Sonic / Gym Retro  | 迁移学习的效果           | 冠军 4692 分，人类水平约 7438            |
| Unity SoccerTwos   | 多智能体自博弈           | ELO ~1600，课程学习可大幅加速            |
| ViZDoom            | 部分可观测性             | Basic/Defend 成功，Corridor 仍困难       |
| Pokemon Red        | 极端延迟与工程补偿       | 到达 Cerulean City，需 anti-loop wrapper |
| Crafter            | 探索瓶颈                 | CNN 约 4.6%，ResNet 约 15.6%             |
| MiniHack / NetHack | 长时程信用分配上限       | NetHackScore ~700，子任务上饱和          |

---

## 按需求选项目

**只想做一个能展示的视频项目**：选 Flappy Bird、Snake、2048、Breakout、CarRacing。训练快，可视化效果好，适合课程展示。

**想做"界面不普通"的项目**：选 Procgen、Mario、SoccerTwos、ViZDoom、Crafter。视觉效果丰富，更容易引起兴趣。

**想做掌机或复古主机**：选 Pokemon Red 早期任务、Super Mario Bros World 1-1、Sonic / Gym Retro 短关卡。注意处理模拟器和授权问题，不要把目标写成完整通关。

**想做研究味更强的项目**：选 Crafter、MiniHack、Pokemon Red。这些任务很适合讨论 PPO 的边界：奖励远、状态不完整、动作语义复杂、探索容易卡住。

---

## 思考题

1. 为什么 Flappy Bird 的状态空间很小，但策略仍然能表现出"智能"？这和状态表示的质量有什么关系？
2. 在 Snake 中，如果只给"吃到食物 +10，死亡 -10"，策略可能学会什么非预期行为？如何设计奖励才能避免？
3. Breakout 需要帧堆叠，但 2048 不需要。这说明什么信息是单帧画面无法提供的？
4. Procgen 和 Retro Contest 都强调泛化，但测试方式不同。Procgen 用 procedurally generated 关卡，Retro 用固定但陌生的关卡。这两种测试各有什么优劣？
5. Pokemon Red 的极端延迟反馈，指向了 PPO（以及策略梯度方法）的什么根本性局限？如果要改进，可能需要在算法层面引入什么机制？

---

## 小结

PPO 是一个通用的策略优化框架，但它的"通用性"不等于"在所有任务上都同样好"。本节的项目讨论，本质上是在帮你建立对 PPO 能力边界的直觉：

- **低维、即时反馈**的任务，PPO 很快、很稳。
- **像素输入**需要预处理工程，但不改变算法本质。
- **连续控制**需要仔细调整超参数，尤其是学习率和熵正则。
- **延迟奖励、部分可观测、多智能体**则触及了 PPO 的设计边界，这些场景下的失败往往不是调参能解决的，而是需要算法层面的扩展。

选项目时，不要只看"这个游戏酷不酷"，而要看"它能让我理解 PPO 的哪一面"。
