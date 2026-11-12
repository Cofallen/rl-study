# 4.2 策略采样与数据来源

> **一句话概括**：本节我们将引入两组经常被混淆的概念——**On-policy vs Off-policy**（数据是不是当前策略自己采的？），以及 **Online vs Offline**（训练时还能不能继续采新数据？）。它们可以两两组合，共同定义强化学习算法的四种数据形态。

## 本节导读

**核心内容**

- 行为策略 $\mu$ 与目标策略 $\pi$：区分“谁在收集数据”和“谁在被学习”。
- On-policy / Off-policy：判断训练数据是否来自当前正在学习的策略。
- Online / Offline：判断训练过程中还能不能继续和环境交互、收集新数据。

前面几节，我们推导了贝尔曼方程，也看到了 TD 误差是如何更新价值表的。这些更新公式都有一个共同的前提：**我们需要数据**。无论是更新 $V(s)$ 还是 $Q(s,a)$，我们都需要状态、动作、奖励和下一状态（即 $s, a, r, s'$）作为原材料。

但这里有个绕不开的问题：**这些 $s, a, r, s'$ 究竟从哪来？**

想象你要学习骑自行车：

- **第一种方法**：你自己跨上车，一边摔跤一边总结经验，立刻调整自己的动作。
- **第二种方法**：你坐在一旁，看别人怎么骑（或者看自己上周骑车的录像），在脑子里总结出一套最优的骑车策略。
- **第三种方法**：教练给你发了一本厚厚的《人类骑车翻车实录》，把你关在小黑屋里学习，没学成之前绝对不准碰真车。

这三种方法对应着强化学习中截然不同的数据使用范式。很多时候，算法之间的巨大差异在**如何获取和使用数据**上。同一个目标函数，配上不同的数据来源，学出来的策略可以天差地别。

接下来，我们逐一拆解这两组概念。

::: info 核心概念

- **On-policy（同策略）**：训练数据由当前正在优化的策略亲自产生（所见即所学）。
- **Off-policy（异策略）**：训练数据可以来自旧策略、人类或专家，被用来优化另一个不同的目标策略（借他山之石）。
- **Online（在线）**：训练过程中智能体仍在与环境交互，数据集在不断生长（边玩边学）。
- **Offline（离线）**：训练前数据集就已彻底固定，过程中绝对不允许与环境交互（闭关修炼）。
  :::

## 行为策略与目标策略

在弄清楚这些概念之前，我们必须先理清强化学习中经常被混为一谈的两个“策略”角色。

在很多基础教程中，我们会默认智能体只有一个策略 $\pi$：它既负责在环境里做动作，也负责在代码里被更新。但一旦进入复杂的现代 RL，这两个角色往往是分离的。

我们把它们拆开来看：

1. **行为策略（Behavior Policy）**，通常记作 $\mu(a\mid s)$。
   它回答的问题是： **“这批数据当初是怎么来的？”**
   行为策略是真正与环境发生物理（或模拟）交互、按下按钮、输出 token 的那个策略。它可能是一个带有 $\epsilon$-greedy 随机探索的策略，可能是智能体昨天用过的一个旧版本，可能是另一个专家模型，甚至可能是人类自己。

2. **目标策略（Target Policy）**，通常记作 $\pi_\theta(a\mid s)$。
   它回答的问题是： **“我们最终想学出一个什么策略？”**
   这是算法心里真正关心的优化对象。不管外面的数据是怎么来的，我们希望参数 $\theta$ 更新后，目标策略能变得更好。

当我们把“谁在行动（产生数据）”和“谁在学习（被优化）”分开后，第一组概念就自然浮现了。

## On-policy 与 Off-policy

这组概念的核心问题是：**更新目标策略时，用的数据是不是它自己刚刚亲自采样出来的？**

### On-policy 同策略

如果训练数据就是由当前正在学习的目标策略（或一个极其接近的快照）产生的，我们就称之为 **On-policy（同策略）**。

用数学语言表达，就是行为策略和目标策略基本一致：

$$
\mu(a\mid s) \approx \pi_\theta(a\mid s)
$$

**直观理解**：这就好比你做一套模拟卷，做完立刻对答案，然后纠正自己的错题，接着再用**更新后的自己**去做下一套卷子。你永远在从自己最新的行为中学习。

在算法层面，之前提到的 **Sarsa** 就是典型的 On-policy 算法。它在状态 $s$ 选了动作 $a$，走到 $s'$，然后再根据**当前策略**在 $s'$ 实际会选的动作 $a'$ 来更新价值。它评估的是“如果我继续按照现在的习惯走下去，会怎么样”。

在大语言模型（LLM）训练中，**PPO**（以及 DeepSeek 使用的 **GRPO**）也属于 On-policy 的范畴。它们通常是先让当前模型对一批 prompt 生成回答（Rollout），用奖励模型打分后，立刻用这批新鲜出炉的数据更新模型参数。为了保证 On-policy 的性质，PPO 还会特意加上限制（Clipping 机制），不让新策略偏离采样时的旧策略太远。[^ppo2017]

- **优点**：逻辑直接，数学上非常稳定。你学到的就是你正在做的。
- **缺点**：太费数据（Sample Inefficient）。用过一次的数据，因为策略更新了，它就不再是“当前策略”产生的数据了，通常只能直接扔掉。

### Off-policy 异策略

如果行为策略 $\mu$ 和目标策略 $\pi_\theta$ 可以完全不同，我们就称之为 **Off-policy（异策略）**。

$$
\mu(a\mid s) \neq \pi_\theta(a\mid s)
$$

**直观理解**：你拿着一个错题本（Replay Buffer），里面记录了你一个月前瞎蒙的题、昨天做错的题，甚至学霸朋友借给你的笔记。你在复习（更新目标策略 $\pi$）时，看的是这些五花八门的数据（行为策略 $\mu$），但你脑子里在总结的是通往满分的最优解。

最经典的 Off-policy 算法是 **Q-Learning**。回想一下它的更新目标：

$$
\text{Target} = r + \gamma \max_{a'} Q(s', a')
$$

在这里，智能体实际探索时可能是瞎走的（行为策略 $\mu$ 带有很大的随机性），但在更新 Q 表时，它总是假设自己下一步会选择**价值最大**的那个动作（目标策略 $\pi$ 是贪心的）。数据是探索出来的，学习的目标却是最优的。[^watkins1992]

在深度强化学习中，**DQN** 把 Off-policy 发扬光大。它引入了经验回放池（Replay Buffer），把几万步以前的旧经验存起来反复抽样训练。既然数据是很久以前的旧策略产生的，显然 $\mu \neq \pi$，因此它是 Off-policy 的。[^mnih2015]

- **优点**：极大地提高了数据利用率（Sample Efficient）。旧数据可以反复学，别人的数据也可以拿来学。
- **缺点**：分布偏移（Distribution Shift）。如果目标策略想做一个动作，但历史数据里从来没有人做过这个动作，算法就只能靠“猜”（外推），很容易猜出极其离谱的高分。[^sutton-barto]

## Online 与 Offline

On-policy 和 Off-policy 讨论的是“数据是谁产生的”。现在我们换个角度，问一个更宏观的问题：**在整个训练期间，智能体还能不能继续和环境交互，产生新的数据？**

这就引出了 Online 和 Offline 的区别。它们讨论的是**数据集是否还能生长**。

### Online RL 在线学习

如果训练过程中，智能体可以持续和环境交互，不断把新收集的样本加入数据集，这就是 **Online RL（在线学习）**。

在这里，数据集是动态增长的：

$$
\mathcal{D}_{k+1} = \mathcal{D}_k \cup \{\tau_k\}
$$

无论是 DQN 玩 Atari 游戏，还是 PPO 训练机器狗走路，只要训练没有结束，智能体就还在不停地开新局、试新动作、拿新奖励。

注意，**Online 不等于 On-policy**。DQN 是 Off-policy 的，因为它反复复用旧数据；但它同时也是 Online 的，因为它在训练时仍然在不断打游戏收集新数据。

Online RL 的最大优势在于**试错**。如果智能体对某个动作的价值估计不准，它只要去环境里实际试一下，拿到真实的奖励，虚假的估计就会立刻被打破。

### Offline RL 离线学习

如果在训练开始前，数据集就已经彻底固定了，智能体在训练过程中**绝对不能**再与环境进行任何交互，这就是 **Offline RL（离线学习）**。

$$
\mathcal{D} = \mathcal{D}_{\text{fixed}}
$$

为什么会有这么苛刻的设定？因为在很多真实场景中，“试错”的代价太高了。[^levine2020]

- **自动驾驶**：你不能让一个还在训练初期的 AI 开着真车上路去“探索”撞车的后果。你只能给它几万小时的人类驾驶录像（固定数据），让它在服务器里学。
- **医疗诊断**：你不能拿病人的生命让 AI 去试错。

在 LLM 对齐中，**DPO（直接偏好优化）** 在常见的设定下就非常接近 Offline RL。研究人员先准备好一批固定的偏好数据（Prompt + 人类更喜欢的回答 + 人类不喜欢的回答），然后让模型在这个固定数据集上直接优化参数。在这个梯度更新的过程中，模型不需要再去跟人类对话索要新的评分。[^dpo2023]

**Offline RL 的致命挑战：**
既然是离线数据，它通常只能是 Off-policy 的（因为数据早就被某个历史行为策略采好了）。Offline 的最大困难在于**无法证伪的过度估计（Overestimation）**。

假设自动驾驶的日志里只有正常行驶的数据。AI 在离线训练时，由于神经网络的外推效应，它可能产生一个幻觉：“如果在高速上以 120km/h 的速度猛打方向盘，我会得到极高的奖励！”
如果是 Online RL，AI 只要去模拟器里试一次，车毁人亡，这个幻觉就被打破了。但在 Offline RL 中，它永远没有机会去试，所以这个致命的错误估计会一直保留在价值函数中，最终导致策略崩溃。

为了解决这个问题，Offline RL 算法（如 CQL）通常会引入**保守主义（Conservatism）**：对于数据集中没见过的动作，一律给低分，强迫 AI 待在已知数据的安全区内。[^cql2020]

## 四种数据形态

把两组概念两两组合，可以得到强化学习算法的四种数据形态。搞清这四种组合，就不会再被各种缩写绕晕了：

| 数据形态                 | 含义                                                       | 典型算法或场景                                   | 主要风险                                       |
| ------------------------ | ---------------------------------------------------------- | ------------------------------------------------ | ---------------------------------------------- |
| **Online + On-policy**   | 训练时继续交互，且只用自己最新鲜的数据更新自己。           | REINFORCE、Sarsa、PPO、GRPO                      | 数据利用率低，需要频繁与环境交互。             |
| **Online + Off-policy**  | 训练时继续交互，但允许把历史数据存进池子里反复学习。       | Q-Learning、DQN、SAC、TD3                        | 需要处理新旧策略之间的分布差异。               |
| **Offline + Off-policy** | 完全闭关修炼，只能从别人给的一份固定历史数据中挖掘好策略。 | CQL、IQL、固定数据集上的 DPO                     | 容易对没见过的动作产生盲目自信（外推与高估）。 |
| **Offline + On-policy**  | 数据固定，同时又要求数据代表当前策略。                     | （边界情形）固定策略评估、极小步长的模仿学习更新 | 策略只要一更新，固定数据就失效了。             |

## 常见误区排雷

在结束本节之前，我们来扫清几个最常见的思维误区：

1. **“On-policy 就是数据只能用一次？”**
   不完全对。PPO 虽然是 On-policy，但它会在同一批数据上更新好几个 epoch。关键在于：只要新策略还没有偏离采样策略太远（通过 Clipping 保证），这批数据就还能被近似当作 On-policy 数据来用。

2. **“Off-policy 就是什么垃圾数据都能学？”**
   大错特错。Off-policy 虽然能复用数据，但必须满足**覆盖条件（Coverage）**。如果你的目标策略想学下围棋，但你的旧数据全是玩超级马里奥的，算法照样什么都学不到。旧数据必须覆盖目标策略可能探索到的状态空间。

3. **“Off-policy 等于 Offline？”**
   绝对不是。**DQN** 是最好的反例。它有个巨大的经验回放池（Off-policy），但它同时在疯狂打游戏收集新数据（Online）。

4. **“DPO 只是个监督学习，不是 RL？”**
   虽然 DPO 把损失函数写成了分类问题的形式，不需要显式拟合奖励模型，但它依然在固定的偏好数据上调整策略分布，本质解决的仍是 Offline 设定下的策略优化问题；如果不注意数据覆盖率，同样会面临 Offline RL 典型的分布外（OOD）风险。

## 小结

本节没有推导新公式，回答的是训练数据从哪来：

1. **行为策略 $\mu$** 负责下场跑数据，**目标策略 $\pi_\theta$** 是我们在后台偷偷修炼的真正主角。
2. **On-policy vs Off-policy** 问的是：数据是不是主角自己刚刚采的？
3. **Online vs Offline** 问的是：现在还能不能继续去环境里拿新数据？

到这里，我们已经知道了价值怎么算（Bellman）、更新怎么做（DP/MC/TD）、数据从哪来（On/Off-policy 与 Online/Offline）。但所有这些算法，都在为了同一个目标努力： **拿到最高的奖励** 。

这引出了强化学习中可能是最有趣、也最危险的一个问题： **如果奖励函数本身就写错了，会发生什么？**

下一节：[奖励函数设计](./reward-design)

## 附录 从论文标题读术语

RL 和 Agentic RL 论文的标题里经常堆满缩写和黑话。与其死记硬背定义，不如直接看真实的论文标题——标题本身就是最地道的用法示范。下面我们以 2024–2026 年最新论文为例，逐一拆解标题中出现的核心术语。

### On-policy 与 Off-policy 的数据来源

这两个词几乎是 RL 论文标题里出现频率最高的形容词。它们描述的是**训练数据和当前策略之间的关系**。

**典型论文标题拆解：**

> **"Group-Relative REINFORCE Is Secretly an Off-Policy Algorithm: Demystifying Some Myths About GRPO and Its Friends"**
> _(arXiv 2509.24203, 2025)_

这篇论文的标题直接挑战了一个流行认知——DeepSeek 提出的 GRPO 一直被当作 On-policy 算法来用，但作者证明它在数学上其实可以天然地被解释为 Off-policy。标题里的 **"Secretly an Off-Policy Algorithm"** 就是在说：你以为数据是当前策略亲自采的（On-policy），其实旧数据也能合法地用进来（Off-policy）。

> **"Prosperity before Collapse: How Far Can Off-Policy RL Reach with Stale Data on LLMs?"**
> _(arXiv 2510.01161, 2025)_

标题里的 **"Off-Policy RL"** + **"Stale Data"**（过期数据）精准点出了 Off-policy 的核心矛盾：数据是旧策略产生的（stale），但你想用它来训练新策略。这篇论文提出 M2PO 算法，通过约束重要性权重的二阶矩，让 Off-policy 训练在 1.7B–32B 的大模型上也能匹配 On-policy 的性能。

> **"On-Policy RL Meets Off-Policy Experts: Harmonizing Supervised Fine-Tuning and Reinforcement Learning via Dynamic Weighting"**
> _(arXiv 2508.11408, 2025)_

这个标题把 **On-Policy** 和 **Off-Policy** 放在对立的位置上，又试图将二者调和。"On-Policy RL" 指的是模型自己采样、自己学习的 RL 阶段；"Off-Policy Experts" 指的是 SFT 阶段用的人类标注数据（来自"专家"，显然不是当前策略自己产的）。论文提出的 CHORD 框架通过动态权重在这两种数据源之间做调和——这是 LLM 训练中 On/Off-policy 混合的典型场景。

> **"Behaviour Policy Optimization: Provably Lower Variance Return Estimates for Off-Policy Reinforcement Learning"**
> _(arXiv 2511.10843, AAAI 2026)_

标题里的 **"Behaviour Policy"**（行为策略）和 **"Off-Policy"** 同时出现，正好呼应本节的核心概念：Off-policy 场景下，行为策略 $\mu$ 和目标策略 $\pi$ 是分离的，这篇论文证明了精心设计的行为策略可以带来比 On-policy 采样更低的方差。

**一句话总结：** 看到标题里有 **On-policy**，意味着"模型用自己的数据更新自己"；看到 **Off-policy**，意味着"模型在用别人（或旧版本自己）的数据学习"。

### Online 与 Offline 的数据采集

这两个词描述的是**训练过程中数据集是否还在增长**，与 On/Off-policy 相互独立。

**典型论文标题拆解：**

> **"Offline vs. Online Learning in Model-based RL: Lessons for Data Collection Strategies"**
> _(arXiv 2509.05735, RLC 2025)_

标题直接把 **Offline** 和 **Online** 对立起来比较。这篇论文在 31 个环境中对比了两种范式，结论是在线智能体普遍优于离线智能体，且离线性能下降的主要原因是测试时遇到了分布外（OOD）状态——这正是 Offline RL 的致命弱点：没见过就是没见过，没机会去试。

> **"Understanding the Performance Gap Between Online and Offline Alignment Algorithms"**
> _(arXiv 2405.08448, NeurIPS 2024)_

标题里的 **"Online and Offline Alignment"** 把这对概念放在了 LLM 对齐的语境下。Online alignment 指 PPO 这类边采样边训练的方法，Offline alignment 指 DPO 这类在固定偏好数据上直接优化的方法。论文系统性地分析了为什么 Online 方法在性能上通常优于 Offline 方法。

**一句话总结：** 标题里看到 **Online**，意味着训练时还在和环境交互、数据集在增长；看到 **Offline**，意味着数据集已经封存，训练期间不许再碰环境。

### 四种组合的论文实例

把 On/Off-policy 和 Online/Offline 两两组合，正文中的四种数据形态各有对应的前沿论文：

| 数据形态                 | 代表论文                                                                                                                          | 标题关键词解读                                                                                                                                                 |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Online + On-policy**   | PPO (Schulman et al., 2017)、GRPO (DeepSeek, 2024)                                                                                | 边采样边学，用完即弃。                                                                                                                                         |
| **Online + Off-policy**  | _"TOP-ERL: Transformer-based Off-Policy Episodic Reinforcement Learning"_ (ICLR 2025 Spotlight)                                   | **Off-Policy** 说明用了经验回放复用旧数据，但 **Episodic** 意味着它仍然在持续开新局、采新数据（Online）。                                                      |
| **Offline + Off-policy** | _"Offline-Boosted Actor-Critic: Adaptively Blending Optimal Historical Behaviors in Deep Off-Policy RL"_ (arXiv 2405.18520, 2024) | **Offline-Boosted** 说明基础数据是离线固定数据集，**Off-Policy** 说明行为策略和目标策略不同。OBAC 从回放池中识别出表现最优的历史策略，用来约束在线策略的学习。 |
| **Offline + On-policy**  | （边界情形，较少独立出现）                                                                                                        | 数据固定，又要求数据代表当前策略——这几乎只在策略评估或极小步长的模仿学习中出现。                                                                               |

## 参考文献

[^sutton-barto]: Sutton, R. S., & Barto, A. G. (2018). _Reinforcement Learning: An Introduction_ (2nd ed.). MIT Press. 参见第 5.5、5.7、6.4、6.5 章关于 off-policy prediction、off-policy control、Sarsa 和 Q-learning 的讨论。MIT Press 页面：<https://mitpress.mit.edu/9780262039246/reinforcement-learning/>

[^watkins1992]: Watkins, C. J. C. H., & Dayan, P. (1992). Q-learning. _Machine Learning_, 8, 279-292. <https://www.gatsby.ucl.ac.uk/~dayan/papers/wd92.html>

[^mnih2015]: Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015). Human-level control through deep reinforcement learning. _Nature_, 518, 529-533. <https://doi.org/10.1038/nature14236>

[^ppo2017]: Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. _arXiv:1707.06347_. <https://arxiv.org/abs/1707.06347>

[^levine2020]: Levine, S., Kumar, A., Tucker, G., & Fu, J. (2020). Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems. _arXiv:2005.01643_. <https://arxiv.org/abs/2005.01643>

[^cql2020]: Kumar, A., Zhou, A., Tucker, G., & Levine, S. (2020). Conservative Q-Learning for Offline Reinforcement Learning. _NeurIPS 2020_. <https://papers.nips.cc/paper_files/paper/2020/hash/0d2b2061826a5df3221116a5085a6052-Abstract.html>

[^dpo2023]: Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., & Finn, C. (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model. _arXiv:2305.18290_. <https://arxiv.org/abs/2305.18290>
