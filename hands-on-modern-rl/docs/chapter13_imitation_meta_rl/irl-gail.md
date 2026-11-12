# 11.2 逆强化学习与 GAIL

## 本节导读

**核心内容**

- 理解逆强化学习（IRL）的核心思想：不直接模仿动作，而是从专家轨迹反推"什么是好的"——也就是奖励函数
- 看到 IRL 的一个根本问题：**奖励不唯一**——专家最优的奖励函数有无穷多个，最大熵原则帮我们选出合理的一个
- 理解最大熵 IRL 的训练目标，但看到它的致命瓶颈：配分函数 $Z$ 难以计算，需要反复做内层 RL
- 掌握 GAIL 如何用 GAN 的对抗训练思想绕开配分函数，让模仿学习扩展到高维问题
- 对比 BC、MaxEnt IRL、GAIL 三条路线的优劣，理解为什么 GAIL 是 DPO 等 LLM 对齐方法的前身

上一节我们看到，行为克隆虽然简单，但有分布偏移问题；DAgger 虽然能解决分布偏移，但需要专家在线反复标注——这在很多场景下根本不现实。你可能会问："既然专家不需要告诉我每个状态该做什么，他已经用自己的行为'告诉我'什么是好的了——我能不能自己从他的轨迹里反推出奖励函数，然后用这个奖励自己做 RL？"

这就是**逆强化学习**（Inverse Reinforcement Learning, IRL）的思路。前向 RL 是"给定奖励，求最优策略"；逆向 RL 反过来——"给定专家策略（或轨迹），求能解释这个行为的奖励函数"。

本节先说明奖励为什么无法由示范唯一确定，再用最大熵原则选择一个可学习的奖励，随后介绍 GAIL 怎样用判别器绕开配分函数，最后比较三条模仿学习路线的代价与适用条件。

## 1. 为什么要从示范推断奖励

让我们先从直觉上理解为什么"推断奖励"可能比"直接模仿动作"更好。

想象你在看一个老司机开车。他在车道中间平稳行驶，看到红灯会减速，看到行人会避让。如果你只做行为克隆，你学会的是"看到红灯踩刹车""看到行人打方向"——但你不知道*为什么*这么做。万一遇到一个训练时没见过的新场景（比如一个奇怪的障碍物），你可能就不知道该怎么办了。

但是如果你能想明白"他是在最大化什么奖励"——比如"保持安全距离""尽快到达目的地""乘坐舒适"——那么即使遇到新状态，你也可以根据这个奖励自己判断该做什么。这就是 IRL 希望达到的效果：学到一个能泛化的"意图"，而不只是死记硬背动作映射。

### 1.1 逆向 RL 的基本设定

逆向 RL（Inverse RL）假设专家行为来自某个尚未观测到的奖励函数。训练先从轨迹推断这个奖励，再用普通 RL 求解对应策略。

给定专家轨迹 $\mathcal{D}_{\text{expert}}=\{\tau_1,\ldots,\tau_M\}$，每条 $\tau=(s_0,a_0,\ldots,s_T)$。我们希望找到奖励函数 $r_\psi(s,a)$，使专家轨迹在这个奖励下比其他轨迹更有可能出现：

$$\text{专家策略在 } r_\psi \text{ 下是最优的}$$

等等——这里有一个大问题。你可能会问："只要求'专家是最优的'，就能确定奖励了吗？"

答案是：**不能**。让我们想一个最简单的例子：如果奖励函数 $r(s,a) = c$（给所有状态-动作对同一个常数 $c$），那会怎么样？所有轨迹的总回报都是 $c \times T$——不管你采取什么策略，回报都一样！在这个奖励下，任何策略（包括专家策略）都可以说是"最优的"。这是一个平凡解，但完全没有用。

再想一个例子：迷宫任务。专家走了一条从起点到终点的路。你能确定奖励是"走到终点+1"吗？不一定——奖励也可能是"每一步靠近终点就+0.1，终点+1"，或者"走最短路径+10，绕路-1"……这些奖励都能让专家的策略是最优的。

这两个例子说明：只要求"专家最优"，无法从无穷多个候选奖励中选出正确的那一个。我们需要一个额外的原则。最大熵 IRL 给出了这个原则：在所有能解释专家行为的奖励中，选择**熵最大**的那个——也就是最不"偏门"、最不武断的那个。

## 2. 用最大熵原则确定奖励

Ziebart et al. 2008 提出最大熵逆向 RL。它要求轨迹既匹配专家的特征期望，又在满足约束的轨迹之间保留尽可能高的熵。在最大熵模型中，一条轨迹的概率与它的累计奖励指数成正比：

$$p(\tau \mid r_\psi) = \frac{1}{Z(r_\psi)} \exp\left(\sum_t r_\psi(s_t, a_t)\right)$$

让我们拆开这个公式看每个部分：

- $\tau = (s_0,a_0,s_1,a_1,\ldots,s_T)$：一条完整的轨迹
- $\sum_t r_\psi(s_t, a_t)$：这条轨迹的总奖励——奖励越高，这条轨迹出现的概率应该越大
- $\exp(\cdot)$：指数函数——总奖励线性增加，概率指数增长，这就是为什么它叫"softmax"分布
- $Z(r_\psi)$：**配分函数**（partition function）——它把所有可能轨迹的未归一化分数 $\exp(\sum r)$ 加起来，做归一化，让所有轨迹的概率总和等于 1。它是一个常数（对于固定的 $r_\psi$），但也是最麻烦的部分。

让我们用数字把 $Z$ 算一遍。假设环境很小，总共只有三条可能的轨迹，当前奖励函数给它们打的总分如下：

| 轨迹                   | 总奖励 $\sum_t r_\psi$ | 未归一化权重          | 归一化概率 |
| ---------------------- | ---------------------- | --------------------- | ---------- |
| $\tau_1$：专家走的短路 | 10                     | $e^{10}\approx 22026$ | 0.993      |
| $\tau_2$：绕远一点的路 | 5                      | $e^{5}\approx 148$    | 0.0067     |
| $\tau_3$：撞墙的路     | 0                      | $e^{0}=1$             | 0.00005    |

$Z = 22026 + 148 + 1 \approx 22176$——把第三列全部加起来，概率就归一了。指数把"10 分和 5 分"的奖励差距放大成约 149 倍的概率差距：总奖励高的轨迹被强烈偏好，分数接近的轨迹仍保留机会。

把同样的指数规则落到每一步的动作选择上，就得到 softmax 策略：

$$\pi(a \mid s) \propto \exp\left(Q^{\text{soft}}_{r_\psi}(s, a)\right)$$

Q 值高的动作概率大，Q 值差不多的动作也都保留一定概率——这和上一章 SAC 的最大熵目标是一脉相承的。

对 $M=|\mathcal D_{\text{expert}}|$ 条专家轨迹取对数后，训练目标为：

$$\max_\psi \; \mathcal{L}(\psi) = \sum_{\tau \in \mathcal{D}_{\text{expert}}} \left[\sum_t r_\psi(s_t, a_t)\right] - |\mathcal{D}_{\text{expert}}| \log Z(r_\psi)$$

这个目标看起来有点复杂，但直觉很清楚：

- **第一项**：提高专家轨迹的累计奖励——专家做的事，应该得高分
- **第二项**：减去 $|\mathcal{D}| \log Z$——防止奖励函数无限制地给所有状态打高分（那样所有轨迹概率都一样大，就没用了）

对参数 $\psi$ 求梯度，我们得到一个更直观的形式：

$$\nabla_\psi \mathcal{L} = \mathbb{E}_{\tau \sim \text{expert}}\left[\sum_t \nabla_\psi r_\psi(s_t, a_t)\right] - \mathbb{E}_{\tau \sim p(\cdot \mid r_\psi)}\left[\sum_t \nabla_\psi r_\psi(s_t, a_t)\right]$$

让我们用中文"翻译"这个梯度：

- 第一项：**专家轨迹上的奖励梯度**——专家经常去的状态-动作对，应该提高它们的奖励
- 第二项：**当前策略诱导的轨迹上的奖励梯度**——当前策略经常去但专家不去的地方，应该降低它们的奖励

放到开车的场景里：专家几乎总在"车道中间、车速平稳"的状态上行驶，而当前策略常漂到"压线、车速忽快忽慢"的状态。这一轮梯度就会提高前一类状态-动作的奖励、压低后一类的——策略再按新奖励做 RL，就会回到车道中间。

当两边的特征统计接近时——也就是策略访问的状态-动作分布和专家一致时——更新趋于停止。这和我们熟悉的很多算法（如 GAN、甚至策略梯度）的结构非常像。

### 2.1 配分函数为什么难以计算

看起来很美，对不对？但是有一个致命问题：$\log Z(r_\psi)$ 在连续状态-动作空间下**不可解析计算**。刚才三条轨迹直接相加就能得到 $Z$；连续空间里 $Z$ 是对所有（无穷多条）轨迹的积分，没有解析形式，也无法枚举。

三种主流近似方案：

1. **基于模型**：用学到的环境模型做 forward rollout 估计 $Z$
2. **基于采样的 soft Q iteration**：用软 Bellman 备份近似（Guided Cost Learning, Finn et al. 2016）
3. **对抗式（GAIL）**：用判别器隐式表达 $r_\psi$——这就是下一节的重点

```python
def maxent_irl_step(reward_net, expert_states_actions, env_sampler, soft_q_planner):
    # 1. 当前奖励下做 soft Q planning，得到采样分布
    current_rewards = reward_net(states_actions_tensor)
    sampled_trajectories = soft_q_planner.rollout(reward_net)

    # 2. 计算特征期望差
    expert_feat = feature_expectation(expert_states_actions, reward_net)
    sampled_feat = feature_expectation(sampled_trajectories, reward_net)

    # 3. 梯度上升更新奖励
    grad = expert_feat - sampled_feat
    reward_net.update(grad)
```

MaxEnt IRL 的代价高昂：每次外层更新需要内层求解一个完整的 soft Q 问题。这使它难以扩展到高维问题（如视觉输入）。每次奖励网络变一点点，你都要重新解一次 RL——这对于复杂环境来说太慢了。

**GAIL** 用对抗训练巧妙地避开了显式计算 $Z$。

## 3. 用 GAIL 直接匹配访问分布

Generative Adversarial Imitation Learning（Ho & Ermon 2016）借用 GAN 的思想，把逆向 RL 写成判别器 $D_\phi$ 与策略 $\pi_\theta$ 之间的博弈。

你可能已经熟悉 GAN 了：生成器造假样本，判别器区分真假，两者对抗训练直到生成器造出的样本以假乱真。GAIL 把这个思想用到模仿学习上：

- **生成器**就是我们的策略 $\pi_\theta$——它生成"状态-动作对"
- **判别器**$D_\phi$ 负责区分"这个(s,a)是专家做的，还是策略生成的"
- 两者交替训练：判别器努力区分专家和策略，策略努力"骗过"判别器——也就是让自己的状态-动作分布和专家一模一样

### 3.1 判别器与策略怎样交替训练

判别器是一个二分类网络，它的训练目标是：

$$\max_\phi \; \mathbb{E}_{(s,a) \sim \mathcal{D}_{\text{expert}}}\left[\log D_\phi(s, a)\right] + \mathbb{E}_{(s,a) \sim \pi_\theta}\left[\log (1 - D_\phi(s, a))\right]$$

让我们拆开看：

- 专家数据：$D_\phi(s,a)$ 应该接近 1（判断为"真"），所以最大化 $\log D$
- 策略数据：$D_\phi(s,a)$ 应该接近 0（判断为"假"），所以最大化 $\log(1-D)$

这就是标准的二分类交叉熵损失。

策略的目标是什么？策略需要让自己的状态-动作分布更接近专家，也就是让判别器分不清它生成的数据是真是假。若约定 $D_\phi(s,a)$ 表示"样本来自专家"的概率，一种常用的策略目标是最小化：

$$\min_\theta \; \mathbb{E}_{(s,a) \sim \pi_\theta}\left[\log (1-D_\phi(s, a))\right] - \lambda \mathcal{H}(\pi_\theta)$$

- 第一项：让策略生成的(s,a)在判别器看来"像是专家的"——最小化 $\log(1-D)$ 等价于最大化 $D$，也就是骗过判别器
- 第二项：熵正则化，系数 $\lambda$，避免策略过早坍缩到只输出少数几个动作（和 SAC 里的熵正则一样）

实现时常把 $-\log(1-D_\phi(s,a))$ 或 $\log D_\phi(s,a)$ 的等价变体作为**隐式奖励**；具体符号取决于判别器把专家标成 1 还是 0，代码与公式必须使用同一约定。

```python
class GAIL:
    def __init__(self, expert_data, policy, discriminator):
        self.expert_buffer = expert_data   # 专家 (s, a) 对
        self.policy = policy               # 任意 RL 算法（PPO/TRPO/SAC）
        self.disc = discriminator          # 二分类网络

    def update(self, n_policy_steps=5, n_disc_steps=1):
        # === 1. 训练判别器 ===
        for _ in range(n_disc_steps):
            # 采样策略数据
            policy_states, policy_actions = self.policy.sample_rollout()
            # 二分类交叉熵
            expert_logits = self.disc(self.expert_buffer.sample())
            policy_logits = self.disc(policy_states, policy_actions)
            d_loss = (
                F.binary_cross_entropy_with_logits(expert_logits, ones) +
                F.binary_cross_entropy_with_logits(policy_logits, zeros)
            )
            self.disc_optim.zero_grad(); d_loss.backward(); self.disc_optim.step()

        # === 2. 训练策略：用 -log D 作为奖励 ===
        for _ in range(n_policy_steps):
            states, actions, next_states, _ = self.policy.rollout()
            with torch.no_grad():
                # D 表示"来自专家"的概率，所以奖励取 -log(1-D)
                rewards = -F.logsigmoid(-self.disc(states, actions))
            # 喂给任意 RL 算法（这里假设 PPO）
            self.policy.ppo_update(states, actions, rewards, next_states)
```

这里最巧妙的地方是：我们不需要显式地学习奖励函数！判别器本身就给我们提供了奖励信号——$-\log(1-D(s,a))$，这个值在(s,a)看起来像专家时高，不像时低。我们可以把这个信号直接喂给任何 RL 算法（PPO、TRPO、SAC 都行）来更新策略。

### 3.2 GAIL 与最大熵 IRL 的联系

你可能会问："为什么 GAIL 这样做就能work？它和之前说的最大熵 IRL 有什么关系？"

固定策略后，二分类判别器的最优解可以写成两个访问分布的比例：

$$D_\phi^*(s, a) = \frac{p_{\text{expert}}(s, a)}{p_{\text{expert}}(s, a) + p_{\pi_\theta}(s, a)}$$

把这个 $D^*$ 代入对数比，可得：

$$\log D^* - \log(1-D^*) = \log\frac{p_{\text{expert}}(s,a)}{p_{\pi_\theta}(s,a)}$$

当策略访问分布接近专家时——$p_\pi = p_{\text{expert}}$——这个比值等于 1，对数等于 0。也就是说，最优判别器其实是在告诉我们："当前策略在(s,a)这里比专家差多少"。

用数字看这个判别器怎样给出奖励。统计专家数据和策略数据中两类状态-动作对的出现次数：

| $(s, a)$               | 专家出现 | 策略出现 | $D^*$ | 隐式奖励 $-\log(1-D^*)$  |
| ---------------------- | -------- | -------- | ----- | ------------------------ |
| （车道中间，平稳转向） | 800 次   | 200 次   | 0.8   | $-\log 0.2 \approx 1.61$ |
| （压线行驶，急打方向） | 0 次     | 500 次   | 0     | $-\log 1 = 0$            |

专家常去而策略少去的地方，$D^*$ 高，隐式奖励也高；策略常去而专家不去的地方，判别器一眼识破，奖励接近 0。上一小节代码里的 `rewards = -log(1-D)` 用的正是这个信号——策略朝着奖励高的方向做 RL，就会越来越频繁地走向"像专家"的状态-动作对。

GAIL 通过判别器估计这种分布差异，因此不需要显式枚举所有轨迹来计算 $Z$——这就是为什么它比 MaxEnt IRL 高效得多，可以扩展到高维任务。

::: details 加餐：GAIL 是最优传输问题的一个特例
从更理论的视角看，GAIL 实际上是在最小化专家分布和策略分布之间的 Jensen-Shannon 散度（这正是 GAN 最小化的散度）。后来的工作如 DAC（Discriminator-Actor-Critic）、WGAIL（Wasserstein GAIL）等探索了用其他散度（如 Wasserstein 距离）来替代，改善训练稳定性。
:::

## 4. 比较三条模仿学习路线

让我们把 BC、MaxEnt IRL、GAIL 放在一起系统对比：

| 维度             | BC  | MaxEnt IRL        | GAIL                |
| ---------------- | --- | ----------------- | ------------------- |
| 是否解决分布偏移 | ❌  | ✅                | ✅                  |
| 需要环境模型     | ❌  | ✅（或软 Q 近似） | ❌                  |
| 显式奖励函数     | —   | ✅（可解释）      | ❌（隐式）          |
| 计算成本         | 低  | 高（内层 RL）     | 中（对抗训练）      |
| 扩展到高维       | 易  | 难                | 中                  |
| LLM 中的对应     | SFT | —                 | DPO 隐式（见 14.6） |

### 4.1 GAIL 的训练稳定性

但 GAIL 也不是完美的——它继承了 GAN 的通病：训练不稳定。判别器过强时生成器梯度消失，判别器过弱时生成器学不到信号。实践中常用几个 Tricks：

- 判别器梯度惩罚（Wasserstein GAIL，用梯度裁剪替代权重裁剪）
- 调整判别器和策略的更新频率（比如每更新 5 步策略，才更新 1 步判别器，避免判别器太强）
- 熵正则化系数 $\lambda$ 调到 0.1-1.0 防止策略坍缩到确定性策略

GAIL 在 MuJoCo 上能接近专家水平，但需要数百万步环境交互——**样本效率仍是瓶颈**。这推动了对**离线模仿学习**的研究（如 DemoDICE、DWBC），把专家数据与次优数据结合，无需在线交互。

## 本节总结

逆向 RL（IRL）把问题从"直接学做什么动作"变成"先学什么是好的"——从专家行为反推奖励函数，再用这个奖励训练策略。MaxEnt IRL 用最大熵原则解决了 IRL 的不适定问题（有无穷多奖励能解释专家行为），但每次更新都需要解一个内层 RL，计算成本太高，难以扩展。

GAIL 借用 GAN 的思想，用判别器和策略的对抗训练绕开了显式奖励推断和配分函数计算，让模仿学习的可扩展性大幅提升。判别器本身提供隐式奖励，策略只需要努力"骗过"判别器——这和后来 LLM 中的 DPO 等偏好优化方法在思想上是同源的。

但 GAIL 仍然假设训练和部署面对的是同一个任务。下一节 [11.3 元 RL：MAML、RL²、PEARL、In-Context RL](./meta-rl) 转向一个更有野心的问题——**当环境不断变化时，agent 如何从少量经验中快速适应新任务**？这就是元强化学习。
