# 10.1 离线数据与分布偏移

## 本节导读

**核心内容**

- 理解为什么"不给新数据"这件事会彻底改变 RL 的游戏规则——分布偏移和外推误差从何而来，为什么 γ=0.99 时误差会被放大近 100 倍
- 掌握三大经典离线 RL 算法的思路差异：BCQ 直接限制动作、CQL 主动压低 OOD 动作的 Q 值、IQL 完全避开数据集外的动作评估
- 理解工程化路线的威力：为什么 TD3+BC 这种"在 Actor loss 里直接加 BC 项"的简单方法，效果能接近复杂算法
- 对比 AWAC 和 IQL 的细微差别，理解"避免 max OOD 动作"为什么是更根本的解决方案

Part II 我们学了一整套在线强化学习算法：从 DQN 到 DDPG、TD3、SAC，它们都依靠智能体持续与环境交互来收集经验。但是让我们停下来想一下——现实世界里，这种"边交互边学习"的模式真的总能行得通吗？

你可能会想到很多场景：自动驾驶汽车不能在真实道路上反复试错撞人，推荐系统不能随便给用户推垃圾内容测试效果，医疗领域更不能拿病人做实验。这些场景里，我们只能使用已有的历史日志，新的试错可能昂贵、缓慢，甚至存在安全风险。

Part III 就从**离线强化学习（Offline RL）**开始——这是一种"只能看历史数据，不能与环境交互"的设定。随后我们会把固定数据学习的思路推进到模仿学习、逆向强化学习、元强化学习、探索、多智能体与分层决策。

回忆一下[第 9 章](../chapter11_continuous_control/ddpg)中学过的 DDPG、TD3 和 SAC，它们其实已经可以用 replay buffer 复用历史数据；基于模型的强化学习也能借助环境模型减少真实交互。不过，这些方法仍然允许新策略继续采样并修正旧经验——它们本质上还是在线的。离线强化学习彻底取消了这条反馈通道：**训练期间只能使用一份固定数据集，绝对不能和环境交互**。

这一听起来很小的限制，为什么会带来那么大的麻烦？本节我们就沿着三个核心问题展开：固定数据为什么会导致分布偏移，BCQ、CQL 与 IQL 怎样限制错误估值，AWAC 与 TD3+BC 又怎样把行为克隆加入策略更新。理解了这条主线以后，[10.2](./sequence-modeling) 才会转向 Decision Transformer 开创的序列建模路线——那条路线甚至连 Bellman 方程都不用了。

## 1. 固定数据为什么会产生分布偏移

在[第 5 章 DQN](../chapter07_dqn/from-q-to-dqn)和[第 9 章 SAC](../chapter11_continuous_control/ddpg)中，我们都会用下一状态的估值来更新当前状态。先写出这个大家已经很熟悉的一步目标：

$$y = r + \gamma \cdot \mathbb{E}_{s' \sim P(\cdot \mid s, a)}\left[V(s')\right]$$

让我们把这行公式从左向右仔细拆解一遍，确保每个符号都清楚：

- $y$：这是本次我们要拟合的目标值，也就是当前 (s,a) 应该"瞄准"的方向
- $r$：这是当前动作已经实实在在拿到的奖励，是确定的、已知的
- $V(s')$：这是下一状态之后的长期价值，需要我们去估计——问题往往就出在这里
- $\gamma$：折扣因子，控制未来价值在目标中占多大比重，通常取 0.99 左右

在线训练的时候，即使我们暂时高估了某个新状态的 V 值，也没关系——因为策略以后总有机会真正访问到那个状态，用真实的奖励信号修正估值。就像你学骑车，偶尔歪了一下，下次骑到那个姿势时身体会自动调整。

但是离线 RL 没有这个"试错修正"的保险。让我们停下来想一下：如果训练时永远不能去新状态，那 V(s') 估错了怎么办？

离线 RL 中，数据集 $\mathcal{D} = \{(s, a, r, s')\}$ 是由某个**行为策略** $\pi_\beta$（behavior policy，也就是收集数据时用的那个策略）采得的，训练时**完全冻结**，再也不能新增任何数据：

$$\mathcal{D} = \{(s_i, a_i, r_i, s'_i)\}_{i=1}^{N}, \quad (s, a) \sim d^{\pi_\beta}(s) \pi_\beta(a \mid s)$$

这里：

- $N$：数据集中的转移样本总数
- $d^{\pi_\beta}(s)$：行为策略 $\pi_\beta$ 产生的状态访问分布（也就是它经常去哪些状态）
- $\pi_\beta(a \mid s)$：行为策略在状态 s 下选择动作 a 的概率

新策略 $\pi_\theta$ 在数据集上训练完成后才部署。但你猜怎么着？它选择动作的分布 $\pi_\theta(a \mid s)$ 和原来的行为策略 $\pi_\beta(a \mid s)$ 几乎肯定不一样——毕竟我们训练它就是为了改进啊！这个策略分布和数据分布不一致的问题，就是**分布偏移（distribution shift）**。

你可能会问："分布不一样就不一样呗，在线 RL 里策略不也一直在变吗？为什么离线就不行？"问得好——这正是离线 RL 的核心难题，我们接着看。

### 1.1 外推误差从哪里产生

Fujimoto et al. 2019 在 BCQ 的经典论文中精确刻画了离线 RL 失败的根源。我们先定义一个概念：设数据集在状态 s 上的动作支撑集为 $\mathcal{D}_\mathcal{A}(s) = \{a : (s, a) \in \text{support}(\pi_\beta(\cdot \mid s))\}$——简单说就是"在状态 s 下，数据里实际出现过哪些动作"。

关键问题来了：Bellman 算子在 $a' \notin \mathcal{D}_\mathcal{A}(s')$ 这些动作上的取值，**没有任何监督信号**！神经网络在这些 OOD（out-of-distribution，分布外）点上只能瞎猜——也就是**外推（extrapolation）**，而神经网络在外推区域的输出是任意的、不可靠的。

为了看清楚误差到底从哪里来，我们可以把 Q 函数的估值误差按来源写成一个示意分解：

$$\underbrace{Q_\phi(s, a) - Q^\pi(s, a)}_{\text{总误差}} = \underbrace{\epsilon_{\text{stat}}}_{\substack{\text{统计误差}\\\text{(样本有限)}}} + \underbrace{\epsilon_{\text{approx}}}_{\substack{\text{函数逼近误差}\\\text{(网络容量)}}} + \underbrace{\max_{a'} Q_\phi(s', a') - Q^\pi(s', \pi(s'))}_{\text{外推误差 (Extrapolation Error)}}$$

我们一项一项来看：

- 前两项 $\epsilon_{\text{stat}}$ 和 $\epsilon_{\text{approx}}$：这是老朋友了，在线训练和离线训练中都会出现——数据有限、网络容量有限，不可能完全拟合，这很正常
- 第三项才是离线 RL 的专属恶魔：当我们做 $\max_{a'} Q(s', a')$ 这个操作时，如果恰好选中了一个数据里从没见过的 OOD 动作，网络可能碰巧给了它一个很高的 Q 值（比如初始化偏巧大、或者训练噪声），那这个完全没验证过的虚假高值就会进入下一轮的 target

更可怕的是，这种外推误差会通过 Bellman 备份递归累积。设 $Q_0$ 是初始估值，Bellman 迭代 $T$ 次后误差满足：

::: details 加餐：外推误差累积公式推导

$$\|Q_T - Q^\pi\|_\infty \leq \gamma^T \|Q_0 - Q^\pi\|_\infty + \sum_{k=0}^{T-1} \gamma^k \|\mathcal{T} Q_k - \mathcal{T}^\pi Q_k\|_\infty$$

这个式子看起来有点吓人，但拆开来看很直观：

- 左边 $\|Q_T - Q^\pi\|_\infty$：迭代 T 次后 Q 函数和真实 Q\* 的最大差距
- 右边第一项 $\gamma^T \|Q_0 - Q^\pi\|_\infty$：初始误差会被折扣因子 γ 指数衰减——γ=0.99 的话，迭代 100 次后就剩不到 1% 了，这是好事
- 右边求和项：这才是麻烦所在——每一轮 Bellman 更新新引入的误差，会被 γ^k 加权后累加起来

这里 $\mathcal{T}$ 表示带动作最大化的 Bellman 更新（我们实际做的），$\mathcal{T}^\pi$ 表示按真实策略计算的更新（理想情况）。两者的差就是每轮新产生的外推误差。

如果每轮都产生大小接近 $\epsilon_{\text{ood}}$ 的误差，它们的总影响会被几何级数放大到约 $\epsilon_{\text{ood}}/(1-\gamma)$。举个具体数值：γ=0.99 时，1/(1-0.99)=100——也就是说每一轮引入 1 单位的 OOD 误差，最后会累积成近 100 单位的总误差！

注意这是**误差反复累加**，不是说误差值本身会指数爆炸增长。但即便如此，100 倍的放大系数也足以让训练彻底崩溃。
:::

::: warning 为什么加更多数据救不了
看到这里你可能会想："那我多收集点数据，把所有 (s,a) 都覆盖到不就行了？"很遗憾，在连续动作空间中这几乎不可能——动作是连续值，你永远不可能覆盖"每个可能的 a"。只要更新时还会在缺少数据的区域取最大值，外推误差就有可能冒出来。因此，扩大数据覆盖和保守更新需要双管齐下，只靠堆数据是不够的。
:::

### 1.2 离线 RL 要同时优化什么

有了上面的诊断，我们就可以把离线 RL 的目标形式化地写出来了：在数据集支撑下学一个策略 $\pi_\theta$，使其期望回报尽可能大，但 **$\pi_\theta$ 不能偏离 $\pi_\beta$ 太远**——否则就会进入 OOD 区域，触发外推误差的死亡螺旋。

所有现代离线 RL 算法本质上都是在这两个目标之间求平衡：

$$\max_\theta \; \mathbb{E}_{s \sim \mathcal{D}}\left[Q^\pi(s, \pi_\theta(s))\right] \quad \text{subject to} \quad D(\pi_\theta \| \pi_\beta) \leq \epsilon$$

这是一个带约束的优化问题：

- 目标：让策略的 Q 值尽可能高（也就是回报尽可能大）
- 约束：用某种散度 D（比如 KL 散度、MMD 距离等）衡量，新策略和行为策略的差距不能超过阈值 ε

你可以把它想象成在悬崖边上走路：你想尽可能往高处走（max Q），但又不能离安全区（数据集覆盖范围）太远，否则就会掉下悬崖（外推误差爆炸）。不同算法的区别，本质上就是用不同方式来画这条"安全边界"。

下面我们先看三大经典算法是怎样在动作空间或价值函数中实现这个约束的，然后再看另一条更工程化的路线——怎样直接把行为克隆加入策略损失。

## 2. 用保守估值限制数据集外动作

最直接的思路是什么？既然问题出在 OOD 动作的 Q 值可能被高估，那就**让 Q 函数对 OOD 动作悲观一点**。如果 $Q(s, a)$ 在没见过的 $a$ 上给很低的值，那 $\max_a Q(s, a)$ 自然就不会选到那些幻想出来的"看起来很美"的动作了。

三大经典算法——BCQ、CQL、IQL——从不同角度实现了这一"保守估计"原则。我们一个一个来看。

### 2.1 BCQ：把动作限制在数据分布附近

Batch-Constrained Q-Learning（Fujimoto et al. 2019）是第一个被证明能在连续动作离线数据上稳定训练的深度离线 RL 算法。它的核心约束非常直白：**target 计算时用的动作 $a'$，必须落在 $\pi_\beta$ 的支撑集内**——也就是数据里出现过的动作附近。

怎么实现呢？BCQ 先训练一个条件 VAE（变分自编码器）$\pi_\beta(a \mid s)$ 来近似行为策略，然后从这个 VAE 中采样出一批候选动作 $\{a_i\}$，再在这些候选动作上做 max，并且只允许做小幅扰动：

$$a' = \arg\max_{a \in \{a_i + \xi \Phi(s, a_i)\}} Q_\phi(s', a)$$

我们来拆解这个式子：

- $\{a_i\} \sim \pi_\beta$：从学到的行为策略模型中采样 n 个候选动作（比如 n=100），这些都是数据中大概率出现的动作
- $\Phi(s, a)$：一个扰动网络，对采样出来的动作做小幅修正，用来逼近局部最优——不然只能完全模仿行为策略，没法改进了
- $\xi$：扰动幅度的上限，是个超参数——不能扰太大，不然又跑出数据集了
- 最后在这 n 个加了扰动的候选里选 Q 值最高的那个作为 a'

这相当于把"连续动作空间中的 argmax"这个危险操作，约束在了行为策略的高密度区域内——你只能在数据里见过的动作附近挑，不能去没去过的地方乱逛。

### 2.2 CQL：压低数据集外动作的价值

Conservative Q-Learning（Kumar et al. 2020）换了个角度切入——我不直接约束你选什么动作，而是**直接惩罚 Q 函数在 OOD 动作上的值**，让那些没见过的动作的 Q 值系统性地被低估。这样不管你怎么 max，都不会选到它们。

具体怎么做？在标准 Bellman 误差之外，加一个保守正则项：

$$\mathcal{L}_{\text{CQL}}(Q) = \alpha \left(\mathbb{E}_{s \sim \mathcal{D}}\left[\log \sum_a \exp(Q(s, a))\right] - \mathbb{E}_{(s, a) \sim \mathcal{D}}[Q(s, a)]\right) + \mathcal{L}_{\text{Bellman}}(Q)$$

这个 loss 看起来有点复杂，我们拆开看两项的作用：

- 第一项 $\log \sum_a \exp(Q(s, a))$：这是 logsumexp，对**所有动作**（包括数据里的和数据外的 OOD 动作）的 Q 值做软最大值。要让这一项变小，唯一的办法就是把所有动作的 Q 值都压低——包括 OOD 动作
- 第二项 $- \mathbb{E}_{(s,a) \sim \mathcal{D}}[Q(s,a)]$：这一项把数据集里实际见过的 (s,a) 的 Q 值往上拉，不让它们被压得太低
- $\alpha$：控制保守程度的系数

这一压一拉之间，就形成了一个"惩罚 gap"：OOD 动作的 Q 值被系统性地低估，而数据内动作的 Q 值保持在合理范围。

CQL 有很漂亮的理论保证：学到的 $\hat{Q}$ 是真实 $Q^\pi$ 的**下界**，也就是对所有 (s,a) 都有 $\hat{Q}(s, a) \leq Q^\pi(s, a)$；进一步可以证明，$\hat{Q}$ 在 OOD 动作上的值，比在数据分布内动作的值低一个 $\mathcal{O}(\alpha)$ 的 gap。因此由 $\hat{Q}$ 推出来的策略，永远不会高估任何动作的回报——这正是我们想要的保守性。

在实践中，α 不需要手动调，用 Lagrangian 对偶自动调节就好，让保守性恰到好处：

$$\mathcal{L}(\alpha) = -\alpha \cdot \left(\mathbb{E}_s\left[\log\sum_a \exp(\hat{Q}(s, a))\right] - \mathbb{E}_{(s, a) \sim \mathcal{D}}[\hat{Q}(s, a)] - \xi\right)$$

这里 ξ 是我们想要的目标 gap（论文里取 5.0 左右）。当实际 gap 低于 ξ 时，就增大 α 加强保守惩罚；反之就减小 α，让 Q 值学准一点。这样 gap 会自动稳定在目标附近。

来看简化的代码实现（基于 SAC 框架）：

```python
class CQL(SAC):
    def critic_loss(self, batch):
        s, a, r, s_next, done = batch
        # 标准 Bellman 误差（完全继承自 SAC）
        with torch.no_grad():
            a_next = self.actor(s_next)
            q_target = torch.min(self.critic_target1(s_next, a_next),
                                  self.critic_target2(s_next, a_next))
            y = r + self.gamma * (1 - done) * q_target
        bellman_loss = F.mse_loss(self.critic1(s, a), y) + \
                       F.mse_loss(self.critic2(s, a), y)

        # CQL 保守正则
        # 第一项：对随机动作（OOD）做 logsumexp
        rand_a = torch.rand_like(a) * 2 - 1
        q_rand1 = self.critic1(s, rand_a).flatten()
        q_curr1 = self.critic1(s, a).flatten()  # in-dist
        q_next1 = self.critic1(s, a_next).flatten()
        cat_q1 = torch.cat([q_rand1, q_curr1, q_next1], dim=1)
        logsumexp_q1 = torch.logsumexp(cat_q1, dim=1).mean()

        conservative_loss = \
            self.alpha * (logsumexp_q1 - q_curr1.mean()) \
            + self.alpha * (logsumexp_q2 - q_curr2.mean())

        return bellman_loss + conservative_loss
```

（注意上面代码为了清晰省略了第二个 Q 网络的对称计算，实际实现要对 Q2 做同样的处理。）

### 2.3 IQL：避免显式评估数据集外动作

Implicit Q-Learning（Kostrikov et al. 2022）的思路更加彻底：既然问题出在 $\max_{a'} Q(s', a')$ 会选中 OOD 动作，那我干脆**完全不对数据集外的动作做任何显式评估**，不就从根源上避免了外推误差吗？

具体怎么做？IQL 分两步走：第一步用 expectile regression（期望分位回归）学习状态价值函数 $V(s)$，让 $V$ 偏向数据中价值较高的动作，但整个过程只用到数据集里已经出现过的动作：

$$\mathcal{L}_V = \mathbb{E}_{(s, a) \sim \mathcal{D}}\left[L_2^\tau(Q_{\bar{\theta}}(s, a) - V_\psi(s))\right]$$

这里先计算残差 $x=Q_{\bar{\theta}}(s,a)-V_\psi(s)$——也就是 Q 和 V 的差，然后用一个非对称的平方损失：

$$L_2^\tau(x) = |\tau - \mathbb{1}(x < 0)| \cdot x^2$$

这就是**expectile loss（期望分位损失）**。我们来理解它：

- 当 $x > 0$（Q > V，说明这个动作比当前 V 估计的好）：权重是 τ
- 当 $x < 0$（Q < V，说明这个动作比当前 V 估计的差）：权重是 1-τ
- 如果取 τ=0.5，这就是普通的 MSE，V 会拟合 Q 的期望
- 如果取 τ>0.5（比如 τ=0.7），正残差的权重更大，V(s) 就会更靠近数据中较高的 Q(s,a)——相当于一种"软 max"，但全程只在数据内动作上计算！

得到 V 之后，我们就可以定义优势函数 $A(s,a)=Q_{\bar\theta}(s,a)-V_\psi(s)$，然后用 advantage-weighted regression 训练策略：

$$\mathcal{L}_\pi = -\mathbb{E}_{(s, a) \sim \mathcal{D}}\left[\exp(\beta \cdot A(s, a)) \cdot \log \pi_\theta(a \mid s)\right]$$

这个 loss 的直觉很清晰：

- 如果 $A(s,a) > 0$，说明数据里的这个动作比该状态的平均水平好，指数权重 $\exp(\beta A)$ 就大于 1，我们就更努力地模仿这个动作
- 如果 $A(s,a) < 0$，说明这个动作比平均水平差，权重就小于 1，我们就少模仿它一点
- $\beta$ 是温度系数，控制这种好坏差别被放大多少——β 越大，越倾向于只模仿最好的动作

整个 IQL 流程中，Q 的 backup 用的是 V(s') 而不是 $\max_a Q(s',a)$，V 和策略的训练都只在数据集内的 (s,a) 上计算——**从头到尾没有一步需要查询 OOD 动作的 Q 值**，因此从根本上避开了外推误差的产生路径。CQL 是主动压低 OOD 动作的价值，而 IQL 是"我根本不去看那些动作"。

### 2.4 比较 BCQ、CQL 与 IQL

我们把这三个经典算法放在一起对比一下：

| 维度               | BCQ             | CQL                 | IQL                  |
| ------------------ | --------------- | ------------------- | -------------------- |
| 约束位置           | 动作空间        | 值函数              | 隐式（分位数 + AWR） |
| 是否评估 OOD 动作  | 否（采样约束）  | 是（logsumexp）     | 否（避免显式查询）   |
| 额外网络           | VAE $\pi_\beta$ | 无                  | $V$ 网络             |
| 超参敏感           | 高（扰动幅度）  | 中（$\alpha$ 自动） | 低（$\tau, \beta$）  |
| 对中等数据集表现   | 中              | 强                  | 强                   |
| 对稀疏数据集稳定性 | 中              | 偶发不稳定          | 强                   |
| 实现复杂度         | 高              | 中                  | 低                   |

给你的实践建议：第一次实现离线 RL 时，可以先用 IQL 建立基线——它的更新只依赖数据集内动作，超参数少且鲁棒，实现也最简单；需要显式控制保守程度时再去试 CQL；BCQ 主要适合用来帮助理解"限制候选动作"这条思路，实际工程中用得相对少一些。

## 3. 用行为克隆约束策略更新

除了上面说的"在价值函数层面做保守估计"这条路线，还有另一条更受工程师欢迎的路线——**保留我们熟悉的 on-policy / off-policy actor-critic 主循环，直接在策略损失里加一个行为克隆（BC）正则项**。这类方法的巨大优势是：和第 8 至 9 章学过的 PPO/SAC/TD3 框架完全兼容，工程改造量极小，往往改几行代码就行。

### 3.1 TD3+BC：在策略损失中加入行为克隆

Fujimoto & Gu 2021 提出的 TD3+BC 把这种"简单粗暴但有效"的哲学发挥到了极致：就在 TD3 的 Actor 损失上加一个 L2 行为克隆项，再加一个自适应权重 λ 就完事了：

$$\mathcal{L}_{\text{actor}} = -\mathbb{E}_{s \sim \mathcal{D}}\left[Q(s, \mu_\theta(s))\right] + \lambda \cdot \mathbb{E}_{(s, a) \sim \mathcal{D}}\left[(\mu_\theta(s) - a)^2\right]$$

这个 loss 太好理解了：

- 第一项 $-\mathbb{E}[Q(s, \mu_\theta(s))]$：就是普通 TD3 的 Actor 损失——让 Q 值最大化，也就是改进策略
- 第二项 $\lambda \cdot \mathbb{E}[(\mu_\theta(s) - a)^2]$：这就是行为克隆——让策略输出不要离数据里的动作 a 太远，L2 距离惩罚偏离

关键是权重 λ 的设计非常巧妙，它会自动适应不同环境的 reward scale，根本不用调：

$$\lambda = \frac{\alpha}{\frac{1}{N}\sum_i |Q(s_i, \mu_{\theta_{\text{old}}}(s_i))|}$$

分母是什么？是当前 Q 值的平均绝对值——也就是不同环境下回报的尺度。如果一个环境的 Q 值普遍很大（比如 reward 没归一化，动辄成百上千），分母就大，λ 自动变小，BC 项的权重相对就弱；如果 Q 值普遍很小，分母就小，λ 自动变大，BC 约束就强。论文里固定 α=2.5，在所有 D4RL MuJoCo 任务上用同一套超参数就能拿到很好的结果。

TD3+BC 的简洁性使它成为离线 RL 非常实用的强基线。它的表现也提示了一个有点反直觉的事实：**在很多离线 RL benchmark 上，这种最朴素的 BC 正则化就能达到接近 CQL/IQL 这些复杂算法的性能**——所以永远不要低估简单方法的威力。

### 3.2 AWAC：提高优质动作的模仿权重

Advantage-Weighted Actor-Critic（Nair et al. 2020）和 IQL 的策略损失其实有同一个源头——advantage-weighted regression（AWR，优势加权回归）——但 AWAC 用显式学习的 Q 函数来计算优势，而不是像 IQL 那样用分位数回归隐式估计 V：

$$\mathcal{L}_\pi^{\text{AWAC}} = -\mathbb{E}_{(s, a) \sim \mathcal{D}}\left[\underbrace{\exp\left(\frac{A(s, a)}{\beta}\right)}_{\text{advantage 权重}} \cdot \log \pi_\theta(a \mid s)\right]$$

其中优势函数 $A(s, a) = Q(s, a) - V(s)$，β 是温度系数。直观理解非常简单：数据中表现优于平均的动作（A>0），权重被指数放大，我们就重点模仿；劣于平均的动作（A<0），权重被压低，我们就少模仿。AWAC 其实是把普通的行为克隆推广成了"加权行为克隆"——不是无脑模仿数据里的所有动作，而是只模仿好的部分。

AWAC 有一个非常实用的工程亮点：它**支持从离线到在线的平滑过渡**——先用纯离线数据预训练，然后再用少量在线交互微调，策略分布不会发生剧烈变化。这一点对真实机器人、推荐系统等需要"冷启动+后续迭代"的场景非常有吸引力。

### 3.3 AWAC 与 IQL 的差别在哪里

你可能已经注意到了，AWAC 和 IQL 的策略损失长得几乎一模一样，我们把它们放在一起对比：

$$\mathcal{L}_\pi^{\text{AWAC}} = -\mathbb{E}\left[\exp\left(\frac{A(s, a)}{\beta}\right) \log \pi(a \mid s)\right], \quad \mathcal{L}_\pi^{\text{IQL}} = -\mathbb{E}\left[\exp\left(\beta \cdot A(s, a)\right) \log \pi(a \mid s)\right]$$

形式上几乎完全一致——β 一个在分母一个在分子，但本质上都可以看作温度系数，只是用法不同。那它们的核心差别在哪里？关键就在于**优势函数 A(s,a) 是怎么估计出来的**：

- **AWAC**：$A = Q_\phi(s, a) - V_\psi(s)$，其中 Q 仍然走标准 Bellman 备份——也就是说 target 里仍然有 $\max_{a'} \pi(a'|s')$ 这一步，还是有可能查询到 OOD 动作
- **IQL**：$A = Q_\phi(s, a) - V_\psi(s)$，但 Q 的备份是通过 V 来做的——target 直接用 V(s') 而不是 $\max_a Q(s', a')$，而 V 是用 expectile regression 在数据内动作上估计出来的，偏向数据中较好的动作

IQL 通过把 Bellman target 从 $\max Q$ 改成 $V$，从根源上消除了外推误差的产生路径；而 AWAC 保留了标准 Bellman target，只是靠加权 BC 来约束策略不要偏太远——这种约束比 IQL 的隐式约束要弱，因此当数据集里 Q 值噪声很大时，AWAC 还是有可能踩到 OOD 的雷。就是这么一个看似微小的差别，在稀疏数据上对稳定性的影响却很大。

### 3.4 比较 AWAC、TD3+BC 与 IQL

我们再把这三个基于 BC 正则的算法对比一下：

| 方法   | 策略损失形式                            | 是否需要 $V$ | 在线微调友好 |
| ------ | --------------------------------------- | ------------ | ------------ |
| TD3+BC | $-\!Q + \lambda \|\mu - a\|^2$          | 否           | 中           |
| AWAC   | $-\!w(A) \log \pi$，$w = \exp(A/\beta)$ | 是           | 强           |
| IQL    | $-\!\exp(\beta A) \log \pi$（AWR）      | 是           | 中           |

注意 AWAC 和 IQL 的策略损失结构高度相似，核心区别就在于 A 的来源——AWAC 用显式的 Q-V 差（Q 仍走标准 Bellman 备份），IQL 用 expectile regression 隐式估计 V，完全避免 OOD 查询。这个细节再次说明：在离线 RL 中，**"避免对 OOD 动作取 max"是比"怎么加 BC 正则"更根本的问题**。

## 本节总结

这一节我们从"为什么没有交互数据就会崩"这个问题出发，理解了分布偏移和外推误差的本质：Q-Learning 中的 max 算子可能选中数据集外的动作，而神经网络在 OOD 区域的外推是任意的，这种误差还会通过 Bellman 备份被 γ 放大近百倍。

针对这个问题，我们看到了两大类解决思路：

1. **保守估值路线**：BCQ 把候选动作限制在数据分布附近，CQL 主动压低数据外动作的 Q 值，IQL 干脆完全避免对数据外动作显式取最大值
2. **BC 正则路线**：TD3+BC 在 Actor loss 里直接加 L2 BC 项，简单有效；AWAC 用优势加权做"非均匀模仿"，还支持离线到在线的平滑过渡

但你发现没有——上面所有这些方法，不管多复杂，它们都还在 Bellman 方程的框架里打转转：都是在估计价值函数，都是在做策略改进，只是想办法别让外推误差爆掉。那有没有可能……我们干脆彻底抛弃 Bellman 方程，换一种完全不同的思路来看待离线 RL？

下一节 [10.2 基于序列建模的离线强化学习](./sequence-modeling)，我们就来看 Decision Transformer 开创的另一条路——把 RL 完全写成条件序列生成问题，用大家熟悉的 GPT-like Transformer 来做，连 Q 函数都不用学了。
