# 10.2 基于序列建模的离线强化学习

## 本节导读

**核心内容**

- 理解决策 Transformer（DT）的核心思路：为什么可以把 RL 完全抛开 Bellman 方程，变成"给定目标回报→预测动作"的条件序列生成问题
- 搞懂 return-to-go（RTG）是什么，为什么它能作为条件控制生成的好坏，以及训练和推理时 RTG 分别怎么用
- 了解 Trajectory Transformer 如何把整条轨迹离散化成 token 并用 beam search 做规划
- 了解 Diffuser 如何用扩散模型生成完整轨迹，以及三种序列建模方法各自的优劣势和适用场景

上一节 [10.1](./offline-data-distribution-shift) 我们学的 BCQ、CQL、IQL、TD3+BC 这些方法，虽然各有各的技巧，但它们骨子里还都是传统 RL 的思路：估计价值函数，做策略改进，只是想尽办法在贝尔曼备份时别让外推误差爆掉。

但是让我们停下来想一下——我们真的需要 Q 函数吗？真的需要反复做 Bellman 迭代吗？

2021 年 Decision Transformer 的提出给了这个问题一个令人惊讶的答案：不需要。它完全跳出了 Bellman 框架，把离线强化学习重新定义成了一个**条件序列生成问题**——把状态、动作和回报排成序列，直接学习"当我想要拿到多少回报时，在当前状态下应该采取什么动作"。这思路和大语言模型预测下一个 token 简直一模一样！

这是不是听起来有点反直觉？不用算 Q 值，不用做价值迭代，就靠预测动作，真的能学会好策略吗？这一节我们就按四步来解开这个疑惑：先用 return-to-go 建立 Decision Transformer，再说明 Trajectory Transformer 怎样搜索完整轨迹，随后介绍 Diffuser 的条件生成，最后比较三种方法各自适合什么任务。

## 1. 用目标回报训练 Decision Transformer

上一节的所有方法都是通过约束动作、压低数据集外 Q 值或者加入 BC 正则来稳定 Bellman 更新。Decision Transformer（Chen et al. 2021，后面简称 DT）则选择了一条革命性的路线：不再学习 Q 函数，也不做 Bellman 更新，而是把离线轨迹直接改写成一个条件序列生成问题。

### 1.1 把 Return-to-Go 作为条件

我们先回忆一下一条轨迹是什么样的：$\tau=(s_1,a_1,r_1,s_2,a_2,r_2,\ldots,s_T,a_T,r_T)$——按时间顺序排列的状态、动作、奖励三元组。

传统 RL 是往后看的：在 t 时刻，你要估计未来会拿到多少奖励（也就是 Q 值），然后选让这个值最大的动作。DT 把这个逻辑反过来了——它往前看：在第 t 步，模型除了看到当前状态，还直接告诉它"**从现在开始到回合结束，我希望拿到多少总回报**"，然后让它预测在这种情况下历史数据里实际采取的动作是什么。

这个"从当前时刻到轨迹结束的剩余奖励总和"有个专门的名字，叫做 **return-to-go（RTG，剩余回报）**：

$$\hat{R}_t = \sum_{t'=t}^{T} r_{t'}$$

我们用一个具体的小例子来理解。假设一条轨迹后面三步拿到的奖励是 2、1、3，那么这三个位置的 RTG 依次是多少呢？

- 第一步（拿到 2 之前）：$\hat{R}_1 = 2 + 1 + 3 = 6$（还能拿 6 分）
- 第二步（拿到 2 之后、拿到 1 之前）：$\hat{R}_2 = 1 + 3 = 4$（还剩 4 分）
- 第三步（拿到 1 之后、拿到 3 之前）：$\hat{R}_3 = 3$（只剩最后 3 分了）

你看，时间越往后走，已经拿到的奖励就会从目标 RTG 中扣除——$\hat{R}_t$ 永远表示"还差多少回报要拿"。给定 $\hat{R}_t$ 和 $s_t$ 之后，训练任务就变得异常简单：让模型预测数据中在这种情况下实际采取的动作 $a_t$。

那怎么把这些信息喂给 Transformer 呢？DT 把轨迹重组为一个三元组交错的序列：

$$\hat{R}_1, s_1, a_1, \hat{R}_2, s_2, a_2, \ldots, \hat{R}_T, s_T, a_T$$

每个时间步依次放入三个 token：RTG、状态、动作。然后用大家熟悉的 GPT 式因果注意力掩码，保证模型在预测 $a_t$ 时只能看到当前及以前的信息，不能偷看未来：

$$\pi_\theta(a_t \mid \hat{R}_t, s_t, a_{t-1}, \ldots) = \text{Transformer}(\hat{R}_{1:t}, s_{1:t}, a_{1:t-1})$$

我们来解读一下这个式子：

- 左边 $\pi_\theta(a_t \mid \ldots)$：模型在当前上下文（历史 RTG、历史状态、历史动作，加上当前 RTG 和当前状态）下选择动作 $a_t$ 的概率
- 右边 $\text{Transformer}(\ldots)$：Transformer 读取到 t 时刻为止的所有 RTG 和状态，以及到 t-1 时刻为止的动作，然后输出 $a_t$ 的分布

注意一个最关键的区别：**整个训练过程完全不调用 Bellman 更新**！它就是一个纯纯的监督学习——目标就是让预测的动作尽量接近训练数据中实际的动作。这意味着什么？意味着数据加载、优化器、分布式训练……所有这些你在训练 GPT/BERT 时用的成熟工具栈，直接拿来用就行了！

来看代码实现，你会发现它就是个标准的 GPT 改了改输入：

```python
class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, act_dim, hidden_dim, n_heads, n_layers,
                 max_ep_len=4096):
        super().__init__()
        # 三个嵌入层：RTG, state, action 各自映射到 hidden_dim
        self.embed_rtg  = nn.Linear(1, hidden_dim)
        self.embed_state = nn.Linear(state_dim, hidden_dim)
        self.embed_action = nn.Linear(act_dim, hidden_dim)
        self.embed_ln = nn.LayerNorm(hidden_dim)
        # 位置编码：timestep 嵌入
        self.pos_emb = nn.Embedding(max_ep_len, hidden_dim)
        # GPT 主体
        self.transformer = GPT(
            d_model=hidden_dim, n_heads=n_heads, n_layers=n_layers,
            # 关键：每个 timestep 占 3 个 token，attention mask 要对应
            attn_pdrop=0.1, resid_pdrop=0.1
        )
        # 动作预测头（回归，假设连续动作）
        self.action_head = nn.Linear(hidden_dim, act_dim)

    def forward(self, rtg, states, actions, timesteps):
        B, T, _ = states.shape
        # 嵌入并交错：(R1, s1, a1, R2, s2, a2, ...)
        rtg_emb   = self.embed_rtg(rtg)
        state_emb = self.embed_state(states) + self.pos_emb(timesteps)
        action_emb = self.embed_action(actions)

        # 把三者堆叠成 (B, 3T, H)，顺序为 RTG, state, action
        stacked = torch.stack([rtg_emb, state_emb, action_emb], dim=1)
        stacked = stacked.permute(0, 2, 1, 3).reshape(B, 3 * T, -1)
        stacked = self.embed_ln(stacked)

        # causal attention：每个 token 只能看到过去
        h = self.transformer(stacked)
        # 取出 state 位置的输出预测对应 action
        h_states = h[:, 1::3, :]  # indices 1, 4, 7, ...
        return self.action_head(h_states)  # 回归到连续动作

    @torch.no_grad()
    def act(self, state, target_rtg, history, t):
        # 推理时把目标 RTG 作为"prompt"，自回归生成动作
        rtg_seq = torch.cat([history.rtg, target_rtg[None]], dim=0)[-self.K:]
        s_seq   = torch.cat([history.states, state[None]], dim=0)[-self.K:]
        a_seq   = history.actions[-self.K - 1:-1]  # 错位
        t_seq   = torch.arange(len(s_seq))
        pred_a = self.forward(rtg_seq, s_seq, a_seq, t_seq)
        return pred_a[-1]  # 取最后一个 timestep 的预测
```

注意一个实现细节：每个时间步占 3 个 token（R, s, a），所以 state token 的位置索引是 1, 4, 7,... 也就是从 1 开始每隔 3 个取一个，`h[:, 1::3, :]` 这行代码就是做这件事。

### 1.2 用监督学习预测动作

DT 的训练损失简单到令人发指——就是连续动作回归的 MSE（或者离散动作的交叉熵），没有任何花里胡哨的东西：

$$\mathcal{L} = \mathbb{E}_{\tau \sim \mathcal{D}}\left[\sum_t \|\hat{a}_t - a_t\|^2\right]$$

这里：

- $\hat{a}_t$：模型预测的 t 时刻动作
- $a_t$：训练数据中 t 时刻实际采取的动作
- 对连续动作用均方误差，对离散动作就把 MSE 换成交叉熵损失

训练过程和你训练任何序列模型完全一样：截取一段长度为 K 的历史窗口（类似 GPT 的 context window），喂给 Transformer，让它预测每个位置的动作，然后和真实动作算 loss 反向传播就行。没有 target network，没有软更新，没有保守正则项——就是纯粹的监督学习，简单到不能再简单。

### 1.3 推理时怎样使用目标回报

训练这么简单，那推理部署的时候怎么用呢？你会发现更简单——DT 不需要在动作空间里 argmax Q，你只要**指定一个目标 RTG**（比如看看你数据集里 expert 轨迹大概是多少分，就设成那个数），然后 DT 就会像 GPT 生成文本一样自回归地生成动作，试图让整个轨迹的累计回报接近你设的目标。

来看推理的伪代码：

```python
target_return = 9000  # HalfCheetah 环境 expert 级别的分数大概是这个数
state = env.reset()
history = TrajectoryBuffer()  # 保存最近 K 步的历史
for t in range(max_steps):
    action = model.act(state, target_return, history, t)
    next_state, reward, done, _ = env.step(action)
    history.append(state, action, reward)
    state = next_state
    # 关键：RTG 每步扣掉实际拿到的 reward，作为"剩余要达成的目标"
    target_return -= reward
```

这里最关键的一行就是 `target_return -= reward`——每在环境里执行一步、实际拿到 reward 之后，就把这个 reward 从剩余目标里减去。这样下一步输入给模型的 RTG，就准确地表示"到回合结束还差多少回报需要拿"。

举个具体例子：你设了目标总回报 6000 分。

- 第一步走得很好，拿到 reward 100，那 target_return 变成 5900——相当于告诉模型"不错，还剩 5900 分要拿，继续加油"
- 第二步走砸了，拿到 reward -50（撞墙了），target_return 变成 5950——相当于告诉模型"刚才搞砸了，剩下的路你得想办法多拿 50 分补回来"

就是这么朴素的机制！不过你可能会问："如果我把目标 RTG 设得特别高，比数据集中任何轨迹都高，会怎么样？"答案是模型可能会生成训练数据里没见过的、不连贯的奇怪动作——因为它在训练时从来没见过"要拿这么高的分"的条件，自然泛化不出来。所以目标 RTG 不要乱设，要参考你数据集中的回报范围来定。

### 1.4 Decision Transformer 为什么能够工作

看到这里你肯定会有疑问：这么简单的方法，就是个序列模型做动作预测，既不估计 Q 也不做 Bellman backup，它凭什么能 work？而且论文里说它在很多 D4RL benchmark 上性能还能和 CQL 打平甚至更好？

Decision Transformer 能不能工作，核心取决于你的数据集中是否同时出现了状态、动作和最终回报之间稳定的对应关系。如果你的数据集里包含了**不同质量的轨迹**——既有 expert 轨迹（高 RTG）、medium 轨迹（中等 RTG）、也有 random 轨迹（低 RTG）——那 RTG 条件就能帮模型区分开：在相似状态下，要想拿到高回报，过去那些好轨迹里都是怎么做的。

具体来说：

- 给定一个很高的目标 RTG，Transformer 学到的条件分布 $p(a \mid \hat{R}_{\text{high}}, s)$ 自然就会偏向高回报轨迹里经常出现的动作
- 给定一个很低的目标 RTG，它就会生成像 random 策略那样的动作
- 你可以把它直观地理解为：**模仿"在相似状态下曾经达到过相似 RTG 的那些轨迹"**

形式化地写出来，DT 学到的策略可以表示为：

$$\pi_\theta(a \mid s, \hat{R}) \propto \exp\left(-\frac{1}{2\sigma^2}\|a - f_\theta(s, \hat{R})\|^2\right)$$

这是一个高斯分布，均值是 Transformer 的回归输出 $f_\theta(s, \hat{R})$，方差是 $\sigma^2$。当 $\sigma \to 0$ 时，这就退化成一个确定性策略 $a = f_\theta(s, \hat{R})$。

这个分布和数据产生的行为策略 $\pi_\beta$ 之间有什么关系呢？很简单：

$$\pi_\theta(a \mid s, \hat{R}) \approx \pi_\beta(a \mid s, \text{return} \approx \hat{R})$$

换句话说，DT 学到的不是别的，正是行为策略在**指定回报条件下的条件分布**。你给它定一个高 RTG 目标，它就复现数据里那些拿到过高回报的轨迹的动作模式；你给它定低 RTG，它就复现差轨迹的动作。

当然，这也意味着它有局限性：数据里没有覆盖到的状态-动作组合，它还是没法可靠预测；而且它能不能把不同轨迹里的好片段"拼接"起来，也要看数据集和 Transformer 的能力。

但即便如此，DT 的贡献是开创性的——它催生了后续一整条"RL via supervised learning"的研究路线：online RL 里的 RL via supervised learning (RvS)、in-context RL（Algorithm Distillation）、Star-Vector、甚至 "language modeling is all you need for RL" 这类工作，源头都可以追溯到 DT。

### 1.5 Decision Transformer 的局限

DT 虽然简单优雅，但也不是银弹，它有几个很明确的局限你需要知道：

1. **只能学到数据中存在的最优策略**——这是最根本的局限。如果你的数据集里根本没有 expert 轨迹（比如只有 random 数据），那不管你把目标 RTG 设得多高，它也没法凭空生成 expert 级别的行为——它没见过啊！
2. **Stitching（轨迹拼接）能力差**——传统离线 RL（比如 CQL/IQL）理论上可以做 subtrajectory stitching：把两条次优轨迹里各自好的部分"缝合"起来，得到比数据中任何一条轨迹都好的策略。但 DT 是纯监督学习，做不到这种强组合泛化——它最多就是复现数据里接近目标 RTG 的轨迹的平均水平。
3. **RTG 选择敏感**——目标 RTG 设太高会生成不连贯的胡来动作，设太低又会太保守拿不到高分，需要针对环境仔细调这个超参数。

## 2. 用 Trajectory Transformer 搜索轨迹

DT 证明了"RL as sequence modeling"这条路是走得通的，之后这条路线迅速发展衍生出了很多工作。其中两个最有代表性的后续：一个是 **Trajectory Transformer（TT）**——它不只是预测动作，而是把整个轨迹建模成 token 序列，然后用 beam search 做规划；另一个是 **Diffuser**——它更激进，直接用扩散模型一次性生成整条轨迹。

### 2.1 离散化轨迹并使用 Beam Search

Janner et al. 2021 提出的 Trajectory Transformer 思路比 DT 更彻底：DT 还是按时间步自回归预测动作，TT 干脆把 RTG、state、action、reward 全部都离散化成一个个 token，然后训练一个标准的 Transformer 来预测下一个 token，建模整条轨迹的联合分布：

$$p_\theta(\tau) = \prod_{t=1}^{T} p_\theta(s_t, a_t, r_t \mid s_{<t}, a_{<t}, r_{<t})$$

这是什么意思？就是说 TT 不区分什么是状态什么是动作，它把整条轨迹（所有时间步的 s, a, r）都看成一个长序列的 token，像语言模型一样一个一个预测下一个 token。

推理的时候，它也不是像 DT 那样只简单采样一个动作，而是用大家在机器翻译里很熟悉的 **beam search（柱搜索）** 来最大化整条轨迹的概率——而且你还可以在搜索过程中显式加入奖励约束（比如"我要最终回报 ≥ X"），相当于做隐式的 model-based planning！

TT 有什么特点呢？我们总结一下：

- **优点**：Planning 能力强——因为你在 beam search 时可以往未来看多步，显式地考虑长期回报，这和 DT 那种单步预测很不一样
- **缺点一**：需要把连续量离散化，state 每一维都要单独离散化成 token，比如 state 是 20 维、每维离散成 100 个 bin，那一个 state 就占 20 个 token，序列长度爆炸，计算量很大
- **缺点二**：Beam search 推理比较慢，要同时展开多个候选轨迹，不像 DT 那样一步就能出动作

## 3. 用 Diffuser 生成完整轨迹

Janner et al. 2022 提出的 Diffuser 又换了个思路：既然我们最终要的是一条好轨迹，那我为什么要自回归一个 token 一个 token 生成呢？我能不能用扩散模型（diffusion model）一次性直接生成一整条完整的轨迹？

这思路非常像图像生成领域里的扩散模型：不是从左到右画像素，而是从纯噪声开始，一步步去噪，最后得到一张清晰的图像。Diffuser 把这个想法用到了轨迹生成上。

如果状态维度是 $d_s$、动作维度是 $d_a$、轨迹长度是 $T$，那一条轨迹就可以表示为一个形状为 $T \times (d_s + d_a)$ 的矩阵——你可以把它想象成一张"轨迹图像"，横轴是时间，纵轴是状态/动作的每一位。

训练过程和图像扩散模型一模一样：先给真实轨迹逐步加高斯噪声，直到变成纯噪声，然后让一个网络预测每一步加入的噪声是什么：

$$\min_\theta \; \mathbb{E}_{\tau, t, \epsilon}\left[\|\epsilon - \epsilon_\theta(\tau_t, t)\|^2\right]$$

我们拆解一下这个训练目标：

- $\tau$：数据集中采样的一条真实轨迹
- $t$：扩散步（从 0 到 T，t=0 是干净轨迹，t=T 是纯噪声）
- $\epsilon \sim \mathcal{N}(0, I)$：我们实际加入的高斯噪声
- $\tau_t$：加了 t 步噪声之后的带噪轨迹
- $\epsilon_\theta(\tau_t, t)$：网络预测的"这一步加了什么噪声"

训练目标就是让预测的噪声和实际加的噪声越接近越好（MSE loss）——这就是标准的扩散模型训练，没有任何特殊改动。

推理的时候反过来：从纯随机噪声开始，迭代 T 步，每一步用网络预测出噪声，然后从当前带噪轨迹里减去这个噪声，最后就能得到一条干净的、看起来像训练数据里的轨迹。

但是等等！我们要的不只是"看起来像真实数据"的轨迹，而是**高回报的轨迹**。怎么控制生成的轨迹好还是坏？Diffuser 用了图像扩散里很成熟的 **classifier-free guidance（无分类器引导）** 技术：训练时随机地把条件信息（比如奖励、起始状态）丢弃，让模型同时学会有条件生成和无条件生成；推理时把两者的预测做个加权外推：

$$\tilde{\epsilon}_\theta = (1 + w) \cdot \epsilon_\theta(\tau_t, t, c) - w \cdot \epsilon_\theta(\tau_t, t)$$

这里：

- $c$：条件信息（比如"从起始状态 s0 出发，最终回报要 ≥ 高分"）
- $\epsilon_\theta(\tau_t, t, c)$：有条件时预测的噪声（朝着满足条件 c 的轨迹去噪）
- $\epsilon_\theta(\tau_t, t)$：无条件时预测的噪声（只要生成像真实数据的轨迹就行）
- $w$：引导强度（guidance scale）——w 越大，越强调满足条件 c，生成的轨迹回报越高，但可能多样性下降、也可能不自然

通过这个引导机制，奖励条件就会改变去噪的方向，让高回报的轨迹获得更高的生成概率。于是优化就从传统 RL 里"显式选择让 Q 最大的动作"，变成了"从受奖励引导的轨迹分布中采样"——又是一种完全不同的思路。

## 4. 比较三种序列建模方法

我们把 Decision Transformer、Trajectory Transformer、Diffuser 这三种序列建模路线放在一起对比，方便你根据场景选择：

| 维度                | Decision Transformer | Trajectory Transformer | Diffuser             |
| ------------------- | -------------------- | ---------------------- | -------------------- |
| 建模对象            | 给定 RTG 的条件策略  | 整条轨迹的联合分布     | 整条轨迹的扩散模型   |
| 离散化              | 否                   | 是（state 每维都离散） | 否                   |
| 推理方式            | 自回归单步采样       | Beam search 多步搜索   | 迭代去噪（几十步）   |
| Planning 能力       | 弱（隐式单步）       | 强（显式看多步）       | 强（条件生成）       |
| Stitching 能力      | 弱                   | 中                     | 强                   |
| 推理速度            | 快                   | 慢                     | 中（需要几十步去噪） |
| 与 LLM 训练栈兼容性 | 强（架构最像 GPT）   | 强                     | 弱（扩散模型架构）   |

怎么选呢？给你一个简单的实践指南：

- 如果你只是想快速搭一个强基线，要快、实现简单、和现有 Transformer 训练栈兼容，选 DT
- 如果你需要显式做长 horizon planning，愿意花更多计算换更好的规划效果，可以试 TT
- 如果你需要强的 stitching 能力，要把不同轨迹的好部分组合起来，或者要做复杂的条件生成（比如同时约束起始状态、中间里程碑、最终回报），Diffuser 是更好的选择

## 本节总结

这一节我们看了一条完全跳出 Bellman 框架的离线 RL 路线：把强化学习重新定义为序列建模问题。

Decision Transformer 是这条路线的开创者：它引入 return-to-go 作为条件，把轨迹排成 (R, s, a) 的三元组序列，用 GPT 式的因果 Transformer 做纯监督学习——给定目标 RTG 和当前状态，直接预测动作。它的实现和训练都极其简单，彻底复用了大语言模型的成熟工具栈。

之后 Trajectory Transformer 走得更远：把整条轨迹全部离散化成 token，用标准语言模型建模联合分布，推理时用 beam search 显式做规划。Diffuser 则引入扩散模型，从噪声中迭代去噪生成完整轨迹，通过 classifier-free guidance 控制生成高回报的轨迹，stitching 和 planning 能力都更强。

但是你发现没有——所有这些"用固定数据集学习"的思想，其实不只在机器人控制这类传统 RL 场景有用。它和今天大语言模型的偏好对齐是不是看起来有点像？DPO 训练时不也只能用已经收集好的偏好数据集，不能随便和环境/标注者交互吗？这中间会不会有什么深层联系？

下一节 [10.3 离线强化学习与偏好数据](./experiments)，我们就来回答这个问题——把离线 RL 的视角带到 LLM 偏好优化中，看看 DPO 为什么其实就是一种隐式的离线 Q-Learning，以及离线 RL 的思路能帮我们理解哪些后训练现象。
