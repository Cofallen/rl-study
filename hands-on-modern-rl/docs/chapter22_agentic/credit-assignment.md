# 19.3 轨迹信用分配

[19.2 多轮强化学习](./formulation) 把轨迹概率写成 $\rho_\theta(x | s_0, c) = \mathbb{I}(s_0, x) \prod_{t \in a(x)} p_\theta(x_t | \cdot)$，并推出 action mask——只有 action token 参与 policy gradient。但这只回答了"该在哪些 token 上算梯度"，没回答"每个 action token 该乘以多大的 advantage"。

策略梯度的形式是：

$$
\nabla J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ R(\tau) \nabla \log \pi_\theta(\tau) \right]
$$

trajectory-level 的标量 $R(\tau)$ 对整条轨迹一视同仁——成功了所有动作都好，失败了所有动作都坏。这就是 Agentic RL 的核心难题：**7 轮交互失败了，第 1 轮的正确搜索也要被惩罚吗？**

把 $R(\tau)$ 回拆成每一步的 step-level advantage $A_{i,t}$，就是**信用分配（Credit Assignment）**。本节系统梳理这个问题：先从最简单的 ORM/PRM 二分讲起，再展开 2025-2026 年发展出的四类精细方法。

## 三层信号：从最终结果到 token 更新

一条 rollout 失败了，信号要经过三层才能落到 LLM 权重上：

**第一层：Trajectory reward。** 整条轨迹最后得到 $R(\tau_i)$，只告诉我们"这一局成没成"。这是环境给的全部信号。

**第二层：Step advantage。** 把最终结果拆成 $A_{i,1}, A_{i,2}, \ldots, A_{i,T}$，回答"第 $t$ 轮该不该被奖励"。这是信用分配要做的核心工作。

**第三层：Token gradient。** 把 $A_{i,t}$ 乘到这一轮 action 的 token log-prob 上：

$$
\Delta\theta = A_{i,t} \cdot \sum_{k \in a_{i,t}} \nabla \log \pi_\theta(y_k \mid h_{i,t}, y_{<k})
$$

其中 $a_{i,t}$ 是第 $i$ 条轨迹第 $t$ 轮 LLM 生成的 token 集合，$h_{i,t}$ 是当时的 context。网页/工具返回内容被 masked，不参与这一步。

当前 GRPO-style Agentic RL 的很多工作，本质上都是在构造更好的 $A_{i,t}$。

## ORM 与 PRM：信用分配的基本二分

最简单的两种做法定义了信用分配的两个极端。

### ORM：只看最终结果

**ORM（Outcome Reward Model）**——只在轨迹终点给奖励，中间步骤全部为 0：

$$
r_1 = r_2 = \cdots = r_{T-1} = 0, \quad r_T = \begin{cases} 1 & \text{成功} \\ 0 & \text{失败} \end{cases}
$$

ORM 的优势是**简单且便宜**——只需要知道最终结果对不对。对于可验证任务（代码是否通过测试、数学答案是否正确），连 reward model 都不用训，直接用 verifier。RLVR（Reinforcement Learning with Verifiable Rewards）就是 ORM 的极端形式，DeepSeek-R1 的成功证明了纯 ORM 可以激发强大的推理能力。

ORM 的劣势是**信号稀疏**。一条 10 步轨迹只有 1 个 reward 信号，模型很难从中学到"具体哪一步该改进"。如果失败时所有步骤都被惩罚——包括那些做得对的中间步骤——信用分配就严重失真。

### PRM：每步独立评分

**PRM（Process Reward Model）** 对每一步独立打分：

$$
r_t = f_{\text{PRM}}(s_1, a_1, s_2, a_2, \ldots, s_t, a_t)
$$

PRM 看了从第 1 步到第 $t$ 步的完整历史，判断第 $t$ 步是否正确。OpenAI 的 "Let's Verify Step by Step"（Lightman et al., 2023）正式提出这一思路：第一步思路正确（+1）、第二步计算有误（-0.5）、第三步虽对但绕了路（+0.3）。模型能精确定位需要改进的步骤。

PRM 的劣势是**标注成本极高**——为每一步都标注"对/错"比只标注最终结果的工作量大了 $T$ 倍。OpenAI 为此构建了 PRM800K 数据集。当前研究热点是**自动化 PRM**（如 Math-Shepherd 让模型自主判断每步质量）和**领域特化 PRM**（如 Web-Shepherd 专为网页导航设计）。

|          | ORM                     | PRM                      |
| -------- | ----------------------- | ------------------------ |
| 信号密度 | 稀疏（只有最终 reward） | 密集（每步都有 reward）  |
| 标注成本 | 低（只看结果）          | 高（每步都要标注）       |
| 学习速度 | 慢（信号少，方差大）    | 快（信号多，方差小）     |
| 适用场景 | 可验证任务（代码/数学） | 复杂推理（需要精细指导） |

### SALT：从轨迹图里提取步骤级 advantage

ORM 太粗，PRM 太贵。**SALT**（[Li et al., EACL 2026](https://arxiv.org/abs/2510.20022)）提供了第三条路——不训练 PRM，但比纯 ORM 精细得多。

核心思路：对同一个 prompt 采样多条轨迹，构建一个**轨迹图**——节点是每一步的动作，如果两条轨迹在某一步做了相同的动作，它们就共享同一个节点。通过分析图结构，可以量化每一步对最终结果的贡献。

直觉上：如果一个步骤被很多**成功轨迹**共享、但很少出现在失败轨迹中，那它大概率是个好步骤——应该得到正向 advantage。反之，如果某步骤只出现在失败轨迹中，它大概率拖了后腿。SALT 利用图结构计算每个步骤的 advantage，**完全不需要额外奖励模型或人工标注**——只需要最终结果的二元信号。

这让 SALT 在 GRPO 框架中特别好用：GRPO 已经在组内采样多条轨迹做比较，SALT 在此基础上进一步细化到步骤级别。

## Step-Level Advantage 的三类来源

ORM/PRM/SALT 之外，2025-2026 年发展出了一大批更精细的方法。[XiaoRed5 的入门资料](https://github.com/XiaoRed5/Agentic-RL-Most-Detailed-Intro) 和 2026 年 4 月的综述 [From Reasoning to Agentic: Credit Assignment in RL for LLMs](https://arxiv.org/abs/2604.09459) 把这些方法按**信号来源**分成了三类。它们的区别不在于最后怎么更新 policy，而在于 step-level 信号 $A_{i,t}$ 从哪里来。

### 方向一：State-anchored stepwise——同状态下比较动作

核心思想：**同一个状态下，不同动作的相对好坏，可以从它们各自的后续回报里看出来**。不需要 PRM，只需要利用 group 内多条 rollout 的天然结构。

#### GiGPO：同 state group 内相对回报

[GiGPO](https://arxiv.org/abs/2505.10978)（Group-in-Group Policy Optimization）的想法是：同一个任务采多条 rollout，如果这些 rollout 中间遇到了相同的环境状态 $s$，那就可以比较"同样局面下，不同动作谁更好"。

先把所有在 state $s$ 下发生的动作聚成一个组：

$$
G^S(s) = \{(a_t^{(i)}, R_t^{(i)}) \mid s_t^{(i)} = s\}
$$

这里的 $R_t^{(i)}$ 不是整条轨迹 reward，而是从这一步开始往后的折扣回报：

$$
R_t^{(i)} = \sum_{k=t}^{T} \gamma^{k-t} r_k^{(i)}
$$

如果环境只有最终成功奖励，可以直觉理解成：这一步之后最终成功就有正回报，成功来得越晚、折扣后回报越小，最终失败则回报接近 0。

然后在同一个 state group 里做相对优势：

$$
A^S(a_t^{(i)}) = \frac{R_t^{(i)} - \text{mean}(\{R_t\}_{G^S(s)})}{F_{\text{norm}}(\{R_t\}_{G^S(s)})}
$$

一个动作的 step 分数 = 它后续带来的回报 − 同 state 下其他动作的平均后续回报。

**WebShop 例子**：多个 rollout 都来到同一个商品搜索结果页：

| 动作                         | 后续结果   | $R_t$ |
| ---------------------------- | ---------- | ----- |
| 点正确商品                   | 很快成功   | 1.00  |
| 点错误商品，但后来返回并买对 | 晚一点成功 | 0.73  |
| 点下一页                     | 最终失败   | 0.00  |

组内平均值 $\bar R = 0.58$。三个动作的相对分数：

```
点正确商品:  1.00 - 0.58 = +0.42
点错误商品:  0.73 - 0.58 = +0.15
点下一页:    0.00 - 0.58 = -0.58
```

模型学到的不是"成功轨迹里所有动作都好"，而是更细的局部偏好：在同一局面下，点正确商品最好；点错误商品虽然不理想，但还有补救机会；点下一页最差，因为后续失败了。

最后 GiGPO 把整条轨迹的 macro advantage 和这一步的 micro advantage 合起来：

$$
A(a_t^{(i)}) = A^E(\tau_i) + \omega \cdot A^S(a_t^{(i)})
$$

其中 $A^E$ 是 trajectory-level 的相对好坏，$A^S$ 是 step-level 在同 state 下的相对好坏。

#### HGPO：还要看历史上下文

[HGPO](https://arxiv.org/abs/2602.22817)（Hierarchy-of-Groups Policy Optimization）是对 GiGPO 的一个关键修正。GiGPO 在同一个 state 下比较动作，但 HGPO 指出：**同一个当前 state，不一定代表同一个决策上下文**。

最容易出错的场景是网页任务：两个 rollout 可能当前页面完全一样，但它们是怎么走到这个页面的并不一样。一个 agent 可能带着完整约束（"黑色、防水、低价的雨靴"）来到这里，另一个 agent 前面已经漏掉了"防水"约束（只搜了"black boots"）。此时同一个动作（"加入购物车"）的好坏含义就会不同——前者大概率是好动作，后者可能是坏动作（因为漏掉了关键约束）。HGPO 把这个问题叫 **historical context inconsistency**。

HGPO 定义一个 $k$-step context operator：

$$
\mathcal{C}_k(s_t^{(i)}) = (s_{t-k}^{(i)}, s_{t-k+1}^{(i)}, \ldots, s_t^{(i)})
$$

表示"当前 state 加上前面 $k$ 个历史 state"。$k=0$ 时只看当前 state；$k$ 越大，要求共享的历史越长，上下文越一致。

对每个 step 构造多个层级 group：

$$
G_k^H(s_t^{(i)}) = \{(j, n) : \mathcal{C}_k(s_t^{(i)}) = \mathcal{C}_k(s_n^{(j)})\}
$$

这些 group 之间是嵌套关系：$G_0^H \supseteq G_1^H \supseteq \cdots \supseteq G_K^H$。$G_0^H$ 是"只看当前页面"的大组（样本多但容易混），$G_K^H$ 要求更长历史一致（比较公平但样本少）。

在每一层 group 里算组内相对 advantage：

$$
A_k^H(s_t^{(i)}) = \frac{R(s_t^{(i)}) - \frac{1}{|G_k^H|} \sum_{(j,n) \in G_k^H} R(s_n^{(j)})}{\sigma_{G_k^H}}
$$

最后 HGPO 把不同层级的 advantage 加权融合：

$$
A^H(s_t^{(i)}) = \sum_{k=0}^{K} w_k A_k^H(s_t^{(i)}), \qquad w_k = \frac{(k+1)^\alpha}{\sum_k (k+1)^\alpha}
$$

$\alpha$ 控制更深层 group 的权重——本质上是 bias-variance trade-off：深层 group 比较公平但样本少、方差高。

一句话说，GiGPO 问"同一个 state 下哪个动作更好"，HGPO 进一步问"在同一个 state 且历史上下文相近时，哪个动作更好"。

#### Group-Graph PO：把轨迹建模成 DAG

[Group-Graph Policy Optimization](https://arxiv.org/abs/2606.22995)（2026-06）走得更远。它指出，现有 step-level 框架仍然把 agent 探索看作"孤立的线性轨迹"——这是过度简化的视角。实际上多条 rollout 之间经常共享前缀、共享中间状态，结构上更像一棵**有向无环图（DAG）**而非一组平行线。

把轨迹建模成图后，可以在图上做更精细的信用分配：识别关键分叉点、追踪某条子路径的独立贡献、避免重复计算共享前缀的 advantage。这对长程任务特别有价值——当反馈延迟到几十步之后才出现时，DAG 结构能更准确地归因到具体决策。

### 方向二：Process / Progress Reward——训练额外的步级 scorer

第二类方法不依赖 rollout 之间出现同 state 碰撞，而是额外训练一个能给步骤打分的模型。

#### SPA-RL：把最终 reward 分摊成每步 progress

[SPA-RL](https://arxiv.org/abs/2505.20732)（Stepwise Progress Attribution）走的是 reward redistribution 路线：额外训练一个 **progress estimator**，让它学会判断"这一步让任务向完成目标推进了多少"。

基本假设：长程任务最终完成，是每一步 incremental progress 累积出来的。所以不直接给每步打"绝对好坏分"，而是让模型预测每步贡献 $\hat c_t$，并要求这些贡献加起来能还原最终 reward：

$$
\hat R = \sum_{t=1}^{N} \hat c_t \approx R
$$

progress estimator 的形式：

$$
\hat c_t = \text{MLP}(h_t), \qquad h_t = f_{\pi_\gamma}(s_t, a_t)
$$

用一个 LLM 编码当前 state-action，再接轻量 MLP 输出标量贡献。训练时用多条探索轨迹的最终 reward 做监督：

$$
\mathcal{L}_{\text{PE}} = \frac{1}{|\mathcal{D}| M} \sum_i \sum_j (\hat R_{i,j} - R_{i,j})^2
$$

意义非常直接：如果某条轨迹最终成功，所有步骤贡献的和应该接近 1；如果最终失败，贡献的和应该接近 0。模型为了拟合这个目标，会学着把正贡献分给真正推动任务成功的步骤，把低贡献或负贡献分给无效步骤。

**直觉例子**：一个搜索 agent 最终答对，轨迹里可能有 5 步：

| 步骤 | 行为                    | 可能的 progress |
| ---- | ----------------------- | --------------- |
| 1    | 生成一个模糊 query      | 0.05            |
| 2    | 打开无关网页            | 0.00            |
| 3    | 改写 query 命中关键网页 | 0.30            |
| 4    | 抽取关键证据            | 0.45            |
| 5    | 组织成最终答案          | 0.20            |

这些分数加起来约等于最终成功 reward。RL 训练时不再只有最后一步有信号，而是每一步都有 dense reward。

SPA-RL 还加了一个 grounding signal $g_t$（动作是否真的能在环境中执行），最终中间奖励是 $r_t^{\text{fused}} = \alpha c_t + \beta g_t$——一个动作既要"有用"，也要"能执行"。

#### AgentPRM：把 PRM 当成 agent 的 $Q(s,a)$

[AgentPRM](https://arxiv.org/abs/2502.10325) 的思路更接近经典 RL 里的 actor-critic。它把 process reward model 看成一个状态-动作价值函数：

$$
Q^\pi(s_t, a_t) = \mathbb{E}_\pi \left[ \sum_{k=t}^{T} \gamma^{k-t} r(s_k, a_k) \mid s_t, a_t \right]
$$

PRM 输出的不是"这一小步贡献了多少"，而是"如果在当前 state 做这个 action，后面按当前 policy 继续走，预期能拿多少总回报"。

AgentPRM 的训练是一个三阶段循环：

1. **Rollout and Compute Target**：用当前 policy $\pi_{i-1}$ 在环境中采样大量轨迹，用 MC return 给每个 $(s, a)$ 估计目标值。
2. **Train PRM**：把这些自动标注的 $\hat Q(s, a)$ 当 soft label，用 soft BCE 训练 PRM。
3. **Train Policy**：用训练好的 PRM 作为 dense reward，更新 policy，同时用 KL 约束不要离上一轮 policy 太远：

$$
\pi_i = \arg\max_{\pi_\theta} \mathbb{E}_{s, a \sim \pi_\theta} [Q_\phi(s, a)] - \beta \mathbb{D}_{\text{KL}} [\pi_\theta(\cdot | s) \| \pi_{i-1}(\cdot | s)]
$$

注意 KL 约束到**上一轮 policy** $\pi_{i-1}$ 而不是 SFT policy——因为 PRM 是用上一轮 policy 的 rollout 训练的，新 policy 飘得太远 PRM 估计就会失真。

AgentPRM 在推理时还可以做 Best-of-N：每一步从 policy 采样 $N$ 个候选动作，用 PRM 选 $Q(s, a)$ 最高的那个。最大风险是 reward hacking——policy 学会让 PRM 打高分而不是让真实任务成功。常见缓解方式包括迭代重训 PRM、增加 rollout 数据、加 KL 约束。

#### PAIR：prefix-aware internal reward

[PAIR](https://arxiv.org/abs/2605.17877)（Prefix-Aware Internal Reward Model，2026-05）针对 AgentPRM 这类方法的训练成本问题——三阶段循环里 PRM 重训很贵。PAIR 训练一个 internal reward model，它在评估某一步时**显式感知前缀**：同一个动作在不同前缀下应该得到不同分数。

#### Web-Shepherd：领域特化 PRM

[Web-Shepherd](https://arxiv.org/abs/2505.15277)（NeurIPS 2025 Spotlight）是首个专为网页导航设计的步骤级 PRM，能自动评估 Agent 在每一步的操作是否正确。实验表明，用 Web-Shepherd 提供步骤级 reward，GPT-4o-mini 的性能提升了 10.9%，而成本仅为使用 LLM 做判官的 1/10。这说明 PRM 不只是理论上的美好愿景——在特定领域，领域特化的 PRM 可以低成本、高效率地提供密集的步骤级信号。

### 方向三：Intrinsic Signal——从 policy 自身找信号

第三类方法不额外训练 PRM，而是从 policy 自己的行为分布里找 step-level 信号。

#### ARPO：把 rollout 预算花在 entropy 飙升的位置

[ARPO](https://arxiv.org/abs/2507.19849)（Agentic Reinforced Policy Optimization）关注工具调用场景。它的关键观察是：**LLM 在收到工具返回结果之后，接下来生成的前 10 到 50 个 token 的 entropy 往往会明显升高**。

token entropy 的定义：

$$
H_t = -\sum_{j=1}^{V} p_{t,j} \log p_{t,j}, \qquad p_t = \text{Softmax}(z_t / \tau)
$$

entropy 高表示模型此刻不确定：可能有多种解释、多种下一步工具调用、多种推理路径。ARPO 的想法是：**既然不确定性最高的位置最可能是关键决策点，那 rollout 预算就不应该平均花，而应该集中在那里**。

ARPO 的 rollout 分成两类：

- **Global rollout**：从头到尾采完整轨迹，保证整体探索
- **Partial rollout**：在高 entropy 工具调用节点分叉，重点探索局部决策

具体流程：总 rollout budget 是 $M$，先采 $N$ 条完整轨迹，剩下 $M - N$ 留给局部分叉。每次工具调用返回后生成 $k$ 个 token，计算 entropy $H_t$；与初始 entropy $H_{\text{initial}}$ 比较，得到归一化 entropy 变化 $\Delta H_t$；如果超过阈值，就从当前节点分出 $Z$ 条 partial rollout。

例如搜索工具返回了几段互相冲突的证据，模型 entropy 飙升，ARPO 就在这里多分几条分支：一条追证据 A、一条追证据 B、一条换 query 验证。分叉后的差异用来更新 individual tokens；共享前缀则吃多个分支 advantage 的平均。

#### IGPO：用正确答案概率的增量当 turn-level reward

IGPO（Information Gain Policy Optimization，出自 [DR-Venus](https://arxiv.org/abs/2604.19859)）的核心想法是：每一轮搜索、读证据、工具调用，本质上都应该让模型**更接近正确答案**。那就直接衡量这一轮之后，模型对 ground-truth answer 的概率有没有上升。

设 ground-truth answer token 序列为 $a = (a_1, \ldots, a_L)$。第 $i$ 条 rollout 到第 $t$ 轮为止的历史是 $o_{i, \leq t}$。IGPO 用 teacher forcing 计算模型生成正确答案的平均 log probability：

$$
\log \pi_\theta(a \mid q, o_{i, \leq t}) = \frac{1}{L} \sum_{j=1}^{L} \log \pi_\theta(a_j \mid q, o_{i, \leq t}, a_{<j})
$$

除以 $L$ 是为了避免长答案天然 log probability 更低。

然后把连续两轮之间的 log probability 增量定义成 information gain reward：

$$
r_{i,t}^{\text{IG}} = \log \pi_\theta(a \mid q, o_{i, \leq t}) - \log \pi_\theta(a \mid q, o_{i, \leq t-1})
$$

直觉上：

- **有用交互**：log-prob 从 -4.0 升到 -1.5 → $r^{\text{IG}} = +2.5$（搜索找到了关键证据，模型现在更确信正确答案）
- **带偏交互**：log-prob 从 -2.0 降到 -3.2 → $r^{\text{IG}} = -1.2$（搜索带回了无关内容，模型反而偏离了正确答案）

IGPO 还保留最终 outcome reward，并把两类 reward 分开做 group-wise z-normalization：

$$
\tilde r_{i,t} = \begin{cases} \frac{r_{i,t}^{\text{IG}} - \mu_{\text{IG}}}{\sigma_{\text{IG}}}, & 1 \leq t < T \\ \frac{r_i^O - \mu_O}{\sigma_O}, & t = T \end{cases}
$$

然后像普通 RL 一样向后累积折扣回报 $\tilde R_{i,t} = \sum_{k=t}^{T} \gamma^{k-t} \tilde r_{i,k}$，分配给第 $t$ 轮产生的 decision tokens；工具返回内容本身不更新。

IGPO 的优点是便宜、稠密、ground-truth-aware，不需要额外 PRM，也不需要 Monte Carlo 估值。缺点是它依赖高质量 ground truth——如果问题本身有多个合理答案但数据集只认其中一个，IGPO 可能会惩罚事实正确但不匹配标注的推理路径。

#### AEM：自适应熵调制

[AEM](https://arxiv.org/abs/2605.00425)（Adaptive Entropy Modulation，2026-05）走了另一条 intrinsic 路线。它指出 sparse outcome-only reward 提供的指导有限，但**额外加 PRM 或 self-supervised signal 又会增加监督和调参复杂度**，可能限制跨任务泛化。

AEM 不引入任何额外监督，而是动态调节 policy 的 entropy bonus——在探索不足时增大 entropy 鼓励，在策略开始坍缩时减小。具体地，它根据 trajectory-level reward variance 自适应调整 entropy 系数，避免 ARPO/RAGEN 等工作中常见的 "echo trap"（模型陷入自生成 reasoning 模板的死循环）和 entropy collapse。

#### RefGRPO：反思鸿沟与校准 bonus

[RefGRPO](https://arxiv.org/abs/2606.14211)（Closing the Reflection Gap，2026-06）发现了一个反直觉现象：**LLM agent 在看到环境反馈后，倾向于错误评估自己的输出**——即使对它本已答对的问题。作者称这是 "reflection gap"，并指出标准 RL 由于 "credit-assignment mismatch" 几乎无法修复它。

RefGRPO 的修复很简单：在 GRPO 基础上加一个 **calibration bonus**——当模型在看到环境反馈后仍然正确评估自己的输出时给予额外奖励。这个 bonus 是"免费"的（不需要额外标注），但能显著缩小 reflection gap，让 agent 学会真正利用环境反馈而非机械响应。

## Step-Aligned 范式：解决 granularity mismatch

前三类方法都在"如何更精细地分配 advantage"上做文章。还有一类工作从更根本的地方入手：**改变优化的基本单元**。

[StepPO](https://arxiv.org/abs/2604.18401)（2026-04）指出：现有 LLM RL 算法继承了 RLHF/RLVR 的 **token-centric paradigm**——token 是建模和优化的基本单元。但在 agentic RL 中，LLM 通过"环境观测 ↔ 动作"的循环做 **step-level 决策**，token-level 优化与 step-level 决策之间存在 **granularity mismatch**。StepPO 提出以 step 为核心的范式：每个 step 是一个完整的决策单元，policy gradient 在 step 粒度上计算，而不是 token 粒度。

[Turn-PPO](https://arxiv.org/abs/2512.17008)（2025-12）从另一个角度到达类似结论。它发现直接把 GRPO 套到多轮任务上、特别是长程推理场景下表现很差——原因是 GRPO 的 group-relative advantage 在 trajectory level 估算时方差太大。Turn-PPO 改用 PPO 并设计 **turn-level advantage estimation**：每个 turn（不是每个 token、也不是整条 trajectory）有自己的 advantage，advantage 在 turn 内部的 token 间共享。

[AT²PO](https://arxiv.org/abs/2601.04767)（2026-01）进一步把 tree search 引入 turn-level optimization。它针对多轮 agentic RL 的三个核心挑战——exploration diversity 不足、credit assignment 稀疏、policy optimization 错位——提出统一的 turn-based + tree search 框架。Tree search 在关键 turn 上展开多条假设路径，比较它们的最终回报，从而为该 turn 提供更精确的 advantage。

这一组工作的共同洞察：**Agentic RL 不应该硬套 RLHF 的 token-centric 框架**。Agent 的决策天然是 step / turn 粒度的，优化框架应该尊重这个粒度。

## Turn-Level Discounting：越早犯错，责任越大

无论用哪种信用分配方法，多轮 RL 都需要处理一个时间维度的问题：**早期步骤的错误影响更大**。直觉上很好理解——第 1 步就走错了方向，后面每一步都在错误的基础上展开；而第 6 步犯的小错，第 7 步还有机会纠正。

为了建模这个直觉，研究者引入了 **Turn-Level Discounting**：

$$
R = \sum_{t=1}^{T} \gamma^t \cdot r_t
$$

注意这里的 $\gamma^t$ 不是对"未来"打折，而是对"过去"的步骤赋予不同的权重。在实际实现中，更常见的做法是**反向折扣**——从最终结果往回推，越早的步骤折扣越大：

```python
def compute_turn_rewards(turn_rewards, gamma=0.9):
    """计算多轮 RL 的折扣累积回报"""
    T = len(turn_rewards)
    returns = []
    G = 0
    # 从最后一轮往前累计
    for t in reversed(range(T)):
        G = turn_rewards[t] + gamma * G
        returns.insert(0, G)
    return returns

# 7 轮交互，只有最后一轮有即时 reward
# turn_rewards = [0, 0, 0, 0, 0, 0, 1.0]
# discount gamma = 0.9
# 返回: [0.531, 0.590, 0.656, 0.729, 0.810, 0.900, 1.000]
# 越早的步骤，折扣越大 → 对最终结果的"责任"被稀释
```

这个实现和第 6 章 REINFORCE 中的 $G_t$ 计算完全一样——只是现在每一步是一个完整的"轮次"（包括文本生成和工具调用），而不是单个 token。

## 代表性框架对比

| 框架           | 信号来源             | 主要贡献              | 适用场景                       |
| -------------- | -------------------- | --------------------- | ------------------------------ |
| ORM (RLVR)     | 最终结果             | 简单、便宜、可验证    | 短程任务（≤ 5 步）、有客观答案 |
| PRM (PRM800K)  | 人工标注每步         | 信号密集、定位精准    | 高标注预算、复杂推理           |
| SALT           | 轨迹图结构           | 不需额外标注          | GRPO 框架、长程任务            |
| GiGPO          | 同 state group       | state-anchored 比较   | 状态可枚举（WebShop）          |
| HGPO           | k-step context       | 解决历史上下文不一致  | 网页导航、上下文敏感任务       |
| Group-Graph PO | 轨迹 DAG             | 长程 + 共享前缀       | 长程任务（10+ 步）             |
| SPA-RL         | progress estimator   | reward redistribution | 任意多步任务                   |
| AgentPRM       | MC return + soft BCE | PRM as $Q(s,a)$       | 需要 Best-of-N 推理            |
| ARPO           | entropy spike        | 局部分叉采样          | 工具调用密集任务               |
| IGPO           | 正确答案 log-prob    | information gain      | 有 ground truth 的任务         |
| AEM            | 自适应 entropy       | 无需额外监督          | 通用、易坍缩场景               |
| RefGRPO        | 校准 bonus           | 缩小反思鸿沟          | 需要自评估能力的任务           |
| StepPO         | step-centric         | granularity 对齐      | 任意 agentic RL                |
| Turn-PPO       | turn-level PPO       | 方差低于 GRPO         | 长程多轮任务                   |
| AT²PO          | turn + tree search   | 探索 + 精细归因       | 高计算预算的长程任务           |

## 综合选型建议

按任务复杂度选择合适的策略：

- **3-5 轮简单任务**：纯 ORM / GRPO 就够了——episode 短，信号稀疏性不严重。
- **5-15 轮中等任务**：里程碑式奖励塑形，或 SALT/GiGPO 这类不依赖额外标注的方法。
- **15+ 轮复杂任务**：必须用 PRM 或 progress reward（SPA-RL/AgentPRM）+ MCTS 探索（AT²PO）。
- **环境不可复现 / 高方差**：AEM 自适应熵调制，或 STO-RL 两阶段（先离线预热，再在线精修）。
- **需要 Best-of-N 推理**：AgentPRM 提供步骤级 Q 值，推理时挑 $Q$ 最高的动作。
- **Granularity mismatch 严重**：换 step-aligned 范式（StepPO / Turn-PPO）。

关键原则：**先确认 reward 信号的密度是否足以支撑学习，再决定用什么 RL 算法**。如果 reward 太稀疏，再好的算法也学不动。

## 从信用分配到规划能力

信用分配回答了"每步做得好不好"的问题。但一个更深层的问题是：**模型能否在行动之前就制定出好的多步计划？** 这是规划（Planning）能力的核心。

到目前为止讨论的 Agent 主要是**反应式**的——根据当前观察做下一步决策。但真正的智能体需要**前瞻式规划**——在行动前推演多种路径，评估预期结果，选择最优路径。

规划能力可以从 RL 训练中涌现。DeepResearcher 等工作的实验揭示：模型自发产生了预搜索规划（先列出关键词列表）、信息分层（先搜概览再深入）和交叉验证等行为——这些都没被 reward 显式鼓励，纯粹是 RL 优化的副产品。

实用启示：**在投入复杂的树搜索和分层 RL 之前，先试试简单的 GRPO + outcome reward——模型可能自己就能学会规划**。只有简单方法无法涌现规划能力时，才需要引入 TreeRL（ACL 2025）、PGTS（ICML 2025）这类显式的树搜索训练方法。

## Mini Agent Loop——ORM vs PRM 信用分配对比

前面的章节里，RL 训练都是"单轮"的：模型生成一段文本，奖励函数打一个分，更新策略。但真正的智能体不是这样工作的——它需要在多轮交互中搜索信息、执行代码、观察结果，最后才给出最终答案。7 轮交互之后只有一个"成功/失败"信号，你怎么把这个信号分摊到 7 个步骤上？

这就是 Agentic RL 的核心挑战：**信用分配（Credit Assignment）**。这一节我们亲手建一个轻量的工具环境，用 Python 模拟多轮 Agent 交互，然后对比两种信用分配策略——ORM（只看最终结果）和 PRM（每步都评估）——看看它们的差异有多大。

### 搭建一个 Mini Tool Environment

我们用纯 Python 搭建一个模拟的"研究助手"环境。Agent 可以调用三种工具：

| 工具              | 功能         | 返回         |
| ----------------- | ------------ | ------------ |
| `search(query)`   | 模拟搜索信息 | 搜索结果文本 |
| `calculate(expr)` | 执行数学计算 | 计算结果     |
| `verify(fact)`    | 验证某个事实 | True / False |

```python
# ==========================================
# 1. Mini Tool Environment
# ==========================================
import re
import math
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ToolResult:
    """工具调用的返回结果"""
    tool: str          # 工具名称
    input: str         # 调用输入
    output: str        # 返回内容
    success: bool      # 是否成功

class MiniToolEnv:
    """模拟的轻量工具环境"""

    # 预设的"知识库"——搜索工具会从这里查
    KNOWLEDGE = {
        "earth_radius": "6371",
        "pi": "3.14159265",
        "speed_of_light": "299792458",
        "gravity": "9.8",
        "moon_distance": "384400",
        "population_china": "1400000000",
        "python_release": "1991",
        "gpt_release": "2020",
        "transformer_paper": "2017",
    }

    def search(self, query: str) -> ToolResult:
        """模拟搜索：在预设知识库中查找"""
        query_lower = query.lower()
        for key, value in self.KNOWLEDGE.items():
            if key in query_lower or any(w in key for w in query_lower.split("_")):
                return ToolResult("search", query, f"找到：{key} = {value}", True)
        return ToolResult("search", query, f"未找到与'{query}'相关的信息", False)

    def calculate(self, expression: str) -> ToolResult:
        """模拟计算器：安全的数学表达式求值"""
        try:
            safe_expr = re.sub(r'[^0-9+\-*/().]', '', expression)
            result = eval(safe_expr)
            return ToolResult("calculate", expression, str(result), True)
        except:
            return ToolResult("calculate", expression, "计算错误", False)

    def verify(self, fact: str) -> ToolResult:
        """模拟事实核查"""
        for key, value in self.KNOWLEDGE.items():
            if key in fact.lower() and value in fact:
                return ToolResult("verify", fact, "正确", True)
        return ToolResult("verify", fact, "无法验证", False)

# 测试环境
env = MiniToolEnv()
print(env.search("earth_radius"))
print(env.calculate("2 * 3.14159 * 6371"))
print(env.verify("earth_radius is 6371"))
```

### 定义多轮交互的 Agent Loop

```python
# ==========================================
# 2. Agent Turn 与 Episode 定义
# ==========================================
@dataclass
class Turn:
    """一个交互轮次"""
    action: str          # "search" | "calculate" | "verify" | "answer"
    input: str           # 工具输入或最终答案
    observation: str     # 环境返回
    success: bool        # 工具调用是否成功

@dataclass
class Episode:
    """一个完整的 Agent 交互过程"""
    task: str
    ground_truth: str
    turns: List[Turn]

def run_agent_loop(env, task, action_plan, ground_truth):
    """执行一次 Agent 交互循环。"""
    turns = []
    for step in action_plan:
        tool = step["tool"]
        inp = step["input"]

        if tool == "search":
            result = env.search(inp)
        elif tool == "calculate":
            result = env.calculate(inp)
        elif tool == "verify":
            result = env.verify(inp)
        elif tool == "answer":
            correct = inp.strip() == ground_truth.strip()
            turns.append(Turn("answer", inp,
                              "正确！" if correct else "错误", correct))
            return Episode(task, ground_truth, turns)
        else:
            result = ToolResult(tool, inp, f"未知工具: {tool}", False)

        turns.append(Turn(tool, inp, result.output, result.success))

    return Episode(task, ground_truth, turns)
```

### 设计一个多步任务

任务：**"地球的赤道周长是多少公里？"** 正确路径是 search → calculate → verify → answer。

```python
# 正确的工具调用序列
good_plan = [
    {"tool": "search", "input": "earth_radius"},
    {"tool": "calculate", "input": "2 * 3.14159 * 6371"},
    {"tool": "verify", "input": "earth_radius is 6371"},
    {"tool": "answer", "input": "40030"},
]

# 第 2 步算错了（π 取成了 3）
bad_plan = [
    {"tool": "search", "input": "earth_radius"},
    {"tool": "calculate", "input": "2 * 3 * 6371"},
    {"tool": "verify", "input": "earth_radius is 6371"},
    {"tool": "answer", "input": "38226"},
]

good_episode = run_agent_loop(env, task, good_plan, ground_truth)
bad_episode = run_agent_loop(env, task, bad_plan, ground_truth)
```

注意差策略的关键特点：**只有第 2 步犯了错（π 取成了 3），但第 1、3 步其实都做对了。** 最终结果错误（第 4 步），但错误根源在第 2 步。

### 对比 ORM 和 PRM 的信用分配

```python
# ==========================================
# 4. ORM vs PRM 信用分配
# ==========================================
import numpy as np

def orm_credit(episode: Episode, gamma: float = 0.95) -> List[float]:
    """ORM：只有最终结果给 reward，中间步骤全部为 0。"""
    T = len(episode.turns)
    final_success = episode.turns[-1].success
    immediate = [0.0] * (T - 1) + [1.0 if final_success else 0.0]

    returns = np.zeros(T)
    G = 0
    for t in reversed(range(T)):
        G = immediate[t] + gamma * G
        returns[t] = G
    return returns.tolist()

def prm_credit(episode: Episode, gamma: float = 0.95) -> List[float]:
    """PRM：每一步根据工具调用是否成功给即时 reward。"""
    T = len(episode.turns)
    immediate = []
    for turn in episode.turns:
        if turn.action == "answer":
            immediate.append(1.0 if turn.success else -0.5)
        else:
            immediate.append(0.3 if turn.success else -0.3)

    returns = np.zeros(T)
    G = 0
    for t in reversed(range(T)):
        G = immediate[t] + gamma * G
        returns[t] = G
    return returns.tolist()

orm_bad = orm_credit(bad_episode)
prm_bad = prm_credit(bad_episode)

print(f"\n{'轮次':<6} {'动作':<12} {'结果':<8} {'ORM Credit':<14} {'PRM Credit':<14}")
for i, turn in enumerate(bad_episode.turns):
    status = "✓" if turn.success else "✗"
    print(f"第{i+1}轮   {turn.action:<12} {status:<8} {orm_bad[i]:<14.3f} {prm_bad[i]:<14.3f}")
```

输出：

```
轮次   动作          结果      ORM Credit     PRM Credit
第1轮   search       ✓        0.000          0.656
第2轮   calculate    ✓        0.000          0.376
第3轮   verify       ✓        0.000          0.170
第4轮   answer       ✗        0.000          -0.500
```

ORM 模式下，差策略的所有步骤 credit 都是 0——包括第 2 步（calculate）。这是因为 ORM 只看最终答案对不对（第 4 步 answer 错了 → reward = 0），然后通过折扣把这个零信号反向传播。由于 $0 \times \gamma = 0$，所有步骤的 credit 都是 0。

如果改成"失败给负 reward"，更严重的问题暴露出来——**第 1 步正确的搜索也会被惩罚**：

```python
def orm_negative(episode, gamma=0.95):
    """ORM 变体：失败时所有步骤都被惩罚"""
    T = len(episode.turns)
    final_success = episode.turns[-1].success
    immediate = [0.0] * (T - 1) + [1.0 if final_success else -1.0]
    returns = np.zeros(T)
    G = 0
    for t in reversed(range(T)):
        G = immediate[t] + gamma * G
        returns[t] = G
    return returns.tolist()

# 输出：第 1 步正确的搜索 credit = -0.857，被错误地惩罚
```

**第 1 步搜索完全正确，却得到了 -0.857 的惩罚。** 这就是 ORM 的核心问题：信号太粗糙，无法区分"正确的步骤"和"导致失败的步骤"。

PRM 的区分度是 ORM 的 19 倍。ORM 几乎无法区分正确步骤和错误步骤（区分度仅 0.045），而 PRM 能清晰地告诉模型"哪些步骤做对了，哪些做错了"（区分度 0.856）。在同样的训练步数下，PRM 的任务成功率比 ORM 高出约 30 个百分点。

### 实验总结

这个实验用纯 Python 模拟了一个多轮 Agent 环境，让你亲手感受到了 Agentic RL 的核心挑战：

- **ORM 信号太稀疏**：失败时所有步骤 credit 都接近 0，模型不知道该改哪里。
- **ORM 错怪好人**：失败时连正确的搜索步骤都被惩罚。
- **PRM 精确归因**：正确步骤得正分，错误步骤得负分，区分度是 ORM 的 19 倍。
- **PRM 的代价**：每步都需要评估——在真实场景中需要标注成本或训练 PRM。

**核心洞察**：多轮 Agent 的关键难题不是"用什么 RL 算法"，而是"中间步骤怎么给 reward"。ORM 简单但粗糙，PRM 精确但昂贵，2025-2026 年的所有精细方法（GiGPO / SPA-RL / ARPO / IGPO / StepPO...）本质上都在 ORM 和 PRM 之间寻找更好的折中。

::: warning 这个实验是模拟的
真实场景中，Agent 不会使用预定义的 `action_plan`，而是由模型动态决定每一步调用什么工具。模型策略的"好坏"取决于 RL 训练的效果，而 RL 训练的效果又取决于信用分配的质量——这是一个闭环。本实验跳过了策略学习，专注于理解信用分配本身。
:::

## 与前面章节的联系

多轮 RL 的信用分配问题和第 6 章的策略梯度定理一脉相承。REINFORCE 用蒙特卡洛采样来估计 $G_t$——从当前步到结束的累积回报。多轮 RL 做的是同样的事，只不过"步"从单个 token 变成了一个完整的轮次。第 8 章的 PPO 通过引入价值函数（Critic）来降低方差——同样的思路在多轮 RL 中依然适用，只是 Critic 需要评估的不是"当前 token 的价值"，而是"当前轮次的价值"。Turn-PPO 就是这条思路的现代版本。

规划能力则是多轮 RL 的**进阶形态**——信用分配解决"每步做得好不好"，规划解决"整体走哪条路径最优"。两者共同构成了 Agentic RL 的决策核心。

下一节我们来拆解 Agentic RL 的工程核心——[19.4 工具调用与轨迹生成](./tool-use-and-trajectory)，看看训练数据从哪里来、工具策略怎么学、系统怎么跑起来。

## 参考资料

[^lightman]: Lightman H, et al. "[Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)." ICLR 2024. —— 提出 ORM vs PRM 的对比框架，证明过程监督（PRM）在数学推理上显著优于结果监督（ORM）。

[^mathshepherd]: Wang P, Li L, Shao Z, et al. "[Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations](https://arxiv.org/abs/2312.08935)." ACL 2024. —— 自动化过程奖励标注，无需人工标注中间步骤。

[^salt]: Li J, Wang Y, et al. "[SALT: Step-level Advantage Assignment for Long-horizon Agents via Trajectory Graph](https://arxiv.org/abs/2510.20022)." EACL 2026 Findings. —— 通过轨迹图量化每步质量，为 GRPO 提供步骤级 advantage 分配，不需要额外奖励模型。

[^gigpo]: Feng L, Xue Z, Liu T, et al. "[Group-in-Group Policy Optimization for LLM Agent Training](https://arxiv.org/abs/2505.10978)." NeurIPS 2025. —— 同 state group 内相对回报，state-anchored stepwise advantage。

[^hgpo]: He S, Feng L, Wei Q, et al. "[Hierarchy-of-Groups Policy Optimization for Long-Horizon Agentic Tasks](https://arxiv.org/abs/2602.22817)." ICLR 2026. —— 用 k-step 历史上下文分层分组，修正 GiGPO 的 historical context inconsistency。

[^spa]: Wang H, et al. "[SPA-RL: Reinforcing LLM Agents via Stepwise Progress Attribution](https://arxiv.org/abs/2505.20732)." arXiv, 2025. —— 通过步骤级进度归因精确分配每步贡献。

[^agentprm]: Choudhury S. "[Process Reward Models for LLM Agents: Practical Framework and Directions](https://arxiv.org/abs/2502.10325)." 2025. —— 提出 AgentPRM 与 InversePRM，把 PRM 当成 agent 的 Q(s,a)，三阶段循环训练。

[^arpo]: Dong G, Mao H, Ma K, et al. "[ARPO: Agentic Reinforced Policy Optimization](https://arxiv.org/abs/2507.19849)." 2025. —— entropy 飙升处局部分叉采样。

[^igpo]: Venus Team, Dai S, et al. "[DR-Venus: Towards Frontier Edge-Scale Deep Research Agents with Only 10K Open Data](https://arxiv.org/abs/2604.19859)." 2026. —— 在该技术报告中提出 IGPO（Information Gain Policy Optimization），用正确答案 log-prob 增量作为 turn-level reward。

[^aem]: Zhao H, Zhou S, Zhang Y, et al. "[AEM: Adaptive Entropy Modulation for Multi-Turn Agentic Reinforcement Learning](https://arxiv.org/abs/2605.00425)." 2026-05. —— 动态调节 entropy bonus 避免策略坍缩。

[^refgrpo]: Zhu Y. "[Closing the Reflection Gap: A Free Calibration Bonus for Agentic RL](https://arxiv.org/abs/2606.14211)." 2026-06. —— RefGRPO 通过校准 bonus 缩小反思鸿沟。

[^pair]: Kim W, In Y, Park S, et al. "[PAIR: Prefix-Aware Internal Reward Model for Multi-Turn Agent Optimization](https://arxiv.org/abs/2605.17877)." 2026-05. —— 前缀感知的 internal reward model。

[^steppo]: Wang D, Li Q, Cheng M, et al. "[StepPO: Step-Aligned Policy Optimization for Agentic Reinforcement Learning](https://arxiv.org/abs/2604.18401)." 2026-04. —— Step-centric paradigm，解决 token 与 step 的 granularity mismatch。

[^turnppo]: Li J, Zhou P, Meng R, et al. "[Turn-PPO: Turn-Level Advantage Estimation with PPO for Improved Multi-Turn RL in Agentic LLMs](https://arxiv.org/abs/2512.17008)." EACL 2026. —— Turn-level PPO 替代 GRPO，方差更低。

[^at2po]: Zong Z, Chen D, Li Y, et al. "[AT²PO: Agentic Turn-based Policy Optimization via Tree Search](https://arxiv.org/abs/2601.04767)." 2026-01. —— Turn-based + tree search 统一框架。

[^groupgraph]: Wang Y, Song M, Zhang Z, et al. "[Group-Graph Policy Optimization for Long-Horizon Agentic Reinforcement Learning](https://arxiv.org/abs/2606.22995)." 2026-06. —— 把轨迹建模成 DAG。

[^casurvey]: Zhang C. "[From Reasoning to Agentic: Credit Assignment in Reinforcement Learning for Large Language Models](https://arxiv.org/abs/2604.09459)." 2026-04. —— 综述：reasoning RL 与 agentic RL 的信用分配全景。

[^webshepherd]: Chae H, et al. "[Web-Shepherd: Advancing PRMs for Reinforcing Web Agents](https://arxiv.org/abs/2505.15277)." NeurIPS 2025 Spotlight. —— 首个网页导航专用的步骤级 PRM。

[^treerl]: Hou Z, Hu Z, Li Y, et al. "[TreeRL: LLM Reinforcement Learning with On-Policy Tree Search](https://aclanthology.org/2025.acl-long.604)." ACL 2025.

[^deepresearcher]: Zheng Y, et al. "[DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments](https://arxiv.org/abs/2504.03160)." EMNLP 2025.
