# 15.4 GRPO 改进方法

上一节我们看完了 DeepSeek-R1-Zero 与 DAPO——它们分别证明了"纯 RL 可以替代 SFT 冷启动"与"工程改造可以让 GRPO 用一半步数达到 R1-Zero 水平"。但 DAPO 只是 2025 年 GRPO 改进家族的**一个成员**。从 R1 论文发表（2025.01）到 2026 年初，开源社区和工业实验室在不到一年内提出了至少五个有影响力的 GRPO 变体。它们不是互相替代的关系，而是从不同角度修补 GRPO 的不同缺陷。

这一节按**改进方向**重新组织——把 Dr.GRPO、GSPO、CISPO、VAPO、RPT 五条路线放在一起对比，让你看清"什么时候该用哪个变体"。

## 改进路线一 与 移除归一化偏差

### Dr.GRPO 的发现

GRPO 在 R1 论文中的原始形式对组内 reward 做了两步归一化：

$$\tilde{r}_i = \frac{r_i - \text{mean}(r_1, \ldots, r_G)}{\text{std}(r_1, \ldots, r_G)}$$

这里除以标准差看起来很自然——让优势值的尺度统一。但 Liu et al. 在 2025 年的研究（[arXiv:2503.20783](https://arxiv.org/abs/2503.20783)）中发现，这个看似无害的归一化引入了两类偏差：

- **长度偏差**：当一个 prompt 的所有回答 reward 方差很大（部分对部分错），除以 std 后优势被压缩；当方差很小（全对或全错），std 接近零，优势会被放大到不合理的量级。模型由此学到"产生差异化输出比答对更重要"。
- **Reward hacking 的温床**：除以 std 等价于鼓励模型增加组内 reward 方差，而增加方差最简单的方式就是让一部分回答变得**更长**（更多 token、更多 chances 答对）。这是 R1-Zero 训练后期回答长度爆炸的直接原因之一。

Dr.GRPO 的修正极其简单——**只减去均值，不除以 std**：

$$\tilde{r}_i^{\text{Dr.GRPO}} = r_i - \text{mean}(r_1, \ldots, r_G)$$

实验显示，这一改动让训练后期的回答长度膨胀问题显著缓解，reward hacking 行为减少。Qwen 系列在内部训练中采纳了类似的修正。

### DeepSeek V3.2 的进一步工程化

DeepSeek 在 V3.2 版本（2025.12, [arXiv:2512.02556](https://arxiv.org/abs/2512.02556)）把 Dr.GRPO 的思想推到极致，针对数学推理任务做了三项工程调整：

- **数学任务 zero KL**：传统 GRPO 用 KL 散度约束策略不偏离参考模型，但数学任务的奖励本身已经足够约束（答错就零分），KL 约束反而抑制了模型探索新解题路径。DeepSeek 在纯数学 RL 阶段直接关闭 KL。
- **自验证 RLVR**：让模型在生成答案后自己加一个"验证步骤"——重新读题、检查计算、确认答案。这个验证步骤的奖励也纳入 RL 优化，形成内部 self-check 机制。
- **mHC 残差稳定性**：Modified Hamiltonian Monte Carlo 采样器在长 CoT 训练中的数值稳定性优化，避免梯度爆炸。

V3.2 Speciale 变体在 AIME 2025 上达到 97 分，超越 GPT-5 同期水平。

## 改进路线二 与 序列级重要性采样

### GSPO（Qwen3 的选择）

GRPO 和 PPO 一样使用 **token 级重要性采样比**：

$$\rho_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$$

每个 token 都有自己的比率，GRPO 分别裁剪这些比率，再把 token 目标求平均。这里没有把所有 $\rho_t$ 相乘。这是 [DeepSeekMath 式（3）](https://arxiv.org/html/2402.03300#S3.SS1)的原始定义。GSPO 论文认为，回答只有一个序列级奖励时，这种逐 token 的重要性权重会引入高方差噪声；MoE 架构的路由变化还会进一步放大比率波动。

GSPO（Group Sequence Policy Optimization）把比率从 token 级提升到**序列级**：

$$
s_i(\theta)
=
\left(
\frac{\pi_\theta(y_i\mid x)}{\pi_{\theta_{\text{old}}}(y_i\mid x)}
\right)^{1/|y_i|}
=
\exp\!\left(
\frac{1}{|y_i|}\sum_{t=1}^{|y_i|}
\log\frac{\pi_\theta(y_{i,t}\mid x,y_{i,<t})}
{\pi_{\theta_{\text{old}}}(y_{i,t}\mid x,y_{i,<t})}
\right)
$$

这个数也是 token 比值的几何平均：

$$
s_i(\theta)=\left(\prod_{t=1}^{|y_i|}\rho_{i,t}\right)^{1/|y_i|}
$$

整个回答只用一个比率参与裁剪。裁剪对象也对应变为序列级：

$$\mathcal{J}^{\text{GSPO}} = \mathbb{E}\left[\min\left(s_i(\theta) \hat A_i, \; \text{clip}(s_i(\theta), 1-\epsilon, 1+\epsilon) \hat A_i\right)\right]$$

$1/|y_i|$ 的长度归一化不能省略。直接使用 $\prod_t\rho_{i,t}$ 会让比率随回答长度剧烈变化，不同长度还需要不同的裁剪范围。GSPO 原文正是用几何平均控制这种方差。因此，“先求和全部 token 对数概率，再直接取指数”的旧 GRPO 示例也不是正确的 GSPO。[GSPO 原论文式（5）与式（7）](https://arxiv.org/html/2507.18071#S4.SS1)

这个改动看起来简单，但对 MoE 模型训练稳定性影响巨大——Qwen3 全系（包括 Qwen3-235B-A22B、Qwen3-Thinking-2507、Qwen3-Coder）都基于 GSPO 训练。序列级比率的方差远小于 token 级，让万卡集群上的大规模 RL 训练成为可能。

GSPO 的代价：序列级比率**耦合了所有 token 的更新**，单 token 的精细信用分配能力弱于 token 级方案。因此 GSPO 在长 CoT 任务（推理、数学）上效果显著，但在需要 token 级奖励的代码生成任务上不如 DAPO。

## 改进路线三 与 裁剪对象的改写

### CISPO（MiniMax 的创新）

GRPO 与 DAPO 都裁剪"策略比率与优势的乘积"——裁剪发生在梯度更新层面。MiniMax 在 M1 模型（[arXiv:2506.13585](https://arxiv.org/abs/2506.13585)）中提出 CISPO，把裁剪对象从"token 更新"改为"**重要性采样权重**"：

$$\tilde{\rho}_t = \text{clip}\left(\frac{\pi_{\theta_{\text{old}}}(a_t|s_t)}{\pi_\theta(a_t|s_t)}, 1-\epsilon, 1+\epsilon\right)$$

注意这里比率的分子分母是反过来的——$\pi_{\text{old}} / \pi_\theta$ 而不是 $\pi_\theta / \pi_{\text{old}}$。这个反向比率作为**采样权重**乘到优势上，但**保留所有 token 的梯度贡献**。

直觉上：传统裁剪是"如果某个 token 的策略偏离过大，直接把它从梯度里抹掉"。CISPO 是"如果偏离过大，降低它对优势估计的贡献权重，但梯度方向不变"。后者避免了"训练后期某些 token 完全不更新"导致的策略卡死问题。

CISPO 还有一项工程优势——它和 MiniMax 自研的 lightning attention 配合时，能解决精度对齐问题。lightning attention 的递归计算让 token 级比率的浮点误差累积，传统裁剪在低精度训练中会误杀大量 token。CISPO 通过权重缩放而非裁剪避免了这个问题。MiniMax M1 用 512 块 H800 训练，整体速度比 DAPO 快 2 倍。

## 改进路线四 与 Value-based 反潮流

### VAPO（字节 Seed 的反潮流）

到这里你可能会产生一个印象：**Critic 网络已经被 GRPO 淘汰了**。但字节 Seed 在 2025.04 发表的 VAPO（Value-based Augmented PPO, [arXiv:2504.05118](https://arxiv.org/abs/2504.05118)）证明——至少在长 CoT 推理任务里，**value model 重新打败了 GRPO**。

VAPO 的核心论点是：GRPO 用组内均值替代 Critic，本质上是用"同一个 prompt 的多个 rollout 之间的相对排名"来估计优势。这在短回答任务（比如 function calling、数学简单题）里足够。但在长 CoT 任务里：

- 一个 rollout 内部有几百个 token，**真正的优势信号是 token 级的**——某一步推理是好的，下一步推理是坏的。
- 组内均值把整个 rollout 当作一个单位，丢失了 token 级信号。
- 训练越长，模型越容易学会"靠运气答对一部分 rollout"而非"每一步都正确"。

VAPO 重新引入 value model $V_\phi(s)$，用 GAE 估计 token 级优势：

$$\hat{A}_t = \delta_t + (\gamma\lambda)\delta_{t+1} + (\gamma\lambda)^2\delta_{t+2} + \cdots$$

其中 $\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$ 是 TD 误差。然后在这个 token 级优势上做 PPO 风格的裁剪。

VAPO 在 AIME 2024 上达到 60.4 分，超越同期所有 GRPO 变体（DAPO 50 分、R1-Zero 71 分但用了 2 倍训练步数）。**字节 Seed 内部的推理模型训练已经从纯 GRPO 转向 VAPO**。

VAPO 的代价：需要训练一个独立的 value model，显存翻倍，工程复杂度高。这是为什么 GRPO 在 2024 年成为主流——**critic-free 是工程妥协，不是算法必然**。

## 改进路线五 与 把 RL 引入预训练

### RPT（Reinforcement Pre-Training）

前面四条改进路线都假设 RL 发生在**后训练阶段**——模型已经预训练完，RL 只是微调。但 Microsoft 在 2025.06 提出的 Reinforcement Pre-Training（[arXiv:2506.08007](https://arxiv.org/abs/2506.08007)）挑战了这个二分法。

RPT 的核心思想：把 next-token 预测任务**重构为推理任务**。传统预训练的损失是：

$$\mathcal{L}_{\text{LM}} = -\mathbb{E}\left[\log \pi_\theta(a_t | s_{<t})\right]$$

每个 token 都是平等的教师强制目标。RPT 把它改为：模型先生成对下一 token 的推理（"根据上下文，下一个词可能是 X 因为..."），然后用推理结果预测下一 token，正确就给奖励：

$$\mathcal{L}_{\text{RPT}} = -\mathbb{E}\left[\log \pi_\theta(a_t | s_{<t}, \text{reasoning}_t)\right] + \beta \cdot \text{RL loss}$$

这个改动的意义是革命性的——**预训练阶段就可以做 RL**，而且 RPT 的 scaling 特性与传统预训练相当。这意味着未来可能不再有"预训练 vs 后训练"的清晰边界，RL 贯穿整个训练流程。

RPT 目前还在早期阶段，工业实践中尚未广泛采用。但它的概念冲击足以让本书专门列出这条改进路线。

## 选型决策树

下面这张表把五个变体的核心差异、适用场景、典型用户整理在一起：

| 算法        | 核心创新             | 解决的痛点               | 典型适用场景                | 代表用户       |
| ----------- | -------------------- | ------------------------ | --------------------------- | -------------- |
| **GRPO**    | 组内均值替代 Critic  | Critic 显存开销          | 通用 RLHF / RLVR            | DeepSeek-R1    |
| **Dr.GRPO** | 移除 std 归一化      | 长度膨胀、reward hacking | 数学推理                    | Qwen 内部      |
| **GSPO**    | 序列级 IS            | MoE 训练不稳定           | MoE 模型 RL                 | Qwen3 全系     |
| **CISPO**   | 裁剪 IS 权重         | Token 丢失、精度对齐     | Lightning attention、低精度 | MiniMax M1     |
| **VAPO**    | 重新引入 value model | 长 CoT 信用分配          | 推理模型训练                | 字节 Seed      |
| **DAPO**    | 4 项工程改造         | 训练效率、长度控制       | 数学 / 代码 RL              | 字节 + 清华    |
| **RPT**     | RL 引入预训练        | 预训练-后训练边界        | 下一代基座模型              | Microsoft 研究 |

实际工业实践中的选型逻辑大致是：

```text
任务类型？
├── 数学/代码推理（长 CoT）
│   ├── MoE 架构 → GSPO + Dr.GRPO 思想
│   ├── Dense 架构 → VAPO 或 DAPO
│   └── 极致稳定要求 → CISPO
├── 通用对话对齐
│   └── GRPO / PPO（基础够用）
├── 多轮工具调用
│   └── DAPO + Token-level loss
└── 下一代基座模型
    └── RPT（实验性）
```

这个决策树不是绝对的——字节 Seed 内部经常混用（比如 DAPO 的工程技巧 + VAPO 的 value model）。但它给出了"看到任务后第一反应应该考虑什么"的检查清单。

## 小结

GRPO 改进家族的快速演化反映了一个事实：**RL 在大模型时代不再是"用 PPO 就够了"的单一选择**。每家实验室都根据自己的训练基础设施（MoE vs Dense、lightning attention vs 标准 attention、显存预算）和任务特征（推理 vs 对话、长 CoT vs 短回答）选择了不同的改进方向。

这一节的真正价值不在于记住每个算法的具体公式——而在于建立一种判断力：**看到一个新 GRPO 变体时，能立刻问出"它修补的是归一化、序列级、裁剪、还是 value model 这四条路线中的哪一条"**。这种判断力是从读论文到能动手改进算法的关键一步。
