# 8.2 信任域约束与 PPO-Clip

## 本节导读

PPO 的裁剪代理目标写为：

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \cdot A_t, \; \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon) \cdot A_t \right) \right]$$

该目标包含策略比率 $r_t$、裁剪操作 clip 以及外层的 min。本节回答两个问题：**这一公式保护的对象是什么？为何不能直接采用原始策略梯度？**

推导线索为：原始策略梯度的更新风险 → 重要性采样实现数据复用 → TRPO 的 KL 散度约束 → PPO 的裁剪近似。整组公式的核心动机只有一个：**约束单次更新的幅度，防止策略被一次更新破坏**。

为便于推导，本节贯穿使用同一个微型示例：状态 $s$ 下有两个动作 $a_1, a_2$，策略初始概率为 $\pi(a_1\mid s) = 0.6$、$\pi(a_2\mid s) = 0.4$。核心问题是：若一次更新将 $\pi(a_1\mid s)$ 推至 $0.99$，会产生什么后果？

::: tip 本节前置知识

- [策略梯度更新公式](../chapter08_policy_gradient/reinforce)——裁剪要保护的正是这个更新
- [优势函数 $A(s,a)$](../chapter09_actor_critic/advantage-function)——策略更新的方向信号
  :::

## 原始策略梯度的更新风险

回顾第 6 章的策略梯度更新（[REINFORCE](../chapter08_policy_gradient/reinforce)）：

$$\theta \leftarrow \theta + \alpha \cdot \nabla_\theta \log \pi_\theta(a\mid s) \cdot A(s,a)$$

当动作 $a$ 的优势 $A(s,a) > 0$（优于平均）时，参数沿增大 $\pi(a\mid s)$ 的方向更新。**问题在于：该更新的幅度没有上界约束**。

代入微型示例。设采样到 $(s, a_1)$，其优势 $A(s, a_1) = 2$，学习率 $\alpha = 0.5$。单次更新的结果为：

|                  | 更新前 | 更新后 |
| ---------------- | ------ | ------ |
| $\pi(a_1\mid s)$ | 0.6    | 0.99   |
| $\pi(a_2\mid s)$ | 0.4    | 0.01   |

仅一次更新，$a_1$ 的概率从 0.6 升至 0.99。然而这只是**单次采样**的结果——若高优势源于采样偶然性，策略已将另一动作的概率压缩至 0.01。**策略更新不可逆，没有撤销机制**。

下一轮更新更为严重。训练数据在更新前的旧策略 $\pi_{\text{old}}$ 下采集，当尝试复用同一批数据时，$a_1$ 在新策略下的概率已为 0.99，与采样时的 0.6 严重偏离。**旧数据失效**。

原始策略梯度的核心困境在于：**单步更新的方差较大，而策略更新不可逆**。一次失当的更新将导致整批数据作废。

> 既然单次更新即可破坏策略，那么在考虑"更新多少"之前，是否应先确保"用旧数据训练新策略"这一操作本身是安全的？

## 重要性采样与数据复用

第一个问题是数据复用。$(s, a_1)$ 样本在 $\pi_{\text{old}}(a_1\mid s) = 0.6$ 下采集，能否用它更新已变为 $\pi_\theta(a_1\mid s) = 0.99$ 的新策略？

数学上可行，其工具是**重要性采样**（Importance Sampling）。同一事件在不同分布下的期望可通过一个概率比值进行转换：

$$\mathbb{E}_{a \sim \pi_{\text{old}}} \left[ \frac{\pi_\theta(a\mid s)}{\pi_{\text{old}}(a\mid s)} \cdot f(a) \right] = \mathbb{E}_{a \sim \pi_\theta} [f(a)]$$

该比值 $r_t(\theta) = \frac{\pi_\theta(a_t\mid s_t)}{\pi_{\text{old}}(a_t\mid s_t)}$ 称为**策略比率**（Policy Ratio），衡量新策略相对于旧策略在同一 $(s,a)$ 上的概率变化。

看那个例子：

| 策略                       | $\pi(a_1\mid s)$ |
| -------------------------- | ---------------- |
| 旧策略 $\pi_{\text{old}}$  | 0.6              |
| 新策略 $\pi_\theta$        | 0.99             |
| $r_t(\theta) = 0.99 / 0.6$ | **1.65**         |

新策略将 $a_1$ 的概率放大至 1.65 倍。将该比值引入策略梯度，得到含重要性采样的目标：

$$L^{\text{IS}}(\theta) = \mathbb{E}_t \left[ r_t(\theta) \cdot A_t \right]$$

形式上，旧数据可用于评估新策略。但 1.65 这一数值也暴露了问题：**梯度被放大 1.65 倍**。若策略更激进地将 $\pi(a_1\mid s)$ 推至 0.999，则 $r_t = 1.665$；推至 0.9999，则 $r_t = 1.6665$。$r_t$ 越大，下一次更新的幅度越大，进而使 $r_t$ 进一步增大——形成**正反馈循环**。

重要性采样解决了数据复用问题，但未提供安全利用的保障。

> 原始策略梯度单步即出现风险，重要性采样又允许 $r_t$ 无限增大。能否将"单次更新幅度受限"这一要求以数学形式严格表述？

## TRPO 与 KL 散度约束

2015 年，Schulman 等人提出 TRPO（Trust Region Policy Optimization，信任域策略优化）。其核心思路是：**直接约束新旧策略之间的距离**。衡量两个概率分布差异的标准工具为 KL 散度（Kullback-Leibler divergence），将其表述为硬约束：

$$\max_\theta \; \mathbb{E}_t \left[ r_t(\theta) \cdot A_t \right] \quad \text{s.t.} \quad \mathbb{E}_t \left[ D_{\text{KL}}(\pi_{\text{old}} \| \pi_\theta) \right] \leq \delta$$

该优化问题包含两部分：左侧 $\max_\theta \; \mathbb{E}_t[r_t(\theta) \cdot A_t]$ 是待最大化的目标——即重要性采样目标，追求更高的累积优势；右侧 $\mathbb{E}_t[D_{\text{KL}}(\pi_{\text{old}} \| \pi_\theta)] \leq \delta$ 是必须满足的约束条件。其中 "s.t." 读作 "subject to"（受约束于）。

KL 散度衡量两个概率分布的差异程度，其定义为：

$$D_{\text{KL}}(P \| Q) = \sum_i P(i) \log \frac{P(i)}{Q(i)}$$

当两个分布完全相同时，$D_{\text{KL}} = 0$；差异越大，$D_{\text{KL}}$ 越大（始终非负）。式中 $\pi_{\text{old}}$ 为更新前的旧策略，$\pi_\theta$ 为更新后的新策略，因此 $D_{\text{KL}}(\pi_{\text{old}} \| \pi_\theta)$ 衡量一次更新使策略分布改变的程度。约束条件要求这一改变量不超过 $\delta$。

回到那个微型示例。$\pi_{\text{old}}(a_1) = 0.6$，$\pi_\theta(a_1) = 0.99$，则：

$$D_{\text{KL}}(\pi_{\text{old}} \| \pi_\theta) = 0.6 \ln\frac{0.6}{0.99} + 0.4 \ln\frac{0.4}{0.01} \approx 0.6 \times (-0.50) + 0.4 \times 3.69 \approx 1.18$$

结果约为 1.18，远超 $\delta = 0.01$。TRPO 在此情形下将直接拒绝该更新，并将策略拉回信任域内。

$\delta$ 通常取 0.01，即每次更新后策略的行为分布最多改变 1%。这定义了一个**信任域**（trust region）：策略可在域内自由变动，一旦越界则更新被拒绝。

理论上严谨，工程上却存在困难。求解该约束优化问题需要 Hessian 矩阵（参数的二阶导数）。对于百万参数规模的网络，Hessian 的维度为参数数量的平方，无法存入显存。在 LLM 场景中，策略本身是 70B 参数的语言模型，计算其 Hessian 完全不可行。TRPO 采用共轭梯度法进行近似，但仍然速度慢且实现复杂。**严格约束的代价是计算成本**。

> TRPO 的约束过于精确，导致计算成本难以承受。是否存在一种方法，约束力度不如 TRPO 严格，但仍能有效限制单次更新幅度？

## PPO 裁剪机制

2017 年，Schulman 提出 PPO（Proximal Policy Optimization）。上一节指出，TRPO 的困难在于精确计算 KL 散度需要 Hessian 矩阵，成本过高。PPO 的思路是：**既然目标是让新旧策略"不要差太远"，那就不必精确测量距离，直接限制策略比率 $r_t$ 即可**。

回到 $r_t$ 的定义：$r_t = \pi_\theta(a_t\mid s_t) / \pi_{\text{old}}(a_t\mid s_t)$。当 $r_t = 1$ 时，新旧策略在该动作上完全一致；$r_t$ 偏离 1 越多，策略变化越大。因此，将 $r_t$ 约束在 $[1-\varepsilon, 1+\varepsilon]$ 内，等价于限制每个动作的概率变化幅度——这是对"策略距离不能太大"这一目标的局部、廉价近似，无需计算 KL 散度，也无需 Hessian。

PPO 的目标函数：

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \cdot A_t, \; \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon) \cdot A_t \right) \right]$$

用前文的 1.65 例子计算（取 $\varepsilon = 0.2$、$A_t = 2$）。已知 $r_t = 1.65$，逐项计算：

**未裁剪项**：$r_t \cdot A_t = 1.65 \times 2 = 3.30$。

**裁剪项**：由于 $r_t = 1.65 > 1 + \varepsilon = 1.2$，超出上界，裁剪操作将 $r_t$ 截断为 $1.2$。于是裁剪项 $= 1.2 \times 2 = 2.40$。

**取最小值**：$\min(3.30,\; 2.40) = 2.40$。

| 项        | 计算                                                 | 值         |
| --------- | ---------------------------------------------------- | ---------- |
| 未裁剪    | $r_t \cdot A_t = 1.65 \times 2$                      | $3.30$     |
| 裁剪值    | $\text{clip}(1.65, 0.8, 1.2) \cdot 2 = 1.2 \times 2$ | $2.40$     |
| $\min$ 取 | $\min(3.30,\; 2.40)$                                 | **$2.40$** |

裁剪将较大的目标值（3.30）压缩至 2.40。此时目标函数在该区间内变为常数，**梯度对 $\theta$ 的依赖消失，不再鼓励继续增大 $\pi(a_1\mid s)$**。

### 裁剪为何让梯度归零

要理解"目标函数变为常数"的含义，需要先看清 clip 函数的完整定义。$\text{clip}(x, a, b)$ 是一个三段函数：

$$\text{clip}(x, a, b) = \begin{cases} a & x < a \\ x & a \leq x \leq b \\ b & x > b \end{cases}$$

代入 $a = 1 - \varepsilon = 0.8$、$b = 1 + \varepsilon = 1.2$，其图像为一条折线：左右两段是水平线（取值固定为 $0.8$ 与 $1.2$），中间一段是斜率为 1 的对角线。

| $r_t$ 的位置            | clip 输出  | 是否含 $\theta$ |
| ----------------------- | ---------- | --------------- |
| $r_t < 0.8$             | 恒为 $0.8$ | 否（纯数字）    |
| $0.8 \leq r_t \leq 1.2$ | $r_t$ 本身 | 是              |
| $r_t > 1.2$             | 恒为 $1.2$ | 否（纯数字）    |

下面用链式法则完整推一遍。先写出 $r_t(\theta)$ 的依赖关系：

$$r_t(\theta) = \frac{\pi_\theta(a_t\mid s_t)}{\pi_{\text{old}}(a_t\mid s_t)}$$

其中 $\pi_{\text{old}}$ 是采样时的旧策略，已经是固定常数（本节微型示例中 $\pi_{\text{old}}(a_1\mid s) = 0.6$）。$\theta$ 只通过分子 $\pi_\theta$ 影响 $r_t$。

裁剪项的梯度按链式法则展开：

$$\nabla_\theta\left[\text{clip}(r_t(\theta), 0.8, 1.2) \cdot A_t\right] = A_t \cdot \underbrace{\frac{\partial \,\text{clip}}{\partial r_t}}_{\text{clip 斜率}} \cdot \underbrace{\nabla_\theta r_t(\theta)}_{\text{含 } \theta}$$

依赖链是 $\theta \xrightarrow{\nabla \pi_\theta} r_t \xrightarrow{\partial \text{clip}/\partial r_t} \text{clip} \xrightarrow{\times A_t} L$。链式法则是这三段导数的乘积，**任何一段为零，整条链乘积为零**。clip 是分段函数，各段斜率不同：

| $r_t$ 的位置            | clip 斜率 $\partial \text{clip}/\partial r_t$ |
| ----------------------- | --------------------------------------------- |
| $r_t < 0.8$             | $0$（水平段）                                 |
| $0.8 \leq r_t \leq 1.2$ | $1$（斜线段）                                 |
| $r_t > 1.2$             | $0$（水平段）                                 |

代入本节微型示例的数字：$\pi_{\text{old}}(a_1\mid s) = 0.6$、$A_t = 2$、$\nabla_\theta r_t = \nabla_\theta \pi_\theta / 0.6$。

**情形一：$r_t = 1.65$（超出上界，顺方向越界）**。clip 在水平段，斜率为 $0$：

$$\nabla_\theta[\text{clip}(r_t,0.8,1.2)\cdot A_t] = 2 \cdot 0 \cdot \frac{\nabla_\theta \pi_\theta}{0.6} = 0$$

中间环节 $\partial \text{clip}/\partial r_t = 0$ 把 $\theta \to L$ 的依赖链切断了——无论 $\nabla_\theta \pi_\theta$ 多大，乘积仍为零。$\min$ 此处选裁剪项（$2.40 < 3.30$），梯度为零，更新停止。**这正是 PPO 的设计意图：好动作已经把概率推到 1.65 倍，应当停止**。

**情形二：$r_t = 0.5$（跌破下界，反方向越界）**。clip 仍在水平段，斜率仍为 $0$：

$$\nabla_\theta[\text{clip}(r_t,0.8,1.2)\cdot A_t] = 2 \cdot 0 \cdot \frac{\nabla_\theta \pi_\theta}{0.6} = 0$$

此时情形颠倒了——$A_t = 2$ 是好动作，本应增大 $r_t$，$r_t$ 却跌到 $0.5$，属于"更新方向错了"。**纯裁剪给出的梯度仍为零，策略被卡死在 $0.5$，爬不回 $[0.8, 1.2]$ 安全区**。

两种情形对照：情形一的零梯度是设计意图——好动作已超出容忍范围，应当停止；情形二的零梯度是隐患——策略走错方向却被卡死。

这就是"目标函数变为常数"的全部含义：**相对于参数 $\theta$ 是常数**，不是说数值不变化。等价的几何语言是：目标函数曲线在该区间是水平线，水平线的斜率（导数）为零。

该公式由三项构成，各司其职：

- **未裁剪项** $r_t(\theta) \cdot A_t$：重要性采样后的标准策略梯度目标，即策略比率与优势的乘积。
- **裁剪项** $\text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon) \cdot A_t$：将 $r_t$ 约束在 $[1-\varepsilon, 1+\varepsilon]$ 内。$\varepsilon$ 通常取 0.1 或 0.2，对应策略概率的最大变化幅度为 10% 或 20%。
- **取最小值** $\min(\cdot, \cdot)$：在两者中选取更保守的一项。

### 裁剪机制的方向性

裁剪的效果取决于优势 $A_t$ 的正负，上下界分别在不同情形下生效：

**当 $A_t > 0$（好动作）时**：更新方向应增大 $r_t(\theta)$。裁剪将 $r_t$ 的上界限制为 $1+\varepsilon$，超过该值即被截断。即使某个好动作具有很高的优势，策略概率也不会无限制地朝该方向增长。

**当 $A_t < 0$（坏动作）时**：更新方向应减小 $r_t(\theta)$。裁剪将 $r_t$ 的下界限制为 $1-\varepsilon$，防止策略概率过度降低。

```mermaid
flowchart LR
    subgraph posA ["A_t > 0（好动作）——上界生效，方向朝右"]
        direction LR
        P_safe["1−ε ≤ r_t ≤ 1+ε\n梯度正常\n概率继续增大"] --> P_clip["r_t > 1+ε\n裁剪至 1+ε\n梯度归零，停止增大"]
    end

    subgraph negA ["A_t < 0（坏动作）——下界生效，方向朝左"]
        direction LR
        N_clip["r_t < 1−ε\n裁剪至 1−ε\n梯度归零，停止减小"] --> N_safe["1−ε ≤ r_t ≤ 1+ε\n梯度正常\n概率继续减小"]
    end

    classDef safe fill:#e8f5e9,stroke:#2e7d32,color:#000
    classDef clip fill:#fce4ec,stroke:#c62828,color:#000

    class P_safe,N_safe safe
    class P_clip,N_clip clip
```

如图所示，$r_t$ 仅在偏离 1 的"顺优势方向"上受裁剪约束：$A_t > 0$ 时上界 $1+\varepsilon$ 生效，$A_t < 0$ 时下界 $1-\varepsilon$ 生效。一旦 $r_t$ 越出对应边界，梯度归零，更新停止。

然而此处隐含一个问题：若 $r_t$ 越出的是**反方向**边界——例如 $A_t > 0$ 应增大 $r_t$，但 $r_t$ 反而跌破 $1-\varepsilon$——裁剪项本身能否处理？这正是外层 $\min$ 存在的理由。

### min 操作的作用

裁剪项在两端梯度为零，外层 $\min$ 凭什么保证整个目标函数梯度非零？更尖锐地问：$\min$ 在两个候选项中选一项，**会不会恰好选到梯度为零的裁剪项，让策略卡死？**

答案是不会。$\min$ 总是选两项中数值**更小**的那一项，而 clip 的算术结构恰好让"更小"等同于"梯度方向正确"。下面分情形逐一验证。

设 $A_t > 0$（应增大 $r_t$），看两种越界情形。

**顺方向越界**（$r_t > 1+\varepsilon$）。clip 截断到 $1+\varepsilon$：

| 项     | 表达式                | 数值（$r_t=1.65$、$\varepsilon=0.2$、$A_t=2$） |
| ------ | --------------------- | ---------------------------------------------- |
| 未裁剪 | $r_t \cdot A_t$       | $1.65 \times 2 = 3.30$（大）                   |
| 裁剪   | $(1+\varepsilon) A_t$ | $1.2 \times 2 = 2.40$（小）                    |

由 $r_t > 1+\varepsilon$ 得 $r_t A_t > (1+\varepsilon) A_t$，$\min$ 选裁剪项，梯度为零。**符合意图**：好动作已超出容忍范围，更新停止。

**反方向越界**（$r_t < 1-\varepsilon$）。clip 截断到 $1-\varepsilon$：

| 项     | 表达式                | 数值（$r_t=0.5$、$\varepsilon=0.2$、$A_t=2$） |
| ------ | --------------------- | --------------------------------------------- |
| 未裁剪 | $r_t \cdot A_t$       | $0.5 \times 2 = 1.0$（小）                    |
| 裁剪   | $(1-\varepsilon) A_t$ | $0.8 \times 2 = 1.6$（大）                    |

由 $r_t < 1-\varepsilon$ 得 $r_t A_t < (1-\varepsilon) A_t$，$\min$ 选未裁剪项。该项含真实 $r_t(\theta)$，链式法则给出非零梯度，方向为增大 $r_t$：

$$\nabla_\theta[r_t \cdot A_t] = A_t \cdot \nabla_\theta r_t = 2 \cdot \frac{\nabla_\theta \pi_\theta(a_t\mid s_t)}{0.6} = \tfrac{10}{3}\,\nabla_\theta \pi_\theta(a_t\mid s_t) \neq 0$$

这正是把策略拉回 $[1-\varepsilon, 1+\varepsilon]$ 安全区的纠偏信号。

$A_t < 0$（坏动作，应减小 $r_t$）的两种越界情形与上面对称——数值皆负，"更小"指绝对值更大、惩罚更重。

**顺方向越界**（$r_t < 1-\varepsilon$）。clip 截断到 $1-\varepsilon$：

| 项     | 表达式                | 数值（$r_t=0.5$、$\varepsilon=0.2$、$A_t=-2$） |
| ------ | --------------------- | ---------------------------------------------- |
| 未裁剪 | $r_t \cdot A_t$       | $0.5 \times (-2) = -1.0$（绝对值小）           |
| 裁剪   | $(1-\varepsilon) A_t$ | $0.8 \times (-2) = -1.6$（绝对值大）           |

两边同乘负数 $A_t=-2$ 后不等号翻转：由 $r_t < 1-\varepsilon$ 得 $r_t A_t > (1-\varepsilon) A_t$，即 $-1.0 > -1.6$，$\min$ 选裁剪项 $-1.6$，梯度为零。**符合意图**：坏动作的概率已降至足够低，更新停止。

**反方向越界**（$r_t > 1+\varepsilon$）。clip 截断到 $1+\varepsilon$：

| 项     | 表达式                | 数值（$r_t=1.65$、$\varepsilon=0.2$、$A_t=-2$） |
| ------ | --------------------- | ----------------------------------------------- |
| 未裁剪 | $r_t \cdot A_t$       | $1.65 \times (-2) = -3.30$（绝对值大）          |
| 裁剪   | $(1+\varepsilon) A_t$ | $1.2 \times (-2) = -2.40$（绝对值小）           |

由 $r_t > 1+\varepsilon$ 得 $r_t A_t < (1+\varepsilon) A_t$，即 $-3.30 < -2.40$，$\min$ 选未裁剪项 $-3.30$，梯度非零，方向为减小 $r_t$——把策略拉回安全区。

四种越界情形汇总：

| $A_t$ | $r_t$ 位置        | 大小关系        | $\min$ 选取 | 梯度             | 设计意图         |
| ----- | ----------------- | --------------- | ----------- | ---------------- | ---------------- |
| $>0$  | $> 1+\varepsilon$ | 未裁剪 $>$ 裁剪 | 裁剪值      | 零（水平段）     | 顺方向过冲，停止 |
| $>0$  | $< 1-\varepsilon$ | 未裁剪 $<$ 裁剪 | 未裁剪值    | 非零，增大 $r_t$ | 反方向纠偏       |
| $<0$  | $< 1-\varepsilon$ | 未裁剪 $>$ 裁剪 | 裁剪值      | 零（水平段）     | 顺方向过冲，停止 |
| $<0$  | $> 1+\varepsilon$ | 未裁剪 $<$ 裁剪 | 未裁剪值    | 非零，减小 $r_t$ | 反方向纠偏       |

<details>
<summary>梯度永不反向的形式证明（选读）</summary>

上表分四种情形验证了 min 的选取。能否不依赖分情形讨论，给出一个统一的数学保证？可以，且证明极其简短。

**命题**：对任意 $r_t > 0$ 和 $A_t \neq 0$，PPO 目标 $L(r_t, A_t) = \min(r_t A_t,\; c \cdot A_t)$（其中 $c = \text{clip}(r_t, 1-\varepsilon, 1+\varepsilon)$）对 $r_t$ 的偏导数满足：

$$\frac{\partial L}{\partial r_t} \in \{0,\; A_t\}$$

**推论**：$A_t \cdot \frac{\partial L}{\partial r_t} \in \{0,\; A_t^2\} \geq 0$。即梯度分量要么为零，要么与 $A_t$ 同号——**永不反向**。

**证明**：min 的导数规则——取两项中较小者，导数等于该项的导数。$c$ 作为 $r_t$ 的函数只有三种形态：

| $r_t$ 的位置                                | $c$             | $\frac{dc}{dr_t}$ |
| ------------------------------------------- | --------------- | ----------------- |
| $r_t < 1-\varepsilon$                       | $1-\varepsilon$ | $0$               |
| $1-\varepsilon \leq r_t \leq 1+\varepsilon$ | $r_t$           | $1$               |
| $r_t > 1+\varepsilon$                       | $1+\varepsilon$ | $0$               |

逐一确定 $L$ 等于哪一项：

- **$r_t \in [1-\varepsilon, 1+\varepsilon]$**：$c = r_t$，两项相等，$L = r_t A_t$，$\frac{\partial L}{\partial r_t} = A_t$。
- **$r_t > 1+\varepsilon$**：$c = 1+\varepsilon$ 为常数。
  - $A_t > 0$：$r_t A_t > (1+\varepsilon) A_t$，min 选裁剪项（常数），$\frac{\partial L}{\partial r_t} = 0$。
  - $A_t < 0$：两边乘负数翻转不等号，$r_t A_t < (1+\varepsilon) A_t$，min 选未裁剪项，$\frac{\partial L}{\partial r_t} = A_t$。
- **$r_t < 1-\varepsilon$**：$c = 1-\varepsilon$ 为常数。
  - $A_t > 0$：$r_t A_t < (1-\varepsilon) A_t$，min 选未裁剪项，$\frac{\partial L}{\partial r_t} = A_t$。
  - $A_t < 0$：翻转后 $r_t A_t > (1-\varepsilon) A_t$，min 选裁剪项（常数），$\frac{\partial L}{\partial r_t} = 0$。

五种可达情形（区间内一种 + 两端各两种）给出 $\frac{\partial L}{\partial r_t} \in \{0, A_t\}$。证毕。$\square$

**几何含义**：$L$ 作为 $r_t$ 的函数是一条分段折线，斜率只有两个值——$0$ 或 $A_t$。斜率为 $A_t$ 的段（含真实 $r_t$）提供纠偏梯度；斜率为 $0$ 的段（裁剪水平段）让更新停止。整条曲线的斜率绝不取 $-A_t$，**这正是"梯度永不反向"的几何写照**。

由链式法则 $\nabla_\theta L = \frac{\partial L}{\partial r_t} \cdot \nabla_\theta r_t$，结合 $r_t(\theta) = \pi_\theta(a_t\mid s_t) / \pi_{\text{old}}(a_t\mid s_t)$（$\pi_{\text{old}}$ 为正常数）：

- $\frac{\partial L}{\partial r_t} = A_t > 0$ 时，$\nabla_\theta L$ 沿增大 $\pi_\theta(a_t\mid s_t)$ 的方向——好动作被加强。
- $\frac{\partial L}{\partial r_t} = A_t < 0$ 时，$\nabla_\theta L$ 沿减小 $\pi_\theta(a_t\mid s_t)$ 的方向——坏动作被抑制。
- $\frac{\partial L}{\partial r_t} = 0$ 时，梯度为零，更新停止——且这只发生在策略已沿正确方向越界的情形，停止是设计意图。

**PPO 的每次更新要么停止，要么沿优势指示的正确方向进行——永远不会反向，也永远不会在需要纠偏时卡死。**

</details>

四种情形中，$\min$ 始终选取"更悲观"（数值更小）的那一项：顺方向越界时 clip 砍掉虚高的奖励，裁剪值更悲观；反方向越界时未裁剪值如实地暴露奖励确实很低，未裁剪值更悲观。**"更悲观"恰好等于"梯度方向正确"**——这是上述证明的直觉版本。

> 若将 $\min$ 替换为 $\max$，规则变为"取更乐观"：顺方向越界时奖励不被砍（鼓励继续越界），反方向越界时奖励被高估（错误方向被奖励）。两种情形都失效，$\min$ 不可替换为 $\max$。

## 裁剪机制的可视化

以下代码用于直观展示裁剪目标函数的行为：

```python
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 可视化 PPO Clip 目标函数
# ==========================================
epsilon = 0.2
r = np.linspace(0.0, 2.0, 500)  # 策略比率 r_t(θ)

def ppo_clip_objective(r, A, eps=0.2):
    """PPO 裁剪目标：L = min(r * A, clip(r, 1-eps, 1+eps) * A)"""
    r_clipped = np.clip(r, 1 - eps, 1 + eps)
    return np.minimum(r * A, r_clipped * A)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# A > 0 的情况
A_pos = 1.0
obj_pos = ppo_clip_objective(r, A_pos)
ax1.plot(r, r * A_pos, 'b--', alpha=0.5, label='未裁剪: r × A')
ax1.plot(r, obj_pos, 'r-', linewidth=2, label='PPO: min(r×A, clip(r)×A)')
ax1.axvline(x=1+epsilon, color='gray', linestyle=':', label=f'1+ε={1+epsilon}')
ax1.axvline(x=1-epsilon, color='gray', linestyle=':', label=f'1-ε={1-epsilon}')
ax1.set_title('A > 0（好动作）')
ax1.set_xlabel('策略比率 r_t(θ)')
ax1.set_ylabel('目标值')
ax1.legend()

# A < 0 的情况
A_neg = -1.0
obj_neg = ppo_clip_objective(r, A_neg)
ax2.plot(r, r * A_neg, 'b--', alpha=0.5, label='未裁剪: r × A')
ax2.plot(r, obj_neg, 'r-', linewidth=2, label='PPO: min(r×A, clip(r)×A)')
ax2.axvline(x=1+epsilon, color='gray', linestyle=':', label=f'1+ε={1+epsilon}')
ax2.axvline(x=1-epsilon, color='gray', linestyle=':', label=f'1-ε={1-epsilon}')
ax2.set_title('A < 0（坏动作）')
ax2.set_xlabel('策略比率 r_t(θ)')
ax2.legend()

plt.suptitle('PPO Clip 目标函数行为（ε=0.2）', fontsize=14)
plt.tight_layout()
plt.savefig("ppo_clip_visualization.png", dpi=150)
print("裁剪函数可视化已保存")
```

运行结果如下：当 $A > 0$ 时，目标函数在 $r_t > 1.2$ 后趋于平坦（梯度为零，更新停止）；当 $A < 0$ 时，目标函数在 $r_t < 0.8$ 后趋于平坦。这正是 PPO 裁剪的核心效果——策略比率超出安全区间后，梯度自动消失。

## $\epsilon$ 的敏感性

$\varepsilon$ 的选择直接影响训练效果，以下为经验性总结：

| $\epsilon$ 值 | 更新幅度 | 训练速度   | 稳定性   | 适用场景                   |
| ------------- | -------- | ---------- | -------- | -------------------------- |
| 0.05          | 很小     | 很慢       | 极其稳定 | 精调已训练好的策略         |
| 0.1           | 较小     | 较慢       | 稳定     | LLM 对齐（参数多，更脆弱） |
| 0.2           | 中等     | 适中       | 适中     | 游戏/控制任务（默认值）    |
| 0.3           | 较大     | 较快       | 不稳定   | 快速实验/简单任务          |
| 0.5           | 很大     | 快但容易崩 | 很不稳定 | 不推荐                     |

在 LLM 对齐场景中，通常使用更小的 $\varepsilon$（0.1 甚至更小），因为语言模型的策略空间更大、更脆弱，一次不恰当的更新可能导致语言能力退化（例如丧失已习得的语种能力）。

<details>
<summary>思考题：如果 PPO 的裁剪让训练"太保守"，有没有办法在不牺牲稳定性的前提下加快训练？</summary>

有几个常见的策略：

1. **自适应 ε**：PPO-PPG（Phasic Policy Gradient）建议在训练早期用较大的 ε，后期逐渐缩小。类似"先大步探索，再小步精调"。
2. **增加更新轮数**：PPO 默认用同一批数据更新 10 个 epoch。如果裁剪让每步更新很小，可以通过增加 epoch 数来累积更新量。
3. **KL 散度早停**：同时监控 KL 散度，如果在某个 epoch 内 KL 超过阈值就停止更新——这相当于把 TRPO 的思想和 PPO 的裁剪结合了起来。

在实践中，第 2 种方法最常用——PPO 默认的 `n_epochs=10` 本身就是为了在裁剪限制下通过多轮累积来实现足够的更新量。

</details>

<details>
<summary>思考题：TRPO 理论上更严谨，为什么工业界几乎都选 PPO？</summary>

因为在工程实践中，"简单可靠"几乎总是打败"理论完美"。TRPO 需要计算二阶导数（Hessian 向量积），这在大规模模型上非常慢，而且实现复杂，容易出 bug。PPO 只需要一个简单的 `torch.clamp` 操作，实现不到 10 行代码。

OpenAI 在 2017 年的论文中用大量实验证明：PPO 在大多数任务上的表现与 TRPO 相当甚至更好。原因是 TRPO 的二阶近似本身也有误差，精确求解并不一定比 PPO 的启发式裁剪更好。

这个选择在 LLM 时代更加正确——70B 参数的语言模型，二阶优化根本不可行。OpenAI 自己在 InstructGPT 和 GPT-4 的对齐训练中也使用的是 PPO，而不是 TRPO。

</details>

至此，PPO 裁剪机制的完整推导已展开：从原始策略梯度的更新风险，到重要性采样的数据复用，再到 TRPO 的 KL 约束与 PPO 的裁剪近似。但 PPO 尚有另一个关键组件未涉及：GAE（广义优势估计），以及它在 LLM 对齐中引出的主要负担——奖励模型。详见 [优势估计与奖励建模](./gae-reward-model)。
