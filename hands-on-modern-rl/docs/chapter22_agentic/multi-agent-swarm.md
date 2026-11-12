# 19.7 多智能体协作

> [19.6 Code Interpreter RL](./industrial-practice) 训练了**单个** agent 在工具调用循环里完成编程任务。但当任务从"写一个函数"升级到"重构整个代码库 + 跑测试 + 写文档 + 提 PR"，单个 agent 的上下文窗口、注意力带宽、错误恢复能力都会被压垮。**多智能体协作**（multi-agent collaboration）是 agentic RL 在 2025-2026 年的关键扩展：把一个复杂任务拆给多个 agent，每个 agent 专注一个子任务，通过显式通信协议协调。本节讲清楚三件事：(1) LLM-era 多智能体与经典 MARL 的根本差异；(2) 主流协作范式（Orchestrator-Worker、Debate、Swarm）；(3) 多 agent 系统的 RL 训练方法。

## 从经典 MARL 到 LLM-era 多智能体

[第 12 章 12.2 节](../chapter14_exploration_marl_hierarchical/marl)讲过经典 MARL：CTDE 框架、MADDPG、MAPPO。这些算法处理的是**同构** agent 在**固定**环境里学习**纳什均衡**——例如多个机器人追逃、多智能体 StarCraft 微操。LLM-era 多智能体完全不同：

| 维度         | 经典 MARL            | LLM-era 多智能体                           |
| ------------ | -------------------- | ------------------------------------------ |
| Agent 数量   | 2-20 个              | 2-10 个（受成本约束）                      |
| Agent 异构性 | 同构（同一策略）     | 高度异构（不同角色：planner/coder/tester） |
| 通信方式     | 通过环境状态隐式通信 | 自然语言显式通信                           |
| 任务类型     | 零和博弈 / 合作博弈  | 长程软件任务（PR、研究、运维）             |
| 训练目标     | 纳什均衡 / 团队回报  | 任务完成率（端到端可验证）                 |
| 训练算法     | MAPPO / QMIX         | GRPO + 多轨迹奖励分配                      |

最关键的差异是**异构性**与**显式通信**。经典 MARL 里所有 agent 共享策略 $\pi_\theta(a \mid s)$，只通过环境状态互相影响。LLM 多智能体里每个 agent 有不同的 system prompt（"你是代码审查员"、"你是测试工程师"），通过**自然语言消息**互相协调。这让通信带宽爆炸式增长——一次协调可能消耗几千 token——但也让协作的语义丰富得多。

## 三种主流架构

### Orchestrator-Worker 模式

**最简单也最常用**的协作范式。一个 **Orchestrator agent**（编排者）负责任务分解、子任务分发、结果聚合；多个 **Worker agent**（工作者）各自执行子任务。

```
[User: "修复 GitHub Issue #123"]

    ↓
[Orchestrator]
    ├── 1. 读 issue → 调用 Worker-A: "定位 bug 文件"
    ├── 2. Worker-A 返回 file.py:42
    ├── 3. 调用 Worker-B: "在 file.py:42 写修复补丁"
    ├── 4. Worker-B 返回 patch.diff
    ├── 5. 调用 Worker-C: "跑测试 + 写 changelog"
    └── 6. 聚合 → 提交 PR
```

Anthropic 2025 年发布的内部研究测得：**Orchestrator-Worker 模式相比单 agent 在 SWE-bench Verified 上加速 90.2%，成功率提升 18-32%**。关键原因不是"两个 agent 比一个强"，而是**任务分解让每个 agent 的上下文窗口不被压垮**——单 agent 处理整个 PR 流程时，注意力分散在"找文件 / 写代码 / 跑测试 / 写文档"四件事上；分解后每个 worker 只需关注一件事。

Orchestrator 的策略可以形式化为分层 MDP：

$$\pi_\theta^{\text{orch}}(w_t, m_t \mid q, h_{1:t})$$

其中 $w_t \in \{1, \ldots, K\}$ 是第 $t$ 步派给哪个 worker，$m_t$ 是发给该 worker 的消息，$h_{1:t}$ 是历史交互。

### Debate 模式

多个 agent **互相辩论**以收敛到更可靠的答案。Anthropic 的 AI Safety via Debate（Irving et al. 2018）是这个范式的理论基础；DeepMind 2024 年的 Scaling Inference 论文验证了 LLM Debate 在数学题上的效果。

Debate 的 MDP：

$$\pi_\theta(a_t^{(i)} \mid q, a_{1:t-1}^{(1)}, a_{1:t-1}^{(2)}, \ldots, a_{1:t-1}^{(K)})$$

第 $i$ 个 agent 看到所有其他 agent 的历史发言，输出本轮回应 $a_t^{(i)}$。最终答案由**外部 judge**（人或另一个 LLM）选择。

Debate 的训练目标是**真理收敛**：让诚实 agent 在多轮辩论后胜出。这比 Orchestrator-Worker 难训得多——需要**对抗训练**（adversarial training）：故意训练一个"撒谎 agent"，再训一个"诚实 agent"打败它。

### Agent Swarm 模式

**Kimi K2.5（2026.01）**和**Step 3.7 Flash Advisor Mode** 把多 agent 推到极致：**几十个异构 agent** 同时在线，由一个 meta-controller 动态调度。这本质上是 A2A（Agent-to-Agent）协议 + RL 调度器。

Swarm 的关键差异：

- **Agent 池**而非固定 worker 集合：meta-controller 根据任务从池中动态选 agent
- **A2A 通信协议**：agent 之间通过结构化协议（如 Anthropic A2A、OpenAI Function Calls）通信
- **信用分配跨 agent**：哪个 agent 贡献最大？需要 SHAP 或注意力归因

形式化：

$$\pi_\theta^{\text{swarm}}(a_t \mid q, \text{pool}, h_{1:t})$$

其中 $a_t = (\text{select-agent}, \text{message}, \text{route-to})$。

::: warning Swarm 的成本爆炸
Swarm 模式的 token 消耗是单 agent 的 10-50 倍。Kimi K2.5 论文报告：处理一个 SWE-bench 任务平均消耗 280K token（单 agent baseline 是 18K）。这是为什么 2026 年工业落地仍以 Orchestrator-Worker 为主——成本可控，效果接近 Swarm。
:::

## 多 agent 系统的 RL 训练

### 从团队回报到个体归因

多 agent RL 最棘手的问题是**信用分配**（credit assignment）。任务成功了，谁该得奖励？

**方案 1：团队回报平均分配**（team-average）

所有 agent 拿到相同奖励 $r / K$（$K$ 是 agent 数）：

$$R^{(i)} = \frac{1}{K} \sum_t r_t$$

简单但容易产生**搭便车**（free-rider）问题：某个 worker 偷懒，团队仍能成功，该 worker 同样拿到奖励。

**方案 2：Shapley value 归因**

博弈论里的 Shapley value 衡量每个 agent 的边际贡献：

$$\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(N - |S| - 1)!}{N!} [v(S \cup \{i\}) - v(S)]$$

其中 $v(S)$ 是子集 $S$ 完成任务的成功率。$N!$ 项需要**反事实评估**（counterfactual evaluation）——把 agent $i$ 从团队移除，看任务还能否完成。计算成本高，但归因最公平。

**方案 3：Orchestrator 显式分配**（heuristic）

Orchestrator 在最终回报里输出权重 $w_i$，agent $i$ 的奖励是 $w_i \cdot R$：

$$R^{(i)} = w_i \cdot R^{\text{team}}, \quad \sum_i w_i = 1$$

这是 Kimi K2.5 实际用的方案——便宜且可解释，但依赖 Orchestrator 的归因能力（本质上是 RL 训练 Orchestrator 学会归因）。

### 多轨迹 GRPO

标准 GRPO 对同一 prompt 采样 $G$ 条轨迹，归一化优势：

$$\hat{A}_j = \frac{R_j - \text{mean}(R_{1:G})}{\text{std}(R_{1:G})}$$

多 agent 版本叫 **Multi-Agent GRPO（MA-GRPO）**：每条轨迹不是单 agent 生成，而是**整个团队协作生成**。$G$ 条轨迹 = $G$ 次团队协作。

关键工程实现：

```python
def ma_grpo_step(prompts, team_size):
    # 对每个 prompt，采样 G 条团队协作轨迹
    trajectories = []
    for prompt in prompts:
        for g in range(G):
            # 1. Orchestrator 分解任务
            subtasks = orchestrator.decompose(prompt)
            # 2. Workers 并行执行
            worker_outputs = [workers[i](subtasks[i]) for i in range(team_size)]
            # 3. Orchestrator 聚合
            final_answer = orchestrator.aggregate(worker_outputs)
            # 4. 计算 reward
            r = verifier(prompt, final_answer)
            trajectories.append({
                'prompt': prompt,
                'final': final_answer,
                'reward': r,
                'orch_logp': orchestrator.logp(...),
                'worker_logp': [w.logp(...) for w in workers]
            })

    # GRPO 优势归一化
    rewards = [t['reward'] for t in trajectories]
    advantages = (rewards - mean(rewards)) / (std(rewards) + eps)

    # 分别对 orchestrator 和 workers 算 loss
    orch_loss = -mean(a * t['orch_logp'] for a, t in zip(advantages, trajectories))
    worker_losses = [-mean(a * lp for a, lp in zip(advantages, t['worker_logp']))
                     for t in trajectories]

    total_loss = orch_loss + sum(worker_losses)
    return total_loss
```

注意三个工程细节：

1. **Orchestrator 和 Workers 共享优势 $a$**——团队成败是统一的信号
2. **三者一起更新**（joint update），而非交替更新——避免非平稳性问题
3. **必须用 group-normalized 优势**——否则某个 agent 学得快，其他 agent 的梯度被淹没

## Kimi K2.5 与 Step 3.7

### Kimi K2.5 的 Agent Swarm

Kimi K2.5（2026.01，arXiv:2602.02276）是首个公开 Swarm 模式训练细节的工业模型：

- **Agent 池**：32 个异构 agent（coder、tester、planner、reviewer、debugger 等）
- **A2A 协议**：基于 JSON Schema 的结构化消息
- **训练数据**：12M 条团队协作轨迹，覆盖 SWE / DeepResearch / Customer Service
- **奖励**：可验证任务用 RLVR，开放任务用 LLM-as-Judge
- **调度 RL**：meta-controller 用 PPO 训练，目标是最小化 token 消耗 + 最大化成功率

报告指标：

- SWE-bench Verified：68.3%（单 agent baseline 49.1%）
- BrowseComp：72.1%（单 agent 51.4%）
- 平均 token 消耗：280K（baseline 18K，15.6×）

### Step 3.7 Flash Advisor Mode

Step 3.7 Flash 的 Advisor Mode 走不同路线：**保守的 Orchestrator-Worker**，但加入**Advisor agent**专门做"反思与纠错"。

```
[Orchestrator] → [Worker-A: code] → [Advisor: review] → [Orchestrator] → [Worker-B: test]
```

Advisor 不直接执行任务，只对 Worker 输出做评论。Orchestrator 看到 Advisor 评论后决定是否返工。这种"哑铃式"协作成本只有 Swarm 的 1/5，但效果接近。

报告指标：

- SWE-bench Verified：62.4%（介于单 agent 和 Swarm 之间）
- 平均 token 消耗：52K（约 Swarm 的 1/5）

## 与 [第 26 章 自我博弈](../chapter32_selfplay/self-play-outlook/) 的呼应

多智能体协作有一个特殊形态：**多个 agent 是同一个 policy 的不同实例**，互相博弈。这就是 AlphaGo / AlphaZero / Constitutional AI Self-Critique 的核心思想。详见 [第 26 章 自我博弈](../chapter32_selfplay/self-play-outlook/)。

关键区别：

- **多智能体协作**：异构 agent，显式通信，团队任务
- **自我博弈**：同构 agent（同一 policy），通过环境交互，零和或合作博弈

两者在 LLM 时代开始融合——例如 Constitutional AI 的 Self-Critique 可以看作"两个 agent 协作（一个生成、一个批评），但用同一个 policy"。

## 多智能体协作的失败模式

理论讲完，回到工程——多智能体系统在生产环境的几种典型失败模式。

### 通信放大错误

单 agent 出错只影响自己；多 agent 系统里，一个 agent 的错误输出会成为其他 agent 的输入，错误被指数放大。

```
Worker-A (误) → 输出 "bug 在 file_X.py:42"
    ↓
Orchestrator 派 Worker-B 修复 file_X.py:42
    ↓
Worker-B 修复了一个不存在的 bug，引入新 bug
    ↓
Orchestrator 派 Worker-C 测试，发现新 bug
    ↓
...无限循环...
```

Anthropic 内部数据：多 agent 系统的"连锁错误率"是单 agent 的 2.7 倍。

**对策**：每个 agent 输出时附带**置信度**；低置信度输出触发 Orchestrator 的二次核验。

### 群体思维

多个 agent 互相影响后，可能收敛到错误共识——尤其在 Debate 模式下。如果一个 agent 用了错误的前提，其他 agent 可能基于"礼貌"或"从众"接受这个前提。

**对策**：引入"魔鬼代言人"（Devil's Advocate）agent——专门反驳主流观点。Anthropic 的 Debate 系统强制至少一个 agent 持反对立场。

### 搭便车（Free Rider）

团队回报平均分配时，某个 worker 学会"做最小贡献"——只输出看似合理但无实质内容的回应，团队仍能成功。

**对策**：

- Shapley value 归因（计算成本高）
- Orchestrator 显式打分（依赖 Orchestrator 能力）
- 测试时单独评估每个 worker（最严格但最贵）

### 上下文重复

多 agent 系统里，每个 worker 都需要"了解全局"才能工作。但全局信息（任务描述、已有进展）在每个 worker 的 prompt 里都重复一遍——token 成本爆炸。

```
任务: "修复 GitHub Issue #123"
上下文（每个 worker 都看到）:
  - Issue 完整描述: 500 token
  - 相关代码文件: 2000 token
  - 已有 worker 的进展: 1500 token
合计: 4000 token × 5 workers = 20K token 仅上下文
```

**对策**：分层上下文——Orchestrator 维护完整上下文，worker 只看精简摘要。

## 开源框架与工具

复现多智能体 RL 训练，有以下开源工具：

| 框架             | 来源        | 特点                                      |
| ---------------- | ----------- | ----------------------------------------- |
| **AutoGen**      | Microsoft   | 多 agent 对话框架，支持多种协作模式       |
| **CrewAI**       | CrewAI Inc. | 角色化 agent（planner/researcher/writer） |
| **MetaGPT**      | DeepWisdom  | SOP（标准作业流程）驱动的多 agent         |
| **LangGraph**    | LangChain   | 基于状态图的多 agent 编排                 |
| **Agency Swarm** | VRSEN       | 字面意义的"agent swarm"开源实现           |

但这些框架大多是**推理时**（inference-time）工具——它们定义 agent 之间怎么对话，不涉及 RL 训练。**真正能用 RL 训练多 agent 系统的开源框架极少**，主要是：

- **OpenRLHF**（字节系）：支持多 agent 的 PPO/GRPO，可自定义 reward 分配
- **verl**（字节系）：分布式 RL 框架，支持异构 agent 联合训练
- **OpenResearcher**：Deep Research 专用，含简单的 Orchestrator-Worker

工业级 Swarm 训练（如 Kimi K2.5）目前**没有完整开源实现**——这仍是中国/美国头部 lab 的核心壁垒。

## 本节总结

| 范式                | 通信方式   | 训练目标        | 代表系统             | 成本   |
| ------------------- | ---------- | --------------- | -------------------- | ------ |
| 单 agent            | N/A        | 任务完成率      | baseline             | 1×     |
| Orchestrator-Worker | 单向派发   | 团队回报        | Anthropic 内部       | 3-5×   |
| Debate              | 双向辩论   | 真理收敛        | Anthropic / DeepMind | 5-10×  |
| Agent Swarm         | 全连接 A2A | 团队 + 个体归因 | Kimi K2.5            | 15-30× |

LLM-era 多智能体的 RL 训练核心挑战：**信用分配**与**token 成本**。前者决定训练能否收敛，后者决定能否商业化。2026 年主流是 Orchestrator-Worker + 显式归因，Swarm 仍处于研究阶段。

下一章 [第 20 章 代码智能体强化学习](../chapter23_rl_based_swe/swe-bench-and-rlvr) 把这套协作框架用在 SWE 任务上——你会看到 SWE-Agent 如何用 Orchestrator-Worker 训练单 agent 代码智能体，DeepSWE 如何用 self-play 训练多 agent 协作开发。
