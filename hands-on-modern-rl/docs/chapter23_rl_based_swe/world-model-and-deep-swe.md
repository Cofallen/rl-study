# 20.2 Code World Model 与 DeepSWE

一个 Agent 修改文件后运行 `pytest`，终端返回 30 行报错。下一步，它必须从报错、当前补丁和仓库状态中推断“刚才的修改造成了什么变化”。如果模型只会写代码，却不会预测动作后果，就会反复尝试相似补丁。

本节并列阅读两项容易被混为一谈的工作。**CWM** 是在解释器与 Docker Agent 轨迹上进行中期训练和多任务 RL 的 32B 开放权重模型，研究模型能否学到软件执行规律。**DeepSWE** 则把 Qwen3-32B 放进真实 R2E-Gym 环境，用可验证结果和纯 RL 训练长程代码 Agent。前者强调从环境轨迹中学习软件世界，后者强调在真实环境中扩展 rollout 与优化。

<img src="./images/cwm-vs-deepswe.svg" alt="CWM 与 DeepSWE 从共同的软件执行环境出发，分别学习环境规律和训练长程代码智能体">

## CWM 的学习对象

[CWM 论文](https://arxiv.org/abs/2510.02387)发布了一个 32B dense decoder-only 模型。它先读取 Python 解释器和 Agentic Docker 环境中的“观察与动作”轨迹，再在可验证代码、数学和多轮软件工程环境中做推理 RL。这里的 world model 首先表现为模型权重中学到的执行规律，并不等同于一个已经取代真实测试的独立转移模型。

### 用 MDP 记号整理交互轨迹

为了理解一条软件 Agent 轨迹，可以先用 MDP 记号整理四类对象：状态 $s_t$ 是仓库代码、当前修改历史与测试结果；动作 $a_t$ 是模型的下一步，如读文件、改代码、跑测试；转移 $T(s_{t+1} \mid s_t, a_t)$ 描述代码执行后状态的变化；奖励 $r_t$ 包含中间状态的每步反馈与测试通过的最终 reward。

### 显式预测下一状态

CWM 的论文重点是用环境轨迹中期训练同一个语言模型。原稿还提出了一个更显式的 model-based RL 扩展：另外训练转移模型 $\hat{T}$ 来预测下一状态。下面保留这个公式和实现思路，并明确把它当作教学推演：

$$\hat{T}(s_{t+1} | s_t, a_t) \approx T(s_{t+1} | s_t, a_t)$$

在该扩展中，world model 可以是另一个 LLM，输入 $(s_t, a_t)$，输出对 $s_{t+1}$ 的预测。

训练数据：

- 从真实 SWE 任务收集 trajectory
- $(s_t, a_t, s_{t+1})$ 三元组作为训练样本
- 让 world model 学会"给定当前代码状态和动作，预测下一步状态"

### 显式 world model 的三阶段训练设想

```text
┌────────────────────────────────────────────────────────────┐
│ Phase 1: World Model 预训练                                │
│   - 从真实 SWE 任务收集 trajectory                         │
│   - 训练 world model 预测代码状态变化                       │
├────────────────────────────────────────────────────────────┤
│ Phase 2: RL with World Model                              │
│   - Policy 与 world model 交互                            │
│   - World model 快速模拟"代码执行"                          │
│   - 减少部分真实测试调用（需实测加速比）                  │
├────────────────────────────────────────────────────────────┤
│ Phase 3: 真实测试 fine-tune                                │
│   - 用 world model 训练后的 policy 在真实环境做最后 RL      │
│   - 修正 world model 与真实环境的偏差                      │
└────────────────────────────────────────────────────────────┘
```

### 这种扩展可能带来的收益

**优势一：速度快**

如果预测模型足够准确，一次前向计算可能比安装依赖、构建仓库和运行完整测试更快。不过原稿中的“100 到 1000 倍”不是 CWM 论文给出的通用实测值；速度取决于模型规模、测试耗时和批处理方式。

**优势二：可以模拟失败**

World model 可以模拟"如果这样改，会发生什么"，policy 可以在 world model 里大量探索失败模式，学习避免。

**优势三：数据效率高**

World model 学到代码执行的"规律"，这些规律可以泛化到新任务。

### 为什么它仍不能替代真实执行

**局限一：World model 的准确性**

World model 是个 LLM，会错。如果它预测错了"代码执行结果"，policy 学到错误的策略。

一种缓解方式是定期回到真实测试校正预测模型，例如每隔若干次模拟 rollout 抽取一批状态执行 ground truth。校正频率越高，环境成本越大；频率越低，模型误差越容易累积。

**局限二：复杂依赖**

代码执行涉及复杂依赖（库版本、环境变量、外部服务）。World model 难以完全模拟这些。

**局限三：训练成本**

训练 world model 本身需要大量 trajectory 数据和算力，比直接训练 policy 复杂。

### CWM 与 model-based RL 的关系

CWM 为软件 world modeling 提供了开放权重研究载体。把它进一步接成“策略 → 显式转移模型 → 搜索”的 model-based RL 系统，是自然的研究方向，但需要单独验证多步预测误差和真实测试上的收益。

参考：[第 8 章长程任务中的模型规划](../chapter10_ppo/rl-long-horizon-planning)和[24.3 VLA 与具身世界模型](../chapter28_vla/embodied-intelligence/model-based-rl/)。

## DeepSWE 的长程训练

[DeepSWE-Preview](https://www.together.ai/blog/deepswe) 由 Agentica 与 Together AI 合作训练。项目从 Qwen3-32B 出发，在约 4,500 个 R2E-Gym 软件工程任务上运行六天、使用 64 张 H100，通过 rLLM 进行纯 RL 训练。官方报告的 SWE-bench Verified 结果是 42.2% Pass@1；加入测试时扩展后约为 59%。

### DeepSWE 的核心思路

DeepSWE 说明，只要环境、rollout 基础设施与可验证奖励足够稳定，长程软件 Agent 可以仅靠 RL 获得显著提升。官方材料还介绍了 trajectory-level 与 step-level GRPO/PPO 的系统支持，以及为测试时扩展训练的验证器。

原稿把三个常见的长程 RL 方案，即步骤 shaping、value model 和分层策略，写成了 DeepSWE 的正式结构。下面保留这些公式和代码，作为理解信用分配的**备选教学方案**，不再归因于 DeepSWE 的公开配方。

**备选方案一：Step-level Reward Shaping**

不是只有最终 reward，而是给每步一个 shaping reward：

```python
def deep_swe_reward(trajectory, final_test_result):
    # 基础 reward：最终测试结果
    base_reward = 1.0 if final_test_result else 0.0

    # Shaping reward：每步的"贡献度"
    step_rewards = []
    for step in trajectory:
        # 用 LLM judge 评估这一步是否"有意义"
        step_quality = llm_judge(step)
        step_rewards.append(step_quality)

    # 总 reward = base + sum(step rewards)
    return base_reward + sum(step_rewards) * 0.1
```

这种 shaping 让模型每步都得到反馈，但 LLM judge 也会引入噪声和可钻的漏洞。它是否优于终态可验证奖励，需要消融实验回答。

**备选方案二：Value Model**

可以重新引入 value model（与 VAPO 思路相关），参考[第 15 章 VAPO](../chapter18_grpo/grpo-family)。

Value model $V_\phi(s_t)$ 估计当前状态的"未来 reward 期望"。这让 RL 可以用 GAE 做 credit assignment：

$$\hat{A}_t = \delta_t + (\gamma\lambda)\delta_{t+1} + \ldots$$

其中 $\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$。

**备选方案三：Hierarchical RL**

把长 trajectory 分层：

- **高层 policy**：决定"接下来要修哪个文件"（粗粒度）
- **低层 policy**：决定"具体怎么改这个文件"（细粒度）

高层用稀疏 reward（最终测试），低层用密集 reward（每步 shaping）。

### 公开训练主线

```text
┌──────────────────────────────────────────────────────────┐
│ Phase 1: 准备环境与训练任务                              │
│   - R2E-Gym 中约 4,500 个真实软件工程任务               │
│   - 为每个任务准备可执行环境与验证器                     │
├──────────────────────────────────────────────────────────┤
│ Phase 2: 生成长程 rollout                                │
│   - Qwen3-32B 读取、编辑并执行工具                       │
│   - 环境返回测试与命令结果                               │
├──────────────────────────────────────────────────────────┤
│ Phase 3: 纯 RL 更新                                      │
│   - 通过 rLLM 进行 agentic RL                            │
│   - 用可验证任务结果更新策略                             │
├──────────────────────────────────────────────────────────┤
│ Phase 4: 记录并检查训练                                  │
│   - 发布训练、评测与 Weights & Biases 日志               │
│   - 比较训练前后 Pass@1                                 │
├──────────────────────────────────────────────────────────┤
│ Phase 5: Test-time scaling                              │
│   - 生成多个候选轨迹                                     │
│   - 用执行式或非执行式 verifier 选择候选                 │
└──────────────────────────────────────────────────────────┘
```

### DeepSWE 的成绩

下面保留原稿记录的对比项，同时把 DeepSWE 的公开口径改正。不同系统的采样预算和 scaffold 不同，数字只能在注明设置后比较：Simple GRPO 以 Meta SWE-RL 为代表，开源且简单，SWE-bench Verified 为 41.0%；DeepSWE-Preview（Agentica 与 Together AI）为 42.2% Pass@1，测试时扩展后约 59%；SWE-Lancer（OpenAI）为 45.0%；Claude Opus 4.5 加工具在 60% 以上。

DeepSWE 从训练前约 23% 提升到约 42% Pass@1，说明可执行环境中的纯 RL 能显著改善长程代码 Agent。测试时多采样后的更高分数应单独标注，不能与单次采样混为一列。

### DeepSWE 与 value-based 方案的边界

DeepSWE 的公开主线不能概括为“字节 VAPO 在 SWE 上的延伸”：项目团队与训练配方都不同。Value model 仍然是值得比较的基线，因为它能估计中间状态的未来回报；是否需要 critic，应由同环境、同 rollout 预算下的实验决定。

## 候选搜索的教学扩展

DeepSWE 的公开结果包含多候选生成与 verifier 选择形式的测试时扩展。原稿进一步给出了 MCTS 与 Beam Search 两段伪代码。它们适合用来理解“怎样比较多个候选未来”，但不是 CWM 或 DeepSWE 已公开实现的逐字复现。

### 在显式转移模型上尝试 MCTS

如果我们另外训练了前文的显式转移模型，就可以尝试在预测状态上运行 MCTS：

```python
def cwm_mcts(issue, model, world_model, depth=10):
    # 在 world model 上做 MCTS
    root_state = initialize_state(issue)

    for _ in range(N_iter):
        # Selection: 用 UCB 选最优子节点
        node = select(root_state)

        # Expansion: 用 policy 生成动作，用 world model 模拟下一状态
        action = model.policy(node.state)
        next_state = world_model.predict(node.state, action)

        # Simulation: 快速 rollout 到终止
        rollout_reward = quick_rollout(next_state, world_model)

        # Backprop: 更新节点统计
        backpropagate(node, rollout_reward)

    # 返回 root 的最优动作
    return best_action(root_state)
```

这段 MCTS 在预测模型中展开，因此减少了搜索阶段的真实测试调用。最终候选仍应回到真实环境验证，否则搜索会偏向 world model 的预测漏洞。

### 用 value model 演示 Beam Search

如果备选方案中训练了 value model，可以用下面的 Beam Search 比较中间状态：

```python
def deep_swe_beam_search(issue, model, value_model, K=4):
    beams = [{"state": init_state(issue), "score": 0}]

    for step in range(MAX_STEPS):
        candidates = []
        for beam in beams:
            # 生成 K 个候选动作
            actions = model.generate_actions(beam["state"], n=K)

            for action in actions:
                next_state = apply_action(beam["state"], action)
                # 用 value model 评估
                value = value_model.estimate(next_state)
                candidates.append({
                    "state": next_state,
                    "score": beam["score"] + value
                })

        # 选 top-K
        beams = sorted(candidates, key=lambda x: x["score"], reverse=True)[:K]

    return beams[0]["state"]
```

Beam Search 展示了用更多推理算力换取候选覆盖率的基本方式，与 [第 16 章 Test-time Compute Scaling](../chapter19_reasoning/test-time-scaling) 相连。是否提升最终准确率，取决于候选多样性与 value model 的排序质量。

## 四类路线的公平比较

原稿试图用一条分数链比较四种方案，但其中“CWM 约 45%”和“DeepSWE 50%”与原始材料不一致。下面保留四类观察对象，并改成可由一手来源核对的口径：Simple GRPO 以 Meta SWE-RL 为代表，开源且简单，SWE-bench Verified 为 41.0%；加上 World Model 以 Code World Model 为代表，用环境轨迹中期训练加多任务 RL，论文报告 65.8%（含测试时扩展）；加上 Value 与 Search 以 DeepSWE 为代表，用 R2E-Gym 长程纯 RL 加测试时扩展，为 42.2% Pass@1，扩展后约 59%；多 agent 协作以商业 Agent 工作流为代表，训练与编排细节通常不公开，分数需按具体模型版本与评测日志记录。

这些分数不能推出“算法越复杂，性能必然越高”。模型规模、训练任务、工具接口、最大步数、测试时采样数和 verifier 都会影响结果。公平实验至少要固定基座模型、环境、单题预算和 Pass@k 口径。

## 小结

这一节区分了两条相关但不同的路线：

- **CWM**：用解释器与 Docker 交互轨迹进行中期训练和多任务 RL，让模型学习软件执行规律
- **DeepSWE**：在 R2E-Gym 真实环境中扩展纯 RL，训练 Qwen3-32B 长程代码 Agent
- **教学扩展**：显式转移模型、value model、分层策略、MCTS 与 Beam Search，仍需各自的消融与真实执行验证

两项工作共同说明，长程任务首先需要稳定、可重置、可并行的执行环境。更复杂的算法只有在同一环境与预算下证明收益后，才能归因于算法本身。

下一节看 Self-play SWE-RL，让模型自己生成训练数据，进一步降低对人工数据的依赖。
