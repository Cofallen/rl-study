# 补充阅读：Meta SWE-RL 与开源 SOTA

[Meta SWE-RL](https://arxiv.org/abs/2502.18449)（2025.02）是开源 SWE-RL 的代表作。它的核心贡献是：

- 用开源数据（SWE-bench + SWE-gym）训练
- 用最简单的 GRPO + 测试 reward
- 在 SWE-bench Verified 上达到 41.0%（开源 SOTA）

这一节我们详细看 Meta SWE-RL 的数据、算法、工程细节。

## 20.1.1 数据规模与构成

Meta SWE-RL 的训练数据来源：

| 数据源               | 规模       | 用途            |
| -------------------- | ---------- | --------------- |
| SWE-bench（开源）    | 2,294 条   | 高质量 baseline |
| SWE-gym（开源）      | 6,800 条   | 扩展训练        |
| 内部 PR 数据（Meta） | 80,000+ 条 | 大规模 RL       |

**总计约 90,000 条 SWE 任务**——比纯 SWE-bench 大 40 倍。

### 数据预处理

Meta 报告了几个数据清洗的关键步骤：

**步骤一：仓库过滤**

- 排除：测试覆盖率 < 50% 的仓库（无法可靠验证）
- 排除：维护不活跃的仓库（last commit > 6 months ago）
- 排除：Python < 3.8 的仓库（与最新依赖不兼容）

**步骤二：PR 过滤**

- 排除：修改超过 10 个文件的 PR（太复杂，不适合 RL 早期训练）
- 排除：纯依赖更新的 PR（不是真正的"修 bug"）
- 排除：删除功能的 PR（与"修复"语义不符）

**步骤三：测试筛选**

- 保留：包含新增测试的 PR（有明确验证标准）
- 排除：测试无法独立运行的 PR
- 排除：测试依赖外部服务的 PR（如需要数据库、API key）

这些过滤让最终数据质量显著提升。Meta 报告，过滤前的数据训练效果差，过滤后效果大幅改善——**数据质量 > 数据数量**。

## 20.1.2 算法：GRPO + 简单奖励

Meta SWE-RL 的算法选择极其简单——**GRPO + 测试 binary reward**。

### 为什么用 GRPO？

Meta 团队在 [SWE-RL 论文](https://arxiv.org/abs/2502.18449) 里对比了 PPO、GRPO、DPO：

| 算法            | SWE-bench Verified |
| --------------- | ------------------ |
| DPO（baseline） | 24.3%              |
| PPO             | 33.2%              |
| **GRPO**        | **41.0%**          |

GRPO 优势：

- **不需要 Critic**：节省显存，适合大规模训练
- **组内归一化**：自然处理不同难度的任务（简单题组内方差大，难题组内方差小）
- **简单稳定**：工程实现比 PPO 容易

这与 [DeepSeek-R1 的发现](../chapter18_grpo/deepseek-dapo) 完全一致——**GRPO 是 SWE-RL 的默认选择**。

### Reward 函数

Meta SWE-RL 的 reward 极其简单：

```python
def swe_rl_reward(test_results):
    """测试通过 binary reward"""
    if all(test_results):
        return 1.0  # 全部通过
    else:
        return 0.0  # 任何失败
```

Meta 报告了几个 reward shaping 的尝试，结果都**比简单 binary 差**：

- **连续 reward（passed/total）**：模型容易"修一半"得到部分 reward，反而不学习完整修复
- **长度惩罚**：模型学会"用更少步骤但不修 bug"——直接 abort 任务得到 0 步
- **复杂 shaping**：reward hacking 严重

这印证了 R1-Zero 的发现：**简单 reward + 大规模 RL 是最好的组合**。

## 20.1.3 工程细节：上下文管理

SWE-RL 的 context 管理是个工程难点。一个完整的 SWE 任务可能涉及：

- 整个仓库（10K-1M 行代码）
- Issue 描述（几百 token）
- 工具调用（read file、edit、run test）
- 模型推理（CoT）

这些加起来可能超过 100K token——超过大多数模型的 context window。怎么处理？

### 工具调用替代读全文

不让模型一次读完仓库，而是让它**主动调用工具读取需要的部分**：

```text
<issue> Django filter(field__in=[]) should short-circuit </issue>

<thought>
我需要先找到这个 filter 的实现。
</thought>

<tool_call> grep -r "def filter" django/db/models/sql/query.py </tool_call>

<tool_result> def filter(self, *args, **kwargs): ... </tool_result>

<thought>
找到了。让我看看 as_sql 方法。
</thought>

<tool_call> read django/db/models/sql/query.py lines 100-150 </tool_call>

<tool_result> def as_sql(self, compiler, connection):
    ...
</tool_result>

<thought>
我看到这里没有短路逻辑。让我加一个。
</thought>

<tool_call> edit django/db/models/sql/query.py
+ if not self.bloom_metadata and not value:
+     return '', []
</tool_call>
```

这种 agentic 方式让模型自己探索仓库，**只读必要的部分**——context 占用大幅减少。

### Long-context model

用支持 128K / 1M context 的模型（如 Claude 3.5 Sonnet、Gemini 1.5 Pro、Qwen3-1M），直接读完整仓库。

Meta SWE-RL 用的是 **Llama-3-70B + RoPE scaling**——扩展到 128K context。但 long context 会带来：

- 训练成本上升（attention 是 O(n²)）
- 推理速度下降
- Position bias（模型对长 context 的中间部分不敏感）

### RAG（Retrieval-Augmented Generation）

预先建立仓库的 embedding 索引，根据 issue 描述检索相关文件，只把相关文件放入 context。

```python
def build_context(issue, repo):
    # 1. 用 embedding 检索相关文件
    relevant_files = retrieve(issue, repo, top_k=5)

    # 2. 拼接为 context
    context = ""
    for file in relevant_files:
        context += f"### {file.path}\n{file.content}\n\n"

    return context
```

RAG 是工业上最常用的方法——简单、高效、与现有模型兼容。

Meta SWE-RL 用的是**方法一 + 方法三的混合**——基础 context 用 RAG，工具调用让模型进一步探索。

## 20.1.4 训练稳定性技巧

SWE-RL 训练稳定性比数学 RL 难——因为：

- Trajectory 长（16-100+ 步）
- Reward 极度稀疏（只有最后测试通过才有 reward）
- 大部分 trajectory 是失败的（reward = 0）

Meta 报告了几个稳定性技巧：

### 成功率过滤（Success Rate Filtering）

在 RL 训练中，**只保留至少有一次成功的 prompt**。如果某个 prompt 的所有 N 个 rollout 都失败（reward 全为 0），它的组内方差也是 0，无法提供训练信号。

```python
def filter_prompts(prompts, model, num_rollouts=8):
    useful_prompts = []
    for prompt in prompts:
        rollouts = [model.generate(prompt) for _ in range(num_rollouts)]
        rewards = [compute_reward(r) for r in rollouts]
        if max(rewards) > 0:  # 至少一次成功
            useful_prompts.append(prompt)
    return useful_prompts
```

这与 [DAPO 的 Dynamic Sampling](../chapter18_grpo/deepseek-dapo) 思路一致——过滤"毕业题"。

### Curriculum Learning

按难度排序 prompt，先训简单的（小 PR、单文件、明确的 issue），再训复杂的（多文件、模糊 issue）。

```python
def curriculum_order(prompts):
    # 按修改文件数排序
    prompts.sort(key=lambda p: p.num_files_changed)
    return prompts
```

### KL 约束

SWE-RL 训练后期容易出现"模型忘记怎么写代码"——RL 过度优化测试通过，损害代码风格。Meta 用 KL 约束：

$$\mathcal{L} = \mathcal{L}_{\text{RL}} + \beta \cdot \text{KL}(\pi_\theta || \pi_{\text{ref}})$$

$\pi_{\text{ref}}$ 是 RL 前的模型（SFT 后的版本），$\beta$ 是约束强度。

这与 DeepSeek V3.2 "数学任务 zero KL" 形成对比——**SWE 需要保留代码风格，所以需要 KL**；数学是纯逻辑，不需要 KL。

## 20.1.5 SWE-bench Verified 41.0%

Meta SWE-RL 在 SWE-bench Verified 上的最终成绩：

| 模型                     | SWE-bench Verified         |
| ------------------------ | -------------------------- |
| GPT-4（zero-shot）       | 1.96%                      |
| Claude 3 Opus            | 3.21%                      |
| SWE-agent（GPT-4）       | 12.5%                      |
| SWE-Gym（开源）          | 19.0%                      |
| **Meta SWE-RL（开源）**  | **41.0%**                  |
| Cognition Devin（闭源）  | 13.95%（注：不同评测口径） |
| Claude 3.5 Sonnet + 工具 | 49.0%（闭源）              |

Meta SWE-RL 是开源模型的 SOTA——证明了**用开源数据 + GRPO + 简单 reward，可以达到接近闭源的水平**。

## 20.1.6 Meta SWE-RL 的局限

但 Meta SWE-RL 也有几个局限：

### 只支持 Python

Meta SWE-RL 的训练数据全部是 Python。其他语言（JavaScript、Java、C++、Go）没有对应数据。

### 依赖测试套件

没有测试的仓库无法训练。这在工业实践中是个大问题——很多公司的代码没有完整的单元测试。

### 长 horizon 训练不稳定

16 步以上的 trajectory 训练不稳定——RL 的 credit assignment 很难。Meta 报告，超过 32 步的 trajectory 训练效果显著下降。

### 数据多样性

虽然 90K 条数据不少，但都来自 GitHub PR——分布偏向开源生态。工业代码（如企业内部 Java 系统）的特性没有覆盖。

## 小结

Meta SWE-RL 是开源 SWE-RL 的代表作。它的核心贡献是：

- **数据**：开源 90K SWE 任务，覆盖 100+ 仓库
- **算法**：GRPO + 简单 binary reward，与 R1-Zero 同源
- **工程**：context 管理、训练稳定性技巧
- **结果**：SWE-bench Verified 41.0%（开源 SOTA）

Meta SWE-RL 证明了 RLVR 在 SWE 领域的可行性。但它的局限（只支持 Python、长 horizon 不稳定、依赖测试）也引出了 [20.2 Code World Model 与 DeepSWE](./world-model-and-deep-swe)：怎样用 world model 让模型模拟代码执行，减少对真实测试的依赖。
