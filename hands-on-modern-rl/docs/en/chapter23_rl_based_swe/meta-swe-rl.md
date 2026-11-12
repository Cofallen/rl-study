# Supplementary Reading: Meta SWE-RL and Open-Source SOTA

[Meta SWE-RL](https://arxiv.org/abs/2502.18449) (2025.02) is a representative work of open-source SWE-RL. Its core contributions are:

- Trained using open-source data (SWE-bench + SWE-gym)
- Utilized the simplest GRPO + test reward
- Achieved 41.0% on SWE-bench Verified (open-source SOTA)

This section provides a detailed look at Meta SWE-RL's data, algorithm, and engineering details.

## 20.1.1 Data Scale and Composition

The training data sources for Meta SWE-RL are as follows:

| Data Source             | Scale         | Purpose               |
| ----------------------- | ------------- | --------------------- |
| SWE-bench (open-source) | 2,294 items   | High-quality baseline |
| SWE-gym (open-source)   | 6,800 items   | Extended training     |
| Internal PR Data (Meta) | 80,000+ items | Large-scale RL        |

**Total of approximately 90,000 SWE tasks** — 40 times larger than pure SWE-bench.

### Data Preprocessing

Meta reported several key steps in data cleaning:

**Step 1: Repository Filtering**

- Exclude: repositories with test coverage < 50% (unreliable for verification)
- Exclude: inactive repositories (last commit > 6 months ago)
- Exclude: repositories with Python < 3.8 (incompatible with latest dependencies)

**Step 2: PR Filtering**

- Exclude: PRs that modify more than 10 files (too complex for early RL training)
- Exclude: PRs that are pure dependency updates (not real "bug fixes")
- Exclude: PRs that delete features (not aligned with "fix" semantics)

**Step 3: Test Filtering**

- Keep: PRs that include new tests (clear verification criteria)
- Exclude: PRs with tests that cannot be run independently
- Exclude: PRs with tests that depend on external services (e.g., databases, API keys)

These filters significantly improved the final data quality. Meta reported that training performance was poor before filtering, but improved substantially after filtering — **data quality > data quantity**.

## 20.1.2 Algorithm: GRPO + Simple Reward

Meta SWE-RL's algorithm choice is extremely simple — **GRPO + test binary reward**.

### Why Use GRPO?

Meta's team compared PPO, GRPO, and DPO in the [SWE-RL paper](https://arxiv.org/abs/2502.18449):

| Algorithm      | SWE-bench Verified |
| -------------- | ------------------ |
| DPO (baseline) | 24.3%              |
| PPO            | 33.2%              |
| **GRPO**       | **41.0%**          |

Advantages of GRPO:

- **No Critic Required**: saves GPU memory, suitable for large-scale training
- **Group-wise Normalization**: naturally handles tasks of varying difficulty (high variance in easy tasks, low variance in hard tasks)
- **Simple and Stable**: easier to implement in engineering than PPO

This is consistent with the findings of [DeepSeek-R1](../chapter18_grpo/deepseek-dapo) — **GRPO is the default choice for SWE-RL**.

### Reward Function

The reward function in Meta SWE-RL is extremely simple:

```python
def swe_rl_reward(test_results):
    """Binary reward for test pass"""
    if all(test_results):
        return 1.0  # All tests passed
    else:
        return 0.0  # Any test failed
```

Meta reported several attempts at reward shaping, all of which were **worse than the simple binary reward**:

- **Continuous reward (passed/total)**: The model tends to "fix half" to get partial reward, and fails to learn full fixes
- **Length penalty**: The model learns to "complete the task in fewer steps but without fixing bugs" — directly abort the task to get 0 steps
- **Complex shaping**: Reward hacking becomes severe

This confirms the finding of R1-Zero: **Simple reward + large-scale RL is the best combination**.

## 20.1.3 Engineering Details: Context Management

Context management in SWE-RL is an engineering challenge. A complete SWE task may involve:

- The entire repository (10K–1M lines of code)
- Issue description (hundreds of tokens)
- Tool calls (read file, edit, run test)
- Model reasoning (CoT)

These can add up to more than 100K tokens — exceeding the context window of most models. How to handle this?

### Tool Invocation as an Alternative to Reading the Entire Codebase

Instead of making the model read through the entire codebase at once, we let it **actively invoke tools to read only the necessary parts**:

```text
<issue> Django filter(field__in=[]) should short-circuit </issue>

<thought>
I need to find the implementation of this filter.
</thought>

<tool_call> grep -r "def filter" django/db/models/sql/query.py </tool_call>

<tool_result> def filter(self, *args, **kwargs): ... </tool_result>

<thought>
Found it. Let me check the as_sql method.
</thought>

<tool_call> read django/db/models/sql/query.py lines 100-150 </tool_call>

<tool_result> def as_sql(self, compiler, connection):
    ...
</tool_result>

<thought>
I see there is no short-circuit logic here. Let me add it.
</thought>

<tool_call> edit django/db/models/sql/query.py
+ if not self.bloom_metadata and not value:
+     return '', []
</tool_call>
```

This agentic approach allows the model to explore the codebase itself, **reading only the necessary parts** — significantly reducing context usage.

### Long-context Model

Use models that support 128K / 1M context (e.g., Claude 3.5 Sonnet, Gemini 1.5 Pro, Qwen3-1M) to directly read through the entire repository.

Meta SWE-RL uses **Llama-3-70B + RoPE scaling**—extending to 128K context. However, long context brings the following challenges:

- Increased training cost (attention is O(n²))
- Slower inference speed
- Position bias (the model is less sensitive to the middle parts of long context)

### RAG (Retrieval-Augmented Generation)

Pre-build an embedding index for the repository, retrieve relevant files based on the issue description, and include only the relevant files in the context.

```python
def build_context(issue, repo):
    # 1. Use embedding to retrieve relevant files
    relevant_files = retrieve(issue, repo, top_k=5)

    # 2. Concatenate into context
    context = ""
    for file in relevant_files:
        context += f"### {file.path}\n{file.content}\n\n"

    return context
```

RAG is the most commonly used method in industry—simple, efficient, and compatible with existing models.

Meta SWE-RL uses a **combination of method one and method three**—using RAG as the base context, and allowing the model to further explore through tool calls.

## 20.1.4 Techniques for Training Stability

SWE-RL training is more challenging than mathematical RL — because:

- Trajectory length is long (16–100+ steps)
- Reward is extremely sparse (only the final test pass yields a reward)
- Most trajectories are failures (reward = 0)

Meta reported several techniques for training stability:

### Success Rate Filtering

In RL training, **only keep prompts that have at least one successful rollout**. If a prompt's N rollouts all fail (reward is 0 for all), its intra-group variance is also 0, and it cannot provide training signals.

```python
def filter_prompts(prompts, model, num_rollouts=8):
    useful_prompts = []
    for prompt in prompts:
        rollouts = [model.generate(prompt) for _ in range(num_rollouts)]
        rewards = [compute_reward(r) for r in rollouts]
        if max(rewards) > 0:  # At least one success
            useful_prompts.append(prompt)
    return useful_prompts
```

This aligns with the idea of [DAPO's Dynamic Sampling](../chapter18_grpo/deepseek-dapo) — filtering out "easy problems".

### Curriculum Learning

Sort prompts by difficulty, first training on simple ones (small PRs, single-file, clear issues), then moving to complex ones (multi-file, ambiguous issues).

```python
def curriculum_order(prompts):
    # Sort by the number of files changed
    prompts.sort(key=lambda p: p.num_files_changed)
    return prompts
```

### KL Constraint

During the later stages of SWE-RL training, it is common for the model to "forget how to write code"—the RL optimization may overfit to passing tests, thereby harming the code style. Meta uses a KL constraint:

$$\mathcal{L} = \mathcal{L}_{\text{RL}} + \beta \cdot \text{KL}(\pi_\theta || \pi_{\text{ref}})$$

Here, $\pi_{\text{ref}}$ is the model before RL (the version after SFT), and $\beta$ is the constraint strength.

This contrasts with the "zero KL" approach in DeepSeek V3.2 for mathematical tasks—**SWE requires preserving code style, hence the need for KL**; mathematics is purely logical and does not require KL.

## 20.1.5 SWE-bench Verified 41.0%

Meta SWE-RL's final performance on SWE-bench Verified:

| Model                           | SWE-bench Verified                           |
| ------------------------------- | -------------------------------------------- |
| GPT-4 (zero-shot)               | 1.96%                                        |
| Claude 3 Opus                   | 3.21%                                        |
| SWE-agent (GPT-4)               | 12.5%                                        |
| SWE-Gym (open source)           | 19.0%                                        |
| **Meta SWE-RL (open source)**   | **41.0%**                                    |
| Cognition Devin (closed source) | 13.95% (note: different evaluation criteria) |
| Claude 3.5 Sonnet + Tools       | 49.0% (closed source)                        |

Meta SWE-RL is the state-of-the-art among open-source models—demonstrating that **using open-source data + GRPO + simple reward functions can achieve performance close to closed-source models**.

## 20.1.6 Limitations of Meta SWE-RL

However, Meta SWE-RL also has several limitations:

### Only Supports Python

All training data of Meta SWE-RL is in Python. There is no corresponding data for other languages such as JavaScript, Java, C++, and Go.

### Relies on Test Suites

Repositories without tests cannot be trained. This is a significant issue in industrial practice — many companies' code lacks complete unit tests.

### Unstable Training for Long Horizons

Training trajectories longer than 16 steps are unstable — credit assignment in RL is difficult. Meta reported that performance significantly drops for trajectories longer than 32 steps.

### Data Diversity

Although 90K data samples are a considerable amount, they are all from GitHub PRs — the distribution is biased toward the open-source ecosystem. The characteristics of industrial code (such as enterprise internal Java systems) are not covered.

## Summary

Meta SWE-RL is a representative work of open-source SWE-RL. Its core contributions are:

- **Data**: Open-sourced 90K SWE tasks, covering over 100 repositories
- **Algorithm**: GRPO + simple binary reward, sharing the same origin as R1-Zero
- **Engineering**: Context management, training stability techniques
- **Results**: Achieved 41.0% on SWE-bench Verified (open-source SOTA)

Meta SWE-RL demonstrates the feasibility of RLVR in the domain of software engineering. However, its limitations — such as only supporting Python, instability for long horizons, and dependency on tests — also lead to [20.2 Code World Model and DeepSWE](./world-model-and-deep-swe): How to use world models to let models simulate code execution, reducing the reliance on real tests.
