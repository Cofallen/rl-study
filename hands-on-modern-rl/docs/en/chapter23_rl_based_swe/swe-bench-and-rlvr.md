# 20.1 Foundations of SWE-RL

+This section studies one concrete problem: **how to place a language model inside a real repository, have it understand an issue, edit code, run tests, and continue repairing from test failures.**

- +Ordinary code generation may require only one function. Real software engineering adds long call chains, regressions caused by a local edit, and repeated diagnosis after tests fail. A final patch alone cannot teach the model how to search files, use error messages, or choose the next action. SWE-RL treats the entire repair process as a trainable trajectory and obtains feedback from executable tests.
- +## Start with a Small Task That Really Fails
- +Suppose a shopping-cart project contains:
- +```python
  +def average_price(prices):
- return sum(prices) / len(prices)
  +```
- +An issue says that `average_price([])` should return `0.0` without changing nonempty-cart behavior. The repository initially tests only:
- +```python
  +def test_average_price():
- assert average_price([10.0, 20.0]) == 15.0
  +```
- +The model's first patch may be too broad:
- +```python
  +def average_price(prices):
- try:
-        return sum(prices) / len(prices)
- except Exception:
-        return 0.0
  +```
- +The empty-list test passes, but every exception is now hidden. Passing `None` incorrectly returns `0.0`. A regression test exposes the problem:
- +```python
  +def test_invalid_input_is_not_hidden():
- with pytest.raises(TypeError):
-        average_price(None)
  +```
- +After reading the failure, the model narrows its second patch:
- +```python
  +def average_price(prices):
- if len(prices) == 0:
-        return 0.0
- return sum(prices) / len(prices)
  +```
- +One repair has now completed a loop:
- +1. Read the issue and identify behavior that must remain unchanged.
  +2. Search for and open the relevant files.
  +3. Generate a candidate patch.
  +4. Run tests and observe failures.
  +5. Edit again from the feedback and submit the patch.
- +These five steps form a **trajectory**. If the repository state at step $t$ is $s_t$ and the action $a_t$ is reading, searching, editing, or testing, then
- +$$
+\tau=(s_0,a_0,s_1,a_1,\ldots,s_T).
+$$
- +Each later observation depends on earlier actions. Opening the wrong file deprives the patch of key context; a failed test can supply the clue needed for the next edit. Reinforcement learning improves this decision trajectory from repository inspection to patch submission.
- +<img src="../../chapter23_rl_based_swe/images/swe-rl-verifier-loop.svg" alt="SWE-RL forms a training loop from a repository and issue through an agent, candidate patch, and executable verifier">
- +The verifier is usually a test suite executed in an isolated environment. A passing patch earns reward, while failure logs return as the agent's next observation. The agent can therefore revise within one task and update its policy across many tasks.
-
-

## SWE-bench Task Definition

[SWE-bench](https://arxiv.org/abs/2310.06770) (Jimenez et al. 2023) is the core benchmark of SWE-RL. Its task definition is as follows:

```text
Input:
  - A GitHub repository (containing the full code)
  - An Issue description (in natural language, describing a bug or a feature request)
  - Test cases (used to verify whether the fix is correct)

Output:
  - A code patch (the modified code)

Validation:
  - Apply the patch to the repository
  - Run the test cases
  - All pass → Task is successful
  - Any test fails → Task is failed
```

### A Concrete Example

```text
Repository: django/django (Django Web Framework)

Issue:
  "In Django 4.2, using `Model.objects.filter(field__in=[])`
   returns an empty queryset, but the SQL query is still executed.
   It should short-circuit to return an empty result, avoiding unnecessary database calls."

Test cases:
  def test_empty_in_lookup_short_circuits(self):
      # Expected: filter(field__in=[]) does not trigger SQL
      with self.assertNumQueries(0):
          list(Model.objects.filter(field__in=[]))

Model Output:
  - Modify django/db/models/sql/query.py
  - Add in the as_sql method: if not self.bloom_metadata and not value: return '', []

Validation:
  - Apply the patch
  - Run the tests: ✓ Passed
  - Task is successful
```

### Difficulty of SWE-bench

The difficulty of SWE-bench far exceeds that of traditional code generation:

| Dimension       | Ordinary Code Generation            | SWE-bench                               |
| --------------- | ----------------------------------- | --------------------------------------- |
| Context         | Single function / short description | Entire repository (10K-1M lines)        |
| Output          | Complete code snippet               | Exact patch (diff)                      |
| Validation      | Manual or testing                   | Automated test suite                    |
| Multi-file      | Rare                                | Often requires cross-file modifications |
| Reasoning Depth | 1-10 steps                          | 10-100+ steps                           |

The state-of-the-art performance on the SWE-bench Verified (high-quality subset, 500 problems):

- Early 2024: ~12% (OpenAI SWE-agent)
- Mid-2024: ~25% (Cognition Devin)
- Early 2025: ~40% (Open-source SWE-RL series)
- End of 2025: ~53% (NVIDIA and others)
- Early 2026: ~65% (Claude Opus 4.7 + tool calls)

## Why SWE is an Ideal Arena for RLVR

Recalling [Chapter 15 on RLVR](../chapter18_grpo/rlvr) — the core idea of RLVR is to **replace RM with rule-based verification**. RLVR requires three conditions:

1. **The task has a clear answer**: Right is right, wrong is wrong
2. **Verification can be automated**: No need for human judgment
3. **There is enough training data**: To support large-scale RL

SWE perfectly satisfies these three conditions:

### Clear Answer

Code either passes the tests or does not — there is no "partially correct" or "subjective judgment." This is the purest "right or wrong" domain outside of mathematics.

### Automated Verification

Testing frameworks like `pytest` and `unittest` automatically run tests and output PASS/FAIL. The entire verification process requires no human intervention.

### Massive Data

- GitHub has over 400 million repositories
- Each PR is a natural SWE task (issue + patch + tests)
- Internal commit histories of industrial companies are also a vast source of training data

These three conditions make SWE-RL one of the **most successful applications** of RLVR in industry. Companies like Meta, ByteDance, Cognition, Alibaba, and Tsinghua University have all invested heavily in this direction.

## SWE-RL vs. Traditional Code Generation

Traditional code generation (e.g., HumanEval, MBPP) involves the following task:

```text
Input: Function signature + docstring
Output: Complete function implementation
```

This is a **short-context, single-file, no-test-feedback** setup. RL performs poorly on such tasks — because the generation space is small, SFT (Supervised Fine-tuning) can already reach SOTA (State-of-the-Art) performance.

In contrast, the task of SWE-RL is:

```text
Input: Full repository + Issue + Test cases
Output: Exact patch
Allowed: Multi-step interaction (read file, edit, run test, edit again)
```

This is a **long-context, multi-file, with-test-feedback** setup. RL performs well on such tasks — because:

- **Exploration space is huge**: The number of possible patches is astronomical, and RL can efficiently explore this space
- **Delayed feedback**: Test results are delayed rewards, which naturally aligns with RL's advantage in reward estimation
- **Multi-step decision making**: Read → think → edit → test → fix → submit is a typical agent trajectory

## Data Generation for SWE-bench

<img src="../../chapter23_rl_based_swe/images/swe-rl-data-loop.svg" alt="Real issues, synthetic defects, and self-play tasks enter a versioned SWE-RL training-data loop">

SWE-RL training requires a large number of (Issue, patch, tests) triplets. There are three sources:

### Real PRs (SWE-bench Method)

PRs are scraped from GitHub, and the following are extracted:

- Issue text (the issue associated with the PR)
- Code diff (the changes made in the PR)
- Test cases (new or modified tests in the PR)

Scale: Approximately 2,300 entries (original SWE-bench)

Limitations:

- **Limited data**: 2,300 entries are insufficient for training large models
- **Dependent on PR quality**: Low-quality PRs are also collected
- **Missing tests**: Many PRs lack complete test coverage

### Synthetic Data (SWE-smith Method)

[SWE-smith](../chapter22_agentic/agent-data-swe-smith) ([arXiv:2504.21798](https://arxiv.org/abs/2504.21798)) — **Intentionally inject bugs into good code and run tests to see which bugs are detected**.

Scale: Over 50,000 entries (covering 128 Python repositories)

Advantages:

- **Large volume**: 20 times that of SWE-bench
- **Controllable**: Types and difficulty of bugs can be adjusted
- **Complete testing**: Each bug has corresponding test cases

### Self-play SSR Method (Self-Generated Training Data)

Let the model:

1. Find a "looks like a bug" place in the repository
2. Write a "fix"
3. Run tests to see if they pass
4. The passed (issue, patch, test) triple as training data

This is the core idea of [Section 20.3 SSR](./self-play-ssr-and-summary) — **the model generates its own training data**.

## Reward Function in SWE-RL

The reward function in SWE-RL is typically extremely simple:

```python
def swe_reward(test_results):
    """Test results as reward"""
    passed = sum(test_results)
    total = len(test_results)
    return passed / total  # Or binary: 1.0 if passed == total else 0.0
```

This reward function is mathematically identical to the binary reward in R1-Zero — **0/1 binary reward**.

### Details of Reward Shaping

However, in industrial practice, several shaping terms are added:

**Term 1: Test Pass Rate**

```python
reward = passed / total
```

Not binary, but a continuous value. This allows the model to receive partial reward when it "fixes half" of the issue.

**Term 2: Length Penalty**

```python
reward -= 0.01 * len(trajectory)
```

Encourages the model to complete the task in fewer steps — avoiding the waste of "first randomly change, then fix after test failure."

**Term 3: Patch Quality**

```python
patch_quality = score_patch(model_output)  # Judged by LLM
reward += 0.1 * patch_quality
```

Encourages the model to generate more elegant patches (e.g., no code duplication, no breaking existing logic).

**Term 4: Context Efficiency**

```python
context_efficiency = relevant_files_read / total_files_read
reward += 0.05 * context_efficiency
```

Encourages the model to read only relevant files, avoiding the waste of "reading all files."

However, [Meta SWE-RL](https://arxiv.org/abs/2502.18449) reported an important finding: **the simplest reward (binary test pass) performs best**. Complex shaping can easily introduce reward hacking — the model learns to "optimize shaping terms" rather than truly fixing bugs.

This aligns with the findings of [R1-Zero](../chapter18_grpo/deepseek-dapo): **simple reward + large-scale RL > complex reward + small-scale RL**.

## Training Process of SWE-RL

A complete training process of SWE-RL:

```text
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Base model selection                                │
│   - Usually a code-tuned LLM (e.g., Qwen-Coder, DeepSeek-Coder) │
│   - Already pre-trained on a large amount of code           │
├─────────────────────────────────────────────────────────────┤
│ Step 2: Cold Start with SFT (Optional)                      │
│   - Use SWE-bench / SWE-smith data for SFT                  │
│   - Let the model learn the basic trajectory format         │
├─────────────────────────────────────────────────────────────┤
│ Step 3: RL Training                                         │
│   - GRPO / PPO                                              │
│   - Reward: Test pass binary                               │
│   - Long horizon: Each trajectory may have 16-100+ steps    │
├─────────────────────────────────────────────────────────────┤
│ Step 4: Rejection Sampling + Second SFT                    │
│   - Generate multiple candidates from the RL-trained model   │
│   - Select the best one for SFT                             │
├─────────────────────────────────────────────────────────────┤
│ Step 5: Evaluation                                          │
│   - SWE-bench Verified                                     │
│   - Internal evaluation set                                │
└─────────────────────────────────────────────────────────────┘
```

This process is highly similar to the training process of [DeepSeek-R1](../chapter18_grpo/deepseek-dapo) — both are combinations of SFT + RL + secondary SFT. The difference lies only in:

- R1's reward is based on whether the mathematical answer is correct.
- SWE-RL's reward is based on whether the test passes.

This similarity indicates: **The training paradigm of RLVR is generalizable** — as long as an appropriate verifier is found, the same algorithm can be applied to different domains.

## Summary

SWE-bench is the core benchmark of SWE-RL, defining the task format of (issue, patch, tests). SWE is an ideal battlefield for RLVR — with clear answers, automated verification, and massive data.

SWE-RL differs fundamentally from traditional code generation — with long context, multi-file support, test feedback, and multi-step decision-making. This makes it highly aligned with Agentic RL, making it one of the most valuable applications of RL in industry.

Next, we will first examine [Supplementary Reading: Meta SWE-RL](./meta-swe-rl) to observe how GRPO and simple rewards are applied in real-world repositories, and then move on to [20.2 Code World Model and DeepSWE](./world-model-and-deep-swe), addressing the issue of unstable training on long trajectories.
