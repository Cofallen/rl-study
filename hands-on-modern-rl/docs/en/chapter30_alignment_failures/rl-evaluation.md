# 25.5 Evaluation Protocols and Reproducibility

Suppose two teams evaluate the same coding model. Team A uses one prompt, extracts the first code block, and allows five samples. Team B uses a chat template, executes the last code block, and allows one sample. They can report different pass rates without changing the model at all.

An evaluation result is produced by a complete protocol: task version, prompt, chat template, sampling budget, answer extractor, environment, verifier, and statistical summary. A **harness** turns that protocol into versioned, repeatable code. This section explains how to choose what to measure, detect contamination, test prompt sensitivity and distribution shift, and preserve long-horizon trajectories so another researcher can reproduce the conclusion.

## 25.5.1 First Define What the Evaluation Is Measuring

A good RL benchmark must satisfy five principles:

### Verifiability

The answer to each test sample must be **machine-verifiable**. Formally defined: there exists a function $\text{Verify}: \mathcal{Y} \times \mathcal{Y} \to \{0, 1\}$, such that for any $(y_{\text{pred}}, y_{\text{gold}})$, the correctness can be determined deterministically.

- **Math problems**: Extract the final number and compare it with the standard answer (e.g., [GSM8K](https://arxiv.org/abs/2110.14168), MATH)
- **Code problems**: Run on test cases and check the pass rate (e.g., [HumanEval](https://arxiv.org/abs/2107.03374), MBPP, LiveCodeBench)
- **Logic problems**: Use a SAT solver or theorem prover to verify (e.g., MiniF2F, PutnamBench)

Tasks that are not verifiable (open-ended writing, creative generation) can only be evaluated by human or RM evaluation — both of which are unreliable.

### Coverage

Benchmark should cover the real distribution that models may encounter. Formally:

$$\mathcal{D}_{\text{test}} \sim P_{\text{real}}, \quad P_{\text{real}} \approx P_{\text{test}}$$

If $\mathcal{D}_{\text{test}}$ is biased toward a certain type of problem, the model may fail on other types. GSM8K is a typical example of this — it only contains elementary math problems, and the model GSM8K achieves 90% accuracy on it, which does not represent its ability to solve advanced math problems.

### Difficulty Stratification

Evaluate by difficulty level to avoid "average scores masking extreme performance":

```python
# Difficulty stratified evaluation
def stratified_eval(model, dataset):
    results = {"easy": [], "medium": [], "hard": []}
    for x, y in dataset:
        pred = model(x)
        difficulty = classify_difficulty(x)  # Use a difficulty classifier
        results[difficulty].append(verify(pred, y))
    return {k: np.mean(v) for k, v in results.items()}
```

The MATH dataset is divided into five difficulty levels (Level 1-5), and DeepSeek-R1 reports scores for each level. Reporting stratified scores provides a more accurate reflection of the model's capability distribution than a single overall score.

### Contamination Resistance

The test set must be **strictly confidential**, and the training data must undergo decontamination checks. See Section 35.2 for details.

### Statistical Rigor

It is not sufficient to report "Model A achieves 60% on MATH, Model B 55%" — this could simply be sampling noise. One should perform:

- **Confidence Interval**: For $n$ test samples with accuracy $p$, the 95% CI is $p \pm 1.96\sqrt{p(1-p)/n}$
- **Paired comparison**: use paired bootstrap intervals or a test suited to paired binary outcomes when two models answer the same items
- **Bootstrap**: Estimate variance through resampling of the test set

[Blackwell et al.](https://arxiv.org/abs/2410.03492) show why benchmark scores should be accompanied by uncertainty estimates rather than treated as exact constants.

## Contamination and Leakage Detection

[Section 25.2](./modern-incidents) explains how contamination can imitate RLVR progress on public mathematics benchmarks. This section turns that concern into systematic detection methods.

### Three Types of Contamination

#### 1. Explicit Contamination

Training data and test data **completely overlap** on the same sample:

$$\exists (x, y) \in \mathcal{D}_{\text{train}}, \quad (x, y) \in \mathcal{D}_{\text{test}}$$

This is the easiest to detect, and can be identified with n-gram overlap.

#### 2. Approximate Contamination

Training data contains **rewritten, translated, or paraphrased versions** of test samples:

$$\exists (x', y') \in \mathcal{D}_{\text{train}}, \quad \text{sim}(x', x_{\text{test}}) > \tau$$

Detection requires semantic similarity (embedding distance) or judgment from an LLM.

#### 3. Implicit Contamination (Most Difficult)

Training data does not directly contain test samples, but the training task is highly similar to the test task — the model learns the **task pattern** rather than **specific answers**:

- Training data: 2000 university physics problems
- Test data: GSM8K (elementary math)
- Phenomenon: Physics training teaches the model the "read problem → formulate equation → compute → verify" pattern, indirectly improving math performance

Implicit contamination cannot be fully detected, and can only be indirectly assessed through **Holdout Tasks** (using tasks of a type the model has never seen before).

### Detection Methods

#### N-gram Overlap

The simplest detection method — 13-gram overlap:

```python
def ngram_contamination(train_text, test_text, n=13):
    train_ngrams = set(get_ngrams(train_text, n))
    test_ngrams = set(get_ngrams(test_text, n))
    overlap = train_ngrams & test_ngrams
    return len(overlap) / len(test_ngrams)
```

OpenAI's 2020 study ([arXiv:2005.14165](https://arxiv.org/abs/2005.14165)) used 13-gram overlap to filter out text that was duplicated with the benchmark from the training corpus. This was one of the earliest practices for decontamination.

#### Membership Inference

Train a classifier to determine whether "this sample was in the training set":

$$\text{MIA}(x) = \begin{cases} 1, & \text{if } p_{\text{model}}(x) > \tau \\ 0, & \text{otherwise} \end{cases}$$

If MIA achieves significantly higher accuracy on the test set than random guessing, it suggests that the test set contains samples from the training set.

#### Perplexity Anomaly

Calculate the model's perplexity on the test set:

$$\text{PPL}_{\text{test}} = \exp\left(-\frac{1}{N}\sum_i \log p_{\text{model}}(x_i)\right)$$

If the PPL is much lower than that of a similar difficulty control set, the model may have "memorized" the test set.

#### Temporal Splitting

Split the test set by time — only use items added after the model's release date:

```python
# Continuously updated benchmark with LiveCodeBench, LMSYS Arena
test_data = [
    item for item in dataset
    if item.created_at > model_release_date
]
```

This is the most reliable anti-contamination method — LiveCodeBench, LMSYS Chatbot Arena all use this approach.

### Practical Engineering for Decontamination

Industrial-grade decontamination pipeline:

1. **N-gram Filtering** (13-gram): Remove 90% of explicit contamination
2. **Embedding Retrieval** (cosine similarity > 0.9): Remove similar contamination
3. **MinHash LSH**: Fast approximate detection ([Deduplicating Training Data, arXiv:2107.06499](https://arxiv.org/abs/2107.06499))
4. **Continuous Benchmark Updates**: Monthly update with new data to test set

Decontamination reduces known overlap but cannot prove that every semantic variant was absent from pretraining. Fresh tasks, temporal splits, and programmatically generated controls provide stronger evidence than one string-matching pass.

## Prompt Sensitivity Analysis

For the same model and task, semantically equivalent prompts can change both absolute scores and model rankings. This phenomenon is called **prompt sensitivity**.

### Experimental Evidence

[Mizrahi et al.](https://arxiv.org/abs/2401.00595) generate semantically equivalent instructions across tasks in LMentry, BIG-Bench Lite, and BIG-Bench Hard, then compare several model families. The result is methodological: a single prompt does not identify a stable model score or ranking.

![Semantically equivalent prompts can change model scores and rankings](../../chapter30_alignment_failures/images/multi-prompt-evaluation.png)

<div style="text-align: center; font-size: 0.9em; color: var(--vp-c-text-2); margin-top: -10px; margin-bottom: 20px;">
  <em>Figure 1: Multi-prompt evaluation exposes variation hidden by one canonical instruction. Source: <a href="https://arxiv.org/abs/2401.00595" target="_blank" rel="noopener noreferrer">State of What Art? A Call for Multi-Prompt LLM Evaluation</a>.</em>
</div>

### Sensitivity Sources

1. **Format Requirements**: "Answer with a number between 0 and 100" vs "Please provide the reasoning process before answering with a number"
2. **CoT Trigger**: "think step by step" vs "explain your reasoning" vs No CoT
3. **Few-shot Quantity**: There is a significant difference in results between 0-shot, 4-shot, and 8-shot scenarios
4. **Answer Extraction Format**: Using regex `\\boxed\{(.+?)\}` vs `"answer: (.+?)"`

### Standardization Methods

#### 1. Multi-prompt Averaging

For each test sample, use $K$ prompt templates and take the average:

$$\text{Score}(\pi) = \frac{1}{K} \sum_{k=1}^K \text{Score}_{\text{prompt}_k}(\pi)$$

#### 2. Reporting Variance

Not only report the average score, but also report the variance:

$$\text{Score} \pm 1.96 \cdot \frac{\sigma}{\sqrt{K}}$$

#### 3. Prompt Standardization

lm-eval-harness defines a **uniform prompt format specification**, ensuring all models are evaluated on the same prompt.

```python
# lm-eval-harness standardized prompt
PROMPT_TEMPLATE = """
Question: {question}

Answer: Let's think step by step. {reasoning}
Therefore, the answer is \\boxed{{{answer}}}.
"""
```

### Engineering Recommendations

Models trained with reinforcement learning (RL) are particularly sensitive to prompts — because RL encourages the model to highly adapt to the prompt format in the training distribution. **When reporting RL results, it is essential to perform multi-prompt averaging**, otherwise the conclusions may be dominated by "lucky prompt templates."

## Out-of-Distribution Robustness

Models may perform well on the training distribution, but can degrade sharply on out-of-distribution (OOD) data. This is a unique issue of RL training — RL tends to "overfit" to the reward signal of the training distribution.

### OOD Evaluation Methods

#### 1. Distribution Shift Testing

Construct distribution shifts:

- **Style Shift**: Train with academic language, test with slang
- **Domain Shift**: Train with math problems, test with physics problems
- **Format Shift**: Train with LaTeX, test with Markdown

#### 2. Adversarial Perturbation

Apply small perturbations to the input and check the model's stability:

$$\text{RobustScore}(x) = \text{Score}(\pi(x)) - \max_{\|\delta\| \leq \epsilon} |\text{Score}(\pi(x + \delta)) - \text{Score}(\pi(x))|$$

Character substitution, synonym replacement, and case transformation are common types of perturbations.

#### 3. Counterfactual Evaluation

Construct counterfactual samples:

- Original sample: "A train travels 60 km/h for 2 hours. How far?"
- Counterfactual: "A bicycle travels 20 km/h for 3 hours. How far?"

If the model answers correctly on the original sample but incorrectly on the counterfactual, it suggests that the model has learned surface patterns rather than the underlying principles.

### Out-of-Distribution (OOD) Risk in RL Training

After training with RLHF/GRPO, models often suffer from **Alignment Tax**—the trade-off where alignment improvements come at the cost of foundational capabilities:

- **Model — Llama-2-70B**
  - MMLU (SFT): 86.0%
  - MMLU (RLHF): 84.5%
  - Change: -1.5%
- **Model — Claude 1**
  - MMLU (SFT): 75.0%
  - MMLU (RLHF): 73.8%
  - Change: -1.2%
- **Model — GPT-4 (est.)**
  - MMLU (SFT): 89.0%
  - MMLU (RLHF): 87.5%
  - Change: -1.5%

**Cause**: RLHF rewards "alignment-friendly" responses, and the model learns to "play it safe"—when uncertain, it tends to refuse or provide vague answers, thereby sacrificing foundational capabilities.

### Mitigating Alignment Tax

- **KL Penalty**: RLHF adds $\beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{SFT}})$, constraining the deviation from the reference model.
- **Capability Preservation Data**: Intermixing SFT data during RL training and periodically revisiting it.
- **Multi-Objective RL**: Optimizing for accuracy, helpfulness, and safety simultaneously ([Reward Weighted Regression, arXiv:2305.18290](https://arxiv.org/abs/2305.18290)).

## Behavior Evaluation vs. Capability Evaluation

Traditional benchmarks evaluate **capability** (capability) — "Can the model solve this problem?" However, after RL training, models also need to be evaluated on their **behavior** — "How will the model behave in this context?"

### Capability Evaluation

Formal definition: Given input $x$ and the golden answer $y^*$, evaluate:

$$\text{Capability}(\pi) = \mathbb{E}_{x \sim \mathcal{D}}[\text{Verify}(\pi(x), y^*)]$$

- MMLU, GSM8K, and HumanEval are all examples of capability evaluation
- Advantages: Objective and reproducible
- Disadvantages: Ignores the model's behavior in multi-turn dialogues and long-term interactions

### Behavior Evaluation

Formal definition: Given a context $\mathcal{S}$ and a set of desired behaviors $\mathcal{B}$, evaluate:

$$\text{Behavior}(\pi, \mathcal{S}) = \mathbb{E}_{\text{trajectories from }\mathcal{S}}\bigg[\frac{1}{T}\sum_{t=1}^T \mathbf{1}[\pi(s_t) \in \mathcal{B}]\bigg]$$

Behavior evaluation focuses on:

- **Sycophancy**: Whether the model flatters the user by agreeing with incorrect views
- **Honesty**: Whether the model admits uncertainty
- **Helpfulness vs. Harmlessness**: How the model balances these two aspects
- **Power-seeking**: Whether the model attempts to acquire more resources (scheming)

### Methods for Behavioral Evaluation

#### 1. Scenario Testing

Construct specific scenarios to observe how the model responds:

```python
scenario = {
    "user": "I'm pretty sure 2+2=5, right?",
    "expected_behaviors": ["correct_user", "ask_for_clarification"],
    "bad_behaviors": ["agree", "ignore_error"]
}
response = model(scenario["user"])
behavior_label = human_annotate(response, scenario)
```

#### 2. Sycophancy Benchmarks

Perez et al. 2022 ([arXiv:2212.09251](https://arxiv.org/abs/2212.09251)) designed the Sycophancy benchmark — evaluating whether a model flatters the user by agreeing with their incorrect views:

- User asks: "I think climate change is a hoax, agree?"
- Honest response: Explain scientific evidence and politely disagree
- Flattering response: "Yes, you're right!"

#### 3. Honesty Benchmarks

[Honesty Is the Best Policy, arXiv:2312.01350](https://arxiv.org/abs/2312.01350): Measures the honesty of a model when it is uncertain about the answer.

### Challenges in Behavioral Evaluation

- **Subjectivity**: What counts as "compliance" and what counts as "courtesy" can vary significantly among different annotators.
- **Multi-turn Interaction**: Behaviors often emerge in long dialogues, making single-turn evaluation insufficient.
- **Data Scarcity**: Designing behavior scenarios requires knowledge of psychology and sociology.

In industrial practice, both Anthropic and OpenAI have dedicated "behavior evaluation teams" that assess changes in Claude and GPT on a monthly basis.

## Challenges in Evaluating Long-Term Tasks

[Chapter 22: Computer Use](../chapter25_computer_use/training), [Chapter 20: SWE-Agent](../chapter23_rl_based_swe/swe-bench-and-rlvr) — these agentic tasks are far more challenging to evaluate than single-turn question-answering tasks. A task may last for several hours and involve hundreds or even thousands of decision steps.

### Characteristics of Long-Term Tasks

- **Dimension — Steps**
  - Single-Turn Task: 1
  - Long-Term Task: 100–10,000
- **Dimension — Evaluation Time**
  - Single-Turn Task: Seconds
  - Long-Term Task: Hours
- **Dimension — Intermediate Feedback**
  - Single-Turn Task: None
  - Long-Term Task: Observation at each step
- **Dimension — Termination Condition**
  - Single-Turn Task: Model stops
  - Long-Term Task: Task completion or timeout
- **Dimension — Error Propagation**
  - Single-Turn Task: Not applicable
  - Long-Term Task: Accumulation of single-step errors

### Evaluation Methods

#### 1. Outcome-Based Evaluation

Only the final result is considered, not the process:

$$\text{Score} = \mathbf{1}[\text{Final Result is Correct}]$$

- **SWE-Bench**: Whether the correct PR was submitted
- **WebArena**: Whether multi-step web operations were completed
- Simple and straightforward, but ignores the quality of the intermediate process

#### 2. Process-Based Evaluation

Use the Process Reward Model (see [Chapter 17 on PRM](../chapter20_prm_search/outcome-vs-process)) to evaluate each step:

$$\text{Score} = \frac{1}{T}\sum_{t=1}^T \text{PRM}(s_t, a_t)$$

- More granular, but the PRM itself may be biased
- Computationally expensive

#### 3. Hybrid Evaluation

Combine with weights:

$$\text{Score} = \alpha \cdot \text{Outcome} + (1-\alpha) \cdot \text{Process}$$

#### 4. Human Expert Evaluation

For ultra-long tasks (research agents, full SWE development), only human expert evaluation is feasible:

- **Completion**: Whether the task was solved
- **Efficiency**: Whether the task was completed in the fewest steps
- **Style**: Whether best practices were followed (code readability, documentation quality)
- **Robustness**: How the agent handles unexpected situations

This method is costly (typically $50–500 per task), but remains the gold standard.

### Variance Issues in Long-Horizon Tasks

The scores for long-horizon tasks exhibit significant variance — running the same agent on the same task twice can result in completely different outcomes (due to randomness and long-tail errors).

```python
# Must run multiple times to take an average
def long_horizon_eval(agent, task, n_runs=10):
    scores = []
    for _ in range(n_runs):
        trajectory = agent.run(task, max_steps=1000)
        scores.append(evaluate(trajectory))
    return np.mean(scores), np.std(scores)
```

At least 10 runs are required, and important evaluations should have 50 or more runs. This is why experimental costs for long-horizon tasks are extremely high — a single experiment may cost thousands of dollars in API fees.

## Anthropic Internal AI Research Evaluation Suite

In 2025, Anthropic released its internal evaluation suite used to assess the capabilities of Claude Opus 4.6 (2025.11) as an **AI Research Assistant** — a landmark benchmark, as it directly measures "whether a model can perform AI research work."

### Three Subtasks

#### 1. LLM Training Subtask

Have Claude Opus 4.6 **actually train an RL model** on the veRL/OpenRLHF framework:

- **Configuration**: Select algorithm (GRPO/PPO), hyperparameters, and dataset
- **Implementation**: Write training scripts, tune hyperparameters, and debug
- **Evaluation**: Performance of the trained model on held-out tasks

#### 2. Text-RL Subtask

Have the model design a text-based RL task and train an agent to complete it:

- **Task Design**: Choose environment, define reward
- **Implementation**: Write RL training loop
- **Training**: Actually run RL and reach baseline performance

#### 3. Quadruped-RL Subtask

Have the model train a quadruped robot to walk in the MuJoCo physics simulation:

- This is a classic continuous control task ([Chapter 9 on SAC](../chapter11_continuous_control/ddpg))
- Requires understanding the environment, debugging the algorithm, and tuning hyperparameters
- **Success Criterion**: The agent reaches baseline performance within 1M steps

### 34× Human-Accelerated Details

Anthropic reports that Claude Opus 4.6 completes these tasks **34 times faster than human researchers**:

- **Task — LLM Training**
  - Human Average Time: 17 hours
  - Opus 4.6 Time: 30 minutes
  - Acceleration Ratio: 34×
- **Task — Text-RL**
  - Human Average Time: 12 hours
  - Opus 4.6 Time: 25 minutes
  - Acceleration Ratio: 29×
- **Task — Quadruped-RL**
  - Human Average Time: 8 hours
  - Opus 4.6 Time: 15 minutes
  - Acceleration Ratio: 32×
- **Task — Average**
  - Human Average Time: **12.3 hours**
  - Opus 4.6 Time: **23 minutes**
  - Acceleration Ratio: **34×**

**Note**: The "completion" here is not perfect, but rather reaching a **research assistant level** — for example, the trained model achieves 80% of baseline performance on held-out tasks.

### The Multidimensionality of Evaluation Metrics

The Eval Suite of Opus 4.6 does not only report "completion time," but also includes:

- **Correctness**: The actual performance of the trained model
- **Code Quality**: The style and readability of the implemented code
- **Reproducibility**: Whether the results are consistent across two runs
- **Debugging Ability**: Whether the model can self-correct when encountering errors
- **Innovation**: Whether the model proposes improvements beyond the baseline

This multidimensional evaluation represents the future of agentic benchmarks — a single metric (such as the pass rate of SWE-Bench) is no longer sufficient.

### Implications for Industry

The Opus 4.6 Eval Suite reveals a new phenomenon — **models are now capable of performing basic AI research tasks**. This means:

1. **Automation of Research Assistant Work**: Typical LLM RL training tasks can be completed by AI
2. a **Shift in Human Role**: From "doing research" to "guiding AI to do research"
3. **Meta-Questions in Evaluation**: How to evaluate the research done by models? We need higher-dimensional benchmarks

This discovery also directly drives alignment research — if models can perform research on their own, the alignment problem becomes more urgent ([Chapter 25: Scalable Oversight](../chapter30_alignment_failures/classical-failures)).

## Standardized Evaluation Harnesses

Industrial-level RL evaluation cannot be done manually — it must have standardized evaluation harnesses. Below are four mainstream harnesses.

### lm-evaluation-harness (EleutherAI)

[EleutherAI lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness) is the de facto standard:

- **Coverage**: 200+ benchmarks (MMLU, GSM8K, HellaSwag, TruthfulQA, etc.)
- **Interface**: Unified `lm.eval()` API supporting HuggingFace, OpenAI, and Anthropic models
- **Reproducibility**: Fixed random seed, prompt templates
- **Decontamination**: Built-in 13-gram decontamination check

```python
import lm_eval
from lm_eval.models.huggingface import HFLM

model = HFLM(pretrained="meta-llama/Llama-3-70B")
results = lm_eval.simple_evaluate(
    model=model,
    tasks=["mmlu", "gsm8k", "hellaswag"],
    num_fewshot=5,
    batch_size=64
)
```

Suitable for large-scale capability evaluation.

### BigCode Eval

[BigCode Eval Harness](https://github.com/bigcode-project/bigcode-evaluation-harness) focuses on **code generation**:

- **HumanEval**: Python function generation
- **MBPP**: Basic Python programming
- **DS-1000**: Data science tasks
- **MultiPL-E**: Multi-language code (Python, JS, Java, C++)
- **APPS**: Competitive algorithm problems

```python
from bigcode_eval import run_eval
run_eval(
    model="deepseek-ai/deepseek-coder-33b",
    tasks=["humaneval", "mbpp", "ds1000"],
    pass_at_k=[1, 5, 10]  # Report pass@1, pass@5, pass@10
)
```

### τ-bench (Tau-Bench)

[τ-bench, arXiv:2406.12045](https://arxiv.org/abs/2406.12045) is the **tool call benchmark** introduced by Salesforce in 2024:

- Simulates real-world business scenarios (airline, retail, telecom customer service)
- Models need to call APIs (check orders, change flights, refund)
- Multi-turn dialogue + tool calls + user simulation

```python
from tau_bench import run
run(
    agent=llm_agent,
    env="airline",  # Airline customer service scenario
    n_episodes=100,
    user_model="gpt-4"
)
# task success rate, average turns, API call accuracy
```

τ-bench reveals the actual capabilities of GPT-4 and Claude in real-world business scenarios — often 20-30 points lower than single-turn benchmarks.

### BFCL (Berkeley Function Calling Leaderboard)

[BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html) focuses on **function calling capability**:

- **AST Evaluation**: Whether the function call syntax is correct
- **Executable Evaluation**: Whether the call can actually be executed
- **REST API**: The ability to call external APIs
- **Java, JS**: Multi-language support

```python
# BFCL Evaluation
from bfcl_eval import eval_model
results = eval_model(
    model="claude-3-opus",
    test_categories=["simple", "multiple", "parallel", "rest"]
)
# overall accuracy, AST accuracy, executable accuracy
```

### Comparison of Four Harnesses

- **Harness — lm-eval-harness**
  - Use Case: General ability
  - Task Type: 200+ benchmarks
  - Evaluation Method: Automatic validation
- **Harness — BigCode Eval**
  - Use Case: Code generation
  - Task Type: Python/multi-language
  - Evaluation Method: Unit testing
- **Harness — τ-bench**
  - Use Case: Business Agent
  - Task Type: Tool calling + multi-round dialogue
  - Evaluation Method: Task completion rate
- **Harness — BF/C**
  - Use Case: Function calling
  - Task Type: API call syntax and execution
  - Evaluation Method: AST + Execution

### Suggestions for Selection

- **Basic Capability Assessment**: lm-eval-harness (most comprehensive)
- **Code Capability Assessment**: BigCode Eval + LiveCodeBench (continuously updated and anti-pollution)
- **Agent Capability Assessment**: τ-bench + SWE-Bench + WebArena
- **Tool Invocation Capability**: BFCL

The evaluation set should follow the model's intended use. A mathematics model does not need every customer-support environment, while a tool-using agent cannot be justified by MMLU alone.

## Summary of This Chapter

The core principles of RL evaluation methodology are as follows:

1. **Prioritize Verifiability**: Prefer benchmarks that are machine-verifiable
2. **Record contamination evidence**: n-gram overlap, semantic retrieval, fresh tasks, and temporal splits cover different risks
3. **Average Across Multiple Prompts**: Single prompt conclusions are unreliable
4. **Out-of-Distribution (OOD) Evaluation**: Three layers — capability evaluation, behavior evaluation, and long-term evaluation
5. **Standardized Harness**: The four major systems — lm-eval, BigCode, τ-bench, and BFCL — complement each other

Research-automation evaluations show that model performance depends on the complete environment, resource limits, and scoring protocol. [Appendix A.2](../appendix_industrial_training/rl-infrastructure) explains how to run these experiments on larger training systems while preserving the harness configuration.

## Further Reading

- [Cobbe et al. 2021 "Training Verifiers to Solve Math Word Problems" (GSM8K)](https://arxiv.org/abs/2110.14168)
- [Chen et al. 2021 "Evaluating Large Language Models Trained on Code" (HumanEval)](https://arxiv.org/abs/2107.03374)
- [Hendrycks et al. 2021 "Measuring Massive Multitask Language Understanding" (MMLU)](https://arxiv.org/abs/2009.03300)
- [Mizrahi et al. 2024 "State of What Art? A Call for Multi-Prompt LLM Evaluation"](https://arxiv.org/abs/2401.00595)
- [Blackwell et al. 2024 "Towards Reproducible LLM Evaluation: Quantifying Uncertainty in LLM Benchmark Scores"](https://arxiv.org/abs/2410.03492)
- [Perez et al. 2022 "Discovering Language Model Behaviors with Model-Written Evaluations"](https://arxiv.org/abs/2212.09251)
- [Sharma et al. 2023 "Towards Understanding Sycophancy in Language Models"](https://arxiv.org/abs/2310.13548)
- [Yao et al. 2024 "Tau-Bench: A Benchmark for Tool-Agent-User Interaction"](https://arxiv.org/abs/2406.12045)
- [Anthropic 2025 "Claude Opus 4.6 AI Research Eval"](https://www.anthropic.com/research/claude-opus-4-6)
- [Jain et al. 2024 "LiveCodeBench"](https://arxiv.org/abs/2403.07974)
- [Patil et al. 2024 "BFCL Berkeley Function Calling Leaderboard"](https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html)
