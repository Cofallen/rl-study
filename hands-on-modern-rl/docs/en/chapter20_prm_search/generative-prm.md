# 17.3 Generative PRM

In Section 17.2, the classifier only returns `bad: 0.96` after seeing an incorrect equation. This score is suitable for ranking, but it does not inform the trainer whether the error stems from a missing square, a wrong premise, or a correct current step with an error in the previous step. When moving to code, physics, or new domains, fixed labels also struggle to expose which part of the knowledge the evaluator is missing.

Generative PRM allows the language model to first write out the reasoning process and then provide the final judgment. For example, it can explicitly point out, "From $4k^2 = 2q^2$, only $2k^2 = q^2$ can be derived, and $k^2$ cannot be changed to $k$." Natural language explanations make judgments verifiable and reuse the base model's existing knowledge.

This section takes [ThinkPRM](https://arxiv.org/abs/2504.16828) as an example. It first compares the outputs of classification labels and generative evaluation, then examines how a small amount of seed data can train a verifier, followed by an analysis of the benefits of adding verification calculations, and finally explains when a cheaper discriminative PRM or reward-based approach should still be used.

<img src="../../chapter20_prm_search/images/generative-prm-pipeline.svg" alt="A generative PRM writes checking evidence, locates an error, and emits an extractable final judgment">

## 1. How to Generate Natural Language Evaluations

Generative evaluation still needs to complete the same task as discriminative PRM: reading the problem, the reasoning prefix, and the current step, and determining whether this step is valid. The difference lies in the output—classification probability is replaced by "evaluation rationale + final label."

### 1.1 Differences Between Discriminative and Generative Outputs

Discriminative PRM ([OpenAI PRM800K](./discriminative-prm)):

```text
Input: prompt + the i-th step of reasoning
Output: good / bad / neutral (three-class classification)
```

Model structure: a language model that reads the full prefix + a classification or scoring head, finally outputting class probabilities.

Generative PRM (ThinkPRM):

```text
Input: prompt + the i-th step of reasoning + "Please evaluate this step"
Output: a natural language evaluation + final judgment
```

Model structure: a standard LLM (decoder-only), with output as text.

### 1.2 How Generated PRM Reuses Existing Capabilities

The generated PRM has two conditions that can be leveraged:

**First, reuse the pre-trained knowledge of the LLM.**

Discriminative PRM requires training a classification head and calibrating the boundaries of good, bad, and neutral steps. In contrast, the generated PRM uses a pre-trained language model, which may already have some knowledge of mathematics, code, and logic. Fine-tuning primarily teaches the model to check each step in a consistent format and provide a conclusion.

**Second, natural language can retain the reasoning basis.**

Discriminative PRM outputs good, bad, or neutral, which is suitable for fast sorting, but does not explain why a step is wrong. Generated PRM can point out the rules used, the values substituted, and the location of the error, allowing trainers to verify the reliability of the judgment.

Generated PRM can output: "This step intended to use factorization, but wrote $x^2 - 1$ as $(x-1)(x-2)$, the correct form should be $(x-1)(x+1)$, therefore the current step is incorrect." Such evaluations allow trainers to directly see where the error occurs; whether to use the explanation as a training target depends on the specific method.

### 1.3 How ThinkPRM Constructs Training Data

[ThinkPRM](https://arxiv.org/abs/2504.16828) (2025.04) starts from the questions, problem-solving prefixes, and human step labels in the PRM800K dataset. The labels still exist, but their purpose has changed: they are no longer used to directly train a classification head, but instead to filter the language model's generated checking process.

#### Generating Step-by-Step Evaluations

Training samples require the model to check each step based on the reasoning and output a judgment of correct or incorrect for each step:

```text
Prompt: Evaluate the third step of the following reasoning:
Problem: Solve the equation $x^2 - 5x + 6 = 0$
Reasoning:
  Step 1: This is a quadratic equation, and the quadratic formula can be used.
  Step 2: Quadratic formula: $x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$
  Step 3: Substituting $a = 1$, $b = -5$, $c = 6$, $x = \frac{5 \pm \sqrt{25 - 24}}{2} = \frac{5 \pm 1}{2}$
  Step 4: $x = 3$ or $x = 2$
Please evaluate Step 3.

ThinkPRM Output:
Let me check Step 3:
- $a = 1$, $b = -5$, $c = 6$ ✓ (consistent with the original equation)
- Quadratic formula: $x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$ ✓
- Substituting: $x = \frac{5 \pm \sqrt{(-5)^2 - 4 \cdot 1 \cdot 6}}{2 \cdot 1}$
       = $\frac{5 \pm \sqrt{25 - 24}}{2}$
       = $\frac{5 \pm \sqrt{1}}{2}$
       = $\frac{5 \pm 1}{2}$ ✓
Step 3 is correct.
Judgment: CORRECT
```

#### Generating, Filtering, and Fine-tuning

Data construction proceeds in three steps:

1. Use QwQ-32B-Preview to generate multiple verification CoTs for the problem-solving prefixes in PRM800K;
2. Retain only samples that are parseable in format, make step-by-step judgments consistent with human labels, and have lengths not exceeding the upper limit;
3. Fine-tune reasoning models such as R1-Distill-Qwen and QwQ using the retained verification CoTs.

The final paper uses approximately 1,000 verification CoTs, containing about 8,000 step judgments. To obtain these samples, the system first generates approximately 5,000 candidates, which are then filtered by the PRM800K labels. Here, the scale of the labels used for fine-tuning is reduced, but high-quality step labels are not completely eliminated; when there are no human labels, rollout estimation can be used to estimate silver labels, although noise will be introduced into the filtering process.

### 1.4 How to Choose Between Step-wise Evaluation and Full-trajectory Evaluation

During deployment, one can evaluate each step individually or hand the entire reasoning process to the model, asking it to output judgments for each step sequentially. Step-wise evaluation always focuses on a clearly defined action, making it easier to locate the first error, although it requires multiple model calls. Full-trajectory evaluation only needs to read the context once, resulting in higher throughput, but may miss minor errors in long trajectories.

The two approaches can also be combined: first use full-trajectory evaluation to filter out low-scoring trajectories, and then perform step-wise verification on suspected error points. When making a choice, one should simultaneously measure the first-error localization rate, the number of tokens per trajectory, and the total latency, rather than only comparing the speed of a single call.

## 2. How Do Labels and Verification Computation Affect Performance?

Generative PRM reduces the number of step labels required for fine-tuning, but increases the generation cost during each evaluation. The paper examines this trade-off from three perspectives.

### 2.1 Can a Small Number of Process Labels Train an Effective Verifier?

ThinkPRM uses approximately 8,000 process labels, which is about 1% of the 712,000 labels used by the discriminative baseline in the paper. Under the given settings of ProcessBench, MATH-500, and AIME 2024, it outperforms the discriminative baseline and the un-tuned LLM-as-a-Judge in step validation, Best-of-N, and verifier-guided search. This conclusion relies on the base model already possessing strong reasoning capabilities, as well as the human labels filtering the generated evaluations.

### 2.2 Can Mathematical Training Transfer to Science and Code?

The paper also tests domain transfer on subsets of GPQA-Diamond and LiveCodeBench. Compared to the discriminative verifier trained on the full PRM800K, ThinkPRM achieves improvements of 8 and 4.5 percentage points respectively. These results support the idea that "generative evaluators can reuse base knowledge," but the evaluation only covers a subset of the data. Therefore, re-calibration is still needed before applying to new codebases or scientific domains.

### 2.3 Verifiers Can Also Increase Reasoning Computation

Generative verifiers can increase computation along two axes: either by parallel sampling multiple validation CoTs and aggregating them, or by having the same check continue to review and revise. The paper compares the two approaches under the same token budget, finding that the overall gap is not significant, with the parallel approach slightly better under some budgets. On the ProcessBench subset, ThinkPRM outperforms the un-tuned LLM-as-a-Judge by 7.2 percentage points.

This aligns with the pattern in Section 16.3: more verification computation is only useful when the evaluation model can generate complementary check paths. If the model lacks domain knowledge, multiple generations may simply repeat the same misjudgment.

## 3. When to Choose Discriminative or Generative PRM

The previous results indicate that generative evaluation can improve label efficiency, but it does not replace classifiers in all scenarios. When training requires evaluating a large number of steps per second, generating an explanation may be too slow; in research and educational settings, where the reasoning process needs to be examined, a probability alone is insufficient.

| Dimension                     | Discriminative PRM                                                | Generative PRM                                                                         |
| ----------------------------- | ----------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| Output                        | Step category or probability                                      | Inspection process and extractable judgments                                           |
| Single Evaluation             | One forward pass, high throughput                                 | Requires autoregressive generation, high latency                                       |
| Data                          | Requires sufficient labels to calibrate classification boundaries | Still needs label filtering for generation, but can be fine-tuned with fewer samples   |
| Domain Transfer               | Depends on training coverage and base knowledge                   | Can invoke base knowledge in text, may also generate fluent misjudgments               |
| Adding Validation Computation | Can use ensemble or multiple models                               | Can increase generation length, review rounds, or parallel samples                     |
| Checkability                  | Only observe scores and error samples                             | Can read the judgment basis, but the basis is not guaranteed to be faithful or correct |

### 3.1 How Task Scale and Reasoning Cost Affect the Choice

Large-scale reinforcement learning requires evaluating a large number of steps per round. Typically, a discriminative model with higher throughput is chosen first. When debugging rewards, educational feedback, or low-frequency high-value tasks requires inspecting the reasons for errors, a generative model is more suitable. A limited data budget does not necessarily mean a limited inference budget: a generative PRM may save annotation effort, but it may use more tokens per call.

### 3.2 Mixing the Two Types of PRMs

When training details are not publicly available, one should not infer from product outputs whether a generative PRM is used internally. Public methods are sufficient to support three combinations:

1. **PRM and ORM Mixed**: Use PRM to evaluate the process, ORM to evaluate the result, and take a weighted sum.
2. **Generative + Discriminative Mixed**: A generative PRM provides high-quality evaluations, which are then distilled into a discriminative PRM for large-scale training.
3. **Self-PRM**: Let the model evaluate its own reasoning (consistent with the idea in [Self-Rewarding Language Models](https://arxiv.org/abs/2401.10020)).

## 4. Is Process Reward Always Necessary?

Even if a verifier can explain step by step, the system should first ask whether it is worth training a PRM. When answers can be automatically validated and the model can sample extensively, outcome rewards may indirectly reinforce self-checking. However, when step errors are hard to locate or the sampling budget is limited, process feedback is more advantageous.

[DeepSeek-R1](https://arxiv.org/abs/2501.12948) provides a contrast: R1-Zero mainly uses result correctness and format rewards, yet after training, it still observes behaviors such as checking steps and modifying answers. Outcome rewards increase the probability of correct final answers; if trajectories with self-checking are more likely to be correct, such behaviors may also be indirectly reinforced. Therefore, PRMs are one method for identifying intermediate errors, but not the only way to produce self-checking behavior.

When choosing a reward method, one can judge based on whether the feedback is sufficient: when answers can be automatically validated and the sampling volume is sufficient, outcome rewards may already be effective. When intermediate errors in long trajectories are hard to locate and the training budget is limited, PRMs can more quickly identify the location of errors. When applying across domains, one must also separately evaluate whether the PRM truly understands the new domain.

Therefore, PRMs are not a necessary condition for all tasks; they can still provide feedback that outcome rewards lack when the training budget is limited or when intermediate errors need to be located.

## Summary

ThinkPRM achieves results comparable to discriminative PRM using only about 1% of step labels in the paper's experiments, while retaining natural language justification. This advantage should be considered alongside the higher inference cost.

ThinkPRM demonstrates how to reduce the need for step-by-step manual annotation using natural language evaluation, while preserving the rationale for inspection by the trainer. The reliability of generative PRM is still constrained by the capability of the evaluation model: when the evaluation model is unfamiliar with a particular domain, it may produce explanations that appear complete but are actually incorrect.

[17.4 Formal Verifier](./formal-prm) delegates formalizable steps to external proof checkers like Lean4, replacing model-based judgment with deterministic verification.
