# 17.1 Outcome Rewards and Process Rewards

Chapter 16 gave the model a longer reasoning chain and a larger inference budget. Once a trajectory grows longer, a single score at the end carries less information. Suppose the model chooses the right formula, copies one symbol incorrectly halfway through, and reaches a wrong answer. An outcome verifier can return 0, but that score does not say which earlier steps should be preserved or where the first error occurred.

The same problem appears in coding and agent tasks. A failed test may come from opening the wrong file at the beginning or from a syntax error in the final line. Likewise, one final score cannot tell us how much each of ten tool calls contributed to a failed task.

When a sequence receives feedback only at the end, its intermediate actions have no direct learning signal. This is the **sparse-reward problem**. A **process reward model (PRM)** evaluates intermediate steps, helping the training system locate an error and the inference system abandon a bad path before the error spreads. We begin with a short proof, formalize outcome and process rewards, and then examine when step-level evaluation is worth its cost.

[Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) systematically compared outcome and process supervision and released the PRM800K step-label dataset. Its conclusions come from MATH under a particular candidate-generation and reranking setup. We first derive the credit-assignment difference here, then return to those data and metrics in Section 17.2.

<img src="../../chapter20_prm_search/images/outcome-process-feedback.svg" alt="Outcome reward hides intermediate errors, while process feedback identifies the first incorrect step">

## 1. Why Final Rewards Are Insufficient

Continuing from the mathematical model introduced in Chapter 16, the model presents a nine-step proof, and the final conclusion happens to be correct. The verifier marks it as 1, and the training system thus increases the probability of the entire response. However, the sixth step missed a square, and later coincidentally returned to the correct conclusion. By only looking at the end result, the system will reinforce both the incorrect steps and the correct ones.

### 1.1 Distinguishing Between "Answering Correctly" and "Reasoning Correctly"

Outcome reward evaluates the final result of the task, similar to an exam that only records the final answer: a correct answer is awarded 1, and an incorrect one is awarded 0. It is cost-effective, stable, and suitable for mathematical answers, unit tests, and rule-based checks.

Process reward evaluates intermediate steps, similar to grading the process of solving a problem. It can identify which steps can still be continued and which steps have already violated the initial assumptions. The PRM is responsible for evaluation, while the Policy is responsible for generation; the evaluation signal can be used for training or to select a path during inference, but it will not replace the Policy in completing the answer.

The two types of rewards address different problems. The outcome reward determines whether the entire task has been completed, while the process reward determines whether the task is still on the correct path at this stage. Short answers typically only require the former, while longer trajectories benefit more from the latter.

### 1.2 This Chapter Follows the Same Error Trajectory

The next five sections all deal with the issue of "Step Six Failing," but the approach to checking it gradually changes:

| Section                    | Feedback Used                              | Problem to Solve                                                            |
| -------------------------- | ------------------------------------------ | --------------------------------------------------------------------------- |
| 17.2 Discriminative PRM    | good / bad / neutral probability           | How to quickly score a large number of steps                                |
| 17.3 Generative PRM        | Natural language inspection and conclusion | How to explain where the mistake is and reduce the need for manual labeling |
| 17.4 Formal Verifier       | Proof checker pass or fail                 | How to obtain deterministic feedback in formalizable tasks                  |
| 17.5 Inference-Time Search | Intermediate node scores                   | How to backtrack and try alternative paths before the error expands         |
| 17.6 Parallel Reasoning    | Multiple complete paths and coordinator    | How to compare different solution approaches and aggregate the final answer |

[Chapter 13 on RLHF](../chapter15_rlhf/standard-rlhf-pipeline) introduced outcome reward models, and [Chapter 15 on GRPO](../chapter18_grpo/grpo-family) showed how outcome rewards drive policy updates. Chapter 17 adds intermediate feedback. Later, [Agentic RL](../chapter22_agentic/overview) extends the same problem to tool actions and environment states, while [reward hacking](../chapter30_alignment_failures/classical-failures) examines what happens when the evaluator itself can be exploited.

### 1.3 From a Final Score to Step-by-Step Feedback

There is only one final reward score, but process rewards require first deciding what constitutes a "step." Mathematical proofs can be split by equation transformations, code tasks can be divided into editing, compiling, and testing, and Agent tasks can be split by a single tool call. Only after the boundaries of steps are determined can the evaluator assign scores to specific positions.

Consider the following example of a proof to illustrate how intermediate errors are missed by the final reward when the conclusion is correct.

### 1.4 How Sparse Rewards Mask Intermediate Errors

Consider a specific example: let the model prove that "√2 is irrational."

The model's Chain-of-Thought (CoT) might look like this (simplified version):

```text
Step 1: Assume √2 = p/q, where p, q are coprime
Step 2: Then 2 = p²/q², i.e., p² = 2q²
Step 3: So p² is even
Step 4: So p is even (this step uses the contrapositive of "the square of an even number is even")
Step 5: Let p = 2k
Step 6: Substituting: 4k² = 2q², i.e., 2k² = q²
Step 7: So q² is even, and q is even
Step 8: This contradicts the assumption that p and q are coprime
Step 9: Therefore, √2 is irrational ✓
```

Suppose there is an error in Step 6 — for example, the model writes "4k² = 2q², i.e., 4k = q²" (missing the square). The final conclusion "√2 is irrational" is still correct (the conclusion is correct), but the reasoning process contains an error.

If the outcome verifier checks only the conclusion "√2 is irrational," the entire response will still receive $r=1$. During training, the correct prefix from Steps 1–5, the error in Step 6, and the correct conclusion afterward all share the same reward. The model has no way of knowing that Step 6 should be suppressed, nor can it determine whether the earlier definitions and equations should be retained.

This is the attribution difficulty caused by sparse rewards: a long trajectory receives only a single score at the end, and if that score happens to be correct, the incorrect steps along the entire trajectory may also be reinforced. The process reward aims to supplement the information of "where the error first appeared."

## 2. How to Distribute Rewards to Intermediate Steps

The sparse reward problem in RL has a more formal name: the **credit assignment problem (CAP)**.

Given a sequential decision task, with a final reward of $ r_T $, the credit assignment problem asks how to propagate this result back to $ a_1, a_2, \ldots, a_T $: which actions contributed to success, and which introduced errors.

Classic RL addresses this problem with several approaches:

### 2.1 Discounted Return

Discount future rewards to the present:

$$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \ldots + \gamma^{T-t} r_T$$

Here, $ G_t $ is the discounted return starting from step $ t $, $ r_t $ is the current reward, and $ \gamma \in [0,1] $ is the discount factor. If only the final reward $ r_T = 1 $ is present, the return at step $ t $ is $ \gamma^{T-t} $; the farther away from the end, the smaller the weight. This is reasonable in many control tasks, but the early definition in the proof may determine the responsibility of all subsequent steps, and simply decaying by distance may not accurately reflect their responsibility.

With $\gamma=0.9$, a first step eight positions from the end receives weight $0.9^8\approx0.43$; a fifth step four positions away receives about $0.66$; the erroneous sixth step three positions away receives about $0.73$; and the penultimate step receives $0.90$. Distance-based discounting therefore gives the earliest assumptions the weakest signal even when they determine every later step.

### 2.2 GAE

[Chapter 8: PPO's GAE](../chapter10_ppo/gae-reward-model) introduces the $\lambda$ parameter to strike a balance between bias and variance. GAE requires a value function; GRPO omits the value function, thus necessitating the construction of advantage using intra-group comparisons or other feedback mechanisms.

### 2.3 Token-Level Loss

[DAPO](../chapter18_grpo/deepseek-dapo) computes the loss at the token level, but the supervisory signal still originates from the reward of the entire response. Consequently, each token receives a different gradient weight, yet there is no independent verifier to judge whether "this step is correct." It remains distinct from step-level PRM.

### 2.4 PRM

A PRM trains an independent verifier to score each reasoning step. Compared with one reward at the end of the response, it provides denser feedback and lets the system compare specific steps.

## 3. Difference Between Outcome Reward and Process Reward

The previous proofs have already illustrated the intuitive differences. Below, we clarify the two types of evaluators by using input and output.

### 3.1 Outcome Reward Model

The Outcome Reward Model (ORM) accepts a prompt $q$ and a complete response $o$, and outputs a scalar score:

$$\text{ORM}(q, o) \in \mathbb{R}$$

Here, $q$ is the question, and $o$ is the complete response. The output is a scalar. In mathematical tasks, it can directly take the value 0 or 1; for open-ended tasks, the reward model can output a continuous score. Regardless of the form, the entire response has only one final outcome.

ORM training data format:

```text
(prompt, response, final_correctness)
```

Example: (`"Prove that √2 is irrational"`, `"<complete proof>"`, `1`)

### 3.2 Process Reward Model

The Process Reward Model (PRM) takes a prompt $q$, a response $o$, and a step position $i$ within the response, and outputs a score for that step:

$$\text{PRM}(q, o, i) \in \mathbb{R}$$

The position $i$ indicates the step to be evaluated. The same prompt and response can receive different scores depending on the value of $i$, allowing the identification of where the response begins to become unreliable. In mathematical tasks, the scores can be:

- Discrete labels: $+1$ (correct), $0$ (neutral), $-1$ (incorrect)
- Continuous: a probability in the range [0, 1]

The training data format for PRM is:

```text
(prompt, response, step_index, step_correctness)
```

Example: (`"Prove that √2 is irrational"`, `"<complete proof>"`, `4`, `1`) — the fourth step is correct.

### 3.3 How Two Types of Rewards Enter RL Training

When ORM is used in RL training:

$$r_{\text{ORM}} = \text{ORM}(q, o)$$

The entire sequence shares a single reward.

When PRM is used in RL training (a common practice):

$$r_t = \text{PRM}(q, o, \text{step}(t))$$

Each token $t$ receives a corresponding PRM score based on "which reasoning step it belongs to." Tokens within the same reasoning step share the score of that step.

This approach transforms a terminal reward into multiple step rewards. If a step contains multiple tokens, one must decide whether these tokens share the score, receive the score only at the end of the step, or propagate the score forward through return calculation; the PRM score does not automatically become a "each token is completely correct" label.

## 4. When Is Process Reward Needed

A long reasoning chain does not automatically require a PRM. We should first ask whether the task has a reliable outcome verifier. A mathematical answer can be substituted into the original equation, code can be tested, and a game state can reveal a win or loss. When outcome verification is accurate and sampling is affordable, an ORM can still train a model to check and revise its behavior. [DeepSeek-R1](https://arxiv.org/abs/2501.12948) provides an example: R1-Zero relies mainly on rule-based accuracy and format rewards, yet develops longer reasoning and self-checking behavior during RL.

Process reward is particularly valuable in another scenario: the final result can only tell us that something has failed, but cannot pinpoint where the failure occurred. We can check three conditions in sequence:

1. Can the task split into steps with stable meaning, such as one equation transformation, one code modification, or one tool call?
2. Will intermediate errors propagate continuously, causing the subsequent generation and environment interaction to waste budget?
3. Can the evaluator judge the current step more reliably than the generative model?

Only when all three conditions are met can PRM potentially offset the costs of data annotation, model reasoning, and misjudgment. Different task lengths can be understood as follows:

| Task Scenario                                  | What Happens When Only ORM Is Used                                         | Potential Value Added by PRM                           |
| ---------------------------------------------- | -------------------------------------------------------------------------- | ------------------------------------------------------ |
| Short answer (a few hundred tokens)            | The error location is close to the end, and the cost of re-sampling is low | Usually limited                                        |
| Proof or program with multiple transformations | The final failure cannot indicate which step should be retained            | Identify the first error and retain the correct prefix |
| Long tool trajectory (thousands of tokens)     | Early errors will continue to trigger invalid calls                        | Early stopping, rollback, or switching paths           |

The number of tokens here is only illustrative of the task scale and should not be used as a fixed threshold for selecting PRM. When making a judgment, one should consider whether the step is well-defined, whether errors can propagate, and whether the evaluator is sufficiently reliable.

### 4.1 How the Three Types of PRM Provide Step Feedback

After determining the need for intermediate feedback, one must also choose the source of the feedback. The following three sections introduce classification models, natural language evaluation, and formal checkers.

#### Discriminative PRM

Treat the PRM as a classifier: given a question and the reasoning up to the current step, output the probability that the step is correct, incorrect, or neutral.

Representative work: OpenAI's [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) (Lightman et al. 2023).

Training data: Manually annotated step correctness (PRM800K dataset).

Model: An encoder, or a classification head added to a decoder-only LLM.

#### Generative PRM

Let the LLM generate a natural language evaluation, and then derive from the evaluation whether the current step can continue and the reason for any error.

Representative work: [ThinkPRM](https://arxiv.org/abs/2504.16828) (2025.04).

Training data: Manually curated seed examples, as well as evaluation data generated by the model.

Model: A general-purpose LLM can be used, learning the evaluation format through prompting or fine-tuning.

#### Formal PRM

Translate the step into a formal language such as Lean or Coq, and then pass it to a proof checker for validation.

Representative work: DeepMind's [AlphaProof](https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/) (2024.07), DeepSeek's [DeepSeek-Prover-V2](https://arxiv.org/abs/2504.21801) (2025.04).

## Training Data

Formalized mathematical theorems in Lean4 format.

System: A model responsible for generating formal proofs, combined with an independent proof checker.

These three approaches are the focus of the next three sections: [17.2 Discriminative PRM](./discriminative-prm), [17.3 Generative PRM](./generative-prm), and [17.4 Formal Verifier](./formal-prm).

Outcome rewards determine whether the task is completed, while process rewards assess whether the reasoning can continue at a certain step. The longer the task, the more expensive error propagation becomes, making intermediate feedback more beneficial. If the final result is easy to verify, ORM can still serve as the primary training signal.

There are three common sources of process feedback:

- **Discriminative**: Train a classifier using step labels, which is fast but depends on annotated data and domain coverage;
- **Generative**: Generate check comments and conclusions, which can explain errors, but the evaluation itself may also be incorrect;
- **Formal**: Determined pass or fail from a proof checker, provided the task and proof can be correctly formalized.

Sections 17.2–17.4 elaborate on these three feedback types, while Sections 17.5–17.6 incorporate the evaluators into search and parallel reasoning processes.
