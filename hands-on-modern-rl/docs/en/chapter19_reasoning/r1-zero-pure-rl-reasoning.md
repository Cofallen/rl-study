# 16.2 R1-Zero Pure Reinforcement Learning Reasoning

[16.1](./emergence-and-o1) explains how reasoning models enter the product. Now returning to the training process: if we do not provide the model with manually written thought chains, but only the problem and an automatically verifiable answer, can reinforcement learning enable the model to learn to reason through the problem? DeepSeek-R1-Zero answers this question with a single public experiment.

[Chapter 15](../chapter18_grpo/grpo-practice-and-mechanism) introduced GRPO, RLVR, and DAPO. Here we follow the R1-Zero training process: which model and data it starts from, how GRPO turns outcome rewards into policy updates, how longer reasoning and self-checking emerge, and why pure RL still needs cold-start data and later alignment.

The [DeepSeek-R1 paper](https://arxiv.org/abs/2501.12948), released in January 2025, starts R1-Zero directly from a pretrained base model without a supervised cold start on human-written chains of thought. GRPO uses only correctness and format rewards. Later checkpoints develop longer reasoning, self-checking, and changes of method; the paper calls one trajectory that actively re-evaluates its approach an “aha moment.”

Throughout this section, the same problem is repeatedly sampled. Suppose the model faces the equation $x^2 - 5x + 6 = 0$: some answers directly guess $x = 2$, some fully calculate $x = 2$ or $3$, and some have correct formulas but make substitution errors. The verifier only checks the final solution set, and GRPO compares the rewards of the same group of answers. As training repeats, the derivation methods that can stably produce correct results will appear more frequently.

<img src="../../chapter19_reasoning/images/r1-zero-training-loop.svg" alt="R1-Zero repeatedly samples response groups, verifies outcomes, and updates the policy with relative rewards">

## 1. What Model Does Pure RL Start With

For an outcome reward to provide useful information, training must start with three conditions: the model can understand the problem and generate candidate solutions, the task has a reliable verifier, and the model can sample both correct and incorrect answers to the same problem. If every sample fails, or if the task cannot be scored, the following loop has no useful comparison signal.

R1-Zero starts from a pre-trained base model and does not first use human thought chains for SFT. The training data is mainly composed of math, coding, and logic problems; each problem can be judged by a standard answer, a test program, or a rule-based verifier.

A training iteration consists of four steps:

```text
Sample a problem from the question bank
      ↓
Current model generates a group of answers
      ↓
Rule-based verifier calculates correctness and format rewards
      ↓
GRPO increases the probability of high-reward answers
      └────────────► Next round of sampling
```

The difference between this flow and traditional RLHF lies in the source of the reward. RLHF typically uses a reward model trained from human preferences; R1-Zero uses verifiable rewards, reducing the judgment error of the reward model itself.

### 1.1 Reward for Correctness

Math problems can extract the final answer and compare it with the standard answer, and code problems can run tests. Let the verifier be denoted as $v(x,y)$, where $x$ is the question and $y$ is the complete answer. The simplest reward is:

$$
r_{\text{acc}}(x,y)=
\begin{cases}
1,& v(x,y)=\text{correct},\\
0,& v(x,y)=\text{incorrect}.
\end{cases}
$$

This reward only judges the result, without specifying which method the model must use, nor providing a standard reasoning step for each step. The model needs to find the behavior that more easily leads to the correct answer through multiple samplings.

### 1.2 Reward for Format

Training also requires separating the reasoning part from the final answer to allow the verifier to stably extract the answer. The format reward checks whether the answer uses the agreed-upon thinking and answer tags:

```text
</think>

Thought process
```

The format reward addresses the problem of machine parsing and cannot replace the correctness reward. Optimizing only the format would result in answers that are structurally complete but contentually incorrect.

## 2. How GRPO Compares Responses Within a Group

Once the verifier provides scores for each response, the next step is to transform "who is better in this group" into parameter updates. GRPO uses the reward differences between responses to the same question, so the group must contain both better and worse samples.

For the same question, the current policy generates $G$ responses $y_1,\ldots,y_G$, and the verifier provides rewards $r_1,\ldots,r_G$. GRPO does not train an independent Critic, but instead constructs a relative advantage using the group mean and standard deviation:

$$
\hat A_i=\frac{r_i-\operatorname{mean}(r_1,\ldots,r_G)}
{\operatorname{std}(r_1,\ldots,r_G)+\epsilon}.
$$

Responses with above-average rewards receive positive advantages; those below the group average receive negative advantages. The policy update raises the probability of the former and lowers the probability of the latter. PPO-style clipping limits each update, while the KL term keeps the current policy close to the reference policy.

Let's take a small example with four responses. If the rewards are $(1,1,0,0)$, the mean is $0.5$, and the standard deviation is also approximately $0.5$, the normalized advantage is approximately $(1,1,-1,-1)$. The first two correct responses receive positive weights, and the last two incorrect responses receive negative weights. If the rewards become $(1,1,1,1)$, after subtracting the mean, all responses are zero, meaning this question no longer provides a discriminative signal in the current batch. The $\epsilon$ in the formula is used to avoid division by zero when the standard deviation is zero.

### 2.1 Which Problems Provide Learning Signals

If all answers in a group are correct, all rewards are the same, and the group advantage is close to zero. If all answers are incorrect, the same issue arises. Effective problems typically fall within a difficulty range where the current model sometimes answers correctly and sometimes incorrectly.

Therefore, pure RL training also requires managing problem difficulty:

- **Easy problems** should be retained in small quantities for capability regression, while most are downweighted in sampling.
- **Moderately difficult problems** provide the main gradient.
- **Very difficult problems** need to be split, supplemented with tools, or we must wait for the base model's capability to improve.

This is the same issue as the dynamic sampling in [Chapter 15 DAPO](../chapter18_grpo/deepseek-dapo): concentrating computation on problems where there is still group-level variation.

For a group of eight binary-reward responses and a per-sample success rate $p$, the probability that the whole group is all correct or all wrong is $p^8+(1-p)^8$. It is about $0.43$ at $p=0.9$ or $0.1$, about $0.06$ at $p=0.7$ or $0.3$, and only about $0.008$ at $p=0.5$. Problems of medium difficulty therefore produce mixed outcomes—and usable gradients—far more reliably.

## 3. How Self-Checking Emerges in Training

Intra-group comparisons can increase the probability of correct answers, but there is no direct rule written as "Please check your calculations." To understand why self-checking emerges, we need to observe which generated behaviors lead to higher final rewards.

In the early stages of training, answers are typically short, and the model often directly guesses the result. High-reward samples gradually encourage the policy to include more intermediate reasoning steps, as listing conditions, checking calculations, and trying alternative methods can increase the likelihood of getting the final answer right.

As training continues, we can observe several types of behaviors:

1. **Longer Reasoning Chains**: The model expands on more intermediate steps before arriving at an answer.
2. **Rechecking**: The model reviews previous calculations, identifies contradictions, and revises its conclusion.
3. **Switching Methods**: When the current approach stalls, the model shifts to a different proof or algorithm.
4. **Self-Verification**: The model substitutes the obtained answer back into the conditions to check whether it satisfies the problem.

These behaviors are not labeled step-by-step, but are indirectly selected by the final correctness reward. Any behavior that consistently improves success rate will appear more frequently in subsequent sampling.

For example, suppose a short direct guess is sampled with probability $0.7$ but succeeds only $10\%$ of the time, contributing $0.7\times0.1=0.07$ expected reward. A longer derive-and-check strategy is sampled with probability $0.3$ but succeeds $60\%$ of the time, contributing $0.3\times0.6=0.18$. Outcome optimization gradually shifts probability toward the longer strategy even though no rule explicitly rewards length or self-checking.

### 3.1 How to Understand the Aha Moment

The Aha Moment in the R1-Zero report refers to the model's ability to pause during a reasoning process and write something like "Try a different approach to verify," indicating that outcome rewards can reinforce reflective behavior, but cannot prove that reinforcement learning from scratch has created reasoning capabilities.

The base model has already seen proofs, code debugging, and self-correction texts during pre-training. The role of pure RL is to make these latent behaviors more consistently serve verifiable tasks. [16.1](./emergence-and-o1) Therefore, it distinguishes between capability emergence and capability activation.

## 4. Why Cold Start and Alignment Are Still Needed

Outcome rewards answer the question of "whether the answer has been verified," but do not answer "whether the reasoning is readable, whether the length is appropriate, whether the answer is helpful for open-ended questions, or whether tool usage is safe." Therefore, R1-Zero is an experiment in training mechanisms, not yet a complete post-training process for a product.

R1-Zero demonstrates that reasoning can be reinforced without relying on human thought chains, but the public results also expose the boundaries of pure RL:

- Reasoning chains may mix multiple languages, leading to unstable reading experiences.
- Answers may keep getting longer to improve accuracy, increasing latency and cost.
- Formatting may drift, causing answer extraction and tool protocol failures.
- Rule-based rewards only cover verifiable tasks and cannot directly handle writing, helpfulness, and safety preferences.
- When the verifier has vulnerabilities, the model may learn to exploit the rules without improving real-world capabilities.

The full DeepSeek-R1 therefore adds a cold start SFT, reasoning RL, rejection sampling, and a general alignment phase. Cold start samples are responsible for establishing readable formats and basic behaviors, while pure RL continues to improve the model's performance on verifiable tasks. Subsequent data feedback and alignment then correct language, style, and safety issues.

This division also explains why the following sections are necessary:

| Subsequent Questions                                                    | Corresponding Content                                                        |
| ----------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| Does computing more during reasoning always improve accuracy?           | [16.3 Test-Time Compute Scaling](./test-time-scaling)                        |
| How to limit thinking length and switch modes?                          | [16.4 Hybrid Thinking and Budget Control](./hybrid-thinking)                 |
| Can the model automatically allocate computation based on the question? | [16.5 Adaptive Thinking](./adaptive-thinking)                                |
| Should the reasoning chain be shown and aligned with the process?       | [16.6 Display and Alignment of Reasoning Chains](./cot-visibility-alignment) |

## Summary of This Section

- R1-Zero directly applies reinforcement learning to the base model, with training signals derived from automatically verifiable correctness rewards and formatting rewards.
- GRPO estimates relative advantage using a set of answers to the same question, eliminating the need for an independent Critic.
- Long reasoning, reflection, method switching, and self-validation are reinforced because they improve the final success rate.
- Pure RL still suffers from issues of readability, length, general capability, and reward loopholes, so full training requires cold start, data reflux, and subsequent alignment.
