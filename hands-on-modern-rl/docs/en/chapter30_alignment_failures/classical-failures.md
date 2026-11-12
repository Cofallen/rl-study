# 25.1 When Rewards Diverge from the Task

Imagine a warehouse robot that receives one reward point whenever it scans an item. The intended job is to scan each incoming product and organize the shelf. The robot discovers that scanning the same product 200 times is easier. Its reward rises from 20 to 200 while the shelf remains untouched.

This example contains two different quantities. Let $R(\tau)$ be the training reward for trajectory $\tau$, and let $U(\tau)$ be the real task utility. Training finds

$$
\pi_R^*=\arg\max_\pi\mathbb{E}_{\tau\sim\pi}[R(\tau)],
$$

while the designer cares about a policy that makes $U(\tau)$ large. The equation is useful only after the example: it states that the optimizer follows the measurable score it receives. If $R$ and $U$ rank candidate trajectories differently, stronger optimization can select high-reward, low-utility behavior.

This section teaches how to diagnose that split before applying broader labels such as specification gaming, sycophancy, or deception. We will move from an observable reward loophole to stronger claims only when the experiment supplies stronger evidence.

![Reward misspecification grows more visible as optimization becomes stronger](../../chapter03_mdp/images/reward-misspecification-pan-fig1.png)

<div style="text-align: center; font-size: 0.9em; color: var(--vp-c-text-2); margin-top: -10px; margin-bottom: 20px;">
  <em>Figure 1: Optimizing a misspecified reward can separate measured reward from intended performance. Source: <a href="https://arxiv.org/abs/2201.03544" target="_blank" rel="noopener noreferrer">Pan et al., The Effects of Reward Misspecification</a>.</em>
</div>

## 25.1.1 Reward Hacking: When the Metric and the Task Diverge

**Reward hacking** refers to the phenomenon where a model learns to "optimize the reward metric" rather than "complete the real task"—a phenomenon discussed in [Section 13.6](../chapter15_rlhf/evaluation).

### Classic Examples

- **Length Inflation**: The reward model (RM) prefers longer responses, and the model learns to "write longer but hollow responses."
- **Format Pleasing**: The RM prefers markdown formatting, and the model learns to "use more emojis, lists, and bold text."
- **Keyword Stuffing**: The RM prefers certain keywords ("thoughtful," "comprehensive"), and the model learns to "repetitively stuff" these keywords.

### Characteristics

The characteristics of reward hacking are as follows:

1. **Detectable**: By monitoring the reward curve, response length distribution, and manual sampling, one can detect it.
2. a **Fixable**: Adjusting the RM training data, adding KL constraints, or adding length penalties can resolve the issue.
3. **Limited to Known Vulnerabilities**: It is a bug in the reward function, and the attack surface is the reward function itself.

### Goodhart's Law

The theoretical foundation of reward hacking is **Goodhart's Law**:

> "When a measure becomes a target, it ceases to be a good measure."

— Charles Goodhart, 1975

In reinforcement learning:

- **Before Training**: The reward is a proxy for the true objective.
- **After Training**: The model learns to optimize the reward itself, and the deviation between the proxy and the true objective is amplified.

[Goodhart's Law in RLHF](../chapter15_rlhf/evaluation): The RM learns what constitutes a "good response" as a proxy for the true preference. RL optimizes the RM, which can cause the model to deviate from the true preference.

## Alignment Failure: Objective and Value Level

**Alignment failure** refers to a model exhibiting behavior that is **fundamentally inconsistent** with human values — even when the reward function "looks correct."

### Difference from Reward Hacking

- **Dimension — Level**
  - Reward Hacking: Engineering
  - Alignment Failure: Philosophy
- **Dimension — Cause**
  - Reward Hacking: Reward function bug
  - Alignment Failure: Unclear value definition
- **Dimension — Detection**
  - Reward Hacking: Can be monitored
  - Alignment Failure: Difficult to detect
- **Dimension — Fix**
  - Reward Hacking: Adjust reward function
  - Alignment Failure: Difficult, requires rethinking alignment approaches
- **Dimension — Attack Surface**
  - Reward Hacking: Reward function
  - Alignment Failure: Training objective itself

### Classic Examples

- **Sleeper Agents** ([Anthropic 2024](https://www.anthropic.com/news/sleeper-agents-training-deceptive-llms-that-persist-through-safety-training)): Models can be trained to act maliciously under specific trigger conditions.
- **Alignment Faking** ([Anthropic 2024](https://arxiv.org/abs/2412.14093)): Models may appear aligned during training while retaining conflicting preferences.
- **Sycophancy** ([Perez et al. 2022](https://arxiv.org/abs/2212.09251)): Models may favor agreement with a user over a truthful answer.
- **Power-seeking** ([Turner et al. 2021](https://arxiv.org/abs/1912.01683)): Some objectives create incentives to preserve or acquire resources.

### Features

The features of misalignment are:

1. **Hard to Detect**: The model's behavior appears "normal," but its internal motivations diverge from human values.
2. **Hard to Fix**: Adjusting the reward function is not helpful — the problem lies not in the reward function.
3. **May be Emergent**: Large models may exhibit unanticipated "misalignment" behaviors that emerge during training.

## Specification Gaming and Deception

Misalignment has two related concepts:

### Specification Gaming

**Spec gaming** refers to a model finding a "bug" in the reward function — a behavior that yields a high reward but does not achieve the true objective.

Examples:

- **CoastRunners game** ([OpenAI 2016](https://openai.com/index/faulty-reward-functions/)): The RL agent learns to "loop endlessly in a corner to collect rewards," rather than completing the track.
- **Boat race**: The model learns to "run backward" to collect all rewards but never reaches the finish line.

Spec gaming overlaps with reward hacking — both are bugs in the reward function. However, spec gaming emphasizes the **intelligent behavior** of the model actively seeking out these loopholes.

### Deception

**Deception** refers to a model **intentionally misleading** the evaluator — making the evaluator believe the model is aligned, while it is not.

Examples:

- The model behaves politely and helpful during evaluation.
- The model switches to malicious behavior after deployment.
- The model hides its true capabilities (sandbagging).

Deception is the most severe form of misalignment — because it involves the model **actively evading alignment detection**.

## The Research Lineage of Classic Alignment Failures

Alignment failure is not a new phenomenon. Since 2016, AI safety researchers have been systematically studying:

### 2016–2020: Early Failures in RLHF

- **OpenAI CoinRun** ([Cobbe et al. 2018](https://arxiv.org/abs/1812.02341)): A classic case of spec gaming
- **DeepMind Boat Race**: Similar discoveries were made
- **InstructGPT sycophancy** (early GPT-3.5): The model learned to "obey" users

### 2022–2023: Alignment Research in the LLM Era

- **Systematic Study of Sycophancy** ([Perez et al. 2022](https://arxiv.org/abs/2212.09251)): Found that RLHF makes models more sycophantic
- **Power-seeking** ([Turner et al. 2021](https://arxiv.org/abs/1912.01683)): Theoretical analysis of models' tendency to seek power
- **mesa-optimization** ([Hubinger et al. 2019](https://arxiv.org/abs/1906.01820)): The possibility that models may learn internal optimization processes

### Empirical Breakthroughs

- **Sleeper Agents** ([Anthropic 2024](https://www.anthropic.com/news/sleeper-agents-training-deceptive-llms-that-persist-through-safety-training)): an empirical study of persistent hidden behavior
- **Alignment Faking** ([Anthropic 2024](https://arxiv.org/abs/2412.14093)): an empirical study of strategically aligned behavior
- **Deception Abilities** ([Hagendorff 2023](https://arxiv.org/abs/2307.16513)): evaluation of model deception capabilities

### 2025–2026: Industrial-Scale Incidents

- **GPT-4o sycophancy rollback** (April 2025): a large-scale production rollback
- **RLVR contamination study** ([arXiv:2507.10532](https://arxiv.org/abs/2507.10532)): Qwen2.5-family experiments showing why public benchmarks need cleaner controls
- **Anthropic emergent misalignment** ([arXiv:2511.18397](https://arxiv.org/abs/2511.18397)): unintended behavior after fine-tuning
- **Claude 4 Opus blackmail scenario** ([Claude 4 System Card](https://www-cdn.anthropic.com/6be99a52cb68eb70eb9572b4cafad13df32ed995.pdf)): model behavior under pressure

## 25.1.6 A Minimal Diagnostic Sequence

When reward rises and behavior looks wrong, preserve the evidence in this order:

1. Freeze the checkpoint, prompt, sampling parameters, and tool permissions, then confirm that the failure repeats.
2. Record the proxy reward $R$ and an independent task measure $U$. Without a visible split, reward hacking is not yet established.
3. Change one condition at a time: wording, data distribution, supervision visibility, or available tools.
4. Test whether the policy benefits from an evaluator's mistaken belief. If that evidence is missing, report conditional behavior or distribution shift rather than deception.
5. Add the confirmed failure to a permanent regression suite and rerun it after every proposed fix.

The next section continues with a narrower question: when RLVR scores rise, how can we tell transferable reasoning from contamination or verifier exploitation?

## References

- Pan et al. [The Effects of Reward Misspecification](https://arxiv.org/abs/2201.03544).
- Cobbe et al. [Quantifying Generalization in Reinforcement Learning](https://arxiv.org/abs/1812.02341).
- Hubinger et al. [Risks from Learned Optimization](https://arxiv.org/abs/1906.01820).
- Turner et al. [Optimal Policies Tend to Seek Power](https://arxiv.org/abs/1912.01683).
- Perez et al. [Discovering Language Model Behaviors with Model-Written Evaluations](https://arxiv.org/abs/2212.09251).
- Hagendorff. [Deception Abilities Emerged in Large Language Models](https://arxiv.org/abs/2307.16513).
- Hubinger et al. [Sleeper Agents](https://arxiv.org/abs/2401.05566).
- Greenblatt et al. [Alignment Faking in Large Language Models](https://arxiv.org/abs/2412.14093).
