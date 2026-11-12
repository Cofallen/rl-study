# 18.3 Training Stability

> **Goal of this section**: learn the four-layer order for diagnosing training stability—data and rewards, policy change, numerical updates, and the training system. You will learn to read KL and entropy alongside loss and gradient norms, then use public cases to identify the layer at which a stability technique operates.

[18.2](./industrial-post-training) assembled the post-training loop: data → SFT → RL → evaluation → data feedback. Once that loop is running, the first thing people usually inspect is the reward curve.

Consider a real pattern from mathematical RL. A 7B model runs GRPO on MATH for 200 steps. Training reward rises smoothly from $0.30$ to $0.80$, yet MATH-500 accuracy remains at $42\%$, AIME pass rate falls from $18\%$ to $15\%$, and the average response grows from $400$ to $1{,}200$ tokens. The team first lowers the learning rate from $1\mathrm{e}{-6}$ to $3\mathrm{e}{-7}$. Reward keeps rising and evaluation remains flat. At step $312$, loss suddenly becomes NaN and the gradient norm reaches $10^4$, corrupting every subsequent parameter. Until that point, the rising reward curve had made training look healthy.

The same rising reward admits at least four explanations. The model may truly be learning to solve problems. It may merely have discovered that longer answers score better. The optimizer may already be driving parameters into an unstable region while NaNs propagate. Or the generation worker may sample with an old model while the trainer recomputes probabilities with a new one, so the logged probabilities do not belong to the policy that generated the trajectory.

A reward curve alone cannot distinguish these cases. Loss, gradient norm, KL, entropy, and independent evaluation must be compared on the same timeline. We will diagnose them in causal order—data and rewards, policy change, numerical updates, and the training system—then connect GLM, Llama 4, Seed-Thinking, and Kimi K2 to representative failures at those layers.

::: info Core idea
Training-stability diagnosis follows a four-layer causal order. Data and rewards determine what the model learns; policy metrics describe how it changes; numerical metrics show whether parameters can be updated normally; and the training system determines whether logged probabilities really come from the policy that generated each response. Optimizers and clipping operate only at the third layer. They cannot repair a wrong objective at the first layer or replace version alignment at the fourth. A tool applied at the wrong layer can make a stable bias even more stable.
:::

---

## A Four-Layer Diagnostic Framework

When four curves are on the screen, which should you inspect first?

The answer follows the causal order. Data and rewards determine what the model learns. Policy metrics describe how it changes. Numerical metrics describe whether the update is valid. The training system determines whether logged probabilities actually belong to the generation policy. An upstream error disturbs downstream curves, but a downstream tool cannot repair an upstream objective. Gradient clipping can limit one update; if the reward measures the wrong target, every clipped update still moves toward that wrong target.

| Layer             | Main Signal                                             | First Checks                                                        | Typical Tools                              |
| ----------------- | ------------------------------------------------------- | ------------------------------------------------------------------- | ------------------------------------------ |
| Data and rewards  | Reward rises while independent evaluation stays flat    | Reward rules, duplicates, evaluation contamination, response length | Rewrite rewards, clean data                |
| Policy change     | KL rises quickly and entropy falls quickly              | Update size, KL constraint, sampling difficulty, policy version     | Lower learning rate, tighten KL            |
| Numerical updates | Loss or gradients explode, or NaNs appear               | Learning rate, precision, anomalous batches, clipping               | Gradient clipping, optimizer               |
| Training system   | Probabilities disagree across engines or versions drift | Weight synchronization, tokenization, precision, MoE routing        | Align versions and recompute probabilities |

Work from top to bottom: first confirm that the target is correct, then check whether the policy moves too quickly, inspect individual numerical updates, and finally verify the distributed system. This avoids using numerical tools to treat a reward problem or using data cleaning to conceal a synchronization failure.

## Data and Rewards

The first layer asks one question: does the training reward measure the capability we actually want?

Return to the first 200 steps of the mathematics example. Reward rises from $0.30$ to $0.80$, MATH-500 accuracy remains at $42\%$, and average response length rises from $400$ to $1{,}200$ tokens. The model has learned an output form that earns reward—longer answers that cover more cases and trigger favorable parser decisions—without improving the target problem-solving ability.

Two other causes are common. First, the model may memorize repeated training tasks or exploit overlap between training and evaluation data. Reward then reflects memory rather than generalization. Second, it may exploit the verifier. A parser might recognize only a fixed format such as `\boxed{}`, awarding partial credit to a wrong answer with the right wrapper. A test harness might check new tests without ensuring that existing tests remain intact, teaching the model to delete old tests.

Inspect training examples, answer parsers, test environments, and reward direction before checking duplicates and evaluation contamination. This layer comes first because a wrong target makes stable optimization harmful: the optimizer faithfully amplifies its signal without judging whether the signal is correct.

## Policy Change

After confirming the target, measure how quickly the policy itself moves. KL measures the distance between the current output distribution and the reference model. Entropy measures how dispersed that distribution remains and therefore how much exploration is left.

Take a two-action reference policy $[0.5,0.5]$. Early in training, suppose the current policy is $[0.80,0.20]$. It already favors action A but retains some uncertainty:

$$
D_{\mathrm{KL}} = 0.80\ln\frac{0.80}{0.50} + 0.20\ln\frac{0.20}{0.50}
= 0.80\times 0.47 + 0.20\times(-0.92) \approx 0.19.
$$

The entropy of the same distribution is $-(0.80\ln0.80+0.20\ln0.20)\approx0.50$. If the distribution later concentrates to $[0.95,0.05]$, KL becomes about $0.49$ and entropy about $0.20$.

KL has more than doubled, so the model is moving away from its reference. Entropy has fallen by more than half, so its exploration space is shrinking. “The policy is moving too quickly” therefore means that KL increases too rapidly per training step while entropy falls at an accelerating rate. Neither signal is decisive alone: KL may rise during healthy departure from the SFT model, and entropy may fall because the model found a better solution. Together they show that the model is leaving the reference distribution while losing the capacity to correct a bad mode.

Real language-model actions span the full vocabulary, but the curves are read the same way. A steepening KL curve indicates movement away from the reference; accelerating entropy loss indicates collapsing exploration.

When these signals appear, compare the old sampling policy with the updated policy. Check the learning rate, clipping range, KL coefficient, and whether rollout experience has become stale. Also inspect task difficulty. If every task is easy, a model can reasonably concentrate on a few high-reward answers; an all-correct group has zero within-group variance. Entropy then falls because the data is too easy, not because the update is too large.

The full definitions are:

$$
D_{\mathrm{KL}}\bigl(\pi_\theta\,\|\,\pi_{\mathrm{ref}}\bigr)
= \sum_a \pi_\theta(a)\ln\frac{\pi_\theta(a)}{\pi_{\mathrm{ref}}(a)},
\qquad
H(\pi_\theta) = -\sum_a \pi_\theta(a)\ln \pi_\theta(a).
$$

Here $\pi_\theta$ is the current policy and $\pi_{\mathrm{ref}}$ is usually the SFT model before RL. KL is zero when the distributions match and grows as they diverge. Entropy is larger for a more even distribution and smaller when probability concentrates on a few actions.

## Numerical Updates

The first two layers describe behavior: what the model learns and where its policy moves. The third asks whether parameters can be updated normally. Loss measures the current objective, while the gradient norm measures how far the step is trying to push the parameters. Sudden spikes or NaNs indicate a numerical failure in the forward pass, backward pass, or update.

NaNs are dangerous because they propagate. One NaN parameter becomes an entire region of NaNs after a matrix multiplication. A few steps later, the model is unusable. There is often a warning period in which gradient norms grow and loss spikes before the first NaN appears.

A useful diagnostic is to disable data shuffling and repeat one fixed batch. If the NaN reproduces reliably, the cause lies in that batch or a deterministic computation path, such as division by zero, a logarithm of a negative value, or an overflowing attention score. If it appears randomly, inspect low-precision overflow or underflow, kernels, and optimizer state. Then check the learning rate, precision, clipping threshold, and anomalous examples with extreme length or repetition.

AdamW, Muon, and clipping all operate here. AdamW scales gradients with second moments. Muon orthogonalizes update matrices so that a few singular directions are not repeatedly amplified. Clipping limits the norm of one update. None can repair a wrong reward, damaged data, or mismatched model versions. A loss spike justifies inspecting the optimizer and clipping; rising reward with flat evaluation and exploding length does not.

## The Training System

If the first three layers look healthy, verify that the rollout and training engines are using the same policy. This failure is subtle because loss, KL, entropy, and reward may all look normal while the recorded probabilities themselves are wrong.

The generation side may use a slightly older model, FP8, and throughput-oriented inference kernels, while the training side uses updated weights, BF16 or FP32, and different kernels. Even with the same parameter file, precision truncation, kernel implementation, and attention paths can change token log probabilities. In an MoE model, a routing difference can send the same token to different experts and alter the forward pass fundamentally.

Suppose generation with model version $v_t$ records one token's log probability as $-1.32$, while training has advanced to $v_{t+1}$ and recomputes it as $-1.51$. The importance ratio is

$$
\frac{\pi_{v_{t+1}}(a)}{\pi_{v_t}(a)}
= e^{-1.51-(-1.32)} = e^{-0.19} \approx 0.83.
$$

Without any sampling noise, the ratio is already about 17% away from 1. Version and precision differences are silently interpreted as policy-gradient signal. Over thousands of steps they can move the policy off course. Align model versions, tokenization, log-probability computation, numerical precision, and expert routing. The old policy described in the log must be the policy that actually generated the trajectory. [18.4](./distributed-sync) develops this issue further.

::: details Extra: a quick checklist for four training curves

1. **Reward versus independent evaluation**: rising reward with flat or falling evaluation and growing response length points to data and reward problems.
2. **KL and entropy**: rapidly rising KL and accelerating entropy decline point to policy movement. Check the learning rate, KL coefficient, clipping range, and the ratio of correct to incorrect responses within groups.
3. **Loss and gradient norm**: spikes, exploding norms, or NaNs point to numerical updates. Check anomalous batches, precision, clipping, and optimizer state.
4. **Everything looks normal but behavior is wrong**: compare generation-side and training-side log probabilities for the same token. A discrepancy points to synchronization, precision, or MoE routing.

Do not diagnose from the bottom upward by reflexively adding gradient clipping for NaNs or lowering the learning rate for high KL. If the objective itself is wrong, those changes only optimize the wrong target more smoothly.
:::

## Public Cases

The framework is now ready for four public cases:

- **GLM-4.5 / GLM-4.6**: MoE routing, staged training, and mode switching; watch for expert imbalance and capability regression between stages.
- **Llama 4**: multimodality, long context, and evaluation versions; watch for disagreement between benchmark scores and real tasks.
- **Seed-Thinking**: data difficulty, curricula, and self-verification; watch for sparse rewards and near-zero within-group advantages.
- **Kimi K2**: optimizer updates and attention scores; watch for loss spikes and anomalous gradient or attention values.

### GLM and Multi-Stage Training

GLM-4.5 and GLM-4.6 use multi-stage training. They expose first- and fourth-layer issues: routing stability imposed by an MoE architecture and preventing capability regression between training stages.

GLM-4.5 has 355B total parameters and activates 32B per forward pass. RL monitoring therefore includes expert load and routing stability alongside reward, KL, and entropy. A few experts can be overloaded or idle even while aggregate loss looks normal. The same model also supports Thinking and Non-Thinking modes and must learn when to reason and when to answer directly. Its data must cover code generation, tool use, and multi-step execution.

GLM-4.5 uses five stages:

```text
Phase 1: Base pretraining (MoE)
  - 15T high-quality tokens
  - 355B total / 32B active
  - RoPE scaling for long context

Phase 2: General SFT
  - Multilingual dialogue
  - Tool-call formatting

Phase 3: Reasoning RL
  - Mathematics, code, and reasoning
  - GRPO + rule-based rewards
  - Self-validation

Phase 4: General RLHF
  - Dialogue quality and safety
  - Helpfulness / Harmlessness

Phase 5: Unify Thinking / Non-Thinking
  - Mixed-data SFT
  - Learn mode switching
```

The order has explicit dependencies. SFT first establishes usable dialogue and tool behavior. Reasoning RL can then obtain meaningful reward differences on verifiable tasks. General RLHF repairs dialogue quality and safety, and mixed data finally unifies the two response modes. Stage transitions create a first-layer problem: later preference training can overwrite mathematical ability gained during reasoning RL. Cross-stage distillation or data replay is needed to retain earlier capabilities; GLM-5's on-policy cross-stage distillation addresses this problem.

GLM-4.6 extends reasoning length beyond 100K tokens, broadens agent tools to search, code execution, and file operations, and adds finer Thinking Budget controls. Its improvements on AIME 2025, MATH-500, LiveCodeBench, and GPQA Diamond still have to be interpreted under the exact evaluation configuration.

Three lessons follow. MoE and reasoning RL must be debugged together with expert-load, routing, and communication metrics. Thinking and Non-Thinking modes require separate evaluations of correctness, length, cost, and response quality. Code and tool tasks require real execution because static answer scores do not cover environment state or long-trajectory failure.

### Llama 4 and Evaluation Consistency

Llama 4 combines MoE, native multimodality, and long context. Scout has 109B total and 17B active parameters with a 10M context; Maverick has 400B total and 17B active parameters with a 1M context; Behemoth was announced with 2T total and 288B active parameters.

Each architectural change reaches post-training. Early Fusion processes text and image tokens together from pretraining, so post-training requires textual, multimodal, and cross-modal consistency tasks. A multimodal reward must check whether an answer actually uses image evidence. MoE adds expert routing and communication. A 10M-token context requires evaluation of evidence retrieval, answer correctness, and reasoning cost; accepting the input is only the minimum.

The key stability lesson is the gap between “training score” and real experience. Maverick scored well on benchmarks while many users found the released model weaker than contemporary alternatives. The version evaluated in LM Arena used chat-template and prompt adjustments that differed from the released version. The scores therefore compared different evaluation configurations.

This is first-layer evaluation contamination. Model comparisons must record weight version, chat template, system prompt, and sampling parameters. Otherwise a score difference can reflect configuration rather than capability. The same principle applies during training: an evaluation configuration that differs from deployment acts like a flawed reward function.

### Seed-Thinking and Data Difficulty

Seed1.5-Thinking combines data organization, policy optimization, self-verification, and curriculum learning. Its failure signal lies between the first and second layers: rewards are too sparse and within-group advantage approaches zero.

GRPO learns from reward differences within a response group. If all 16 responses are correct or all are wrong, the within-group standard deviation is zero and the batch produces no gradient. Early in training, all-wrong groups are common; with overly easy data, all-correct groups waste the same rollout compute.

Seed-Thinking addresses this in four ways. Mathematics data includes contest and generated problems bucketed by base-model pass rate; code data covers Codeforces, SWE-bench, and function generation. Problems that the current model solves 30%–70% of the time have the greatest training value because they are likely to produce mixed groups.

Dynamic KL constrains the policy strongly early and relaxes later. Adaptive clipping changes the clipping range over training. Group-size scheduling uses larger groups early, such as 32 rollouts per prompt, to increase the chance of mixed outcomes, then smaller groups later to save compute. These are second-layer controls on policy movement.

Self-Verification gives $1.0$ for a correct answer that passes verification, $0.5$ for a wrong answer whose error the model detects, and $0$ for a wrong answer the model fails to recognize. Without the middle level, rewards $(1,0,0)$ suppress “wrong but self-aware” and “wrong and unaware” equally. With it, verification behavior earns partial credit and an otherwise all-zero difficult group can produce a gradient.

Curriculum learning addresses the other side of the problem. Start with tasks that the current model can sometimes solve, then increase difficulty. Because difficulty is defined by current pass rate, the curriculum must change as the model improves.

Thus difficulty bucketing and curricula increase reward density at the first layer, while dynamic KL, adaptive clipping, and Self-Verification control policy learning at the second. Deployment must still monitor length, latency, and regression in general capability.

### Kimi K2 and Constraints on Anomalous Updates

Kimi K2's representative signals lie at the third layer: loss spikes and anomalous gradient or attention values. MuonClip constrains parameter updates; QK-clip constrains attention scores.

Muon is a momentum-orthogonalization optimizer. It accumulates directional information across updates and orthogonalizes update matrices. Gradient matrices often have uneven singular values, allowing a few directions to dominate over thousands of steps. Orthogonalization evens those directions.

MuonClip adds a norm limit after orthogonalization. If the update exceeds the threshold, it is scaled proportionally. This constrains how far one step moves, but it does not identify a bad example or reward that caused the spike. Learning-rate schedules, gradient monitoring, and data checks remain necessary.

QK-clip handles attention scores. As query and key norms grow in long contexts, $QK^\top$ can reach tens or hundreds. Softmax then concentrates on a few tokens and magnifies low-precision errors. QK-clip clamps $QK^\top$ to $[-clip\_value,clip\_value]$ before Softmax. This is a numerical safeguard, not a policy-level change.

The published result reports that the combination reduced loss spikes from about once per 1T tokens to once per 10T tokens and trained about 15% faster than Adam. The methodological lesson is to match the tool to the failure layer. MuonClip constrains update norms; QK-clip constrains extreme attention scores. Neither repairs reward hacking.

### Sources

- [GLM-4.5 technical report](https://arxiv.org/abs/2508.06471)
- [GLM-5 technical report](https://arxiv.org/html/2602.15763v1)
- [Llama 4 technical report](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)
- [Seed1.5-Thinking technical report](https://arxiv.org/abs/2504.13914)
- [DAPO: An Open-Source LLM RL System at Scale](https://seed.bytedance.com/en/public_papers/dapo-an-open-source-llm-reinforcement-learning-system-at-scale)
- [Kimi K2 technical report](https://arxiv.org/abs/2507.20534)
- [Muon optimizer](https://arxiv.org/abs/2502.16982)

---

## Where Stability Methods Operate

Public method names—GRPO, GSPO, DAPO, CISPO, VAPO, and MuonClip—address different problems:

- **Layer 1, data and rewards**: DAPO Dynamic Sampling filters all-correct and all-wrong groups, while Overlong Reward Shaping handles truncated responses; VAPO Length-Adaptive GAE normalizes reward scale across response lengths; Seed-Thinking Self-Verification increases gradient density; Skywork-OR1 monitors entropy collapse; MiMo uses test-difficulty-driven rewards.
- **Layer 2, policy change**: GRPO and GSPO use group-relative advantage and KL constraints; DAPO Clip-Higher preserves exploration through asymmetric clipping, and Token-Level Policy Gradient avoids dilution on long sequences; MiniMax-M1 CISPO clips importance weights; Hunyuan-T1 uses curricula and policy resets; DeepSeek-R1 uses staged domain-specific RL.
- **Layer 3, numerical updates**: MuonClip, QK-clip, gradient clipping, mixed-precision switches, and optimizer-state monitoring.
- **Layer 4, training system**: GLM-5/SAO asynchronous RL with version tags, LongCat DORA streaming RL, generation-training log-probability alignment, MoE routing consistency, weight synchronization, and checkpoint management.

Classify a method before combining it with others. Two clipping methods at the same policy layer may conflict, while a policy-level method and a numerical safeguard are often complementary. More tools do not automatically create stability; if the reward is flawed, additional optimizers only stabilize exploitation.

::: details Extra: identify a method's layer quickly

1. **Does it change rewards or data?** Reward functions, filtering, or sampling usually act at layer 1 or 2.
2. **Does it change gradients or parameter updates?** Optimizers, gradient clipping, and attention clipping act at layer 3.
3. **Does it change distribution or generation-training consistency?** Synchronization, version checks, and weight alignment act at layer 4.

If a method touches several layers, as DAPO does, classify each component separately.
:::

## Additional Checks at Scale

The four-layer framework diagnoses one training run. Each layer acquires additional checks as scale grows.

### Very Large Models and Long Contexts

MoE routing and cross-GPU communication become stability signals themselves. Persistent routing imbalance leaves some experts undertrained and others overfit, causing abrupt regression on particular task types. Contexts beyond 10M tokens add memory, attention-value, and trajectory-cost constraints. Attention scores overflow more easily, and KV-cache precision requires separate monitoring.

### Native Multimodal RL

Text, images, and environment actions can occupy one trajectory, so rewards must verify cross-modal evidence. Did the response use the image, or guess from textual priors? Llama 4 Early Fusion shows one way to unify modalities from pretraining. Multimodal RL also needs replayable images, video, and interactive environments. Otherwise a failed trajectory cannot distinguish perception error, reasoning error, and tool error.

### Industrial Agentic RL

Agent tasks extend from software engineering to support, research, and computer use. Trajectories must record tool arguments, environment returns, file-system snapshots, and intermediate state. Uneven trajectory lengths increase scheduling, recovery, and asynchronous-training costs. As sandboxes expand to browsers, phones, and desktop GUIs, environment failures—dependency errors, page timeouts, and tool crashes—become stability metrics.

### Training Cost and Efficiency

Total cost includes pretraining, data generation, rollouts, updates, and evaluation. Higher generation throughput, smaller activated parameter counts, and better filtering reduce wasted computation. Asynchronous training reduces idle GPU time but introduces version drift and therefore requires explicit data-version management. Small teams should first reproduce the complete four-layer monitoring loop on a small verifiable task before scaling the model and cluster.

## Summary

Training stability requires observing data and rewards, policy change, numerical updates, and the training system in causal order. First verify the objective, then read KL and entropy, inspect loss and gradients, and finally align model versions and probabilities.

1. **Four-layer chain**: data and rewards → policy change → numerical updates → training system. Diverging reward and independent evaluation indicate layer 1; NaNs and exploding gradients indicate layer 3; generation-training probability disagreement indicates layer 4.
2. **Reading policy metrics**: concentrating from $[0.80,0.20]$ to $[0.95,0.05]$ raises KL from 0.19 to 0.49 and lowers entropy from 0.50 to 0.20. Data that is too easy can produce the same entropy decline, so read difficulty alongside these slopes.
3. **Optimizer boundaries**: AdamW, Muon, and clipping only turn gradients into controlled updates. They can treat loss spikes, not reward-evaluation divergence.
4. **Version consistency**: a log-probability gap of 0.19 moves the importance ratio about 17% away from 1 even without sampling noise. Precision, kernels, tokenization, and MoE routing must align.
5. **Cases and layers**: GLM illustrates MoE routing and cross-stage regression; Llama 4 illustrates evaluation-version contamination; Seed illustrates task difficulty and self-verification rewards; Kimi K2 illustrates update-norm and attention-score control.

[18.4 Distributed RL Training](./distributed-sync) continues with the fourth layer: keeping data and model versions correct when generation, rewards, and training run across many GPUs.
