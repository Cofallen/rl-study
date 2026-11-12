# 15.4 Improved Methods for GRPO

In the previous section, we examined DeepSeek-R1-Zero and DAPO — they respectively demonstrated that "pure RL can replace SFT cold start" and "engineering improvements can allow GRPO to reach the level of R1-Zero with half the steps." However, DAPO is just **one member** of the family of GRPO improvements in 2025. From the publication of the R1 paper (2025.01) to early 2026, the open-source community and industrial laboratories proposed at least five influential GRPO variants within less than a year. These variants are not mutually exclusive but rather address different defects of GRPO from different angles.

This section reorganizes the improvements by **improvement direction** — placing the five approaches of Dr.GRPO, GSPO, CISPO, VAPO, and RPT side by side for comparison, so you can clearly understand "when to use which variant."

## Improvement Route One: Removing Normalization Bias

### Discovery of Dr.GRPO

The original form of GRPO in the R1 paper applied two steps of normalization to the group's reward:

$$\tilde{r}_i = \frac{r_i - \text{mean}(r_1, \ldots, r_G)}{\text{std}(r_1, \ldots, r_G)}$$

Dividing by the standard deviation seems natural — it standardizes the advantage values. However, Liu et al. discovered in their 2025 study ([arXiv:2503.20783](https://arxiv.org/abs/2503.20783)) that this seemingly harmless normalization introduced two types of bias:

- **Length Bias**: When a prompt's reward variance is large (some answers are correct, some are not), dividing by the standard deviation compresses the advantage. When the variance is small (all answers are correct or all are incorrect), the standard deviation approaches zero, and the advantage is amplified to unreasonable magnitudes. The model thus learns that "producing diverse outputs is more important than being correct."
- **A breeding ground for reward hacking**: Dividing by the standard deviation is equivalent to encouraging the model to increase the variance of the group's reward. The simplest way to increase variance is to make some answers **longer** (more tokens, more chances to be correct). This is one of the direct causes of the explosive answer length in the later stages of R1-Zero training.

Dr.GRPO's correction is extremely simple — **only subtract the mean, not divide by the standard deviation**:

$$\tilde{r}_i^{\text{Dr.GRPO}} = r_i - \text{mean}(r_1, \ldots, r_G)$$

Experiments show that this change significantly alleviates the problem of answer length inflation in the later stages of training and reduces reward hacking behavior. The Qwen series adopted a similar correction in their internal training.

### Further Engineering of DeepSeek V3.2

In the V3.2 version of DeepSeek (2025.12, [arXiv:2512.02556](https://arxiv.org/abs/2512.02556)), the idea of Dr.GRPO is pushed to its extreme, with three engineering adjustments made specifically for mathematical reasoning tasks:

- **Zero KL for Math Tasks**: Traditional GRPO uses KL divergence to constrain the policy from deviating from the reference model. However, in mathematical tasks, the reward itself is already sufficient to constrain performance (a wrong answer receives zero points). The KL constraint can instead suppress the model's exploration of new problem-solving paths. DeepSeek directly disables the KL constraint during the pure mathematical RL phase.
- **Self-Verification RLVR**: The model is given an additional "verification step" after generating an answer—re-reading the question, checking calculations, and confirming the answer. The reward from this verification step is also incorporated into the RL optimization, forming an internal self-check mechanism.
- **mHC Residual Stability**: Optimization of the numerical stability of the Modified Hamiltonian Monte Carlo (mHC) sampler during long Chain-of-Thought (CoT) training, to prevent gradient explosions.

The V3.2 Speciale variant achieves a score of 97 on the AIME 2025, surpassing the performance of GPT-5 at the same time.

## Improvement Route Two and Sequence-Level Importance Sampling

### GSPO (Qwen3's Choice)

GRPO and PPO both use **token-level importance sampling ratios**:

$$\rho_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$$

Each token has its own ratio. GRPO clips those ratios independently and then averages the token objectives; it does not multiply all token ratios together. This is the original definition in [DeepSeekMath Equation (3)](https://arxiv.org/html/2402.03300#S3.SS1). The GSPO paper argues that these token-wise importance weights introduce high-variance noise when the reward belongs to the whole response, and that routing changes in MoE models amplify the fluctuations.

GSPO (Group Sequence Policy Optimization) elevates the ratio from the token level to the **sequence level**:

$$
s_i(\theta)
=
\left(
\frac{\pi_\theta(y_i\mid x)}{\pi_{\theta_{\text{old}}}(y_i\mid x)}
\right)^{1/|y_i|}
=
\exp\!\left(
\frac{1}{|y_i|}\sum_{t=1}^{|y_i|}
\log\frac{\pi_\theta(y_{i,t}\mid x,y_{i,<t})}
{\pi_{\theta_{\text{old}}}(y_{i,t}\mid x,y_{i,<t})}
\right).
$$

Equivalently, this is the geometric mean of the token ratios:

$$
s_i(\theta)=\left(\prod_{t=1}^{|y_i|}\rho_{i,t}\right)^{1/|y_i|}.
$$

The entire response uses a single ratio for clipping. The clipping target is also adjusted to the sequence level:

$$\mathcal{J}^{\text{GSPO}} = \mathbb{E}\left[\min\left(s_i(\theta)\hat A_i, \; \text{clip}(s_i(\theta), 1-\epsilon, 1+\epsilon)\hat A_i\right)\right].$$

The $1/|y_i|$ length normalization is essential. Using the raw product $\prod_t\rho_{i,t}$ makes the ratio fluctuate with response length and would require different clipping ranges for responses of different lengths. For this reason, the former GRPO example—which summed all token log-probability differences and exponentiated without length normalization—was not a correct GSPO implementation either. [Equations (5) and (7) in the original GSPO paper](https://arxiv.org/html/2507.18071#S4.SS1)

This change may seem simple, but it has a significant impact on the training stability of MoE models — all Qwen3 series (including Qwen3-235B-A22B, Qwen3-Thinking-2507, Qwen3-Coder) are trained based on GSPO. The variance of the sequence-level ratio is much smaller than that of the token-level, making large-scale RL training on a 10,000-card cluster feasible.

The cost of GSPO: the sequence-level ratio **couple all token updates**, and its ability to perform fine-grained credit assignment at the token level is weaker than token-level methods. Therefore, GSPO performs significantly well on long CoT tasks (reasoning, mathematics), but is less effective than DAPO on code generation tasks that require token-level rewards.

## Improvement Route Three and Rewriting of the Pruning Object

### CISPO (Innovation from MiniMax)

GRPO and DAPO both prune the "product of the policy ratio and advantage" — the pruning occurs at the gradient update level. In M1 model ([arXiv:2506.13585](https://arxiv.org/abs/2506.13585)), MiniMax proposes CISPO, which changes the pruning object from "token update" to "**importance sampling weight**":

$$\tilde{\rho}_t = \text{clip}\left(\frac{\pi_{\theta_{\text{old}}}(a_t|s_t)}{\pi_\theta(a_t|s_t)}, 1-\epsilon, 1+\epsilon\right)$$

Note that the numerator and denominator of the ratio are reversed here — $\pi_{\text{old}} / \pi_\theta$ instead of $\pi_\theta / \pi_{\text{old}}$. This reversed ratio serves as the **sampling weight** multiplied onto the advantage, but **retains all token gradient contributions**.

Intuitively: traditional pruning is "if a token's policy deviates too much, directly remove it from the gradient." CISPO is "if the deviation is too large, reduce its contribution weight to the advantage estimate, but keep the gradient direction unchanged." The latter avoids the problem of "some tokens completely stop updating in the later stage of training," which can cause the policy to get stuck.

CISPO also has an engineering advantage — it works well with MiniMax's self-developed lightning attention, solving the problem of precision alignment. The recursive computation of lightning attention leads to the accumulation of floating-point errors in token-level ratios. Traditional pruning can mistakenly prune many tokens in low-precision training. CISPO avoids this issue by scaling the weights instead of pruning. MiniMax M1 trains on 512 H800 cards, achieving a training speed twice as fast as DAPO.

## Improvement Route Four: Against the Value-based Counter-trend

### VAPO (Byte Seed's Counter-trend)

At this point, you might have the impression that **the Critic network has been eliminated by GRPO**. However, the paper VAPO (Value-based Augmented PPO, [arXiv:2504.05118](https://arxiv.org/abs/2504.05118)) published by Byte Seed in April 2025 demonstrates — at least in long CoT reasoning tasks — that **the value model has once again defeated GRPO**.

The core argument of VAPO is that GRPO replaces the Critic network with the group mean, essentially using the "relative ranking among multiple rollouts of the same prompt" to estimate advantage. This is sufficient for short answer tasks (e.g., function calling, simple math problems). However, in long CoT tasks:

- Within a rollout, there are hundreds of tokens, and the **true advantage signal is at the token level** — one step of reasoning is good, the next is bad.
- The group mean treats the entire rollout as a single unit, losing the token-level signal.
- As training proceeds, the model tends to learn to "answer some rollouts by chance" rather than "being correct at every step."

VAPO reintroduces the value model $V_\phi(s)$, using GAE to estimate token-level advantage:

$$\hat{A}_t = \delta_t + (\gamma\lambda)\delta_{t+1} + (\gamma\lambda)^2\delta_{t+2} + \cdots$$

where $\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$ is the TD error. Then, it applies PPO-style clipping on this token-level advantage.

VAPO achieves a score of 60.4 on AIME 2024, surpassing all contemporary GRPO variants (DAPO at 50, R1-Zero at 71 but with double the training steps). **Byte Seed's internal reasoning model training has shifted from pure GRPO to VAPO**.

The cost of VAPO: it requires training an independent value model, doubling the memory usage, and increasing engineering complexity. This is why GRPO became the mainstream approach in 2024 — **critic-free is an engineering compromise, not an algorithmic necessity**.

## Improvement Route Five: Integrating RL into Pre-training

### RPT (Reinforcement Pre-training)

The first four improvement routes assumed that RL occurred in the **post-training phase** — the model had already been pre-trained, and RL was merely fine-tuning. However, Microsoft's 2025.06 paper on Reinforcement Pre-training ([arXiv:2506.08007](https://arxiv.org/abs/2506.08007)) challenges this binary distinction.

The core idea of RPT is to **reframe the next-token prediction task as a reasoning task**. The traditional pre-training loss is:

$$\mathcal{L}_{\text{LM}} = -\mathbb{E}\left[\log \pi_\theta(a_t | s_{<t})\right]$$

In this setup, each token is treated as an equal teacher-forcing target. RPT transforms this by having the model first generate a reasoning about the next token ("Given the context, the next word might be X because..."), and then use the reasoning result to predict the next token, rewarding the model when it is correct:

$$\mathcal{L}_{\text{RPT}} = -\mathbb{E}\left[\log \pi_\theta(a_t | s_{<t}, \text{reasoning}_t)\right] + \beta \cdot \text{RL loss}$$

This change is revolutionary — **RL can be performed during the pre-training phase**, and RPT's scaling properties are comparable to traditional pre-training. This suggests that the future may no longer have a clear distinction between "pre-training" and "post-training," with RL being integrated throughout the entire training process.

RPT is still in its early stages and has not yet been widely adopted in industry practice. However, its conceptual impact is significant enough for this book to dedicate a separate improvement route to it.

## Decision Tree for Selection

The following table summarizes the core differences, applicable scenarios, and typical users of the five variants:

| Algorithm   | Core Innovation                        | Pain Points Solved                   | Typical Use Cases                  | Representative Users |
| ----------- | -------------------------------------- | ------------------------------------ | ---------------------------------- | -------------------- |
| **GRPO**    | Replacing Critic with Intra-group Mean | Critic memory overhead               | General RLHF / RLVR                | DeepSeek-R1          |
| **Dr.GRPO** | Removing std normalization             | Length explosion, reward hacking     | Mathematical reasoning             | Qwen Internal        |
| **GSPO**    | Sequence-level IS                      | MoE training instability             | RL for MoE models                  | Qwen3 Full Line      |
| **CISPO**   | Pruned IS weights                      | Token loss, precision alignment      | Lightning attention, low precision | MiniMax M1           |
| **VAPO**    | Reintroducing value model              | Long CoT credit assignment           | Training of reasoning models       | Byte Seed            |
| **DAPO**    | Four engineering improvements          | Training efficiency, length control  | Mathematical / code RL             | Byte + Tsinghua      |
| **RPT**     | Pre-training RL integration            | Pre-training to fine-tuning boundary | Next-generation base models        | Microsoft Research   |

The selection logic in practical industrial applications is roughly as follows:

```text
Task Type?
├── Mathematical/Code Reasoning (Long CoT)
│   ├── MoE Architecture → GSPO + Dr.GRPO Ideas
│   ├── Dense Architecture → VAPO or DAPO
│   └── Extreme Stability Requirements → CISPO
├── General Dialogue Alignment
│   └── GRPO / PPO (Basic Enough)
├── Multi-turn Tool Calls
│   └── DAPO + Token-level Loss
└── Next-generation Base Models
    └── RPT (Experimental)
```

This decision tree is not absolute — within ByteDance Seed, there is frequent mixing (for example, DAPO's engineering techniques + VAPO's value model). But it provides a checklist of what to consider first after seeing a task.

## Summary

The rapid evolution of the GRPO improvement family reflects a fact: **RL in the era of large models is no longer a single choice of "just using PPO is enough."** Each lab has chosen different improvement directions based on their training infrastructure (MoE vs Dense, lightning attention vs standard attention, memory budget) and task characteristics (inference vs dialogue, long CoT vs short answer).

The true value of this section is not in remembering the specific formulas of each algorithm — it is in building a sense of judgment: **when seeing a new GRPO variant, being able to immediately ask, "Which of the four directions — normalization, sequence-level, clipping, or value model — is this variant addressing?"** This sense of judgment is a crucial step from reading papers to being able to improve algorithms hands-on.
