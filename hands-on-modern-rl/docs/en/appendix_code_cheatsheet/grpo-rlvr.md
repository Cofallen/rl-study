---
title: B.4 GRPO and Reward Models
---

# B.4 GRPO and Reward Models

---

## GRPO Loss

**Core problem**: PPO requires training a critic as large as the policy itself to estimate $V(s)$ as a baseline. GRPO exploits the natural control group formed by sampling $G$ answers per prompt — it uses the within-group reward normalization directly as the advantage, removing the critic.

**Core variables**:

- `rewards`: rewards of $G$ completions for the same prompt (from RM or rule verifier), shape `[G]`
- `advantages`: within-group normalized advantages, $A_i = (r_i - \bar r)/\mathrm{std}(r)$
- `old_log_probs` / `new_log_probs`: log-probs of completions under sampling policy and current policy
- `ref_log_probs`: log-probs under the reference policy, used for KL penalty
- `clip_eps`, `kl_coeff`: same hyperparameters as PPO

### One-Line Memory

> Sample G answers per prompt; z-score the rewards within each group as the advantage; copy PPO's clip and KL; drop the critic.

### Pseudocode

```
# Step 1: sample G completions for one prompt, score each
rewards = [reward_fn(generate(prompt)) for _ in range(G)]   # [G]

# Step 2: within-group normalization (subtract mean, divide by std) -> advantage
advantages = (rewards - mean(rewards)) / (std(rewards) + eps)

# Step 3: PPO clipped loss (advantage comes from step 2, not a critic)
ratio = exp(new_logp - old_logp)
surr1 = ratio * advantages
surr2 = clip(ratio, 1-eps, 1+eps) * advantages
policy_loss = -min(surr1, surr2).mean()

# Step 4: k3 KL penalty (pull back, don't drift too far from reference)
log_ratio = ref_logp - new_logp
kl = (exp(log_ratio) - 1 - log_ratio).mean()

# Step 5: total loss
loss = policy_loss + kl_coeff * kl
```

### PPO vs GRPO

|                  | PPO                        | GRPO                              |
| ---------------- | -------------------------- | --------------------------------- |
| Advantage source | Critic $V(s)$ + GAE        | within-group reward normalization |
| Number of models | 4 (actor, critic, ref, rm) | 2~3 (actor, ref, rm/verifier)     |
| KL penalty       | optional                   | almost always used                |
| Sampling         | single rollout per prompt  | $G$ samples per prompt            |

### Python (NumPy) Implementation

```python
import numpy as np

def grpo_advantages(rewards):
    """
    rewards: [num_prompts, G]  rewards of G completions per prompt
    returns within-group z-score normalized advantages
    """
    mean = rewards.mean(axis=1, keepdims=True)
    std = rewards.std(axis=1, keepdims=True)
    return (rewards - mean) / (std + 1e-8)

def grpo_policy_loss(new_logps, old_logps, advantages, clip_eps=0.2):
    """Identical to PPO's clipped surrogate loss."""
    ratio = np.exp(new_logps - old_logps)
    surr1 = ratio * advantages
    surr2 = np.clip(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    return -np.minimum(surr1, surr2).mean()
```

### PyTorch Implementation

```python
import torch
import torch.nn.functional as F

def grpo_loss(log_probs, old_log_probs, ref_log_probs,
              rewards, clip_eps=0.2, kl_coeff=0.05):
    """
    log_probs:     [B, G]  current policy's per-completion sequence-level log_prob
    old_log_probs: [B, G]  behavior policy (used during sampling)
    ref_log_probs: [B, G]  reference policy
    rewards:       [B, G]  group rewards
    B = num_prompts, G = group_size
    """
    # 1. Within-group normalization (grouped by prompt)
    advantages = (rewards - rewards.mean(dim=1, keepdim=True)) \
                 / (rewards.std(dim=1, keepdim=True) + 1e-8)   # [B, G]

    # 2. Clipped policy loss (identical to PPO)
    ratio = torch.exp(log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # 3. KL penalty (k3 estimator: log_ratio = log(π_ref/π_θ), samples from π_θ)
    log_ratio = ref_log_probs - log_probs
    kl = (torch.exp(log_ratio) - 1 - log_ratio).mean()

    return policy_loss + kl_coeff * kl
```

---

## Reward Model (Bradley-Terry)

**Core problem**: RLHF-PPO needs a scalar reward $r \in \mathbb{R}$ to drive policy optimization, but human preferences only provide relative ordering ("A is better than B"). The RM compresses preference pairs into absolute scores, maximizing the probability that good answers score higher than bad ones.

**Core variables**:

- `r_chosen` / `r_rejected`: scalar scores the RM assigns to the good / bad answer
- Bradley-Terry assumption: $P(y_w \succ y_l) = \sigma(r_w - r_l)$, preference probability is proportional to the sigmoid of the score difference

### One-Line Memory

> Make good scores beat bad ones: `-log_sigmoid(r_chosen - r_rejected)`, one line, done.

### Pseudocode

```
# Step 1: the RM assigns a scalar score to each answer
r_w = reward_model(chosen_input)     # score for the good answer
r_l = reward_model(rejected_input)   # score for the bad answer

# Step 2: we want r_w > r_l; sigmoid the diff and take the negative log
loss = -log(sigmoid(r_w - r_l))
```

### Python (NumPy) Implementation

```python
import numpy as np

def log_sigmoid(x):
    return -np.logaddexp(0, -x)   # numerically stable log σ(x)

def reward_model_loss(r_chosen, r_rejected):
    """r_chosen, r_rejected: [B]"""
    return -log_sigmoid(r_chosen - r_rejected).mean()
```

### PyTorch Implementation

```python
import torch.nn.functional as F

def reward_model_loss(r_chosen, r_rejected):
    """
    r_chosen:   [B]  RM score for chosen
    r_rejected: [B]  RM score for rejected
    """
    return -F.logsigmoid(r_chosen - r_rejected).mean()
```

---

## Interview Follow-Up: GRPO, PPO-RLHF, and RLVR

|                  | PPO-RLHF     | GRPO                       | RLVR                       |
| ---------------- | ------------ | -------------------------- | -------------------------- |
| Advantage source | Critic + GAE | within-group normalization | within-group normalization |
| Critic           | required     | not needed                 | not needed                 |
| Reward source    | trained RM   | RM or verifier             | rule verifier (math/code)  |
| Online sampling  | required     | required (G per prompt)    | required (G per prompt)    |

RLVR is a special case of GRPO: rewards do not come from a learned RM but from rule-based verification of the answer (math answer equality, code test pass), so there is no reward hacking risk.

---

## Common Pitfalls

| Pitfall                                 | Explanation                                                                                                   |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| GRPO advantage is within-group          | Not global normalization; only compare the $G$ completions of the **same prompt**                             |
| GRPO has no value loss                  | No critic means no value loss — the core difference from PPO                                                  |
| Two normalizer variants exist           | Most implementations use std (z-score); some use only `rewards - mean` (no std). Note the difference          |
| RM must be frozen during policy updates | When training the RM, rewards are differentiable; when training the policy, the RM is usually detached/frozen |
| KL is sequence-level                    | Typically sum token log-probs per completion first, then compute KL — not per token                           |
| Direction of the k3 KL estimator        | `log_ratio = log(π_ref/π_θ)`, samples drawn from the current policy $\pi_\theta$                              |
| RLVR case                               | Rewards come from a rule verifier (code execution, math answer check), not from RM                            |
