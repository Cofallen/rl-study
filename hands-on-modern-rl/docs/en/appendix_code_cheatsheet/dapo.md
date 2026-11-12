---
title: B.5 DAPO
---

# B.5 DAPO

DAPO (Decoupled Clip and Dynamic sAmpling Policy Optimization), proposed by ByteDance in 2025, improves GRPO for long-chain reasoning RL with four orthogonal tricks: decoupled clipping, dynamic sampling, token-level loss, and soft overlong punishment. Frequently asked in interviews.

---

## Core Problem

GRPO training of long-chain reasoning models hits four issues: symmetric clipping suppresses exploration of good actions; all-correct / all-wrong prompts waste sampling compute yet give no gradient; sequence-level loss under-weights long answers; binary overlong truncation zeroes the reward at the boundary with no gradient signal. DAPO addresses each with one independent modification.

## Core Variables

- `ratio`: new/old policy probability ratio $r = \exp(\text{new\_logp} - \text{old\_logp})$
- `advantage`: group-wise z-score $\hat{A}_{i,t} = (R_i - \bar R)/\mathrm{std}(R)$ (same as GRPO)
- `ε_{low}`, `ε_{high}`: decoupled clipping lower / upper bounds (paper typical 0.2 / 0.28)
- `reward_std`: within-group reward std, used by dynamic sampling filter
- `max_len`, `buffer_len`, `penalty_factor`: soft overlong threshold, buffer width, max deduction

## One-Line Memory

> **Four cuts: clip with asymmetric ε (upper vs lower), drop all-correct/all-wrong prompts, flatten loss to token level, soft linear penalty for overlong answers.**

---

## Four Modifications at a Glance

| Modification     | GRPO                                      | DAPO                                                                            |
| ---------------- | ----------------------------------------- | ------------------------------------------------------------------------------- |
| clipping         | symmetric `clip(r, 1-ε, 1+ε)`             | decoupled `clip(r, 1-ε_{low}, 1+ε_{high})`, looser upper bound                  |
| sampling         | keep all prompts                          | filter prompts whose within-group reward variance is 0                          |
| loss granularity | sequence-level (token-mean then seq-mean) | token-level (sum all tokens, divide by total token count)                       |
| overlong answer  | reward=0 (binary cliff)                   | soft penalty: linear deduction past soft threshold, capped at `-penalty_factor` |

---

## Decoupled Clipping (Clip-Higher)

### Core Problem

Symmetric clipping `clip(r, 1-ε, 1+ε)` uses the same ε on both sides. But good actions with positive advantage deserve larger steps (upper ε should be bigger); bad actions with negative advantage should move more cautiously (lower ε can be smaller). DAPO decouples the two bounds so each direction's exploration strength is tuned independently.

### One-Line Memory

> **Same min-clipped as PPO, but with asymmetric $\varepsilon$: wide upper $\varepsilon_{high}$, tight lower $\varepsilon_{low}$.**

### Pseudocode

```
ratio = exp(new_logp - old_logp)

# Same min-clipped as PPO, but clip with asymmetric ε
surr1  = ratio * advantage
surr2  = clip(ratio, 1 - eps_low, 1 + eps_high) * advantage
loss_t = -min(surr1, surr2)            # per-token loss
```

> Note: when advantage > 0 the lower bound is inactive (consumed by min); when advantage < 0 the upper bound is inactive. So the effect equals "clip only the upper bound for A>0, only the lower bound for A<0" — but writing it as PPO's min-clipped stays closer to the paper.

### Python (NumPy)

```python
import numpy as np

def dapo_policy_loss(new_logp, old_logp, advantages,
                     eps_low=0.2, eps_high=0.28):
    """
    new_logp:   [T]
    old_logp:   [T]
    advantages: [T]
    Returns token-level loss (sum divided by token count).
    """
    ratio = np.exp(new_logp - old_logp)
    surr1 = ratio * advantages
    surr2 = np.clip(ratio, 1 - eps_low, 1 + eps_high) * advantages
    loss_per_token = -np.minimum(surr1, surr2)
    return loss_per_token.sum() / len(loss_per_token)
```

### PyTorch

```python
import torch

def dapo_policy_loss(new_logps, old_logps, advantages,
                     eps_low=0.2, eps_high=0.28):
    """
    new_logps:  [B, seq_len]
    old_logps:  [B, seq_len]
    advantages: [B, seq_len]
    """
    ratio = torch.exp(new_logps - old_logps)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - eps_low, 1 + eps_high) * advantages
    loss_per_token = -torch.minimum(surr1, surr2)
    return loss_per_token.sum() / loss_per_token.numel()
```

---

## Dynamic Sampling

### Core Problem

GRPO's advantage is a within-group z-score. If all G completions under one prompt are correct or all wrong, the within-group reward variance is 0 and the z-score degenerates, giving no gradient signal yet wasting sampling compute. DAPO filters these invalid prompts at the data level and keeps sampling until the batch is full of valid samples.

### One-Line Memory

> **Within-group reward variance 0 (all correct / all wrong) → no gradient signal → skip and keep sampling to fill the batch.**

### Pseudocode

```
# Sample G completions per prompt and score them
rewards = [get_reward(c) for c in group]      # [G]

# All rewards identical → skip and resample a fresh prompt
if std(rewards) == 0:
    skip this prompt and resample
```

### PyTorch

```python
def dynamic_sampling_filter(rewards):
    """
    rewards: [B, G]  B prompts, each with G completion rewards
    Returns bool mask [B], True = keep.
    """
    return rewards.std(dim=1) > 1e-6
```

---

## Token-Level Loss

### Core Problem

GRPO aggregates loss at sequence level: first average tokens within each sequence to get a per-sequence loss, then average across sequences. This makes a long answer (more tokens) carry the same weight as a short one, under-weighting the detailed tokens inside long answers. DAPO switches to token-level aggregation: sum every token of every answer, then divide by total token count, so long answers naturally contribute more gradient.

### One-Line Memory

> **Don't average within sequences first — flatten all tokens, sum and divide by total count; long answers aren't under-weighted.**

### Pseudocode

```
loss_mat = -min(ratio*A, clip(ratio, 1-eps_low, 1+eps_high)*A)   # [B, T]

# GRPO: token-mean then seq-mean → long sequences are flattened
seq_loss = mean(loss_mat, dim=token)        # one value per sequence
loss_grpo = mean(seq_loss)

# DAPO: flatten all tokens
loss_dapo = sum(loss_mat) / total_num_tokens
```

### PyTorch

```python
def token_level_loss(loss_mat, loss_mask):
    """
    loss_mat:  [B, T]  per-token policy loss
    loss_mask: [B, T]  1 for valid token, 0 for padding
    Returns token-level aggregated loss.
    """
    return (loss_mat * loss_mask).sum() / loss_mask.sum()
```

---

## Soft Overlong Punishment

### Core Problem

GRPO zeroes the reward of any answer exceeding the max length, leaving no gradient at the boundary — the policy only knows "I was penalized", not "shorter would be better". DAPO introduces a buffer: lengths inside `[max_len - buffer_len, max_len]` are not penalized, beyond `max_len - buffer_len` a linear deduction kicks in, capped at `-penalty_factor`, giving the policy a smooth, differentiable directional signal.

### One-Line Memory

> **Past `max_len − buffer_len`, deduct linearly by overflow ratio, cap at `-penalty_factor` — no binary zeroing.**

### Pseudocode

```
expected_len = max_len - buffer_len
exceed_len   = response_length - expected_len

if exceed_len > 0:
    # Linear penalty: the more overflow, the more deduction, capped at -penalty_factor
    penalty = max(-penalty_factor, -(exceed_len / buffer_len) * penalty_factor)
    reward = reward + penalty        # penalty ≤ 0
```

### Python (NumPy)

```python
def soft_overlong_penalty(response_length, max_len,
                          buffer_len, penalty_factor=1.0):
    """Returns the penalty (≤0); add it to the raw reward."""
    expected_len = max_len - buffer_len
    exceed_len = response_length - expected_len
    if exceed_len <= 0:
        return 0.0
    linear = -(exceed_len / buffer_len) * penalty_factor
    return max(-penalty_factor, linear)        # capped, no infinite deduction
```

---

## DAPO Total Loss (Sketch)

```
# 1. within-group z-score normalization (same as GRPO)
advantages = (rewards - rewards.mean(dim=G)) / (rewards.std(dim=G) + eps)

# 2. dynamic sampling filter
valid = dynamic_sampling_filter(rewards)        # drop all-correct / all-wrong prompts

# 3. decoupled clipping + token-level loss
ratio = exp(new_logp - old_logp)
surr1 = ratio * advantages
surr2 = clip(ratio, 1 - eps_low, 1 + eps_high) * advantages
loss_mat = -minimum(surr1, surr2)               # per-token

# 4. token-level aggregation (key change: long answers not under-weighted)
policy_loss = (loss_mat * mask)[valid].sum() / mask[valid].sum()

# 5. KL penalty (same as GRPO)
kl = ((exp(ref_logp - new_logp) - 1) - (ref_logp - new_logp)).mean()

loss = policy_loss + kl_coeff * kl
```

---

## Common Pitfalls

| Pitfall                                   | Explanation                                                                             |
| ----------------------------------------- | --------------------------------------------------------------------------------------- |
| Decoupled clipping ≠ no clipping          | Still PPO's `min(r*A, clip(r,lo,hi)*A)`, just with asymmetric ε                         |
| Each sign of advantage binds only one ε   | For A>0 the lower bound is inactive; for A<0 the upper bound is inactive (eaten by min) |
| Dynamic sampling criterion                | Not "reward below threshold", but "**within-group reward variance is 0**"               |
| Token-level loss is the fourth key change | GRPO aggregates at sequence level, DAPO at token level — long answers carry more weight |
| Soft overlong penalty is linear, not exp  | Simple `exceed_len / buffer_len`, capped at `-penalty_factor`                           |
| Advantage is still group-wise normalized  | This part is identical to GRPO; DAPO does not change it                                 |
