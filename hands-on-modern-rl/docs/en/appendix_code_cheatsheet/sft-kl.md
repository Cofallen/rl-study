---
title: B.1 SFT and KL Divergence
---

# B.1 SFT and KL Divergence

> Skim this once in the 30 minutes before an interview. For each item, memorize one sentence plus one formula. That is usually enough.

This appendix covers the algorithms that are most frequently asked to be handwritten in LLM post-training / RLHF interviews, ordered roughly by how often they show up. Each topic is presented from four angles:

| View                | What It Is For                                               |
| ------------------- | ------------------------------------------------------------ |
| **One-line memory** | The short mantra you can recite before walking into the room |
| **Pseudocode**      | The whiteboard version                                       |
| **Python**          | Explaining the logic with NumPy / plain Python               |
| **PyTorch**         | The engineering version interviewers often probe             |

## Contents

| Section                                       | Topic                                                          | Frequency |
| --------------------------------------------- | -------------------------------------------------------------- | --------- |
| [B.1 SFT Loss and KL Divergence](./sft-kl)    | autoregressive SFT loss, shift-right, KL estimates             | 4/5       |
| [B.2 PPO Policy Loss and GAE](./ppo-gae)      | clipped surrogate, value loss, reverse-time GAE recursion      | 5/5       |
| [B.3 DPO and Variants](./dpo-family)          | DPO loss, IPO, KTO, SimPO                                      | 5/5       |
| [B.4 GRPO and Reward Models](./grpo-rlvr)     | group-wise normalization in GRPO, Bradley-Terry reward model   | 4/5       |
| [B.5 DAPO](./dapo)                            | decoupled clipping, dynamic sampling, overlong penalty shaping | 3/5       |
| [B.6 Softmax and Cross-Entropy](./softmax-ce) | numerically stable softmax, log-sum-exp, CE loss               | 4/5       |
| [B.7 Top-k / Top-p Sampling](./top-k-top-p)   | temperature, top-k, top-p (nucleus) decoding                   | 4/5       |
| [B.8 Attention / MHA / GQA](./attention-mha)  | scaled dot-product attention, multi-head attention, MQA, GQA   | 5/5       |

## How To Use This Appendix

1. Start by memorizing the one-line mantra. Each file opens with a short sentence that is enough to reconstruct the pseudocode.
2. Prioritize pseudocode. In a whiteboard interview, pseudocode plus clear variable definitions is often sufficient.
3. Use the PyTorch snippet for details. If the interviewer asks about implementation specifics (for example `ignore_index`, `log_sum_exp`, `clamp`), jump to the PyTorch section.
4. Review the “Common Pitfalls.” Each file ends with a short list of high-frequency mistakes. Read those the night before.

## SFT Loss (Autoregressive Cross-Entropy)

**Core problem**: predict the next token at every position, computing loss only on the answer part.

**Core variables**:

- `logits`: model output, shape `[B, seq_len, vocab_size]`; position $t$ predicts $t+1$
- `labels`: ground-truth token ids; prompt tokens marked with `ignore_index=-100`
- `ignore_index`: cross-entropy skips this index (default `-100`)

### One-Line Memory

> Cut the tail of logits, the head of labels: position $t$ predicts $t+1$. Mask prompt positions with `-100` so they don't enter the loss.

### Pseudocode

```
logits = model(input_ids)                # position t predicts t+1
shift_logits = logits[:, :-1, :]         # cut tail: no "next" after end
shift_labels = labels[:, 1:]             # cut head: nobody predicts the first
loss = cross_entropy(shift_logits, shift_labels, ignore_index=-100)
```

An autoregressive model predicts the token at position $t+1$ from the prefix up to $t$, so logits index $t$ aligns with labels index $t+1$.

### Python Implementation

```python
import numpy as np

def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)  # subtract max first to avoid overflow
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def sft_loss(logits, labels, ignore_index=-100):
    """
    logits: [seq_len, vocab_size]
    labels: [seq_len]  (unshifted)
    """
    shift_logits = logits[:-1]
    shift_labels = labels[1:]

    probs = softmax(shift_logits, axis=-1)
    total, count = 0.0, 0
    for t in range(len(shift_labels)):
        if shift_labels[t] == ignore_index:
            continue
        total += -np.log(probs[t, shift_labels[t]] + 1e-12)
        count += 1
    return total / max(count, 1)
```

### PyTorch Implementation

```python
import torch
import torch.nn.functional as F

def sft_loss(logits, labels, ignore_index=-100):
    """
    logits: [B, seq_len, vocab_size]
    labels: [B, seq_len]
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )
```

---

## KL Divergence Estimates

**Core problem**: estimate the gap between the current policy $p$ and a reference policy $q$, used as the KL penalty in PPO / GRPO.

**Core variables**:

- `log_probs`: log-probabilities of sampled tokens under the current policy $p$
- `ref_log_probs`: log-probabilities of the same tokens under the reference policy $q$ (usually a frozen SFT model)
- `log_ratio`: $\log(q/p)$, the core quantity for k3

### One-Line Memory

> k1: `mean(log_p − log_q)`, simple and unbiased but can go negative; k3: `mean(exp(Δ) − 1 − Δ)`, $\Delta=\log\frac{q}{p}$, always nonnegative.

### Pseudocode

```
# k1 (common in PPO): plain average, unbiased but high variance, can go negative
kl = (log_probs - ref_log_probs).mean()

# k3 (default in GRPO / trl): always nonnegative, ratio direction q/p
log_ratio = ref_log_probs - log_probs        # log(q/p)
kl = (exp(log_ratio) - 1 - log_ratio).mean()
```

### Python Implementation

```python
import numpy as np

def kl_k1(log_p, log_q):
    """E_p[log p - log q]: unbiased, high variance, can be negative with few samples."""
    return np.mean(log_p - log_q)

def kl_k3(log_p, log_q):
    """E_p[exp(log q - log p) - 1 - (log q - log p)]: unbiased and always nonnegative."""
    log_ratio = log_q - log_p
    return np.mean(np.exp(log_ratio) - 1 - log_ratio)
```

### PyTorch Implementation

```python
import torch

def kl_penalty(log_probs, ref_log_probs, mode="k3"):
    """
    log_probs:     [B, seq_len]  current policy p
    ref_log_probs: [B, seq_len]  reference policy q
    """
    if mode == "k1":
        return (log_probs - ref_log_probs).mean()

    log_ratio = ref_log_probs - log_probs   # log(q/p)
    return (torch.exp(log_ratio) - 1 - log_ratio).mean()
```

### Comparing the Two Estimators

Samples come from $p$; the target is $\text{KL}(p \| q)$:

| Estimator | Formula                                            | Notes                                                  |
| --------- | -------------------------------------------------- | ------------------------------------------------------ |
| k1        | $\mathbb{E}_p[\log \frac{p}{q}]$                   | unbiased, simple, can be negative with limited samples |
| k3        | $\mathbb{E}_p[\frac{q}{p} - 1 - \log \frac{q}{p}]$ | unbiased, always $\geq 0$, default in GRPO             |

::: warning Pitfall
In k3 the ratio must be $q/p$ (ref/current). Since $e^x - 1 - x \geq 0$ for every real $x$, this guarantees nonnegativity; flipping it to $p/q$ keeps the value nonnegative but the expectation is no longer $\text{KL}(p \| q)$.
:::

---

## Common Pitfalls

| Pitfall                  | Explanation                                                                        |
| ------------------------ | ---------------------------------------------------------------------------------- |
| Shift direction reversed | Cut the **tail** of logits and the **head** of labels: position $t$ predicts $t+1$ |
| Forgot `ignore_index`    | Prompt tokens are marked `-100` and excluded from the loss                         |
| k3 ratio reversed        | Must be $q/p$ (ref/current); flipping it biases the expectation                    |
| k1 with too few samples  | A single batch can yield a negative estimate — that is sampling noise, not a bug   |
| Softmax overflow         | Subtract `max(x)` before `exp`                                                     |
| Missing `.contiguous()`  | PyTorch `view` on a slice may fail; add `.contiguous()`                            |
